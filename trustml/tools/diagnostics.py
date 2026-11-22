"""Model diagnostics on OUT-OF-FOLD training predictions (the test set is never used):

- screen_interactions: does a numeric feature act differently inside one category level?
- fairness_report:     error rates per group of a (possibly sensitive) column, with a bootstrap CI on the gap.
- calibration_report:  are predicted probabilities trustworthy? (Brier score, ECE, reliability table)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from trustml.io import load_predictions
from trustml.schemas import TaskType

SENSITIVE_TOKENS = {"gender", "sex", "race", "ethnicity", "ethnic", "religion", "religious", "nationality",
                    "marital", "disability", "disabled"}
_AGE_OK = {"age", "group", "band", "bucket", "bin", "applicant", "customer", "client", "patient", "person", "member",
           "user", "employee", "borrower", "of", "at"}
_AGE_NOT = {"house", "home", "building", "property", "vehicle", "car", "account", "loan", "garage",
            "roof", "machine", "equipment", "policy", "contract"}


def _tokens(name: str) -> list[str]:
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
    return [t for t in re.split(r"[^0-9a-zA-Z]+", name.lower()) if t]


def _oof(ctx, exp_id: str) -> pd.DataFrame:
    e = ctx.experiments[exp_id]
    return load_predictions(ctx.run_dir / e.oof_path).set_index("row_index")


def _masked_numeric(ctx, col: str, sentinels: dict) -> pd.Series:
    x = pd.to_numeric(ctx.data[col].astype(str).str.strip(), errors="coerce") \
        if ctx.kinds.get(col) == "numeric_text" else ctx.data[col].astype(float)
    vals = (sentinels or {}).get(col)
    return x.mask(x.isin(vals)) if vals else x


# ---------------------------------------------------------------- interactions
def screen_interactions(ctx, exp_id: str, exclude: list[str], sentinels: dict, z_crit: float = 3.0,
                        top_k: int = 3, min_diff: float = 0.15) -> dict:
    """For each (category level L, numeric feature X): compare corr(X, residual) inside L vs outside L.
    A candidate needs both a significant difference (Fisher z-test, |z| >= z_crit) and a material one
    (|r_in - r_out| >= min_diff): on large data tiny differences are 'significant' but not worth testing."""
    oof = _oof(ctx, exp_id)
    resid = (oof["y_true"] - (oof["y_score"] if "y_score" in oof else oof["y_pred"])).astype(float)
    data = ctx.data.loc[oof.index]
    cats = [c for c in ctx.candidate_features if c not in exclude and ctx.kinds.get(c) in ("categorical", "boolean")
            and 2 <= data[c].nunique() <= 12]
    nums = [c for c in ctx.candidate_features if c not in exclude and ctx.kinds.get(c) in ("numeric", "numeric_text")
            and data[c].nunique() > 10]
    rows = []
    for c in cats:
        g = data[c].astype(str)
        for level in g.unique():
            inside = (g == level).to_numpy()
            n_in, n_out = int(inside.sum()), int((~inside).sum())
            if n_in < max(50, 0.03 * len(g)) or n_out < 50:
                continue
            if ctx.task == TaskType.BINARY and oof["y_true"].to_numpy()[inside].sum() < 15:
                continue
            for x in nums:
                xv = _masked_numeric(ctx, x, sentinels).loc[oof.index].to_numpy()
                ok = ~np.isnan(xv)
                a, b = ok & inside, ok & ~inside
                if a.sum() < 40 or b.sum() < 40 or np.std(xv[a]) == 0 or np.std(xv[b]) == 0:
                    continue
                r_in = float(np.corrcoef(xv[a], resid.to_numpy()[a])[0, 1])
                r_out = float(np.corrcoef(xv[b], resid.to_numpy()[b])[0, 1])
                z = (np.arctanh(np.clip(r_in, -0.999, 0.999)) - np.arctanh(np.clip(r_out, -0.999, 0.999))) / \
                    np.sqrt(1 / (a.sum() - 3) + 1 / (b.sum() - 3))
                rows.append({"feature": x, "group_col": c, "level": str(level), "n_in": int(a.sum()),
                             "r_in": round(r_in, 4), "r_out": round(r_out, 4), "z": round(float(z), 2)})
    rows.sort(key=lambda r: -abs(r["z"]))
    return {"exp_id": exp_id, "method": "Fisher z-test of corr(feature, OOF residual) inside vs outside a category level",
            "z_threshold": z_crit, "min_corr_diff": min_diff,
            "candidates": [r for r in rows if abs(r["z"]) >= z_crit and abs(r["r_in"] - r["r_out"]) >= min_diff][:top_k],
            "n_tested": len(rows), "top10": rows[:10]}


def group_metric(ctx, exp_id: str, col: str, level: str) -> dict:
    """OOF metric of one group (recall for binary, MAE for regression)."""
    oof = _oof(ctx, exp_id)
    g = ctx.data.loc[oof.index, col].astype(str) == level
    if ctx.task == TaskType.BINARY:
        pos = g & (oof["y_true"] == 1)
        return {"metric": "recall", "value": float((oof.loc[pos, "y_pred"] == 1).mean()), "n": int(pos.sum())}
    err = (oof["y_true"] - oof["y_pred"]).abs()
    return {"metric": "mae", "value": float(err[g].mean()), "n": int(g.sum())}


# ---------------------------------------------------------------- fairness
def detect_sensitive(columns: list[str]) -> list[str]:
    """Conservative name-based detection of protected attributes (whole-token matches only)."""
    out = []
    for c in columns:
        t = _tokens(c)
        if set(t) & SENSITIVE_TOKENS or t == ["senior", "citizen"]:
            out.append(c)
        elif "age" in t and set(t) <= _AGE_OK and not set(t) & _AGE_NOT:
            out.append(c)
    return out


def fairness_report(ctx, exp_id: str, column: str, n_boot: int = 300, min_pos: int = 20) -> dict:
    oof = _oof(ctx, exp_id)
    raw = ctx.data.loc[oof.index, column]
    edges = None
    if pd.api.types.is_numeric_dtype(raw) and raw.nunique() > 12:
        cats, bins = pd.qcut(raw, 4, duplicates="drop", retbins=True)
        g = cats.astype(str).where(cats.notna(), "__missing__")
        edges = {str(iv): [float(bins[i]), float(bins[i + 1]), i == 0] for i, iv in enumerate(cats.cat.categories)}
    else:
        g = raw.astype(str).where(raw.notna(), "__missing__")
    y, p = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy()
    groups = []
    for level in sorted(g.unique()):
        m = (g == level).to_numpy()
        row = {"group": str(level), "n": int(m.sum()), "interval": (edges or {}).get(str(level))}
        if ctx.task == TaskType.BINARY:
            pos, neg = m & (y == 1), m & (y == 0)
            row.update(n_pos=int(pos.sum()), actual_rate=round(float(y[m].mean()), 4),
                       flagged_rate=round(float(p[m].mean()), 4),
                       recall=round(float(p[pos].mean()), 4) if pos.any() else None,
                       false_alarm_rate=round(float(p[neg].mean()), 4) if neg.any() else None)
        else:
            err = y[m] - p[m]
            mae, scale = float(np.abs(err).mean()), float(np.abs(y[m]).mean())
            row.update(mae=round(mae, 4), mean_error=round(float(err.mean()), 4), mean_actual=round(scale, 4),
                       relative_mae=round(mae / scale, 4) if scale else None)
        groups.append(row)
    res = {"exp_id": exp_id, "column": column, "source": "out-of-fold training predictions", "groups": groups}
    rng = np.random.default_rng(ctx.seed)
    if ctx.task == TaskType.BINARY:
        ok = [r for r in groups if (r.get("n_pos") or 0) >= min_pos and r["recall"] is not None]
        if len(ok) >= 2:
            hi, lo = max(ok, key=lambda r: r["recall"]), min(ok, key=lambda r: r["recall"])
            gaps = []
            mh, ml = (g == hi["group"]).to_numpy() & (y == 1), (g == lo["group"]).to_numpy() & (y == 1)
            ih, il = np.where(mh)[0], np.where(ml)[0]
            for _ in range(n_boot):
                gaps.append(p[rng.choice(ih, len(ih))].mean() - p[rng.choice(il, len(il))].mean())
            ci = (round(float(np.percentile(gaps, 2.5)), 4), round(float(np.percentile(gaps, 97.5)), 4))
            fa = [r for r in groups if r.get("false_alarm_rate") is not None and r["n"] - (r.get("n_pos") or 0) >= min_pos]
            fa_gap = fa_ci = fa_hi = fa_lo = None
            if len(fa) >= 2:
                h2, l2 = max(fa, key=lambda r: r["false_alarm_rate"]), min(fa, key=lambda r: r["false_alarm_rate"])
                fa_hi, fa_lo = h2["group"], l2["group"]
                fa_gap = round(h2["false_alarm_rate"] - l2["false_alarm_rate"], 4)
                jh = np.where((g == fa_hi).to_numpy() & (y == 0))[0]
                jl = np.where((g == fa_lo).to_numpy() & (y == 0))[0]
                bs = [p[rng.choice(jh, len(jh))].mean() - p[rng.choice(jl, len(jl))].mean() for _ in range(n_boot)]
                fa_ci = (round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4))
            rec_mat = bool(hi["recall"] - lo["recall"] >= 0.10 and ci[0] > 0)
            fa_mat = bool(fa_gap is not None and fa_gap >= 0.10 and fa_ci[0] > 0)
            res.update(metric="recall", best_group=hi["group"], worst_group=lo["group"],
                       gap=round(hi["recall"] - lo["recall"], 4), gap_ci95=ci, false_alarm_gap=fa_gap,
                       false_alarm_gap_ci95=fa_ci, false_alarm_highest=fa_hi, false_alarm_lowest=fa_lo,
                       recall_gap_material=rec_mat, false_alarm_gap_material=fa_mat, material=rec_mat or fa_mat)
        else:
            res.update(metric="recall", material=False, note=f"fewer than 2 groups with >= {min_pos} positives")
    else:
        ok = [r for r in groups if r["n"] >= 30 and r.get("relative_mae") is not None]
        if len(ok) >= 2:
            # judged on RELATIVE error (MAE / mean actual value in the group): a group with larger values has
            # larger absolute errors by construction, which is not a fairness problem on its own
            hi, lo = max(ok, key=lambda r: r["relative_mae"]), min(ok, key=lambda r: r["relative_mae"])
            amax, amin = max(ok, key=lambda r: r["mae"]), min(ok, key=lambda r: r["mae"])
            res.update(metric="mae", worst_group=hi["group"], best_group=lo["group"],
                       gap=round(amax["mae"] - amin["mae"], 4),
                       ratio=round(amax["mae"] / amin["mae"], 3) if amin["mae"] else None,
                       relative_ratio=round(hi["relative_mae"] / lo["relative_mae"], 3) if lo["relative_mae"] else None,
                       material=bool(lo["relative_mae"] and hi["relative_mae"] / lo["relative_mae"] >= 1.25))
    return res


# ---------------------------------------------------------------- calibration
def calibration_report(ctx, exp_id: str, n_bins: int = 10) -> dict:
    oof = _oof(ctx, exp_id)
    y, s = oof["y_true"].to_numpy().astype(float), oof["y_score"].to_numpy().astype(float)
    brier = float(np.mean((s - y) ** 2))
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, n_bins - 1)
    table, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        mp, obs = float(s[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(mp - obs)
        table.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(m.sum()), "mean_predicted": round(mp, 4),
                      "observed_rate": round(obs, 4)})
    e = ctx.experiments[exp_id]
    return {"exp_id": exp_id, "brier": round(brier, 6), "ece": round(float(ece), 6), "n_bins": n_bins, "table": table,
            "mean_predicted": round(float(s.mean()), 4), "actual_rate": round(float(y.mean()), 4),
            "class_weighted": e.spec.class_weight == "balanced", "source": "out-of-fold training predictions",
            "well_calibrated": bool(ece < 0.05)}
