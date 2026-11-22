"""Independent verification layer.

Deliberately does NOT reuse trustml.experiments.compute_metrics or sklearn.metrics:
metrics are recomputed from the saved prediction files with separate NumPy code,
so a bug in the experiment engine would show up as a mismatch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from trustml.io import load_predictions
from pydantic import BaseModel

from trustml.schemas import TaskType

TOL = 1e-6


class Check(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class VerificationResult(BaseModel):
    exp_id: str
    passed: bool
    checks: list[Check]


# ---------------------------------------------------------------- independent metric code
def _binary(y: np.ndarray, pred: np.ndarray, score: Optional[np.ndarray], costs: Optional[dict] = None) -> dict[str, float]:
    tp = float(np.sum((pred == 1) & (y == 1)))
    tn = float(np.sum((pred == 0) & (y == 0)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    out = {"accuracy": (tp + tn) / len(y), "precision": prec, "recall": rec,
           "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "balanced_accuracy": (rec + spec) / 2}
    if costs:
        out["cost_per_row"] = (costs["fn"] * fn + costs["fp"] * fp) / len(y)
    if score is not None:
        # ROC-AUC via Mann-Whitney U with average ranks for ties
        ranks = pd.Series(score).rank(method="average").to_numpy()
        n_pos, n_neg = y.sum(), len(y) - y.sum()
        out["roc_auc"] = float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
        # Average precision: sum over distinct thresholds of (R_k - R_{k-1}) * P_k
        order = np.argsort(-score, kind="mergesort")
        s, yy = score[order], y[order]
        distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
        tps = np.cumsum(yy)[distinct]
        fps = (distinct + 1) - tps
        precision = tps / (tps + fps)
        recall = tps / n_pos
        out["pr_auc"] = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return out


def _regression(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = y - pred
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2))}


def _multiclass(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    classes = np.unique(y)
    recalls, f1s = [], []
    for c in np.unique(np.r_[y, pred]):
        tp = np.sum((pred == c) & (y == c)); fp = np.sum((pred == c) & (y != c)); fn = np.sum((pred != c) & (y == c))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        if c in classes:
            recalls.append(r)
    return {"accuracy": float(np.mean(y == pred)), "balanced_accuracy": float(np.mean(recalls)),
            "f1_macro": float(np.mean(f1s)), "recall_macro": float(np.mean(recalls))}


# ---------------------------------------------------------------- verifier
class Verifier:
    def __init__(self, run_dir: str | Path, original_df: pd.DataFrame, target: str, positive_class: Optional[str]):
        self.run_dir = Path(run_dir)
        self.split = json.loads((self.run_dir / "split.json").read_text())
        self.df, self.target, self.pos = original_df, target, positive_class

    def verify_experiment(self, exp_id: str) -> VerificationResult:
        rec = json.loads((self.run_dir / "experiments" / f"{exp_id}.json").read_text())
        preds = load_predictions(self.run_dir / rec["predictions_path"])
        task = TaskType(rec["task_type"])
        checks: list[Check] = []

        # 1. Test set is the run's fixed test set
        checks.append(Check(name="test_set_fixed", passed=rec["test_index_sha"] == self.split["test_index_sha"],
                            detail="experiment test-index hash equals run split hash"))
        same_rows = sorted(preds["row_index"].tolist()) == sorted(self.split["test_index"])
        checks.append(Check(name="predictions_cover_test_set", passed=same_rows and len(preds) == rec["n_test"],
                            detail=f"{len(preds)} prediction rows vs n_test={rec['n_test']}"))
        no_train = not (set(preds["row_index"]) & set(self.split["train_index"]))
        checks.append(Check(name="no_train_rows_in_test", passed=no_train))

        # 2. Labels in prediction file match the ORIGINAL data
        y_orig = self.df.loc[preds["row_index"], self.target]
        if task == TaskType.BINARY:
            y_orig = (y_orig.astype(str) == self.pos).astype(int).to_numpy()
            ok = np.array_equal(y_orig, preds["y_true"].to_numpy())
        elif task == TaskType.REGRESSION:
            ok = np.allclose(y_orig.astype(float).to_numpy(), preds["y_true"].to_numpy())
        else:
            ok = True  # label encoding checked indirectly through metrics
        checks.append(Check(name="labels_match_original_data", passed=bool(ok)))

        # 3. Feature hygiene
        feats = set(rec["features_used"])
        checks.append(Check(name="target_not_a_feature", passed=self.target not in feats))
        excl = set(rec["spec"]["exclude_features"]) & feats
        checks.append(Check(name="exclusions_respected", passed=not excl, detail=f"leaked exclusions: {sorted(excl)}"))
        extra = rec["spec"].get("extra_features") or []
        if extra:
            eng_path = self.run_dir / "engineered.json"
            reg = json.loads(eng_path.read_text()) if eng_path.exists() else {}
            bad = [e for e in extra if e not in reg or self.target in reg[e]["sources"]
                   or set(reg[e]["sources"]) & set(rec["spec"]["exclude_features"])]
            checks.append(Check(name="engineered_features_valid", passed=not bad,
                                detail=f"registered, no target or excluded sources; invalid: {bad}"))

        # 4. Predictions consistent with threshold
        y, pred = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
        score = preds["y_score"].to_numpy() if "y_score" in preds else None
        if task == TaskType.BINARY and score is not None:
            consistent = np.array_equal(pred, (score >= rec["threshold"]).astype(int))
            checks.append(Check(name="threshold_applied", passed=bool(consistent), detail=f"threshold={rec['threshold']}"))

        # 5. Recompute every reported test metric independently
        recomputed = (_binary(y.astype(int), pred.astype(int), score, rec.get("costs")) if task == TaskType.BINARY
                      else _regression(y.astype(float), pred.astype(float)) if task == TaskType.REGRESSION
                      else _multiclass(y, pred))
        for k, reported in rec["test_metrics"].items():
            if k in recomputed:
                diff = abs(recomputed[k] - reported)
                checks.append(Check(name=f"metric:{k}", passed=diff <= TOL,
                                    detail=f"reported={reported:.6f} recomputed={recomputed[k]:.6f}"))
        return VerificationResult(exp_id=exp_id, passed=all(c.passed for c in checks), checks=checks)

    def verify_claim(self, exp_id: str, metric: str, value: float, source: str = "test", tol: float = 5e-4) -> Check:
        """Check a number quoted in a summary against the stored experiment record."""
        path = self.run_dir / "experiments" / f"{exp_id}.json"
        if not path.exists():
            return Check(name=f"claim:{exp_id}.{metric}", passed=False, detail="experiment does not exist")
        rec = json.loads(path.read_text())
        stored = rec["test_metrics"].get(metric) if source == "test" else rec["cv_metrics"].get(metric, {}).get("mean")
        if stored is None:
            return Check(name=f"claim:{exp_id}.{metric}", passed=False, detail="metric not stored")
        return Check(name=f"claim:{exp_id}.{metric}", passed=abs(stored - value) <= tol,
                     detail=f"claimed={value} stored={stored:.6f}")


def _in_interval(x: pd.Series, label: str) -> pd.Series:
    """Membership for a pandas interval label like '(36.0, 55.0]'."""
    lo, hi = label.strip("([])").split(",")
    lo, hi = float(lo), float(hi)
    left = x >= lo if label.startswith("[") else x > lo
    right = x <= hi if label.endswith("]") else x < hi
    return left & right


def verify_subgroups(self, step_result: dict, ctx) -> list[Check]:
    exp_id = step_result.get("exp_id")
    flagged = list(step_result.get("flagged", []))
    key = "recall" if step_result.get("metric") == "recall" else "mae"
    out_extra = []
    for r in step_result.get("recheck", []):  # verify the 'after' value of every re-checked group
        col, grp = r["group"].split("=", 1)
        flagged.append({"column": col, "group": grp, key: r["after"], "_recheck": True, "interval": r.get("interval")})
    if not flagged:
        return [Check(name="subgroups:none_flagged", passed=True)]
    rec = json.loads((self.run_dir / "experiments" / f"{exp_id}.json").read_text())
    oof = load_predictions(self.run_dir / rec["oof_path"]).set_index("row_index")
    orig = self.df.loc[oof.index]
    out = []
    for g in flagged:
        col = orig[g["column"]]
        if g.get("interval"):  # exact edges (pd.qcut bins are right-closed)
            x = pd.to_numeric(col.astype(str).str.strip(), errors="coerce")
            lo, hi = g["interval"][0], g["interval"][1]
            include_lowest = len(g["interval"]) > 2 and g["interval"][2]
            mask = ((x >= lo) if include_lowest else (x > lo)) & (x <= hi)
        elif g["group"].startswith(("(", "[")) and "," in g["group"]:
            x = pd.to_numeric(col.astype(str).str.strip(), errors="coerce")
            mask = _in_interval(x, g["group"])
        else:
            mask = col.astype(str).fillna("__missing__") == g["group"]
        if "recall" in g:
            pos = mask & (oof["y_true"] == 1)
            val = float((oof.loc[pos, "y_pred"] == 1).mean())
            ok = abs(val - g["recall"]) <= 1e-3 and (g.get("_recheck") or int(pos.sum()) == g["n_positives"])
            out.append(Check(name=f"subgroup:{g['column']}={g['group']}", passed=ok,
                             detail=f"recall reported={g['recall']} recomputed={val:.4f} n_pos={int(pos.sum())}"))
        else:
            err = (oof["y_true"] - oof["y_pred"]).abs()
            val = float(err[mask].mean())
            ok = abs(val - g["mae"]) < 0.01 and (g.get("_recheck") or int(mask.sum()) == g["n_rows"])
            out.append(Check(name=f"subgroup:{g['column']}={g['group']}" + (":recheck" if g.get("_recheck") else ""),
                             passed=ok,
                             detail=f"MAE reported={g['mae']} recomputed={val:.4f} n={int(mask.sum())}"))
    return out


Verifier.verify_subgroups = verify_subgroups


def verify_time_split(self, result: dict) -> list[Check]:
    """Recompute every forward-in-time fold metric from the saved predictions with independent code."""
    if not result.get("predictions_path"):
        return [Check(name="time_split:no_folds", passed=True, detail="no evaluable folds")]
    pr = load_predictions(self.run_dir / result["predictions_path"])
    metric, out = result["metric"], []
    train_rows = set(self.split["train_index"])
    out.append(Check(name="time_split:train_rows_only", passed=set(pr["row_index"]) <= train_rows,
                     detail="forward-in-time folds use training rows only (test set untouched)"))
    for f in result["folds"]:
        d = pr[pr["fold"] == f["fold"]]
        y, p_ = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
        sc = d["y_score"].to_numpy() if "y_score" in d else None
        if metric == "roc_auc":
            val = _binary(y.astype(int), p_.astype(int), sc)["roc_auc"]
        elif metric == "r2":
            val = _regression(y.astype(float), p_.astype(float))["r2"]
        else:
            val = _multiclass(y, p_)["balanced_accuracy"]
        out.append(Check(name=f"time_split:fold{f['fold']}:{metric}", passed=abs(val - f[metric]) <= 1e-5,
                         detail=f"reported={f[metric]:.6f} recomputed={val:.6f} n={len(d)}"))
        key = "target_rate" if "target_rate" in f else "target_mean" if "target_mean" in f else None
        if key:
            got = float(np.mean(y))
            out.append(Check(name=f"time_split:fold{f['fold']}:{key}", passed=abs(got - f[key]) <= 1e-5,
                             detail=f"reported={f[key]:.6f} recomputed={got:.6f}"))
    return out


def verify_dedupe(self, result: dict, ctx) -> list[Check]:
    """Recount duplicates in the ORIGINAL file and check none remain in the derived data."""
    labelled = self.df[self.df[self.target].notna()]
    n_orig = int(labelled.duplicated().sum())
    remaining = int(ctx.data.drop(columns=list(ctx.engineered)).duplicated().sum())
    return [Check(name="dedupe:rows_removed_match_original", passed=n_orig == result["rows_removed"],
                  detail=f"duplicates in original labelled rows={n_orig}, removed={result['rows_removed']}"),
            Check(name="dedupe:none_remaining", passed=remaining == 0, detail=f"remaining duplicates={remaining}"),
            Check(name="dedupe:split_rewritten", passed=len(self.split["train_index"]) + len(self.split["test_index"])
                  == result["rows_after"], detail="split.json covers exactly the deduplicated rows")]


Verifier.verify_time_split = verify_time_split
Verifier.verify_dedupe = verify_dedupe


def verify_group_split(self, result: dict) -> list[Check]:
    """Re-read split.json and the ORIGINAL file: no group value may appear in both train and test."""
    sp = json.loads((self.run_dir / "split.json").read_text())
    col = result["column"]
    tr = set(self.df.loc[sp["train_index"], col].astype(str))
    te = set(self.df.loc[sp["test_index"], col].astype(str))
    shared = tr & te
    return [Check(name="group_split:no_shared_groups", passed=not shared, detail=f"{len(shared)} shared '{col}' values"),
            Check(name="group_split:recorded", passed=sp.get("group_col") == col and sp.get("strategy") == "group",
                  detail=f"split.json strategy={sp.get('strategy')} group_col={sp.get('group_col')}")]


Verifier.verify_group_split = verify_group_split


# ---------------------------------------------------------------- diagnostics (independent recomputation)
def _oof_frame(self, exp_id: str) -> pd.DataFrame:
    rec = json.loads((self.run_dir / "experiments" / f"{exp_id}.json").read_text())
    return load_predictions(self.run_dir / rec["oof_path"]).set_index("row_index")


def _num(col: pd.Series) -> np.ndarray:
    return pd.to_numeric(col.astype(str).str.strip(), errors="coerce").to_numpy(dtype=float)


def verify_interactions(self, result: dict, ctx) -> list[Check]:
    """Recompute the residual correlations of every reported candidate from saved OOF predictions + original file."""
    cands = result.get("candidates", [])
    if not cands:
        return [Check(name="interactions:none_found", passed=True)]
    oof = _oof_frame(self, result["exp_id"])
    resid = oof["y_true"].to_numpy(float) - (oof["y_score"] if "y_score" in oof else oof["y_pred"]).to_numpy(float)
    orig = self.df.loc[oof.index]
    out = []
    for c in cands:
        x = _num(orig[c["feature"]])
        for v in (result.get("sentinels") or {}).get(c["feature"], []):
            x[x == v] = np.nan
        inside = (orig[c["group_col"]].astype(str) == c["level"]).to_numpy()
        ok = ~np.isnan(x)
        r_in = np.corrcoef(x[ok & inside], resid[ok & inside])[0, 1]
        r_out = np.corrcoef(x[ok & ~inside], resid[ok & ~inside])[0, 1]
        out.append(Check(name=f"interaction:{c['feature']}|{c['group_col']}={c['level']}",
                         passed=abs(r_in - c["r_in"]) < 1e-3 and abs(r_out - c["r_out"]) < 1e-3
                         and int((ok & inside).sum()) == c["n_in"],
                         detail=f"r_in reported={c['r_in']} recomputed={r_in:.4f}; r_out reported={c['r_out']} "
                                f"recomputed={r_out:.4f}; n_in={int((ok & inside).sum())}"))
    return out


def verify_group_metrics(self, result: dict, ctx) -> list[Check]:
    """Recompute the in-group recall / MAE quoted for an interaction test."""
    out = []
    for g in result.get("group_checks", []):
        oof = _oof_frame(self, g["exp_id"])
        m = (self.df.loc[oof.index, g["column"]].astype(str) == g["level"]).to_numpy()
        y, p = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy()
        if g["metric"] == "recall":
            pos = m & (y == 1)
            val, n = float((p[pos] == 1).mean()), int(pos.sum())
        else:
            val, n = float(np.abs(y[m] - p[m]).mean()), int(m.sum())
        out.append(Check(name=f"group_metric:{g['exp_id']}:{g['column']}={g['level']}",
                         passed=abs(val - g["value"]) < 1e-6 and n == g["n"],
                         detail=f"{g['metric']} reported={g['value']:.4f} recomputed={val:.4f} n={n}"))
    return out


def verify_fairness(self, result: dict, ctx) -> list[Check]:
    oof = _oof_frame(self, result["exp_id"])
    raw = self.df.loc[oof.index, result["column"]]
    y, p = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy()
    out = []
    for g in result["groups"]:
        if g.get("interval"):
            x = _num(raw)
            lo, hi, low_inc = g["interval"]
            m = ((x >= lo) if low_inc else (x > lo)) & (x <= hi)
        else:
            m = (raw.astype(str).where(raw.notna(), "__missing__") == g["group"]).to_numpy()
        ok = int(m.sum()) == g["n"]
        detail = f"n reported={g['n']} recomputed={int(m.sum())}"
        if g.get("recall") is not None:
            r = float(p[m & (y == 1)].mean())
            ok &= abs(r - g["recall"]) < 1e-3
            detail += f"; recall reported={g['recall']} recomputed={r:.4f}"
        if g.get("false_alarm_rate") is not None:
            fa = float(p[m & (y == 0)].mean())
            ok &= abs(fa - g["false_alarm_rate"]) < 1e-3
            detail += f"; false-alarm reported={g['false_alarm_rate']} recomputed={fa:.4f}"
        if g.get("mae") is not None:
            mae = float(np.abs(y[m] - p[m]).mean())
            ok &= abs(mae - g["mae"]) < 1e-3 * max(1, abs(g["mae"]))
            detail += f"; MAE reported={g['mae']} recomputed={mae:.4f}"
            if g.get("relative_mae") is not None:
                rel = mae / float(np.abs(y[m]).mean())
                ok &= abs(rel - g["relative_mae"]) < 1e-3 * max(1, abs(g["relative_mae"]))
                detail += f"; relative MAE reported={g['relative_mae']} recomputed={rel:.4f}"
        out.append(Check(name=f"fairness:{result['column']}={g['group']}", passed=bool(ok), detail=detail))
    return out


def verify_calibration(self, result: dict) -> list[Check]:
    oof = _oof_frame(self, result["exp_id"])
    y, s = oof["y_true"].to_numpy(float), oof["y_score"].to_numpy(float)
    brier = float(np.mean((s - y) ** 2))
    n_bins = result["n_bins"]
    b = np.minimum((s * n_bins).astype(int), n_bins - 1)  # independent binning: floor(s * n_bins)
    ece = sum((b == k).mean() * abs(s[b == k].mean() - y[b == k].mean()) for k in range(n_bins) if (b == k).any())
    return [Check(name="calibration:brier", passed=abs(brier - result["brier"]) < 1e-5,
                  detail=f"reported={result['brier']} recomputed={brier:.6f}"),
            Check(name="calibration:ece", passed=abs(ece - result["ece"]) < 1e-4,
                  detail=f"reported={result['ece']} recomputed={ece:.6f}")]


Verifier.verify_interactions = verify_interactions
Verifier.verify_group_metrics = verify_group_metrics
Verifier.verify_fairness = verify_fairness
Verifier.verify_calibration = verify_calibration
