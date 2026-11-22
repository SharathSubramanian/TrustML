"""B3: subgroup error analysis on out-of-fold TRAINING predictions.

Using OOF predictions keeps the test set out of every decision. Groups are the
levels of categorical/boolean columns (<= 20 levels) and quartile bins of numeric
columns. A group is flagged only if it has enough support and is clearly worse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from trustml.io import load_predictions
from scipy import stats

from trustml.schemas import TaskType

MIN_POS = 20       # binary: min positives in group
MIN_ROWS = 50      # regression: min rows in group


def _groups(ctx, row_index: pd.Index, features: list[str]) -> tuple[dict[str, pd.Series], dict[str, dict]]:
    """Group labels per column, plus EXACT bin edges for numeric bins (display labels are rounded)."""
    data = ctx.data.loc[row_index]
    out, edges = {}, {}
    for c in [f for f in features if f in ctx.candidate_features]:  # original columns used by the model
        s = data[c]
        kind = ctx.kinds.get(c)
        if kind in ("categorical", "boolean") and s.nunique() <= 20:
            out[c] = s.astype(str).fillna("__missing__")
        elif kind in ("numeric", "numeric_text") and s.nunique() > 10:
            try:
                cats, bins = pd.qcut(s, 4, duplicates="drop", retbins=True)
            except ValueError:
                continue
            out[c] = cats.astype(str).where(cats.notna(), "__missing__")
            # category labels are rounded (precision=3); rows are assigned with the exact bins:
            # bin i = (bins[i], bins[i+1]], and the first bin also includes its lowest value
            edges[c] = {str(iv): [float(bins[i]), float(bins[i + 1]), i == 0]
                        for i, iv in enumerate(cats.cat.categories)}
    return out, edges


def subgroup_errors(ctx, exp_id: str, z_crit: float = 3.0, max_flags: int = 5) -> dict:
    e = ctx.experiments[exp_id]
    oof = load_predictions(ctx.run_dir / e.oof_path).set_index("row_index")
    groups, edges = _groups(ctx, oof.index, e.features_used)
    rows, flagged = [], []
    if ctx.task == TaskType.BINARY:
        pos = oof["y_true"] == 1
        overall = float((oof.loc[pos, "y_pred"] == 1).mean())
        metric = "recall"
        for col, g in groups.items():
            for level in g.unique():
                m = (g == level) & pos
                n = int(m.sum())
                if n < MIN_POS:
                    continue
                r = float((oof.loc[m, "y_pred"] == 1).mean())
                se = np.sqrt(max(overall * (1 - overall), 1e-9) / n)
                z = (r - overall) / se
                rows.append({"column": col, "group": str(level), "n_positives": n, "recall": round(r, 4),
                             "z": round(float(z), 2), "interval": edges.get(col, {}).get(str(level))})
    elif ctx.task == TaskType.REGRESSION:
        err = (oof["y_true"] - oof["y_pred"]).abs()
        overall = float(err.mean())
        metric = "mae"
        for col, g in groups.items():
            for level in g.unique():
                m = g == level
                n = int(m.sum())
                if n < MIN_ROWS or n == len(g):
                    continue
                t = stats.ttest_ind(err[m], err[~m], equal_var=False)
                ratio = float(err[m].mean() / err[~m].mean())
                z = float(-np.sign(t.statistic) * stats.norm.isf(t.pvalue / 2)) if t.pvalue > 0 else -40.0
                rows.append({"column": col, "group": str(level), "n_rows": n, "mae": round(float(err[m].mean()), 4),
                             "mae_ratio_vs_rest": round(ratio, 3), "z": round(z, 2),
                             "interval": edges.get(col, {}).get(str(level))})
    else:
        return {"exp_id": exp_id, "supported": False, "reason": "subgroup analysis supports binary and regression"}

    for r in sorted(rows, key=lambda r: r["z"]):
        bad = r["z"] <= -z_crit and (ctx.task != TaskType.REGRESSION or r["mae_ratio_vs_rest"] >= 1.3)
        if bad and len(flagged) < max_flags:
            flagged.append(r)
    return {"exp_id": exp_id, "supported": True, "metric": metric, "overall": round(overall, 4),
            "source": "out-of-fold training predictions", "n_groups_tested": len(rows),
            "z_threshold": -z_crit, "flagged": flagged, "all_groups": rows}
