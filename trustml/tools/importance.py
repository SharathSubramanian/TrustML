"""Permutation importance per CV fold (training split only).

Importance is measured on each fold's validation part for the model fitted on
that fold's training part, so it reflects generalisation, not memorisation.
It identifies candidates for ablation; the ablation experiment is what tests
whether removing them actually costs anything.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.inspection import permutation_importance

from trustml.schemas import TaskType


def cv_permutation_importance(ctx, exp_id: str, n_repeats: int = 3) -> dict:
    e = ctx.experiments[exp_id]
    feats = e.features_used
    X_tr, y_tr = ctx.data.iloc[ctx.train_pos][feats], ctx.y[ctx.train_pos]
    groups = ctx.data.iloc[ctx.train_pos][ctx.group_col].astype(str) if ctx.group_col else None
    scoring = {TaskType.BINARY: "roc_auc", TaskType.REGRESSION: "r2", TaskType.MULTICLASS: "balanced_accuracy"}[ctx.task]
    per_fold = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a, b in ctx._cv().split(X_tr, y_tr, groups):
            model = ctx._pipeline(e.spec, feats).fit(X_tr.iloc[a], y_tr[a])
            r = permutation_importance(model, X_tr.iloc[b], y_tr[b], scoring=scoring, n_repeats=n_repeats,
                                       random_state=ctx.seed, n_jobs=1)
            per_fold.append(r.importances_mean)
    M = np.array(per_fold)
    rows = [{"feature": f, "mean": round(float(M[:, i].mean()), 5), "std": round(float(M[:, i].std(ddof=1)), 5),
             "folds_positive": int((M[:, i] > 0).sum())} for i, f in enumerate(feats)]
    rows.sort(key=lambda r: r["mean"], reverse=True)
    k = len(per_fold)
    noise = [r["feature"] for r in rows if r["mean"] < 0.002 or (r["folds_positive"] <= k // 2 and r["mean"] < 0.005)]
    return {"exp_id": exp_id, "scoring": scoring, "n_folds": k, "n_repeats": n_repeats, "ranking": rows,
            "noise_candidates": noise,
            "rule": "noise = mean importance < 0.002, or < 0.005 and positive in at most half the folds"}
