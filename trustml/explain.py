"""SHAP explanations for the final model (global + local), with an additivity check.

SHAP describes how THIS MODEL uses its inputs. It is not causal: a high SHAP
value does not mean changing the feature would change the outcome.
Explanations are computed on test rows for reporting only; no decision uses them.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from trustml.io import load_predictions

from trustml.schemas import TaskType


def _source_map(pre, features: list[str]) -> list[str]:
    """Map each transformed column back to its original feature."""
    out = []
    by_len = sorted(features, key=len, reverse=True)
    for name in pre.get_feature_names_out():
        n = name.split("__", 1)[1]
        n = n.replace("missingindicator_", "", 1)
        src = next((f for f in by_len if n == f or n.startswith(f + "_")), n)
        out.append(src)
    return out


def explain_model(ctx, exp_id: str, n_explain: int = 300, n_background: int = 200) -> dict:
    import shap

    pipe, feats = ctx._models[exp_id]
    if ctx.experiments[exp_id].spec.model == "random_forest":
        n_explain = min(n_explain, 150)  # TreeExplainer cost grows with rows x trees
    e = ctx.experiments[exp_id]
    pre, model = pipe.named_steps["pre"], pipe.named_steps["model"]
    rng = np.random.default_rng(ctx.seed)
    te = ctx.test_pos[rng.permutation(len(ctx.test_pos))[:n_explain]]
    tr = ctx.train_pos[rng.permutation(len(ctx.train_pos))[:n_background]]
    X_te_raw = ctx.data.iloc[te][feats]
    prep = pipe[:-1]  # sentinel fix + preprocessing, exactly as in training
    X_te = np.asarray(prep.transform(X_te_raw), dtype=float)
    X_bg = np.asarray(prep.transform(ctx.data.iloc[tr][feats]), dtype=float)
    log_target = hasattr(model, "regressor_")
    if log_target:
        model = model.regressor_  # explain the fitted regressor, i.e. in log1p(target) space
    sources = _source_map(pre, feats)
    binary = ctx.task == TaskType.BINARY
    multi = ctx.task == TaskType.MULTICLASS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if e.spec.model in ("random_forest", "gradient_boosting"):
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_te, check_additivity=False)
            base = explainer.expected_value
            if binary:
                per_class = isinstance(sv, list) or np.ndim(sv) == 3   # random forest: one output per class
                if per_class:
                    sv = sv[1] if isinstance(sv, list) else sv[..., 1]
                    base = np.atleast_1d(base)[-1]
                    model_out = model.predict_proba(X_te)[:, 1]
                    space = "probability"
                else:                                                  # gradient boosting: one log-odds output
                    base = float(np.atleast_1d(base)[0])
                    model_out = model.decision_function(X_te)
                    space = "log-odds"
            elif multi:
                if e.spec.model == "gradient_boosting":
                    model_out = model.decision_function(X_te)
                    space = "log-odds of the predicted class"
                else:
                    model_out = model.predict_proba(X_te)
                    space = "probability of the predicted class"
            else:
                base = float(np.atleast_1d(base)[0])
                model_out = model.predict(X_te)
                space = "log1p(prediction)" if log_target else "prediction"
        elif e.spec.model == "logreg":
            explainer = shap.LinearExplainer(model, shap.maskers.Independent(X_bg, max_samples=len(X_bg)))
            sv = explainer.shap_values(X_te)
            base = explainer.expected_value if multi else float(np.atleast_1d(explainer.expected_value)[0])
            model_out = model.decision_function(X_te) if (binary or multi) else model.predict(X_te)
            space = "log-odds" if binary else ("log-odds of the predicted class" if multi else
                                               ("log1p(prediction)" if log_target else "prediction"))
        else:
            return {"supported": False, "reason": "no explanation for the dummy model"}
    if multi:
        # explain each row's PREDICTED class: shap values (n, f, K) -> (n, f); base value per row
        sv3 = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv, dtype=float)
        k = np.argmax(model_out, axis=1)
        rows_ = np.arange(len(k))
        sv = sv3[rows_, :, k]
        base_rows = np.atleast_1d(np.asarray(base, dtype=float))[k]
        model_out = model_out[rows_, k]
        base = float(np.mean(base_rows))
    else:
        sv = np.asarray(sv, dtype=float)
        base = float(base)
        base_rows = np.full(len(sv), base)

    # aggregate expanded columns (one-hot levels, missing indicators) back to source features
    srcs = list(dict.fromkeys(sources))
    agg = np.column_stack([sv[:, [i for i, s in enumerate(sources) if s == f]].sum(axis=1) for f in srcs])

    additivity_err = float(np.max(np.abs(base_rows + sv.sum(axis=1) - model_out)))
    global_imp = sorted(({"feature": f, "mean_abs_shap": round(float(np.abs(agg[:, j]).mean()), 6)}
                         for j, f in enumerate(srcs)), key=lambda r: r["mean_abs_shap"], reverse=True)

    # local explanations: the cases a reader cares about
    y_true = ctx.y[te]
    pred_rows = load_predictions(ctx.run_dir / e.predictions_path).set_index("row_index")
    y_pred = pred_rows.loc[ctx.data.index[te], "y_pred"].to_numpy()
    picks = []
    if binary:
        fn = np.where((y_true == 1) & (y_pred == 0))[0]
        tp = np.where((y_true == 1) & (y_pred == 1))[0]
        if len(fn):
            picks.append(("missed positive (false negative)", int(fn[np.argmax(model_out[fn])])))
        if len(tp):
            picks.append(("correctly caught positive", int(tp[np.argmax(model_out[tp])])))
    elif multi:
        wrong = np.where(y_true != y_pred)[0]
        right = np.where(y_true == y_pred)[0]
        if len(wrong):
            picks.append(("misclassified row", int(wrong[0])))
        if len(right):
            picks.append(("correctly classified row", int(right[0])))
    else:
        err = np.abs(y_true - y_pred)
        picks.append(("largest error", int(np.argmax(err))))
        picks.append(("typical case", int(np.argsort(err)[len(err) // 2])))
    local = []
    for label, i in picks:
        order = np.argsort(-np.abs(agg[i]))[:6]
        local.append({
            "case": label, "row_index": int(ctx.data.index[te][i]), "y_true": float(y_true[i]),
            "y_pred": float(y_pred[i]), "model_output": float(model_out[i]), "output_space": space,
            "base_value": float(base_rows[i]),
            "contributions": [{"feature": srcs[j], "value": _val(X_te_raw.iloc[i][srcs[j]]) if srcs[j] in X_te_raw else None,
                               "shap": round(float(agg[i, j]), 6)} for j in order],
        })
    return {"supported": True, "exp_id": exp_id, "model": e.spec.model, "output_space": space,
            "n_rows_explained": int(len(te)), "base_value": base, "global": global_imp, "local": local,
            "additivity_max_error": additivity_err, "additivity_ok": additivity_err < 1e-4 * max(1.0, abs(base)) + 1e-3,
            "note": "SHAP explains model behaviour, not causation."}


def _val(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    return str(v)
