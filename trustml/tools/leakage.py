"""Potential-leakage screen.

This produces SUSPICIONS, not proof. A feature is suspicious when, on its own,
it predicts the target almost perfectly, or far better than every other
feature. Confirming leakage needs domain knowledge (was the value available at
prediction time?) and an ablation experiment (milestone 2).

Call this on TRAINING data only, so the held-out test set stays untouched.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from trustml.schemas import AuditReport, DatasetProfile, Finding, Severity, TaskType

POST_OUTCOME_RE = re.compile(
    r"(cancel|terminat|churn|refund|closed|outcome|result|final|status|post_|_after|after_|resolution|"
    r"chargeback|collection|writeoff|write_off|default)", re.I)
SKIP_KINDS = {"id_like", "constant", "text", "datetime"}


def _single_feature_score(x: pd.Series, y: np.ndarray, task: TaskType, seed: int = 0) -> float:
    is_num = pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
    X = x.to_frame()
    if is_num:
        pre = ColumnTransformer([("num", SimpleImputer(strategy="median", add_indicator=True), [x.name])])
    else:
        X = X.astype("object").where(X.notna(), "__MISSING__").astype(str)
        pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), [x.name])])

    if task == TaskType.REGRESSION:
        model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=seed)
        cv, scoring = KFold(5, shuffle=True, random_state=seed), "r2"
    else:
        model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=seed)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        scoring = "roc_auc" if task == TaskType.BINARY else "roc_auc_ovr"
    pipe = Pipeline([("pre", pre), ("model", model)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
    score = float(np.mean(scores))
    if task == TaskType.BINARY:
        score = max(score, 1 - score)  # direction-agnostic
    return score


def screen_leakage(df: pd.DataFrame, profile: DatasetProfile, high: float = 0.97, medium: float = 0.90,
                   gap: float = 0.15) -> AuditReport:
    if profile.target is None:
        raise ValueError("Leakage screen needs a target.")
    tgt, task = profile.target.name, profile.target.task_type
    data = df[df[tgt].notna()]
    y_raw = data[tgt]
    y = y_raw.to_numpy() if task == TaskType.REGRESSION else LabelEncoder().fit_transform(y_raw.astype(str))

    kinds = {c.name: c.inferred_kind for c in profile.columns}
    candidates = [c for c in data.columns if c != tgt and kinds.get(c) not in SKIP_KINDS]
    def col(c: str) -> pd.Series:
        return pd.to_numeric(data[c].astype(str).str.strip(), errors="coerce") if kinds.get(c) == "numeric_text" else data[c]

    scores = {c: round(_single_feature_score(col(c), y, task), 4) for c in candidates}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    metric = "r2" if task == TaskType.REGRESSION else "roc_auc"

    findings: list[Finding] = []
    for i, (col, sc) in enumerate(ranked):
        next_best = ranked[i + 1][1] if i + 1 < len(ranked) else None
        name_hint = bool(POST_OUTCOME_RE.search(col))
        sev = None
        reasons = []
        if sc >= high:
            sev, reasons = Severity.HIGH, [f"single-feature {metric}={sc:.3f} ≥ {high}"]
        elif sc >= medium:
            sev, reasons = Severity.MEDIUM, [f"single-feature {metric}={sc:.3f} ≥ {medium}"]
        elif i == 0 and next_best is not None and sc - next_best >= gap and sc >= 0.80:
            sev, reasons = Severity.MEDIUM, [f"{metric}={sc:.3f} exceeds next-best feature by {sc-next_best:.3f}"]
        if sev is None:
            continue
        if name_hint:
            reasons.append("column name suggests a post-outcome value")
        findings.append(Finding(
            check="suspicious_predictor", severity=sev, columns=[col],
            message=f"'{col}' alone predicts '{tgt}' unusually well ({'; '.join(reasons)}). Possible target leakage.",
            evidence={"metric": metric, "single_feature_score": sc, "rank": i + 1,
                      "next_best_score": next_best, "name_hint_post_outcome": name_hint,
                      "method": "5-fold CV, depth-4 decision tree on this feature only"},
            recommendation="Check whether this value is known at prediction time. Then run an ablation "
                           "(retrain without it) and compare. High predictive power alone does not prove leakage.",
        ))

    # Name-only hints, reported as INFO so a human can review them
    for col in candidates:
        if POST_OUTCOME_RE.search(col) and col not in {c for f in findings for c in f.columns}:
            findings.append(Finding(
                check="post_outcome_name_hint", severity=Severity.INFO, columns=[col],
                message=f"'{col}' has a name that may indicate a post-outcome value, but its predictive power is "
                        f"not unusual ({metric}={scores[col]:.3f}).",
                evidence={"single_feature_score": scores[col]},
                recommendation="Confirm timing with the data owner.",
            ))

    findings.append(Finding(
        check="single_feature_ranking", severity=Severity.INFO,
        message=f"Single-feature {metric} for {len(ranked)} candidate features (excluded: id/constant/text/datetime).",
        evidence={"ranking": dict(ranked)},
    ))
    return AuditReport(tool="screen_leakage", findings=findings)
