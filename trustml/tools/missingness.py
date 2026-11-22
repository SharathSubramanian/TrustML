"""B2: is missingness associated with the target? (training split only)

Informative missingness is not an error, but it matters: the model can learn from
the missingness pattern, and if values go missing BECAUSE of the outcome, that is
a form of leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from trustml.schemas import AuditReport, DatasetProfile, Finding, Severity, TaskType


def missingness_vs_target(train: pd.DataFrame, profile: DatasetProfile, min_pct: float = 1.0,
                          alpha: float = 0.01) -> AuditReport:
    tp = profile.target
    findings: list[Finding] = []
    y = train[tp.name]
    if tp.task_type == TaskType.BINARY:
        y = (y.astype(str) == tp.minority_class).astype(int)
    for c in profile.columns:
        if c.name == tp.name or c.pct_missing < min_pct or c.name not in train:
            continue
        miss = train[c.name].isna()
        if miss.sum() < 10 or (~miss).sum() < 10:
            continue
        if tp.task_type == TaskType.REGRESSION:
            a, b = y[miss].astype(float), y[~miss].astype(float)
            stat = stats.ttest_ind(a, b, equal_var=False)
            diff, p = float(a.mean() - b.mean()), float(stat.pvalue)
            ev = {"target_mean_when_missing": float(a.mean()), "target_mean_when_present": float(b.mean())}
            material = abs(diff) >= 0.1 * float(y.std())
        else:
            if tp.task_type == TaskType.MULTICLASS:
                table = pd.crosstab(miss, train[tp.name])
                p = float(stats.chi2_contingency(table)[1]); diff = float("nan"); material = True
                ev = {"contingency": {str(k): v for k, v in table.to_dict().items()}}
            else:
                r1, r0 = float(y[miss].mean()), float(y[~miss].mean())
                table = pd.crosstab(miss, y)
                p = float(stats.chi2_contingency(table)[1]); diff = r1 - r0
                ev = {"target_rate_when_missing": round(r1, 4), "target_rate_when_present": round(r0, 4)}
                material = abs(diff) >= 0.03
        ev.update({"n_missing_train": int(miss.sum()), "p_value": p, "test": "Welch t" if tp.task_type == TaskType.REGRESSION else "chi-square"})
        if p < alpha and material:
            findings.append(Finding(
                check="informative_missingness", severity=Severity.MEDIUM, columns=[c.name],
                message=(f"Missingness in '{c.name}' is associated with '{tp.name}' (p={p:.2g}"
                         + (f"; target rate {ev['target_rate_when_missing']:.1%} when missing vs "
                            f"{ev['target_rate_when_present']:.1%} when present" if "target_rate_when_missing" in ev else "")
                         + ")."),
                evidence=ev,
                recommendation="Keep a missing-indicator (TrustML's pipeline adds one). Ask why values are missing: "
                               "if they go missing because of the outcome, this is leakage.",
            ))
    return AuditReport(tool="missingness_vs_target", findings=findings)
