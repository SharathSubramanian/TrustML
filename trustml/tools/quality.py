"""Data-quality auditor. Reports issues with evidence; never modifies the data."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trustml.schemas import AuditReport, DatasetProfile, Finding, Severity, TaskType

SENTINELS = {-1, -9, -99, -999, -9999, 99, 999, 9999, 99999}


def _is_number(x: str) -> bool:
    try:
        float(x.replace(",", ""))
        return True
    except ValueError:
        return False


def _missing_severity(pct: float) -> Severity:
    if pct >= 30:
        return Severity.HIGH
    if pct >= 5:
        return Severity.MEDIUM
    return Severity.LOW


def audit_quality(df: pd.DataFrame, profile: DatasetProfile) -> AuditReport:
    findings: list[Finding] = []
    target = profile.target.name if profile.target else None
    n = len(df)

    # 0. File without a header row: column names look like data values
    names = [str(c) for c in df.columns]
    numeric_names = [c for c in names if _is_number(c)]
    value_like = [c for c in names if (df[c].astype(str).str.strip() == c.strip()).any()]
    if len(names) >= 3 and (len(numeric_names) / len(names) >= 0.3 or len(value_like) / len(names) >= 0.5):
        findings.append(Finding(
            check="possible_missing_header", severity=Severity.HIGH, columns=numeric_names[:5] or value_like[:5],
            message=f"The column names look like data values ({len(numeric_names)} numeric names; {len(value_like)} "
                    f"names also appear as values in their own column). The file may have no header row, in which "
                    f"case its first data row was used as column names.",
            evidence={"numeric_names": numeric_names[:10], "names_found_as_values": value_like[:10]},
            recommendation="Add a header row with real column names and re-run; until then column names in all "
                           "findings are first-row values and one row of data is missing.",
        ))

    # 1. Duplicate rows. Identical rows are only implausible (likely copy errors) if at least one column carries
    #    high information (an ID or continuous values); with only low-cardinality columns they are expected.
    if profile.n_duplicate_rows:
        dup_idx = df.index[df.duplicated(keep="first")].tolist()
        dup_frac = profile.n_duplicate_rows / n
        informative = [c.name for c in profile.columns if c.name != target and c.pct_unique >= 50]
        actionable = bool(informative) and dup_frac <= 0.30
        if actionable:
            sev, msg = Severity.MEDIUM, (f"{profile.n_duplicate_rows} exact duplicate rows ({100 * dup_frac:.2f}%). "
                                         f"Rows are identical even in high-information columns ({informative[:3]}), "
                                         f"so these are likely copy errors.")
            rec = "Deduplicate BEFORE splitting so copies cannot land in both train and test (the agent does this)."
        else:
            why = ("no column is high-information (IDs or continuous values), so identical rows are expected"
                   if not informative else f"they are {100 * dup_frac:.0f}% of the data - too many to remove automatically")
            sev, msg = Severity.INFO, (f"{profile.n_duplicate_rows} identical rows ({100 * dup_frac:.2f}%), but {why}. "
                                       f"Not removed.")
            rec = "Confirm with the data owner whether identical rows are real repeated cases or copies."
        findings.append(Finding(
            check="duplicate_rows", severity=sev, message=msg,
            evidence={"count": profile.n_duplicate_rows, "fraction": round(dup_frac, 4), "actionable": actionable,
                      "high_information_columns": informative[:5], "example_row_indices": dup_idx[:10]},
            recommendation=rec,
        ))

    for col in profile.columns:
        name = col.name
        s = df[name]

        # 2. Missing values
        if col.n_missing:
            findings.append(Finding(
                check="missing_values", severity=_missing_severity(col.pct_missing), columns=[name],
                message=f"'{name}' is {col.pct_missing:.2f}% missing ({col.n_missing} rows).",
                evidence={"n_missing": col.n_missing, "pct_missing": col.pct_missing},
                recommendation="Impute inside the training pipeline (fit on train only) or add a missing-indicator; "
                               "investigate whether missingness is related to the target.",
            ))

        if name == target:
            continue

        # 3. Constant columns
        if col.inferred_kind == "constant":
            findings.append(Finding(
                check="constant_column", severity=Severity.LOW, columns=[name],
                message=f"'{name}' has at most one distinct non-null value.",
                evidence={"n_unique": col.n_unique, "values": col.sample_values[:1]},
                recommendation="Exclude from modelling; it carries no information.",
            ))

        # 4. Identifier-like columns
        if col.inferred_kind == "id_like":
            findings.append(Finding(
                check="id_like_column", severity=Severity.MEDIUM, columns=[name],
                message=f"'{name}' looks like an identifier ({col.pct_unique:.1f}% unique).",
                evidence={"pct_unique": col.pct_unique, "dtype": col.dtype},
                recommendation="Exclude from features. If rows share an ID (same customer repeated), use a group split.",
            ))

        # 5. High-cardinality categoricals
        if col.inferred_kind == "categorical" and col.n_unique > 50:
            findings.append(Finding(
                check="high_cardinality", severity=Severity.LOW, columns=[name],
                message=f"'{name}' has {col.n_unique} categories.",
                evidence={"n_unique": col.n_unique, "pct_unique": col.pct_unique},
                recommendation="Group rare levels or use target/frequency encoding fit on training folds only.",
            ))

        # 6. Numbers stored as text with some invalid entries
        if col.inferred_kind == "numeric_text":
            nn = s.dropna().astype(str)
            coerced = pd.to_numeric(nn, errors="coerce")
            frac = float(coerced.notna().mean()) if len(nn) else 0.0
            if frac < 1.0:
                bad = nn[coerced.isna()].value_counts().head(5)
                findings.append(Finding(
                    check="mixed_type_column", severity=Severity.MEDIUM, columns=[name],
                    message=f"'{name}' is {100*frac:.1f}% numeric but stored as text; other entries are non-numeric.",
                    evidence={"numeric_fraction": round(frac, 4), "non_numeric_examples": {str(k): int(v) for k, v in bad.items()}},
                    recommendation="Decide how to treat the non-numeric tokens (likely missing) and convert to numeric.",
                ))
            else:
                findings.append(Finding(
                    check="numeric_stored_as_text", severity=Severity.LOW, columns=[name],
                    message=f"'{name}' holds only numbers but is stored as text.",
                    recommendation="Convert to a numeric type.",
                ))

        if col.inferred_kind == "numeric":
            nn = s.dropna()
            # 7. Sentinel / placeholder values
            for sv in SENTINELS:
                cnt = int((nn == sv).sum())
                if cnt >= 3:
                    rest = nn[nn != sv]
                    if len(rest) and (sv < rest.quantile(0.01) or sv > rest.quantile(0.99)):
                        findings.append(Finding(
                            check="sentinel_value", severity=Severity.MEDIUM, columns=[name],
                            message=f"'{name}' contains {cnt} occurrences of {sv}, far outside its normal range — "
                                    f"likely a placeholder for missing.",
                            evidence={"value": sv, "count": cnt, "p01_without": float(rest.quantile(0.01)),
                                      "p99_without": float(rest.quantile(0.99))},
                            recommendation=f"Confirm with the data owner; if a placeholder, treat {sv} as missing.",
                        ))
            # 8. Rare negative values in an otherwise non-negative column
            neg = int((nn < 0).sum())
            if 0 < neg <= 0.02 * len(nn) and not any(f.check == "sentinel_value" and name in f.columns for f in findings):
                findings.append(Finding(
                    check="unexpected_negative", severity=Severity.MEDIUM, columns=[name],
                    message=f"'{name}' is non-negative except for {neg} rows ({100*neg/len(nn):.2f}%).",
                    evidence={"n_negative": neg, "examples": nn[nn < 0].head(5).tolist()},
                    recommendation="Check whether negative values are valid for this variable.",
                ))
            # 9. Extreme outliers (3 x IQR), ignoring values already reported as sentinels
            flagged_sent = [f.evidence["value"] for f in findings if f.check == "sentinel_value" and name in f.columns]
            nn = nn[~nn.isin(flagged_sent)]
            q1, q3 = nn.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
                out = int(((nn < lo) | (nn > hi)).sum())
                if out:
                    findings.append(Finding(
                        check="extreme_outliers", severity=Severity.LOW if out / len(nn) < 0.01 else Severity.MEDIUM,
                        columns=[name],
                        message=f"'{name}' has {out} values beyond 3×IQR fences [{lo:.3g}, {hi:.3g}].",
                        evidence={"count": out, "lower_fence": float(lo), "upper_fence": float(hi)},
                        recommendation="Inspect before acting; outliers may be errors or genuine extremes. Do not drop silently.",
                    ))

        # 10. Datetime columns → temporal-validation warning
        if col.inferred_kind == "datetime":
            findings.append(Finding(
                check="datetime_present", severity=Severity.INFO, columns=[name],
                message=f"'{name}' is a date/time column. A random train/test split may leak future information.",
                evidence={"samples": col.sample_values[:3]},
                recommendation="If the model will be used on future rows, check it forward in time: the agent's "
                               "check_time_split trains on earlier rows and tests on later ones.",
            ))

    # 11. Target checks
    tp = profile.target
    if tp is not None:
        if tp.n_missing:
            findings.append(Finding(
                check="target_missing", severity=Severity.HIGH, columns=[tp.name],
                message=f"Target '{tp.name}' is missing in {tp.n_missing} rows.",
                evidence={"n_missing": tp.n_missing},
                recommendation="Rows without a label cannot be used for supervised training; they will be excluded and logged.",
            ))
        if tp.task_type in (TaskType.BINARY, TaskType.MULTICLASS) and tp.minority_pct is not None and tp.minority_pct < 20:
            findings.append(Finding(
                check="class_imbalance", severity=Severity.HIGH if tp.minority_pct < 5 else Severity.MEDIUM,
                columns=[tp.name],
                message=f"Minority class '{tp.minority_class}' is {tp.minority_pct:.2f}% of labelled rows "
                        f"(imbalance ratio {tp.imbalance_ratio:.1f}:1).",
                evidence={"class_counts": tp.class_counts, "minority_pct": tp.minority_pct,
                          "imbalance_ratio": tp.imbalance_ratio},
                recommendation="Use stratified splits; evaluate with recall/precision/PR-AUC, not accuracy; "
                               "consider class weighting or threshold tuning.",
            ))

    if tp is not None:
        labelled = int(len(df) - tp.n_missing)
        rare = min(tp.class_counts.values()) if tp.class_counts else None
        if labelled < 200 or (rare is not None and rare < 30):
            findings.append(Finding(
                check="small_sample", severity=Severity.MEDIUM, columns=[tp.name],
                message=(f"Small sample: {labelled} labelled rows" + (f", rarest class has only {rare}" if rare is not None else "")
                         + ". Metrics and comparisons will have wide uncertainty."),
                evidence={"labelled_rows": labelled, "rarest_class_count": rare},
                recommendation="Treat results as indicative; collect more data before relying on them.",
            ))
        if tp.task_type == TaskType.REGRESSION:
            yv = pd.to_numeric(df[tp.name], errors="coerce").dropna()
            q1, q3 = yv.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                out = yv[(yv < q1 - 3 * iqr) | (yv > q3 + 3 * iqr)]
                if len(out):
                    findings.append(Finding(
                        check="target_outliers", severity=Severity.HIGH if len(out) / len(yv) >= 0.002 else Severity.MEDIUM,
                        columns=[tp.name],
                        message=f"Target '{tp.name}' has {len(out)} extreme values beyond 3×IQR "
                                f"(max {yv.max():,.4g} vs median {yv.median():,.4g}); squared-error metrics (RMSE, R²) "
                                f"will be dominated by them.",
                        evidence={"count": int(len(out)), "examples": [float(v) for v in out.head(5)],
                                  "upper_fence": float(q3 + 3 * iqr), "lower_fence": float(q1 - 3 * iqr)},
                        recommendation="Check whether these are data errors (e.g. wrong units). If genuine, consider "
                                       "modelling log(target) and judging with MAE. Do not drop them silently.",
                    ))

    order = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2, Severity.INFO: 3}
    findings.sort(key=lambda f: order[f.severity])
    return AuditReport(tool="audit_quality", findings=findings)
