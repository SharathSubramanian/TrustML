"""Dataset profiler: deterministic description of the data. No modelling."""
from __future__ import annotations

import re
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from trustml.schemas import ColumnProfile, DatasetHandle, DatasetProfile, TargetProfile, TaskType

ID_NAME_RE = re.compile(r"(^id$|_id$|^id_|uuid|guid|identifier|record_?no|row_?num)", re.I)


def _py(v: Any) -> Any:
    """Convert numpy scalars to JSON-friendly Python values."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    return v


def _looks_datetime(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    sample = s.dropna().astype(str).head(200)
    if sample.empty or sample.str.len().mean() < 6:
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() >= 0.9


def infer_kind(name: str, s: pd.Series) -> str:
    non_null = s.dropna()
    n_unique = non_null.nunique()
    n = len(non_null)
    if n_unique <= 1:
        return "constant"
    if pd.api.types.is_bool_dtype(s) or n_unique == 2:
        return "boolean"
    if _looks_datetime(s):
        return "datetime"
    pct_unique = n_unique / max(n, 1)
    if pd.api.types.is_integer_dtype(s):
        sorted_range = n_unique == n and (non_null.max() - non_null.min() + 1) == n
        if pct_unique >= 0.99 and (ID_NAME_RE.search(name) or sorted_range):
            return "id_like"
        return "numeric"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    # object / string: mostly-numeric text (e.g. "12.5", " ") is numbers stored as text
    coerced = pd.to_numeric(non_null.astype(str).str.strip(), errors="coerce")
    if coerced.notna().mean() >= 0.8:
        return "numeric_text"
    avg_len = non_null.astype(str).str.len().mean()
    if pct_unique >= 0.95:
        return "text" if avg_len > 30 else "id_like"
    if ID_NAME_RE.search(name) and pct_unique >= 0.5:
        return "id_like"
    return "categorical"


def profile_column(name: str, s: pd.Series) -> ColumnProfile:
    n = len(s)
    n_missing = int(s.isna().sum())
    non_null = s.dropna()
    n_unique = int(non_null.nunique())
    kind = infer_kind(name, s)
    prof = ColumnProfile(
        name=name,
        dtype=str(s.dtype),
        inferred_kind=kind,
        n_missing=n_missing,
        pct_missing=round(100 * n_missing / n, 3) if n else 0.0,
        n_unique=n_unique,
        pct_unique=round(100 * n_unique / max(len(non_null), 1), 3),
        sample_values=[_py(v) for v in non_null.head(5).tolist()],
    )
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s) and len(non_null):
        q = non_null.quantile([0.25, 0.5, 0.75])
        prof.mean = _py(non_null.mean())
        prof.std = _py(non_null.std())
        prof.min = _py(non_null.min())
        prof.p25 = _py(q.loc[0.25])
        prof.median = _py(q.loc[0.5])
        prof.p75 = _py(q.loc[0.75])
        prof.max = _py(non_null.max())
    if kind in ("categorical", "boolean"):
        prof.top_values = {str(k): int(v) for k, v in non_null.value_counts().head(10).items()}
    return prof


def infer_task_type(y: pd.Series, override: Optional[TaskType] = None) -> TaskType:
    if override is not None:
        return override
    y = y.dropna()
    n_unique = y.nunique()
    if n_unique == 2:
        return TaskType.BINARY
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return TaskType.MULTICLASS
    if pd.api.types.is_integer_dtype(y) and n_unique <= 10:
        return TaskType.MULTICLASS
    return TaskType.REGRESSION


def profile_target(df: pd.DataFrame, target: str, task_type: Optional[TaskType] = None) -> TargetProfile:
    if target not in df.columns:
        import difflib
        close = [c for c in df.columns if c.strip().lower() == target.strip().lower()] or \
            difflib.get_close_matches(target, list(df.columns), n=3)
        hint = f" Did you mean {close}?" if close else ""
        raise KeyError(f"Target column '{target}' not found.{hint} Columns: {list(df.columns)}")
    y = df[target]
    if y.dropna().nunique() < 2:
        from trustml.io import DatasetError
        raise DatasetError(f"Target '{target}' has fewer than 2 distinct values "
                           f"({y.dropna().unique()[:3].tolist()}); there is nothing to predict.")
    tt = infer_task_type(y, task_type)
    tp = TargetProfile(name=target, task_type=tt, n_missing=int(y.isna().sum()))
    if tt == TaskType.REGRESSION:
        tp.mean = _py(y.mean())
        tp.std = _py(y.std())
    else:
        counts = y.dropna().value_counts()
        tp.class_counts = {str(k): int(v) for k, v in counts.items()}
        tp.minority_class = str(counts.idxmin())
        tp.minority_pct = round(100 * counts.min() / counts.sum(), 3)
        tp.imbalance_ratio = round(counts.max() / counts.min(), 3)
    return tp


def profile_dataset(df: pd.DataFrame, handle: DatasetHandle, target: Optional[str] = None,
                    task_type: Optional[TaskType] = None) -> DatasetProfile:
    return DatasetProfile(
        handle=handle,
        n_duplicate_rows=int(df.duplicated().sum()),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1e6, 3),
        columns=[profile_column(c, df[c]) for c in df.columns],
        target=profile_target(df, target, task_type) if target else None,
    )
