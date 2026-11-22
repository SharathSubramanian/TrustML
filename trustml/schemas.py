"""Structured outputs for every TrustML tool.

Every tool returns one of these Pydantic models, never free text. The LLM later
reads these objects; it never computes numbers itself.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TaskType(str, Enum):
    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"


class DatasetHandle(BaseModel):
    """Reference to an immutable loaded dataset."""
    path: str
    sha256: str = Field(description="Hash of the original file bytes; proves the original was not modified.")
    n_rows: int
    n_cols: int
    sheet: Optional[str] = None


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    inferred_kind: Literal["numeric", "categorical", "boolean", "datetime", "text", "id_like", "constant", "numeric_text"]
    n_missing: int
    pct_missing: float
    n_unique: int
    pct_unique: float
    sample_values: list[Any]
    # numeric-only stats (None otherwise)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p25: Optional[float] = None
    median: Optional[float] = None
    p75: Optional[float] = None
    max: Optional[float] = None
    # categorical-only
    top_values: Optional[dict[str, int]] = None


class TargetProfile(BaseModel):
    name: str
    task_type: TaskType
    n_missing: int
    class_counts: Optional[dict[str, int]] = None
    minority_class: Optional[str] = None
    minority_pct: Optional[float] = None
    imbalance_ratio: Optional[float] = Field(None, description="majority count / minority count")
    mean: Optional[float] = None
    std: Optional[float] = None


class DatasetProfile(BaseModel):
    handle: DatasetHandle
    n_duplicate_rows: int
    memory_mb: float
    columns: list[ColumnProfile]
    target: Optional[TargetProfile] = None


class Finding(BaseModel):
    """One evidence-backed issue. `check` names the rule that produced it."""
    check: str
    severity: Severity
    columns: list[str] = Field(default_factory=list)
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommendation: str = ""


class AuditReport(BaseModel):
    tool: str
    findings: list[Finding]

    def by_check(self, check: str) -> list[Finding]:
        return [f for f in self.findings if f.check == check]

    def flagged_columns(self, check: str) -> set[str]:
        return {c for f in self.by_check(check) for c in f.columns}
