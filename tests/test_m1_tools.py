"""M1 tests: tools must detect every injected issue and avoid known false positives."""
from __future__ import annotations

import hashlib

import numpy as np

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from benchmarks.make_churn import make_churn
from trustml.io import DatasetError, load_dataset
from trustml.schemas import Severity, TaskType
from trustml.tools.leakage import screen_leakage
from trustml.tools.profiler import profile_dataset
from trustml.tools.quality import audit_quality


@pytest.fixture(scope="module")
def churn(tmp_path_factory):
    df, truth = make_churn(seed=42)
    p = tmp_path_factory.mktemp("data") / "churn.csv"
    df.to_csv(p, index=False)
    loaded, handle = load_dataset(p)
    prof = profile_dataset(loaded, handle, target="churned")
    return p, loaded, handle, prof, truth


def test_loader_hash_and_immutability(churn):
    p, df, handle, _, _ = churn
    assert handle.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    before = p.read_bytes()
    audit_quality(df, profile_dataset(df, handle, target="churned"))
    assert p.read_bytes() == before, "tools must never modify the original file"


def test_loader_rejects_bad_input(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "missing.csv")
    (tmp_path / "x.json").write_text("{}")
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "x.json")


def test_excel_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]})
    p = tmp_path / "t.xlsx"
    df.to_excel(p, index=False)
    loaded, h = load_dataset(p)
    assert h.n_rows == 3 and list(loaded.columns) == ["a", "y"]


def test_profiler_numbers_match_pandas(churn):
    _, df, _, prof, truth = churn
    assert prof.n_duplicate_rows == int(df.duplicated().sum()) == 30
    col = {c.name: c for c in prof.columns}
    assert col["annual_income"].n_missing == int(df["annual_income"].isna().sum())
    assert col["monthly_charges"].mean == pytest.approx(df["monthly_charges"].mean())
    assert col["customer_id"].inferred_kind == "id_like"
    assert col["country"].inferred_kind == "constant"
    assert col["signup_date"].inferred_kind == "datetime"
    t = prof.target
    assert t.task_type == TaskType.BINARY
    assert t.minority_pct == pytest.approx(truth["churn_rate_pct"], abs=0.01)


def test_quality_detects_all_injected_non_leakage_issues(churn):
    _, df, _, prof, truth = churn
    rep = audit_quality(df, prof)
    for issue in truth["injected_issues"]:
        if issue["check"] == "suspicious_predictor":
            continue
        found = rep.by_check(issue["check"])
        assert found, f"missed injected issue: {issue['check']}"
        for c in issue["columns"]:
            assert c in rep.flagged_columns(issue["check"]), f"{issue['check']} missed column {c}"


def test_quality_no_false_sentinel_or_mixed_type(churn):
    _, df, _, prof, _ = churn
    rep = audit_quality(df, prof)
    assert rep.flagged_columns("sentinel_value") == {"age"}
    assert rep.flagged_columns("mixed_type_column") == {"total_charges"}
    assert rep.flagged_columns("constant_column") == {"country"}


def test_leakage_flags_planted_leak_on_train_only(churn):
    _, df, _, prof, truth = churn
    train, _ = train_test_split(df, test_size=0.2, random_state=42, stratify=df["churned"])
    rep = screen_leakage(train, prof)
    high = {c for f in rep.by_check("suspicious_predictor") if f.severity == Severity.HIGH for c in f.columns}
    assert "cancellation_fee_charged" in high
    flagged = rep.flagged_columns("suspicious_predictor")
    for legit in truth["must_not_flag_as_leakage"]:
        assert legit not in flagged, f"false leakage alarm on legitimate driver {legit}"


def test_leakage_quiet_on_clean_data():
    rng = np.random.default_rng(0)
    n = 2000
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(0.8 * x1 + 0.5 * x2)))).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "noise": rng.normal(size=n), "y": y})
    from trustml.schemas import DatasetHandle
    prof = profile_dataset(df, DatasetHandle(path="mem", sha256="0", n_rows=n, n_cols=4), target="y")
    rep = screen_leakage(df, prof)
    assert rep.by_check("suspicious_predictor") == []


def test_regression_target_and_leak():
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.normal(size=n)
    y = 3 * x + rng.normal(size=n)
    df = pd.DataFrame({"x": x, "final_price_paid": y + rng.normal(scale=0.05, size=n), "z": rng.normal(size=n), "y": y})
    from trustml.schemas import DatasetHandle
    prof = profile_dataset(df, DatasetHandle(path="mem", sha256="0", n_rows=n, n_cols=4), target="y")
    assert prof.target.task_type == TaskType.REGRESSION
    rep = screen_leakage(df, prof)
    assert "final_price_paid" in rep.flagged_columns("suspicious_predictor")


def test_sentinel_not_double_counted_as_outlier(churn):
    _, df, _, prof, _ = churn
    assert "age" not in audit_quality(df, prof).flagged_columns("extreme_outliers")


def test_prediction_files_round_trip_exactly(tmp_path):
    from trustml.io import load_predictions, save_predictions
    x = np.random.default_rng(0).random(50000)
    df = pd.DataFrame({"row_index": np.arange(len(x)), "y_score": x})
    save_predictions(df, tmp_path / "p.csv")
    assert np.array_equal(load_predictions(tmp_path / "p.csv")["y_score"].to_numpy(), x)
