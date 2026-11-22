"""Regression tests for gaps found by blind test #2 (hospital readmission), written after scoring it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustml.agent.loop import Budget, infer_recall_target, run_agent


def test_infer_recall_target():
    assert infer_recall_target("Catch at least 70% of readmissions.")[0] == 0.7
    assert infer_recall_target("We need 80% recall on fraud")[0] == 0.8
    assert infer_recall_target("recall of 0.85 please")[0] == 0.85
    assert infer_recall_target("Reduce costs by 30%") is None
    assert infer_recall_target("improve accuracy to 95%") is None


@pytest.fixture(scope="module")
def dominant_leak(tmp_path_factory):
    """A post-outcome column that scores 'only' ~0.96 alone but towers over every other feature."""
    rng = np.random.default_rng(5)
    n = 2500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.2 + 0.7 * x1 + 0.4 * x2)))).astype(int)
    leak = np.where(y == 1, rng.choice(["a", "b"], n, p=[.92, .08]), rng.choice(["a", "b", "c"], n, p=[.04, .56, .40]))
    bmi = rng.normal(28, 5, n).round(1)
    bmi[rng.random(n) < 0.04] = 999
    d = tmp_path_factory.mktemp("dom")
    pd.DataFrame({"x1": x1, "x2": x2, "followup_kind": leak, "bmi": bmi, "y": y}).to_csv(d / "d.csv", index=False)
    return run_agent(str(d / "d.csv"), "y", "Catch at least 75% of positives.", run_root=str(d))


def test_dominant_feature_is_treated_as_likely_leakage(dominant_leak):
    v = next(v for v in dominant_leak.facts["verdicts"] if v["topic"] == "leakage")
    assert v["feature"] == "followup_kind" and v["verdict"] == "likely_leakage"
    assert "followup_kind" not in dominant_leak.summary["final_model"]["features_used"]


def test_tiny_loss_does_not_block_sentinel_fix(dominant_leak):
    v = next(v for v in dominant_leak.facts["verdicts"] if v.get("fix") == "sentinels")
    assert v["verdict"] == "fix_adopted"


def test_recall_target_read_from_objective(dominant_leak):
    assert dominant_leak.budget.min_recall == 0.75 and dominant_leak.recall_source.startswith("inferred")
    assert any("Target recall 0.75" in c for c in dominant_leak.summary["caveats"])


def test_explicit_recall_target_wins(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=600)})
    df["y"] = (df.a + rng.normal(size=600) > 1).astype(int)
    df.to_csv(tmp_path / "e.csv", index=False)
    run = run_agent(str(tmp_path / "e.csv"), "y", "Catch at least 90% of positives", run_root=str(tmp_path),
                    budget=Budget(max_steps=1, min_recall=0.5))
    assert run.budget.min_recall == 0.5 and run.recall_source == "user"


def test_regression_fairness_uses_relative_error(tmp_path):
    """Blind test 3: older members have larger costs, so larger absolute errors; that alone is not a gap."""
    from trustml.experiments import ExperimentSpec, RunContext
    from trustml.io import load_dataset
    from trustml.tools.diagnostics import fairness_report
    from trustml.tools.profiler import profile_dataset
    rng = np.random.default_rng(2)
    n = 2000
    age = rng.integers(18, 80, n)
    y = np.exp(7 + 0.03 * age + rng.normal(0, 0.4, n))  # same relative noise at every age
    pd.DataFrame({"age": age, "x": rng.normal(size=n), "y": y}).to_csv(tmp_path / "r.csv", index=False)
    df, h = load_dataset(str(tmp_path / "r.csv"))
    ctx = RunContext(df, profile_dataset(df, h, target="y"), tmp_path / "run")
    e = ctx.run(ExperimentSpec(name="b", target_transform="log1p"))
    fr = fairness_report(ctx, e.exp_id, "age")
    assert fr["ratio"] > 1.5 and fr["relative_ratio"] < 1.25 and not fr["material"]


def test_number_guard_accepts_magnitudes_and_scientific():
    from trustml.narrative import numbers_supported
    assert numbers_supported("a negative correlation of 0.183 (z 21.3)", ["r=-0.183, z=-21.3"])[0]
    assert numbers_supported("max about 1.075 × 10^5", ["max 1.075e+05"])[0]
    assert not numbers_supported("AUC 0.95", ["AUC 0.80"])[0]
