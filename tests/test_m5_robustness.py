"""Regression tests for problems found by running TrustML on real public datasets and messy edge cases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trustml.agent.loop import run_agent
from trustml.io import DatasetError, load_dataset
from trustml.report import finalize_run
from trustml.tools.profiler import profile_dataset
from trustml.tools.quality import audit_quality

rng = np.random.default_rng(5)


def _csv(tmp_path, name, df):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_headerless_file_detected(tmp_path):
    n = 300
    df = pd.DataFrame({"a": rng.integers(18, 90, n), "b": rng.choice(["State-gov", "Private"], n),
                       "c": rng.integers(1, 16, n), "y": rng.choice(["<=50K", ">50K"], n)})
    p = tmp_path / "noheader.csv"
    df.to_csv(p, index=False, header=False)  # first data row becomes the header
    d, h = load_dataset(p)
    rep = audit_quality(d, profile_dataset(d, h, target=d.columns[-1]))
    assert rep.by_check("possible_missing_header")


def test_normal_header_not_flagged(tmp_path):
    df = pd.DataFrame({"age": rng.integers(18, 90, 300), "job": rng.choice(["x", "y"], 300), "y": rng.integers(0, 2, 300)})
    d, h = load_dataset(_csv(tmp_path, "ok.csv", df))
    assert not audit_quality(d, profile_dataset(d, h, target="y")).by_check("possible_missing_header")


def test_constant_target_and_wrong_name_messages(tmp_path):
    p = _csv(tmp_path, "c.csv", pd.DataFrame({"x": rng.normal(size=50), "y": np.ones(50, int)}))
    d, h = load_dataset(p)
    with pytest.raises(DatasetError, match="fewer than 2 distinct values"):
        profile_dataset(d, h, target="y")
    with pytest.raises(KeyError, match=r"Did you mean \['y'\]"):
        profile_dataset(d, h, target="Y")


def test_small_sample_and_fold_guard(tmp_path):
    m = 120
    y = np.zeros(m, int); y[:4] = 1; rng.shuffle(y)
    p = _csv(tmp_path, "rare.csv", pd.DataFrame({"a": rng.normal(size=m) + y, "b": rng.normal(size=m), "y": y}))
    run = run_agent(str(p), "y", "x", run_root=str(tmp_path / "runs"))
    assert run.ctx.cv_folds < 5
    assert any(t["op"] == "reduce_cv_folds" for t in run.ctx.transform_log)
    assert any(f.check == "small_sample" for f in run.quality.findings)
    assert run.summary["all_claims_verified"]


def test_target_outliers_flagged(tmp_path):
    n = 600
    df = pd.DataFrame({"size": rng.uniform(50, 300, n)})
    df["price"] = 1000 * df["size"] + rng.normal(0, 20000, n)
    df.loc[:9, "price"] *= 20
    d, h = load_dataset(_csv(tmp_path, "o.csv", df))
    assert audit_quality(d, profile_dataset(d, h, target="price")).by_check("target_outliers")


def test_multiclass_end_to_end_with_report(tmp_path):
    n = 600
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "kind": rng.choice(["p", "q"], n),
                       "species": np.select([x < -0.5, x > 0.5], ["A", "C"], "B")})
    run = run_agent(str(_csv(tmp_path, "m.csv", df)), "species", "classify", run_root=str(tmp_path / "runs"))
    assert run.ctx.task.value == "multiclass_classification"
    out = finalize_run(run)
    assert out["explanations"]["final"]["additivity_ok"]
    assert "final" in {c["id"] for c in out["charts"]}
    assert run.summary["all_claims_verified"]


def test_subgroup_bins_verified_at_rounding_boundary(tmp_path):
    """qcut labels are rounded to 3 decimals; verification must use the exact edges."""
    n = 4000
    inc = np.round(rng.uniform(0.5, 15, n), 4)
    inc[:200] = 4.77305  # many rows just above a rounded label edge
    df = pd.DataFrame({"income": inc, "zone": rng.choice(list("ABC"), n)})
    df["value"] = 30000 * np.log1p(df["income"]) + np.where(df["income"] > 6, 40000, 0) + rng.normal(0, 8000, n)
    run = run_agent(str(_csv(tmp_path, "s.csv", df)), "value", "x", run_root=str(tmp_path / "runs"))
    sub_steps = [s for s in run.steps if s.action["action"] == "analyze_subgroups"]
    assert sub_steps and all(s.verification["passed"] for s in sub_steps)
    assert run.summary["all_claims_verified"]


def test_tied_folds_still_count_as_improvement(tmp_path):
    m = 80
    y = np.zeros(m, int); y[:8] = 1; rng.shuffle(y)
    p = _csv(tmp_path, "t.csv", pd.DataFrame({"a": rng.normal(size=m) + 1.2 * y, "b": rng.normal(size=m), "y": y}))
    run = run_agent(str(p), "y", "x", run_root=str(tmp_path / "runs"))
    w = [v for v in run.facts["verdicts"] if v["topic"] == "imbalance"]
    assert w and w[0]["verdict"] in ("recall_improved", "no_consistent_recall_gain")
    assert run.summary["all_claims_verified"]


def test_engineered_feature_logged_with_correct_op(tmp_path):
    from benchmarks.make_housing import make_housing
    df, _ = make_housing(seed=7)
    run = run_agent(str(_csv(tmp_path, "h.csv", df)), "sale_price", "x", run_root=str(tmp_path / "runs"))
    eng = [t for t in run.ctx.transform_log if t.get("name") == "lot_width_m_x_lot_depth_m"]
    assert eng and eng[0]["op"] == "engineered_feature" and eng[0]["recipe"]["op"] == "product"


def test_try_client_without_key(monkeypatch):
    from trustml.llm import try_client
    monkeypatch.delenv("NAVIGATOR_TOOLKIT_API_KEY", raising=False)
    assert try_client() is None
