"""M3 tests: thresholds/costs, missingness, subgroups, importance, engineered features, full investigations."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from benchmarks.make_churn import make_churn
from benchmarks.make_housing import make_housing
from trustml.agent.loop import Budget, run_agent
from trustml.agent.planner import LLMPlanner
from trustml.experiments import ExperimentSpec, RunContext
from trustml.io import load_dataset
from trustml.tools.feature_eng import apply_recipe, suggest_engineered
from trustml.tools.importance import cv_permutation_importance
from trustml.tools.missingness import missingness_vs_target
from trustml.tools.profiler import profile_dataset
from trustml.tools.subgroups import subgroup_errors
from trustml.verify import Verifier

LEAK_C, LEAK_H = ["cancellation_fee_charged"], ["assessed_value_after_sale"]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("bench")
    c, _ = make_churn(seed=42)
    h, _ = make_housing(seed=7)
    c.to_csv(d / "churn.csv", index=False)
    h.to_csv(d / "housing.csv", index=False)
    return d / "churn.csv", d / "housing.csv"


def _ctx(path, target, tmp, **kw):
    df, h = load_dataset(path)
    prof = profile_dataset(df, h, target=target)
    return df, prof, RunContext(df, prof, tmp, **kw)


# ------------------------------------------------------------------ thresholds and costs
def test_target_recall_threshold_meets_target_on_oof(files, tmp_path):
    _, _, c = _ctx(files[0], "churned", tmp_path)
    r = c.run(ExperimentSpec(name="t", exclude_features=LEAK_C, threshold_strategy="target_recall", target_recall=0.8))
    oof = pd.read_csv(tmp_path / r.oof_path)
    assert (oof.loc[oof.y_true == 1, "y_pred"] == 1).mean() >= 0.8
    assert r.threshold < 0.5


def test_min_cost_threshold_not_worse_than_default_on_oof(files, tmp_path):
    df, _, c = _ctx(files[0], "churned", tmp_path, costs={"fn": 5, "fp": 1})
    d = c.run(ExperimentSpec(name="d", exclude_features=LEAK_C))
    m = c.run(ExperimentSpec(name="m", exclude_features=LEAK_C, threshold_strategy="min_cost"))
    def oof_cost(r):
        o = pd.read_csv(tmp_path / r.oof_path)
        return (5 * ((o.y_pred == 0) & (o.y_true == 1)).sum() + ((o.y_pred == 1) & (o.y_true == 0)).sum()) / len(o)
    assert oof_cost(m) <= oof_cost(d)
    v = Verifier(tmp_path, df, "churned", c.positive_class).verify_experiment(m.exp_id)
    assert v.passed and any(ch.name == "metric:cost_per_row" for ch in v.checks)


def test_min_cost_requires_costs(files, tmp_path):
    _, _, c = _ctx(files[0], "churned", tmp_path)
    with pytest.raises(ValueError):
        c.run(ExperimentSpec(name="x", threshold_strategy="min_cost"))


# ------------------------------------------------------------------ missingness (B2)
def test_missingness_positive_and_negative_control(files, tmp_path):
    _, prof_h, ch = _ctx(files[1], "sale_price", tmp_path / "h")
    rep = missingness_vs_target(ch.data.iloc[ch.train_pos], prof_h)
    assert rep.flagged_columns("informative_missingness") == {"garage_area_m2"}
    _, prof_c, cc = _ctx(files[0], "churned", tmp_path / "c")
    assert missingness_vs_target(cc.data.iloc[cc.train_pos], prof_c).findings == []  # weak effect, p≈0.1


# ------------------------------------------------------------------ engineered features
def test_engineered_product_found_and_target_blocked(files, tmp_path):
    _, _, c = _ctx(files[1], "sale_price", tmp_path)
    s = suggest_engineered(c, exclude=LEAK_H)
    assert s["suggestions"][0]["name"] == "lot_width_m_x_lot_depth_m"
    with pytest.raises(ValueError):
        c.add_engineered("bad", {"op": "ratio", "sources": ["living_sqft", "sale_price"]}, c.data["living_sqft"])
    with pytest.raises(ValueError):
        c.run(ExperimentSpec(name="x", extra_features=["not_registered"]))


def test_engineered_feature_using_excluded_column_rejected(files, tmp_path):
    _, _, c = _ctx(files[1], "sale_price", tmp_path)
    r = {"op": "log1p", "sources": ["assessed_value_after_sale"]}
    c.add_engineered("log_leak", r, apply_recipe(c.data, r))
    with pytest.raises(ValueError):
        c.run(ExperimentSpec(name="x", exclude_features=LEAK_H, extra_features=["log_leak"]))


# ------------------------------------------------------------------ importance + subgroups
def test_importance_finds_planted_noise(files, tmp_path):
    _, _, c = _ctx(files[0], "churned", tmp_path)
    r = c.run(ExperimentSpec(name="w", exclude_features=LEAK_C, class_weight="balanced"))
    imp = cv_permutation_importance(c, r.exp_id)
    assert set(imp["noise_candidates"]) == {"age", "region", "annual_income"}
    assert imp["ranking"][0]["feature"] in {"contract_type", "tenure_months"}


def test_subgroup_failure_found_for_linear_and_verified(files, tmp_path):
    df, _, c = _ctx(files[1], "sale_price", tmp_path)
    r = c.run(ExperimentSpec(name="lin", exclude_features=LEAK_H))
    sg = subgroup_errors(c, r.exp_id)
    assert sg["flagged"][0]["column"] == "neighborhood" and sg["flagged"][0]["group"] == "historic"
    v = Verifier(tmp_path, df, "sale_price", None)
    step = {"exp_id": r.exp_id, "flagged": sg["flagged"], "metric": sg["metric"]}
    assert all(ch.passed for ch in v.verify_subgroups(step, c))
    step["flagged"] = [dict(sg["flagged"][0], mae=sg["flagged"][0]["mae"] * 0.8)]  # tampered
    assert not all(ch.passed for ch in v.verify_subgroups(step, c))


# ------------------------------------------------------------------ full investigations
def test_churn_full_investigation(files, tmp_path):
    run = run_agent(str(files[0]), "churned", "Predict churn", run_root=str(tmp_path))
    acts = [s.action["action"] for s in run.steps]
    for a in ["run_baseline", "ablate_feature", "apply_class_weighting", "rank_feature_importance",
              "ablate_feature_group", "analyze_subgroups", "finish"]:
        assert a in acts
    excl = set(run.summary["final_model"]["spec"]["exclude_features"])
    assert {"cancellation_fee_charged", "age", "region", "annual_income"} <= excl
    assert run.summary["all_claims_verified"]
    assert all(s.verification["passed"] for s in run.steps if s.verification)


def test_housing_full_investigation(files, tmp_path):
    run = run_agent(str(files[1]), "sale_price", "Predict price", run_root=str(tmp_path))
    v = {x["topic"]: x for x in run.facts["verdicts"] if x["topic"] != "subgroups"}
    assert v["leakage"]["verdict"] == "likely_leakage"
    assert v["feature_addition"]["verdict"] == "engineered_feature_improves"
    # the planted subgroup failure (age effect reversed in 'historic') is now found as an explicit interaction,
    # which fixes the linear model before subgroup analysis runs, so the group is no longer flagged
    assert v["interaction"]["verdict"] == "interaction_confirmed"
    assert v["interaction"]["group"] == "neighborhood=historic" and "age_years" in v["interaction"]["feature"]
    assert v["interaction"]["group_after"] < v["interaction"]["group_before"]  # MAE in the group falls
    flagged = [g for h in run.analyses["subgroups_history"] for g in h.get("flagged", [])]
    assert not any(g["column"] == "neighborhood" and g["group"] == "historic" for g in flagged)
    assert run.summary["all_claims_verified"]
    assert "lot_width_m_x_lot_depth_m" in run.summary["final_model"]["features_used"]


def test_costs_drive_threshold(files, tmp_path):
    run = run_agent(str(files[0]), "churned", "Minimise cost", run_root=str(tmp_path), costs={"fn": 5, "fp": 1})
    thr = [s for s in run.steps if s.action["action"] == "tune_threshold"]
    assert thr and thr[0].action["strategy"] == "min_cost"
    assert run.summary["all_claims_verified"]


def test_llm_cannot_invent_engineered_feature(files, tmp_path):
    from tests.test_m2_agent import FakeClient
    fake = FakeClient([
        ("deduplicate_rows", {"rationale": "duplicates"}),
        ("run_baseline", {"model": "logreg", "rationale": "ref"}),
        ("add_engineered_feature", {"feature_name": "made_up_feature", "base_exp_id": "E02_baseline_linear",
                                    "rationale": "?"}),
        ("finish", {"reason": "done"}),
    ])
    run = run_agent(str(files[1]), "sale_price", "Predict price", run_root=str(tmp_path),
                    planner=LLMPlanner(client=fake, model="fake"), budget=Budget(max_steps=4))
    assert run.steps[2].planner == "rule(fallback)" and "not one of the engineered suggestions" in run.steps[2].planner_note


def test_random_forest_tied_scores_verify(files, tmp_path):
    """Regression test: RF scores have many ties; CSV float parsing must round-trip exactly."""
    df, _, c = _ctx(files[0], "churned", tmp_path)
    r = c.run(ExperimentSpec(name="rf", model="random_forest", exclude_features=LEAK_C, class_weight="balanced"))
    assert Verifier(tmp_path, df, "churned", c.positive_class).verify_experiment(r.exp_id).passed
