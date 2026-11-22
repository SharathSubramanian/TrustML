"""Regression tests for gaps found in the first blind test (loan default dataset)."""
from __future__ import annotations

import json

import pytest

from trustml.agent.actions import RunBaseline, TuneThreshold
from trustml.agent.loop import AgentRun, infer_costs, run_agent
from trustml.agent.planner import validate_action
from trustml.followup import FollowUp
from trustml.narrative import numbers_supported
from trustml.report import finalize_run

LOANS = "data/benchmarks/loan_applications.csv"
OBJ = ("We approve loans and want to predict which applicants will default. Check this data for anything that would "
       "make a model look better than it really is, tell me which applicants the model gets wrong, and recommend a "
       "cut-off given that a missed default costs us 10 times more than rejecting a good applicant.")


@pytest.fixture(scope="module")
def loan_run(tmp_path_factory):
    run = run_agent(LOANS, "defaulted", OBJ, run_root=str(tmp_path_factory.mktemp("loans")))
    finalize_run(run)
    return run


def test_costs_inferred_from_objective():
    assert infer_costs(OBJ)[0] == {"fn": 10.0, "fp": 1.0}
    assert infer_costs("Predict churn") is None
    assert infer_costs("A false alarm costs us 3 times more than a miss")[0] == {"fn": 1.0, "fp": 3.0}


def test_loan_pipeline_order_and_fixes(loan_run):
    acts = [s.action["action"] for s in loan_run.steps]
    assert acts[:3] == ["deduplicate_rows", "use_group_split", "run_baseline"]
    assert loan_run.ctx.costs == {"fn": 10.0, "fp": 1.0}
    assert any("inferred from the objective" in c for c in loan_run.summary["caveats"])
    v = {(x["topic"], x.get("fix")): x for x in loan_run.facts["verdicts"]}
    assert v[("leakage", None)]["verdict"] == "likely_leakage"
    assert v[("data_fix", "sentinels")]["columns"] == ["employment_years"]
    assert v[("data_fix", "group_split")]["column"] == "customer_id"
    assert loan_run.summary["all_claims_verified"]


def test_group_split_has_no_shared_customers(loan_run):
    sp = json.loads((loan_run.run_dir / "split.json").read_text())
    df = loan_run.df
    assert not set(df.loc[sp["train_index"], "customer_id"]) & set(df.loc[sp["test_index"], "customer_id"])
    step = next(s for s in loan_run.steps if s.action["action"] == "use_group_split")
    assert step.verification["passed"]


def test_target_rate_drift_detected(loan_run):
    v = next(x for x in loan_run.facts["verdicts"] if x["topic"] == "time_split")
    assert v["verdict"] in ("target_drift", "degrades_forward_in_time")
    dr = loan_run.analyses["time_split"]["target_drift"]
    assert dr["late_rate"] > dr["early_rate"] and dr["material"]


def test_baseline_blocked_until_dedupe_and_group_split(tmp_path):
    run = AgentRun(LOANS, "defaulted", "x", run_root=str(tmp_path))
    assert "deduplicate_rows must come before" in validate_action(RunBaseline(), run)


def test_target_recall_threshold_blocked_when_met(loan_run):
    w = loan_run.facts["working_id"]
    if loan_run.ctx.experiments[w].cv_metrics["recall"].mean >= loan_run.budget.min_recall:
        err = validate_action(TuneThreshold(base_exp_id=w, strategy="target_recall"), loan_run)
        assert err and "already meets the target" in err


def test_rounding_accepted_invented_rejected():
    assert numbers_supported("recall 0.12 vs 0.69", ["recall 0.118 vs overall 0.690"])[0]
    assert not numbers_supported("recall 0.25", ["recall 0.118"])[0]


def test_ask_repeated_customers(loan_run):
    a = FollowUp(loan_run.run_dir).ask("Should I be worried about customers applying more than once?")
    assert a.intent == "data_quality" and a.text.split(":")[1].strip().startswith("repeated_ids")
    assert "group_split by 'customer_id'" in a.text


def test_identical_rows_in_low_cardinality_data_are_not_removed(tmp_path):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 900
    df = pd.DataFrame({"plan": rng.choice(["a", "b", "c"], n), "region": rng.choice(["n", "s"], n),
                       "contract": rng.choice(["m", "y"], n)})
    df["churn"] = np.where((df.contract == "m") & (rng.random(n) < 0.5), "yes", "no")
    p = tmp_path / "cat.csv"; df.to_csv(p, index=False)
    run = run_agent(str(p), "churn", "x", run_root=str(tmp_path / "runs"))
    assert "deduplicate_rows" not in [s.action["action"] for s in run.steps]
    f = run.quality.by_check("duplicate_rows")[0]
    assert f.severity.value == "info" and not f.evidence["actionable"]
    assert run.split.n_train + run.split.n_test == n if hasattr(run, "split") else \
        run.ctx.split.n_train + run.ctx.split.n_test == n


def test_scope_guard_allows_future_validity_questions():
    from trustml.agent.scope import scope_guard
    assert not scope_guard("Will this model still work on next year's applicants?")
    assert scope_guard("Forecast next month's defaults")
    assert scope_guard("How many customers will churn next month?")


def test_report_text_reflects_actions(loan_run):
    caveats = " ".join(loan_run.summary["caveats"])
    assert "may not hold up on future data" in caveats and "outcome rate changed" in caveats
    assert "random train/test split may leak future" not in caveats          # replaced by the time-check result
    assert "appear in more than one row" not in caveats                      # handled by the group split
    concl = " ".join(loan_run.summary["conclusions"])
    assert "outcome rate" in concl and "%" in concl
    html = (loan_run.run_dir / "report.html").read_text()
    assert "rerun with a group split" not in html and "gradient boosting" in html
