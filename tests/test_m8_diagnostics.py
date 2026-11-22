"""V1 upgrades: interaction screen/test, fairness check, calibration check, repeated CV."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trustml.agent.actions import CheckFairness, InteractionTest
from trustml.agent.loop import run_agent
from trustml.agent.planner import validate_action
from trustml.report import finalize_run
from trustml.tools.diagnostics import detect_sensitive

LOANS = "data/benchmarks/loan_applications.csv"
OBJ = "Predict which applicants default; catch at least 70% of defaulters. A missed default costs 5x a declined applicant."


@pytest.fixture(scope="module")
def loan_run(tmp_path_factory):
    run = run_agent(LOANS, "defaulted", OBJ, run_root=str(tmp_path_factory.mktemp("loans8")))
    out = finalize_run(run)
    return run, out


def _step(run, name):
    return next(s for s in run.steps if s.action["action"] == name)


def test_planted_interaction_is_found_and_confirmed(loan_run):
    run, _ = loan_run
    screen = _step(run, "screen_interactions")
    top = screen.result["candidates"][0]
    assert (top["feature"], top["group_col"], top["level"]) == ("employment_years", "purpose", "small_business")
    assert screen.verification["passed"] and screen.verification["analysis_checks"]
    v = next(v for v in run.facts["verdicts"] if v["topic"] == "interaction")
    assert v["verdict"] == "interaction_confirmed" and v["group_improved"]
    assert "employment_years" in v["readded"]  # it had been removed as noise on average
    assert _step(run, "test_interaction").verification["passed"]
    # the sentinel 99 is masked inside the interaction term too
    rec = run.ctx.engineered[v["feature"]]
    assert rec["mask"] == [99]
    assert not (run.ctx.data[v["feature"]] == 99).any()


def test_interaction_must_come_from_screen(loan_run):
    run, _ = loan_run
    a = InteractionTest(feature="credit_score", group_col="state", level="CA", base_exp_id=run.facts["working_id"])
    assert "not a screened interaction candidate" in validate_action(a, run)


def test_calibration_checked_and_verified(loan_run):
    run, out = loan_run
    s = _step(run, "check_calibration")
    assert s.verification["passed"] and {c["name"] for c in s.verification["analysis_checks"]} == {
        "calibration:brier", "calibration:ece"}
    cal = run.analyses["calibration"]
    assert cal["class_weighted"] and not cal["well_calibrated"]  # weighting inflates probabilities
    assert any("not calibrated" in c for c in run.summary["caveats"])
    assert "calibration" in {c["id"] for c in out["charts"]}


def test_calibration_verifier_catches_tampering(loan_run):
    run, _ = loan_run
    res = dict(run.analyses["calibration"], ece=run.analyses["calibration"]["ece"] + 0.05)
    assert not all(c.passed for c in run.verifier.verify_calibration(res))


def test_report_has_trust_sections(loan_run):
    _, out = loan_run
    html = open(out["report"], encoding="utf-8").read()
    assert "Trust checks: fairness and calibration" in html and "Group-specific effects" in html
    assert "window.print()" in html and "@media print" in html


def test_detect_sensitive_is_conservative():
    found = detect_sensitive(["age", "age_years", "garage_area_m2", "gender", "SeniorCitizen", "marital-status",
                              "race", "customer_age", "sex", "page_views", "average", "storage", "home_age"])
    assert found == ["age", "gender", "SeniorCitizen", "marital-status", "race", "customer_age", "sex"]


@pytest.fixture(scope="module")
def fair_run(tmp_path_factory):
    """Synthetic data where the model misses far more positives in one gender group."""
    rng = np.random.default_rng(0)
    n = 2500
    g = rng.choice(["F", "M"], n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    logit = -1.6 + np.where(g == "F", 1.8 * x1, 0.2 * x1) + 0.4 * x2
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    d = tmp_path_factory.mktemp("fair")
    pd.DataFrame({"gender": g, "x1": x1, "x2": x2, "region": rng.choice(list("ABC"), n), "y": y}).to_csv(
        d / "fair.csv", index=False)
    return run_agent(str(d / "fair.csv"), "y", "predict y", run_root=str(d), fairness_cols=["region"])


def test_fairness_auto_and_user_columns(fair_run):
    run = fair_run
    assert run.facts["fairness_columns"] == ["region", "gender"]
    steps = [s for s in run.steps if s.action["action"] == "check_fairness"]
    assert {s.action["column"] for s in steps} == {"region", "gender"}
    assert all(s.verification["passed"] for s in steps)
    g = run.analyses["fairness"]["gender"]
    assert g["material"] and g["gap_ci95"][0] > 0 or g["false_alarm_gap_material"]
    assert any("gender" in c for c in run.summary["caveats"])
    a = CheckFairness(column="x1", base_exp_id=run.facts["working_id"])
    assert "not a fairness column" in validate_action(a, run)


def test_fairness_verifier_catches_tampering(fair_run):
    run = fair_run
    res = json.loads(json.dumps(run.analyses["fairness"]["gender"]))
    res["groups"][0]["recall"] = (res["groups"][0]["recall"] or 0) + 0.1
    assert not all(c.passed for c in run.verifier.verify_fairness(res, run.ctx))


def test_repeated_cv_wired_through(tmp_path):
    run = run_agent(LOANS, "defaulted", OBJ, run_root=str(tmp_path), cv_repeats=2, group_col=None,
                    budget=__import__("trustml.agent.loop", fromlist=["Budget"]).Budget(max_steps=1))
    # group split is forced first on this data (customer_id repeats), so repeats are not applied there
    assert json.loads((run.run_dir / "run.json").read_text())["cv_repeats"] == 2


def test_repeated_cv_folds(tmp_path):
    rng = np.random.default_rng(1)
    n = 800
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = (df.a + rng.normal(size=n) > 0.8).astype(int)
    df.to_csv(tmp_path / "r.csv", index=False)
    run = run_agent(str(tmp_path / "r.csv"), "y", "predict y", run_root=str(tmp_path), cv_repeats=3)
    b = run.ctx.experiments[run.facts["baseline_id"]]
    assert len(b.cv_metrics["roc_auc"].folds) == 15
    assert all(s.verification["passed"] for s in run.steps if s.verification)
    assert run.summary["all_claims_verified"]


def test_llm_cannot_invent_an_interaction(tmp_path):
    from trustml.agent.loop import Budget
    from trustml.agent.planner import LLMPlanner
    from tests.test_m2_agent import FakeClient
    fake = FakeClient([
        ("deduplicate_rows", {"rationale": "duplicates were found"}),
        ("use_group_split", {"group_col": "customer_id", "rationale": "ids repeat"}),
        ("run_baseline", {"model": "logreg", "rationale": "reference"}),
        ("test_interaction", {"feature": "credit_score", "group_col": "state", "level": "CA",
                              "base_exp_id": "E02_baseline_logreg", "rationale": "a hunch"}),
        ("finish", {"reason": "done"}),
    ])
    run = run_agent(LOANS, "defaulted", OBJ, run_root=str(tmp_path), planner=LLMPlanner(client=fake, model="fake"),
                    budget=Budget(max_steps=4))
    s = run.steps[3]
    assert s.planner == "rule(fallback)" and "not a screened interaction candidate" in s.planner_note
