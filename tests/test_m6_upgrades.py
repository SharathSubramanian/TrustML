"""M6 tests: data-fix actions, gradient boosting, time-split check, trusted features, Ask TrustML."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmarks.make_churn import make_churn
from benchmarks.make_housing import make_housing
from trustml.agent.loop import run_agent
from trustml.experiments import ExperimentSpec, RunContext
from trustml.explain import explain_model
from trustml.followup import FollowUp
from trustml.io import load_dataset
from trustml.report import finalize_run
from trustml.tools.profiler import profile_dataset
from trustml.verify import Verifier

LEAK = ["cancellation_fee_charged"]


@pytest.fixture(scope="module")
def churn_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("u")
    df, _ = make_churn(seed=42)
    df.to_csv(d / "churn.csv", index=False)
    run = run_agent(str(d / "churn.csv"), "churned", "Predict churn", run_root=str(d / "runs"), costs={"fn": 5, "fp": 1})
    finalize_run(run)
    return d / "churn.csv", run


@pytest.fixture(scope="module")
def housing_run(tmp_path_factory):
    d = tmp_path_factory.mktemp("h")
    df, _ = make_housing(seed=7)
    df.to_csv(d / "housing.csv", index=False)
    run = run_agent(str(d / "housing.csv"), "sale_price", "Predict price", run_root=str(d / "runs"))
    finalize_run(run)
    return d / "housing.csv", run


def test_dedupe_first_and_verified(churn_run):
    _, run = churn_run
    assert run.steps[0].action["action"] == "deduplicate_rows"
    assert run.steps[0].result["rows_removed"] == 30 and run.steps[0].verification["passed"]
    assert run.ctx.split.n_train + run.ctx.split.n_test == 5000
    assert not [f for f in run.split_findings if f.check == "train_test_overlap"]


def test_dedupe_refused_after_experiments(churn_run, tmp_path):
    csv, _ = churn_run
    df, h = load_dataset(csv)
    ctx = RunContext(df, profile_dataset(df, h, target="churned"), tmp_path)
    ctx.run(ExperimentSpec(name="x", model="dummy"))
    with pytest.raises(ValueError, match="before the first experiment"):
        ctx.deduplicate()


def test_sentinel_fix_adopted_and_applied_in_pipeline(churn_run):
    _, run = churn_run
    v = next(v for v in run.facts["verdicts"] if v["topic"] == "data_fix" and v["fix"] == "sentinels")
    assert v["verdict"] == "fix_adopted" and v["columns"] == ["age"]
    spec = run.ctx.experiments[v["new"]].spec
    pipe = run.ctx._pipeline(spec, run.ctx.experiments[v["new"]].features_used)
    X = run.ctx.data.iloc[run.ctx.train_pos][run.ctx.experiments[v["new"]].features_used]
    assert pipe.named_steps["fix"].transform(X)["age"].eq(-999).sum() == 0


def test_time_split_detects_churn_drift_and_is_verified(churn_run):
    _, run = churn_run
    v = next(v for v in run.facts["verdicts"] if v["topic"] == "time_split")
    assert v["verdict"] == "degrades_forward_in_time"
    step = next(s for s in run.steps if s.action["action"] == "check_time_split")
    assert step.verification["passed"] and len(step.verification["analysis_checks"]) >= 3


def test_housing_gradient_boosting_and_stable_time(housing_run):
    _, run = housing_run
    v = {(x["topic"], x.get("model")): x for x in run.facts["verdicts"]}
    # gradient boosting is tested; it no longer wins because the explicit interaction already captures the planted
    # non-linearity (historic houses gain value with age), and the linear model with it is better on every fold
    assert ("model_choice", "gradient_boosting") in v
    assert next(x for x in run.facts["verdicts"] if x["topic"] == "time_split")["verdict"] == "stable_over_time"
    assert "age_years_if_neighborhood_historic" in run.summary["final_model"]["features_used"]
    assert run.summary["all_claims_verified"]


def test_log_target_and_gb_explanations(housing_run, tmp_path):
    csv, _ = housing_run
    df, h = load_dataset(csv)
    ctx = RunContext(df, profile_dataset(df, h, target="sale_price"), tmp_path)
    l = ctx.run(ExperimentSpec(name="l", exclude_features=["assessed_value_after_sale"], target_transform="log1p"))
    assert Verifier(tmp_path, df, "sale_price", None).verify_experiment(l.exp_id).passed
    assert explain_model(ctx, l.exp_id)["additivity_ok"]


def test_log_target_action_on_outlier_target(tmp_path):
    rng = np.random.default_rng(2)
    n = 900
    df = pd.DataFrame({"size": rng.uniform(50, 300, n), "zone": rng.choice(list("ABC"), n)})
    df["price"] = np.exp(8 + df["size"] / 60 + rng.normal(0, 0.5, n))  # heavy right tail
    p = tmp_path / "o.csv"; df.to_csv(p, index=False)
    run = run_agent(str(p), "price", "x", run_root=str(tmp_path / "runs"))
    acts = [s.action["action"] for s in run.steps]
    assert "log_transform_target" in acts
    v = next(v for v in run.facts["verdicts"] if v.get("fix") == "log_target")
    assert v["verdict"] == "log_target_improves_mae" and run.summary["all_claims_verified"]


def test_trusted_feature_not_excluded(churn_run, tmp_path):
    csv, _ = churn_run
    run = run_agent(str(csv), "churned", "x", run_root=str(tmp_path), trusted_features=["cancellation_fee_charged"])
    assert "cancellation_fee_charged" not in run.facts["excluded_as_leak"]
    assert "cancellation_fee_charged" in run.summary["final_model"]["features_used"]
    assert any("marked" in c for c in run.summary["caveats"])


@pytest.mark.parametrize("q,intent", [
    ("Why does the model miss churners?", "weak_spots"), ("What is wrong with the data?", "data_quality"),
    ("Which features matter most?", "top_features"), ("Is there leakage?", "leakage"),
    ("Does it hold up over time?", "time_drift"), ("Which model was chosen?", "model_choice"),
    ("What threshold gives 90% recall?", "threshold_for_recall"), ("what if we remove tenure_months", "remove_feature"),
])
def test_followup_routing(churn_run, q, intent):
    _, run = churn_run
    assert FollowUp(run.run_dir).route(q)[0] == intent


def test_followup_new_experiments_verified_on_same_split(churn_run):
    _, run = churn_run
    fu = FollowUp(run.run_dir)
    a = fu.ask("What if we remove contract_type?")
    assert a.verification["passed"] and len(a.new_experiments) == 2
    assert fu._context().split.test_index_sha == run.ctx.split.test_index_sha
    b = fu.ask("What threshold gives 90% recall?")
    assert b.verification["passed"] and "threshold is" in b.text
    assert (run.run_dir / "followups.jsonl").read_text().count("\n") >= 2


def test_followup_refuses_changed_data(churn_run, tmp_path):
    import json, shutil
    csv, run = churn_run
    rd = tmp_path / "copy"; shutil.copytree(run.run_dir, rd)
    changed = tmp_path / "changed.csv"
    pd.read_csv(csv).iloc[:-1].to_csv(changed, index=False)
    meta = json.loads((rd / "run.json").read_text()); meta["data_path"] = str(changed)
    (rd / "run.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="changed"):
        FollowUp(rd).ask("What if we remove contract_type?")


def test_dashboard_demo_preset_and_ask(churn_run, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import trustml.llm
    # force the deterministic chat: with a real NaviGator key the LLM phrases the answer in its own words
    monkeypatch.setattr(trustml.llm, "try_client", lambda *a, **k: None)
    _, run = churn_run
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.run()
    at.selectbox(key="demo").set_value("Churn – leakage, imbalance and cost").run()
    assert at.session_state["target"] == "churned" and at.session_state["use_cost"] is True
    at = AppTest.from_file("../app.py", default_timeout=300)
    at.session_state["run_dir"] = str(run.run_dir)
    at.run()
    assert not at.exception and len(at.tabs) == 9
    at.chat_input[0].set_value("Is there leakage?").run()
    assert not at.exception
    assert any("likely_leakage" in m.markdown[0].value for m in at.chat_message if m.markdown)
    assert any(m.name == "assistant" for m in at.chat_message)  # conversational chat, tool trace below
