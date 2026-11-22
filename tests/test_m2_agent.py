"""M2 tests: experiment engine, verifier, planners, agent loop."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score, roc_auc_score

from benchmarks.make_churn import make_churn
from trustml.agent.actions import AblateFeature, Finish, RunBaseline, tool_definitions
from trustml.agent.loop import Budget, run_agent
from trustml.agent.planner import LLMPlanner, RulePlanner
from trustml.agent.scope import scope_guard
from trustml.experiments import ExperimentSpec, RunContext
from trustml.io import load_dataset
from trustml.tools.profiler import profile_dataset
from trustml.verify import Verifier


@pytest.fixture(scope="module")
def churn_csv(tmp_path_factory):
    df, _ = make_churn(seed=42)
    p = tmp_path_factory.mktemp("d") / "churn.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="module")
def ctx(churn_csv, tmp_path_factory):
    df, h = load_dataset(churn_csv)
    prof = profile_dataset(df, h, target="churned")
    c = RunContext(df, prof, tmp_path_factory.mktemp("run"))
    return df, c


# ------------------------------------------------------------------ experiment engine
def test_split_fixed_and_disjoint(ctx):
    _, c = ctx
    assert not set(c.train_pos) & set(c.test_pos)
    assert len(c.train_pos) + len(c.test_pos) == len(c.data)
    a = c.run(ExperimentSpec(name="a", model="dummy"))
    b = c.run(ExperimentSpec(name="b", model="logreg"))
    assert a.test_index_sha == b.test_index_sha == c.split.test_index_sha


def test_metrics_match_sklearn(ctx):
    _, c = ctx
    r = c.run(ExperimentSpec(name="m", model="logreg"))
    p = pd.read_csv(c.run_dir / r.predictions_path)
    assert r.test_metrics["recall"] == pytest.approx(recall_score(p.y_true, p.y_pred))
    assert r.test_metrics["roc_auc"] == pytest.approx(roc_auc_score(p.y_true, p.y_score))
    assert len(r.cv_metrics["recall"].folds) == 5 and r.cv_metrics["recall"].std >= 0
    lo, hi = r.test_ci95["recall"]
    assert lo <= r.test_metrics["recall"] <= hi


def test_dummy_baseline_is_majority(ctx):
    _, c = ctx
    r = c.run(ExperimentSpec(name="d", model="dummy"))
    assert r.cv_metrics["recall"].mean == 0.0 and r.cv_metrics["roc_auc"].mean == pytest.approx(0.5)


def test_preprocessing_fit_on_train_only(ctx):
    _, c = ctx
    feats = c._features(ExperimentSpec(name="x"))
    pipe = c._pipeline(ExperimentSpec(name="x"), feats).fit(c.data.iloc[c.train_pos][feats], c.y[c.train_pos])
    num_imputer = pipe.named_steps["pre"].named_transformers_["num"].named_steps["impute"]
    num_cols = pipe.named_steps["pre"].transformers_[0][2]
    i = num_cols.index("annual_income")
    assert num_imputer.statistics_[i] == pytest.approx(c.data.iloc[c.train_pos]["annual_income"].median())
    assert num_imputer.statistics_[i] != pytest.approx(c.data["annual_income"].median())


def test_threshold_tuned_on_train_not_test(ctx):
    _, c = ctx
    r = c.run(ExperimentSpec(name="t", model="logreg", exclude_features=["cancellation_fee_charged"],
                             threshold_strategy="max_f1"))
    assert r.threshold != 0.5 and 0.05 <= r.threshold <= 0.95


def test_split_checks_find_overlap_and_repeated_ids(ctx):
    _, c = ctx
    checks = {f.check: f for f in c.split_checks()}
    assert checks["repeated_ids"].evidence["n_repeated_values"] == 30
    assert checks["train_test_overlap"].evidence["n_overlap_test_rows"] > 0


def test_group_split_keeps_ids_together(churn_csv, tmp_path):
    df, h = load_dataset(churn_csv)
    c = RunContext(df, profile_dataset(df, h, target="churned"), tmp_path, group_col="customer_id")
    tr = set(c.data.iloc[c.train_pos]["customer_id"])
    te = set(c.data.iloc[c.test_pos]["customer_id"])
    assert not tr & te
    assert not [f for f in c.split_checks() if f.check == "train_test_overlap"]


def test_unknown_exclusion_rejected(ctx):
    _, c = ctx
    with pytest.raises(ValueError):
        c.run(ExperimentSpec(name="bad", exclude_features=["no_such_column"]))


def test_regression_experiment(tmp_path):
    rng = np.random.default_rng(3)
    n = 800
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": 2 * x + rng.normal(size=n)})
    from trustml.schemas import DatasetHandle
    prof = profile_dataset(df, DatasetHandle(path="m", sha256="0", n_rows=n, n_cols=3), target="y")
    c = RunContext(df, prof, tmp_path)
    r = c.run(ExperimentSpec(name="lin", model="logreg"))
    assert r.cv_metrics["r2"].mean > 0.7
    assert Verifier(tmp_path, df, "y", None).verify_experiment(r.exp_id).passed


# ------------------------------------------------------------------ verifier
def test_verifier_passes_and_detects_tampering(ctx):
    df, c = ctx
    r = c.run(ExperimentSpec(name="v", model="logreg"))
    v = Verifier(c.run_dir, df, "churned", c.positive_class)
    assert v.verify_experiment(r.exp_id).passed
    path = c.run_dir / "experiments" / f"{r.exp_id}.json"
    rec = json.loads(path.read_text())
    rec["test_metrics"]["roc_auc"] += 0.02
    path.write_text(json.dumps(rec))
    res = v.verify_experiment(r.exp_id)
    assert not res.passed and any(ch.name == "metric:roc_auc" and not ch.passed for ch in res.checks)
    rec["test_metrics"]["roc_auc"] -= 0.02
    rec["features_used"].append("churned")
    path.write_text(json.dumps(rec))
    assert not v.verify_experiment(r.exp_id).passed


def test_verify_claim(ctx):
    df, c = ctx
    r = c.run(ExperimentSpec(name="c", model="logreg"))
    v = Verifier(c.run_dir, df, "churned", c.positive_class)
    assert v.verify_claim(r.exp_id, "recall", r.test_metrics["recall"]).passed
    assert not v.verify_claim(r.exp_id, "recall", r.test_metrics["recall"] + 0.05).passed
    assert not v.verify_claim("E99_missing", "recall", 0.5).passed


# ------------------------------------------------------------------ scope guard
@pytest.mark.parametrize("text,expected", [
    ("Forecast churn for next month", True), ("Use a neural network", True),
    ("Cluster customers into segments", True), ("Predict churn and explain weak recall", False),
    ("Which customer segments have the worst recall?", False),
])
def test_scope_guard(text, expected):
    assert bool(scope_guard(text)) is expected


# ------------------------------------------------------------------ agent loop
def test_agent_rule_planner_end_to_end(churn_csv, tmp_path):
    run = run_agent(str(churn_csv), "churned", "Predict churn", run_root=str(tmp_path))
    actions = [s.action["action"] for s in run.steps]
    assert actions[:3] == ["deduplicate_rows", "run_baseline", "ablate_feature"] and actions[-1] == "finish"
    assert "apply_class_weighting" in actions
    assert run.steps[2].result["verdict"] == "likely_leakage"
    assert next(s for s in run.steps if s.action["action"] == "apply_class_weighting").result["verdict"] == "recall_improved"
    assert all(s.verification["passed"] for s in run.steps if s.verification)
    s = run.summary
    assert s["all_claims_verified"]
    assert "cancellation_fee_charged" in s["final_model"]["spec"]["exclude_features"]
    assert (run.run_dir / "steps.jsonl").read_text().count("\n") == len(run.steps)


def test_agent_planner_never_sees_test_metrics(churn_csv, tmp_path):
    run = run_agent(str(churn_csv), "churned", "Predict churn", run_root=str(tmp_path),
                    budget=Budget(max_steps=2))
    view = json.dumps(run.planner_view())
    assert "test_metrics" not in view and "test_ci95" not in view


def test_agent_step_budget(churn_csv, tmp_path):
    run = run_agent(str(churn_csv), "churned", "Predict churn", run_root=str(tmp_path), budget=Budget(max_steps=1))
    assert [s.action["action"] for s in run.steps] == ["deduplicate_rows", "finish"]
    assert "budget" in run.stop_reason.lower()


def test_agent_threshold_path_when_weighting_insufficient(churn_csv, tmp_path):
    run = run_agent(str(churn_csv), "churned", "Predict churn", run_root=str(tmp_path),
                    budget=Budget(min_recall=0.9))
    actions = [s.action["action"] for s in run.steps]
    assert "tune_threshold" in actions and actions[-1] == "finish"
    thr = next(s for s in run.steps if s.action["action"] == "tune_threshold")
    assert thr.action["strategy"] == "target_recall"
    assert thr.result["verdict"] == "recall_improved"
    tuned = run.ctx.experiments[thr.result["new_experiments"][0]]
    assert tuned.cv_metrics["recall"].mean >= 0.85  # target 0.9 met on OOF (fold means can differ slightly)
    assert run.summary["all_claims_verified"]


# ------------------------------------------------------------------ LLM planner (fake client, no network)
class FakeClient:
    """Mimics the OpenAI client; replays scripted tool calls."""

    def __init__(self, script):
        self.script, self.calls = list(script), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        assert kwargs["tools"] and "test_metrics" not in kwargs["messages"][1]["content"]
        item = self.script.pop(0) if self.script else ("finish", {"reason": "done"})
        if isinstance(item, Exception):
            raise item
        name, args = item
        call = SimpleNamespace(function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call], content=None))])


def test_tool_definitions_valid():
    names = {t["function"]["name"] for t in tool_definitions()}
    assert names == {"run_baseline", "ablate_feature", "ablate_feature_group", "apply_class_weighting",
                     "tune_threshold", "rank_feature_importance", "add_engineered_feature", "try_model",
                     "analyze_subgroups", "finish", "deduplicate_rows", "treat_sentinels_as_missing",
                     "log_transform_target", "check_time_split", "use_group_split",
                     "screen_interactions", "test_interaction", "check_fairness", "check_calibration"}
    for t in tool_definitions():
        assert t["function"]["parameters"]["type"] == "object"


def test_llm_planner_follows_valid_calls(churn_csv, tmp_path):
    fake = FakeClient([
        ("deduplicate_rows", {"rationale": "30 duplicates found"}),
        ("run_baseline", {"model": "logreg", "rationale": "need a reference"}),
        ("ablate_feature", {"feature": "cancellation_fee_charged", "base_exp_id": "E02_baseline_logreg",
                            "rationale": "AUC 0.983 alone"}),
        ("finish", {"reason": "enough"}),
    ])
    run = run_agent(str(churn_csv), "churned", "Predict churn", run_root=str(tmp_path),
                    planner=LLMPlanner(client=fake, model="fake"))
    assert [s.planner for s in run.steps] == ["llm:fake"] * 4
    assert [s.action["action"] for s in run.steps] == ["deduplicate_rows", "run_baseline", "ablate_feature", "finish"]


def test_llm_planner_invalid_call_falls_back(churn_csv, tmp_path):
    fake = FakeClient([
        ("ablate_feature", {"feature": "x", "base_exp_id": "E99", "rationale": "skip baseline"}),  # invalid
        ("run_baseline", {"rationale": "missing model is fine"}),  # valid once the fallback has deduplicated
        ("ablate_feature", {"feature": "not_a_column", "base_exp_id": "E02_baseline_logreg", "rationale": "?"}),
        ConnectionError("network down"),
        ("finish", {"reason": "done"}),
    ])
    run = run_agent(str(churn_csv), "churned", "Predict churn", run_root=str(tmp_path),
                    planner=LLMPlanner(client=fake, model="fake"))
    planners = [s.planner for s in run.steps]
    assert planners[0] == "rule(fallback)" and "run_baseline must come first" in run.steps[0].planner_note
    assert run.steps[0].action["action"] == "deduplicate_rows"          # the fallback chose the required step
    assert planners[1] == "llm:fake" and run.steps[1].action["action"] == "run_baseline"
    assert planners[2] == "rule(fallback)" and "not used by" in run.steps[2].planner_note
    assert planners[3] == "rule(fallback)" and "network down" in run.steps[3].planner_note
    assert run.summary["all_claims_verified"]


def test_llm_planner_retries_without_temperature(churn_csv, tmp_path):
    """GPT-5-style models reject temperature=0; the planner must retry instead of silently falling back."""
    inner = FakeClient([("deduplicate_rows", {"rationale": "dupes"}), ("run_baseline", {"model": "logreg", "rationale": "ref"}),
                        ("finish", {"reason": "done"})])

    class Strict:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._c))

        def _c(self, **kw):
            if "temperature" in kw:
                raise ValueError("Unsupported value: 'temperature' does not support 0 with this model.")
            return inner._create(**kw)

    run = run_agent(str(churn_csv), "churned", "Predict churn", run_root=str(tmp_path),
                    planner=LLMPlanner(client=Strict(), model="gpt-5.2"))
    assert [s.planner for s in run.steps] == ["llm:gpt-5.2"] * 3


class _ToolRejectingClient:
    """Mimics a vLLM server without tool-call parsing: tool requests fail, plain chat works."""

    def __init__(self, script):
        self.script = list(script)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._c))

    def _c(self, **kw):
        if "tools" in kw:
            raise ValueError('"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set')
        content = self.script.pop(0) if self.script else '{"action": "finish", "reason": "done"}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=content))])


def test_llm_planner_switches_to_json_mode(churn_csv, tmp_path):
    client = _ToolRejectingClient([
        'Duplicates first: {"action": "deduplicate_rows", "rationale": "30 duplicate rows"}',
        'Reasoning first... {"action": "run_baseline", "model": "logreg", "rationale": "need a reference"}',
        '{"action": "ablate_feature", "feature": "cancellation_fee_charged", "base_exp_id": "E02_baseline_logreg", '
        '"rationale": "single-feature AUC 0.983"}',
        'not json at all',
        '{"action": "finish", "reason": "done"}',
    ])
    planner = LLMPlanner(client=client, model="local-model")
    run = run_agent(str(churn_csv), "churned", "Predict churn", run_root=str(tmp_path), planner=planner)
    assert planner.mode == "json"
    assert any("tools_required failed" in m for m in planner.mode_log)
    assert [s.planner for s in run.steps][:3] == ["llm:local-model"] * 3
    assert run.steps[3].planner == "rule(fallback)" and "no JSON action" in run.steps[3].planner_note
    assert run.summary["all_claims_verified"]


def test_check_llm_with_fake_server(churn_csv):
    from scripts.check_llm import test_model
    client = _ToolRejectingClient([
        "OK",
        '{"action": "deduplicate_rows", "rationale": "duplicates first"}',
        '{"action": "run_baseline", "model": "logreg", "rationale": "reference"}',
    ])
    r = test_model(client, "local-model", str(churn_csv))
    assert r["mode"] == "json" and r["step1_ok"] and r["step2_ok"] and r["agrees_with_rules"] == 2


def test_planner_view_lists_executed_actions(churn_csv, tmp_path):
    run = run_agent(str(churn_csv), "churned", "Predict churn", run_root=str(tmp_path), budget=Budget(max_steps=3))
    view = run.planner_view()
    assert [a["action"] for a in view["actions_already_executed"]][:3] == ["deduplicate_rows", "run_baseline",
                                                                            "ablate_feature"]
