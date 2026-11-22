"""V2: conversational Ask TrustML. The LLM plans tool calls and phrases replies; numbers are guarded."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from trustml.agent.loop import Budget, run_agent
from trustml.chat import Conversation, _extract_json
from trustml.report import finalize_run


class FakeLLM:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return NS(choices=[NS(message=NS(content=out, tool_calls=None))])


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    rng = np.random.default_rng(3)
    n = 1200
    df = pd.DataFrame({"tenure": rng.integers(1, 60, n), "fee": rng.normal(50, 10, n),
                       "plan": rng.choice(["a", "b", "c"], n), "gender": rng.choice(["F", "M"], n)})
    df["churn"] = (rng.random(n) < 1 / (1 + np.exp(0.08 * df.tenure - 0.06 * (df.fee - 50) - 1.5))).astype(int)
    d = tmp_path_factory.mktemp("chat")
    df.to_csv(d / "c.csv", index=False)
    run = run_agent(str(d / "c.csv"), "churn", "predict churn", run_root=str(d), budget=Budget(max_steps=20))
    finalize_run(run)
    return run.run_dir


def test_extract_json_handles_fences_and_reasoning():
    assert _extract_json('thinking... ```json\n{"calls": [], "reply": "hi"}\n```') == {"calls": [], "reply": "hi"}
    assert _extract_json("no json here") is None


def test_deterministic_mode_routes_and_remembers(run_dir):
    c = Conversation(run_dir)
    t = c.send("Can I trust the predicted probabilities?")
    assert t.tools[0]["tool"] == "calibration" and "Brier" in t.content
    t = c.send("Is it fair across gender?")
    assert t.tools[0]["tool"] == "fairness" and "gender" in t.content
    assert len(c.history) == 4
    assert (run_dir / "chat.jsonl").exists()


def test_small_talk_needs_no_tools(run_dir):
    t = Conversation(run_dir).send("thanks!")
    assert not t.tools and "welcome" in t.content.lower()


def test_llm_plans_multiple_tools_and_experiment_is_verified(run_dir):
    fake = FakeLLM([json.dumps({"calls": [{"tool": "remove_feature", "question": "remove tenure"},
                                          {"tool": "top_features", "question": "which features matter?"}]}),
                    "Removing tenure hurts the model, and tenure is also its most important feature."])
    c = Conversation(run_dir, llm_client=fake, llm_model="fake")
    t = c.send("what if we drop it, and what matters most?")
    assert [x["tool"] for x in t.tools] == ["remove_feature", "top_features"]
    assert t.verification["experiments_verified"] is True and t.verification["n_new_experiments"] == 2
    assert t.content.startswith("Removing tenure") and "llm:fake" in t.source


def test_llm_reply_with_invented_number_is_replaced(run_dir):
    fake = FakeLLM([json.dumps({"calls": [{"tool": "calibration", "question": "calibrated?"}]}),
                    "Calibration error is 0.0123, excellent."])
    t = Conversation(run_dir, llm_client=fake, llm_model="fake").send("calibrated?")
    assert "rejected" in t.source and "0.0123" not in t.content and "Brier" in t.content


def test_bad_llm_plan_falls_back_to_router(run_dir):
    fake = FakeLLM([json.dumps({"calls": [{"tool": "delete_everything", "question": "x"}]}),
                    RuntimeError("down")])
    t = Conversation(run_dir, llm_client=fake, llm_model="fake").send("Is there leakage?")
    assert t.tools[0]["tool"] == "leakage" and "plan rejected" in t.source and "LLM unavailable" in t.source


def test_history_is_sent_to_llm(run_dir):
    fake = FakeLLM([json.dumps({"calls": [{"tool": "leakage", "question": "leakage?"}]}), "No leak was found.",
                    json.dumps({"calls": [{"tool": "top_features", "question": "top features"}]}), "Tenure matters most."])
    c = Conversation(run_dir, llm_client=fake, llm_model="fake")
    c.send("any leakage?")
    c.send("and what matters?")
    msgs = fake.calls[2]["messages"]
    assert any(m["role"] == "user" and m["content"] == "any leakage?" for m in msgs)
