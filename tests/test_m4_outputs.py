"""M4/M5 tests: explanations, charts, report, narrative guard, dashboard, benchmark scoring."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmarks.make_churn import make_churn
from benchmarks.make_housing import make_housing
from trustml.agent.loop import Budget, run_agent
from trustml.narrative import numbers_supported, write_narratives
from trustml.report import finalize_run


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    d = tmp_path_factory.mktemp("o")
    c, _ = make_churn(seed=42); h, _ = make_housing(seed=7)
    c.to_csv(d / "churn.csv", index=False); h.to_csv(d / "housing.csv", index=False)
    rc = run_agent(str(d / "churn.csv"), "churned", "Predict churn", run_root=str(d / "runs"))
    rh = run_agent(str(d / "housing.csv"), "sale_price", "Predict price", run_root=str(d / "runs"))
    return rc, rh, finalize_run(rc), finalize_run(rh)


def test_shap_additivity_and_leak_visible_in_baseline(runs):
    rc, rh, oc, oh = runs
    for out in (oc, oh):
        assert out["explanations"]["final"]["additivity_ok"]
    assert oc["explanations"]["baseline"]["global"][0]["feature"] == "cancellation_fee_charged"
    assert all(g["feature"] != "cancellation_fee_charged" for g in oc["explanations"]["final"]["global"])


def test_charts_are_purposeful(runs):
    rc, rh, oc, oh = runs
    ids_c = {c["id"] for c in oc["charts"]}
    assert {"target", "leak_screen", "importance", "final", "shap_final"} <= ids_c
    for c in oc["charts"] + oh["charts"]:
        assert c["interpretation"] and c["title"]
    assert "interactions" in {c["id"] for c in oh["charts"]}


def test_report_contains_required_sections_and_no_unverified_flag(runs):
    rc, rh, oc, oh = runs
    html = open(oc["report"], encoding="utf-8").read()
    for sec in ["Executive summary", "Dataset overview", "Data quality and leakage", "Preprocessing and validation",
                "Agent investigation", "Experiments", "Feature recommendations", "Error analysis",
                "Explainability", "Limitations", "Recommendations", "Appendix – verification"]:
        assert sec in html, sec
    assert "independently verified" in html and "verification failed" not in html


def test_narrative_number_guard():
    ok, bad = numbers_supported("Recall rose from 0.096 to 0.741 across 5 folds.", ["recall 0.0957 → 0.7415"])
    assert ok
    ok, bad = numbers_supported("Recall rose to 0.95 and revenue grew 12%.", ["recall 0.0957 → 0.7415"])
    assert not ok and 0.95 in bad


class _LLM:
    def __init__(self, summary, collection):
        self.out = [summary, collection]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._c))

    def _c(self, **kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.out.pop(0)))])


def test_llm_summary_rejected_when_inventing_numbers(runs):
    rc = runs[0]
    n = write_narratives(rc, _LLM("Churn will fall by 37% next quarter.", '{"suggestions": []}'), "fake")
    assert n["summary_meta"]["source"] == "template" and 37.0 in n["summary_meta"]["unsupported_numbers"]


def test_llm_collection_validated_and_existing_columns_dropped(runs):
    rc = runs[0]
    good = {"suggestions": [
        {"name": "customer_satisfaction_score", "rationale": "r", "how_to_collect": "survey",
         "available_at_prediction_time": "yes", "timing_note": "t", "feasibility": "moderate", "privacy_fairness": "p"},
        {"name": "tenure_months", "rationale": "dup", "how_to_collect": "x", "available_at_prediction_time": "yes",
         "timing_note": "t", "feasibility": "easy", "privacy_fairness": "p"}]}
    n = write_narratives(rc, _LLM(rc.summary["conclusions"][0], json.dumps(good)), "fake")
    assert [s["name"] for s in n["collection"]] == ["customer_satisfaction_score"]
    bad = write_narratives(rc, _LLM("x", '{"suggestions": [{"name": "a"}]}'), "fake")
    assert bad["collection"] == [] and "validation" in bad["collection_meta"]["error"]


def test_dashboard_renders_saved_run(runs, monkeypatch):
    from streamlit.testing.v1 import AppTest
    rc = runs[0]
    at = AppTest.from_file("../app.py", default_timeout=120)
    at.session_state["run_dir"] = str(rc.run_dir)
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Data quality", "Agent investigation", "Planner", "Models",
                                          "Trust checks", "Explainability", "Ask TrustML", "Report"]


def test_benchmark_scoring(tmp_path, monkeypatch, runs):
    import scripts.benchmark as B
    rh = runs[1]
    truth = make_housing(seed=7)[1]
    det = B.trustml_detections(rh.run_dir)
    s = B.score_detection(truth, det)
    assert s["found"] == s["total"] and not s["false_leakage_alarms"]
    bad = B.score_detection(truth, [{"check": "leakage", "columns": ["living_sqft"]}])
    assert bad["found"] == 0 and bad["false_leakage_alarms"] == ["living_sqft"]
    facts = B.reference_facts(str(rh.run_dir.parent.parent / "housing.csv"), "sale_price")
    n = B.score_numbers([{"metric": "n_rows", "value": 3015}, {"metric": "duplicate_rows", "value": 99}], facts)
    assert n["correct"] == 1 and n["checked"] == 2
