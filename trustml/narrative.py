"""LLM-written text, constrained and checked.

1. Executive summary: the LLM rewrites the verified conclusions in plain language.
   Every number in its text must match a number in the verified conclusions,
   otherwise the text is rejected and a deterministic summary is used.
2. Data-collection suggestions: NEW variables worth collecting. The LLM sees only
   column names, the target and the objective (never data rows). Output is
   schema-validated and labelled as untested suggestions.
"""
from __future__ import annotations

import json
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

NUM_RE = re.compile(r"(?<![\w.])[-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?%?", re.I)


def _numbers(text: str) -> list[float]:
    out = []
    for m in NUM_RE.findall(text):
        t = m.replace(",", "").rstrip("%")
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def _tokens(text: str) -> list[tuple[float, int]]:
    """(value, number of decimals as written) for every number in text."""
    out = []
    for m in NUM_RE.findall(text):
        t = m.replace(",", "").rstrip("%")
        try:
            v = float(t)
        except ValueError:
            continue
        dec = len(t.split(".")[1]) if "." in t and "e" not in t.lower() else 0
        out.append((v, dec))
    return out


def numbers_supported(text: str, sources: list[str]) -> tuple[bool, list[float]]:
    """Every number in `text` must equal a source number, allowing for rounding to the precision written
    (0.12 is accepted for 0.118) and for fraction <-> percentage (12% for 0.118)."""
    text = text.replace("\u2212", "-")          # unicode minus
    allowed = set()
    for s in sources:
        s = s.replace("\u2212", "-")
        allowed.update(_numbers(s))
        # mantissas of scientific notation, so "1.075 × 10^5" matches 1.075e+05
        allowed.update(float(m) for m in re.findall(r"(\d+\.\d+)e[-+]?\d+", s, re.I))
    allowed |= {abs(a) for a in allowed}       # magnitudes written in words ("a negative correlation of 0.18")
    pool = allowed | {a * 100 for a in allowed if abs(a) <= 1} | {a / 100 for a in allowed}
    bad = []
    for x, dec in _tokens(text):
        if float(x).is_integer() and 0 <= x <= 10:
            continue  # small counts ("5 folds", "two steps") are allowed
        tol = 0.5 * 10 ** (-dec) + 1e-9          # rounding to the written number of decimals
        if not any(abs(x - a) <= max(tol, 5e-4) or abs(x - a) <= 0.006 * abs(a) for a in pool):
            bad.append(x)
    return (not bad), bad


def deterministic_summary(run) -> str:
    s, f = run.summary, run.facts
    parts = [f"TrustML analysed {run.handle.n_rows:,} rows to predict '{run.ctx.target}' "
             f"({run.ctx.task.value.replace('_', ' ')})."]
    for v in f["verdicts"]:
        t = v["topic"]
        if t == "leakage" and v["verdict"] == "likely_leakage":
            parts.append(f"'{v['feature']}' behaves like target leakage and was excluded (to be confirmed with the data owner).")
        elif t in ("imbalance", "threshold") and v["verdict"] in ("recall_improved", "cost_reduced"):
            parts.append(f"{v['intervention'].replace('_', ' ')} improved {v['metric'].replace('_', ' ')} on every CV fold.")
        elif t == "feature_ablation" and v["verdict"] == "simpler_model_no_material_loss":
            parts.append(f"Removing {v['features']} cost nothing, so the simpler model was kept.")
        elif t == "feature_addition" and v["verdict"] == "engineered_feature_improves":
            parts.append(f"The engineered feature '{v['feature']}' improved the model on every fold.")
        elif t == "model_choice" and v["verdict"] == "model_improves":
            parts.append(f"A {v['model'].replace('_', ' ')} outperformed the linear model on every fold.")
    if s.get("final_model"):
        parts.append(s["conclusions"][-1])
    parts.append("All reported numbers were independently re-verified." if s.get("all_claims_verified")
                 else "WARNING: some numbers failed verification; see the appendix.")
    return " ".join(parts)


SUMMARY_PROMPT = """Write a 4-6 sentence executive summary of an ML analysis for a non-specialist manager.
Use ONLY the facts below. Do not add any number that is not in the facts. Do not claim causation.
Answer the user's objective directly in the first sentence. Report the FINAL model's test results as the result;
if a leak was found, say the baseline score was inflated by it and never present that score as an achievement.
Mention leakage suspicions as suspicions, and mention the most important caveat (caveats are listed by importance).
Facts (JSON):
"""


def llm_summary(run, client, model: str) -> tuple[str, dict]:
    facts = {"objective": run.objective, "target": run.ctx.target, "rows": run.handle.n_rows,
             "conclusions": run.summary["conclusions"], "caveats": run.summary["caveats"][:4],
             "out_of_scope": [i["matched"] for i in run.summary["out_of_scope"]]}
    sources = run.summary["conclusions"] + run.summary["caveats"] + [str(run.handle.n_rows), f"{run.handle.n_rows:,}"]
    try:
        from trustml.llm import chat
        r = chat(client, model, messages=[{"role": "user", "content": SUMMARY_PROMPT + json.dumps(facts)}])
        text = (r.choices[0].message.content or "").strip()
        ok, bad = numbers_supported(text, sources)
        if text and ok:
            return text, {"source": f"llm:{model}", "numbers_checked": True}
        return deterministic_summary(run), {"source": "template", "rejected_llm_text": text,
                                             "unsupported_numbers": bad}
    except Exception as e:  # noqa: BLE001
        return deterministic_summary(run), {"source": "template", "llm_error": f"{type(e).__name__}: {e}"}


class CollectionSuggestion(BaseModel):
    name: str
    rationale: str
    how_to_collect: str
    available_at_prediction_time: Literal["yes", "no", "depends"]
    timing_note: str
    feasibility: Literal["easy", "moderate", "hard"]
    privacy_fairness: str


class CollectionSuggestions(BaseModel):
    suggestions: list[CollectionSuggestion] = Field(max_length=5)


COLLECT_PROMPT = """You advise which NEW variables (not in the dataset) would most help predict the target.
Dataset columns: {cols}
Target: {target}. Objective: {objective}
Rules: suggest 3-5 variables that are NOT already present; never invent values; say whether each would be
known at prediction time; flag privacy/fairness risks (e.g. protected attributes, proxies).
Return ONLY JSON: {{"suggestions": [{{"name": str, "rationale": str, "how_to_collect": str,
"available_at_prediction_time": "yes"|"no"|"depends", "timing_note": str,
"feasibility": "easy"|"moderate"|"hard", "privacy_fairness": str}}]}}"""


def llm_collection_suggestions(run, client, model: str) -> tuple[list[dict], dict]:
    cols = [c.name for c in run.profile.columns if c.name != run.ctx.target]
    try:
        from trustml.llm import chat
        r = chat(client, model, messages=[{"role": "user", "content":
                 COLLECT_PROMPT.format(cols=cols, target=run.ctx.target, objective=run.objective)}])
        text = r.choices[0].message.content or ""
        text = text[text.find("{"): text.rfind("}") + 1]
        parsed = CollectionSuggestions.model_validate_json(text)
        existing = {c.lower() for c in cols}
        keep = [s.model_dump() for s in parsed.suggestions if s.name.lower() not in existing]
        return keep, {"source": f"llm:{model}", "validated": True, "dropped_existing": len(parsed.suggestions) - len(keep)}
    except (ValidationError, ValueError) as e:
        return [], {"source": "none", "error": f"LLM output failed validation: {e}"}
    except Exception as e:  # noqa: BLE001
        return [], {"source": "none", "error": f"{type(e).__name__}: {e}"}


def write_narratives(run, client=None, model: Optional[str] = None) -> dict:
    if client is None:
        return {"summary": deterministic_summary(run), "summary_meta": {"source": "template"},
                "collection": [], "collection_meta": {"source": "none", "error":
                "LLM text was not enabled for this run - add NAVIGATOR_TOOLKIT_API_KEY to .env, or tick "
                "'Use NaviGator for summary wording...' in the dashboard"}}
    summ, sm = llm_summary(run, client, model)
    coll, cm = llm_collection_suggestions(run, client, model)
    return {"summary": summ, "summary_meta": sm, "collection": coll, "collection_meta": cm}
