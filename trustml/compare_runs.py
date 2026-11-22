"""Side-by-side comparison of two saved runs (read-only; numbers come from the stored records)."""
from __future__ import annotations

import json
from pathlib import Path


def _load(d: Path) -> dict:
    j = lambda n: json.loads((d / n).read_text()) if (d / n).exists() else {}  # noqa: E731
    steps = [json.loads(x) for x in (d / "steps.jsonl").read_text().splitlines()] if (d / "steps.jsonl").exists() else []
    return {"run": j("run.json"), "summary": j("summary.json"), "steps": steps}


def compare_runs(a: Path, b: Path) -> dict:
    A, B = _load(a), _load(b)
    out = {"same_data": A["run"].get("data_sha256") == B["run"].get("data_sha256"),
           "same_split": (A["summary"] and B["summary"] and
                          A["run"].get("split", {}).get("test_index_sha") == B["run"].get("split", {}).get("test_index_sha"))}
    rows = []

    def add(label, fa, fb):
        rows.append({"": label, "Run A": fa(A), "Run B": fb(B) if fb else fa(B)})

    fin = lambda R: (R["summary"].get("final_model") or {})  # noqa: E731
    add("dataset", lambda R: Path(R["run"].get("data_path", "?")).name, None)
    add("target", lambda R: R["run"].get("target"), None)
    add("planner", lambda R: R["run"].get("planner"), None)
    add("steps", lambda R: R["summary"].get("steps"), None)
    add("LLM decisions", lambda R: sum(s["planner"].startswith("llm") for s in R["steps"]), None)
    add("rule fallbacks", lambda R: sum("fallback" in s["planner"] for s in R["steps"]), None)
    add("final model", lambda R: f"{fin(R).get('exp_id')} ({(fin(R).get('spec') or {}).get('model')})", None)
    keys = sorted(set(fin(A).get("test_metrics", {})) | set(fin(B).get("test_metrics", {})))
    for k in [k for k in ("recall", "precision", "roc_auc", "pr_auc", "cost_per_row", "r2", "mae", "rmse",
                          "balanced_accuracy") if k in keys]:
        add(f"test {k}", lambda R, k=k: round(fin(R).get("test_metrics", {}).get(k, float("nan")), 4), None)
    add("features used", lambda R: len(fin(R).get("features_used", [])), None)
    add("excluded as leakage", lambda R: ", ".join(sorted({v["feature"] for v in R["summary"].get("verdicts", [])
                                                         if v.get("topic") == "leakage" and v["verdict"] == "likely_leakage"})) or "–", None)
    add("all claims verified", lambda R: R["summary"].get("all_claims_verified"), None)
    add("elapsed (s)", lambda R: R["summary"].get("elapsed_s"), None)
    out["rows"] = [{k: str(v) for k, v in r.items()} for r in rows]
    for side, R in (("a", A), ("b", B)):
        out[side] = {"label": f"{R['run'].get('planner')} · {R['run'].get('run_id')}",
                     "path": [s["action"]["action"] for s in R["steps"]],
                     "conclusions": R["summary"].get("conclusions", [])}
    sa = {s["action"]["action"] for s in A["steps"]}
    sb = {s["action"]["action"] for s in B["steps"]}
    out["only_a"], out["only_b"] = sorted(sa - sb), sorted(sb - sa)
    return out
