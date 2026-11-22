"""Check NaviGator access and find a model that can plan for TrustML.

  python -m scripts.check_llm                      # list models available to YOUR key
  python -m scripts.check_llm --model gpt-oss-120b # test one model
  python -m scripts.check_llm --candidates         # test the usual local candidates, print a ranking

For each model: (1) plain chat, (2) two REAL planning decisions on the churn benchmark using TrustML's own
prompt and action schema (tool calling if the server supports it, otherwise JSON mode); each decision must be
valid, and agreement with the rule planner's choice is reported.
"""
from __future__ import annotations

import argparse
import tempfile
import time

from trustml.llm import chat, get_client, planner_model

CANDIDATES = ["gpt-oss-120b", "llama-3.3-70b-instruct", "nemotron-3-super-120b-a12b", "gemma-4-31b-it",
              "mistral-small-3.1", "gpt-oss-20b"]


def test_model(client, model: str, data: str) -> dict:
    from trustml.agent.loop import AgentRun, Budget
    from trustml.agent.planner import LLMPlanner, RulePlanner, signature
    out = {"model": model}
    t = time.time()
    try:
        r = chat(client, model, messages=[{"role": "user", "content": "Reply with exactly: OK"}])
        out["chat"] = (r.choices[0].message.content or "").strip()[:40]
    except Exception as e:  # noqa: BLE001
        out["chat"] = f"ERROR {type(e).__name__}: {str(e)[:120]}"
        return out
    run = AgentRun(data, "churned", "Predict churn and explain why the model misses churners",
                   run_root=tempfile.mkdtemp(prefix="trustml_check_llm_"), budget=Budget(max_steps=4))
    planner, rules = LLMPlanner(client=client, model=model), RulePlanner()
    agree = 0
    for k in (1, 2):  # two real decisions; each executed before the next
        expected = rules.decide(run).action.action
        d = planner.decide(run)
        out["mode"], out["mode_log"] = planner.mode or "none", planner.mode_log
        out[f"step{k}"] = (f"{d.planner}: {d.action.action} (rule planner: {expected})"
                           + (f"  ({d.note[:150]})" if d.note else ""))
        out[f"step{k}_ok"] = d.planner.startswith("llm")
        agree += d.action.action == expected
        out["rationale"] = (getattr(d.action, "rationale", "") or getattr(d.action, "reason", ""))[:200]
        if not out[f"step{k}_ok"] or d.action.action == "finish":
            break
        run.pending = d
        run.facts["tried"].add(signature(d.action))
        run.act_node()
        run.verify_node()
    out["agrees_with_rules"] = agree
    out["seconds"] = round(time.time() - t, 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--candidates", action="store_true")
    ap.add_argument("--data", default="data/benchmarks/churn_synthetic_v1.csv")
    a = ap.parse_args()
    client = get_client()
    ids = sorted(m.id for m in client.models.list().data)
    print(f"{len(ids)} models available to this key:\n  " + "\n  ".join(ids))
    if not a.model and not a.candidates:
        print(f"\nNext: python -m scripts.check_llm --candidates   (or --model <name>)")
        return
    models = [m for m in CANDIDATES if m in ids] if a.candidates else [a.model or planner_model()]
    for m in models:
        if m not in ids:
            print(f"\n[!] '{m}' is not available to this key - pick an exact name from the list above.")
            return
    results = []
    for m in models:
        print(f"\n=== {m} ===")
        r = test_model(client, m, a.data)
        results.append(r)
        for k in ("chat", "mode", "step1", "step2", "rationale", "seconds"):
            if k in r:
                print(f"  {k:9}: {r[k]}")
        for line in r.get("mode_log", []):
            print(f"  mode-log : {line}")
    good = [r for r in results if r.get("step1_ok") and r.get("step2_ok")]
    good.sort(key=lambda r: -r.get("agrees_with_rules", 0))
    print("\n=== Result ===")
    for r in results:
        verdict = "PASS" if r in good else ("PARTIAL" if r.get("step1_ok") else "FAIL")
        print(f"  {verdict:7} {r['model']:30} mode={r.get('mode', '-'):15} agrees-with-rules="
              f"{r.get('agrees_with_rules', 0)}/2  {r.get('seconds', '-')} s")
    if good:
        best = good[0]  # candidates are listed in expected-quality order
        print(f"\nRecommended: put TRUSTML_PLANNER_MODEL={best['model']} in .env")


if __name__ == "__main__":
    main()
