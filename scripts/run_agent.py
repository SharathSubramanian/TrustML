"""Run the TrustML agent from the command line.

  python -m scripts.run_agent data/benchmarks/churn_synthetic_v1.csv --target churned \
      --objective "Predict churn and find out why the model misses churners" --planner rule
  ... --planner llm --model gpt-oss-120b          (needs NAVIGATOR_TOOLKIT_API_KEY in .env)
"""
from __future__ import annotations

import argparse

from trustml.agent.loop import Budget, StepRecord, run_agent
from trustml.agent.planner import LLMPlanner, RulePlanner
from trustml.report import finalize_run


def show(s: StepRecord) -> None:
    a = s.action
    print(f"\n── Step {s.step}  [{s.planner}]")
    print(f"   Observed : {s.observation}")
    print(f"   Decided  : {a['action']} {({k: v for k, v in a.items() if k not in ('action', 'rationale', 'reason')})}")
    print(f"   Why      : {a.get('rationale') or a.get('reason')}")
    if s.planner_note:
        print(f"   Note     : {s.planner_note}")
    if s.result:
        print(f"   Result   : {s.result.get('interpretation') or 'experiments ' + str(s.result['new_experiments'])}")
    if s.verification:
        v = s.verification
        n = sum(x["n_checks"] for x in v["experiments"].values()) + len(v.get("analysis_checks", []))
        print(f"   Verified : {'PASS' if v['passed'] else 'FAIL'} ({n} independent checks)"
              + (f" – {v['note']}" if v.get("note") else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--target", required=True)
    ap.add_argument("--objective", default="Build a reliable model and investigate its weaknesses.")
    ap.add_argument("--planner", choices=["rule", "llm"], default="rule")
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-steps", type=int, default=22)
    ap.add_argument("--cv-repeats", type=int, default=1, help="repeat K-fold CV with different shuffles (1-3)")
    ap.add_argument("--fairness-col", action="append", default=[], help="column to check for error-rate gaps (repeatable)")
    ap.add_argument("--max-seconds", type=float, default=600)
    ap.add_argument("--cost-fn", type=float, default=None, help="cost of a missed positive (binary)")
    ap.add_argument("--cost-fp", type=float, default=1.0, help="cost of a false alarm (binary)")
    ap.add_argument("--min-recall", type=float, default=None,
                    help="target minority recall (default: read from the objective, else 0.6)")
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--no-report", action="store_true", help="skip SHAP, charts and the HTML report")
    ap.add_argument("--no-llm-text", action="store_true",
                    help="do not use NaviGator for summary wording / data-collection suggestions")
    a = ap.parse_args()

    planner = LLMPlanner(model=a.model) if a.planner == "llm" else RulePlanner()
    costs = {"fn": a.cost_fn, "fp": a.cost_fp} if a.cost_fn else None
    run = run_agent(a.path, a.target, a.objective, planner=planner, group_col=a.group_col, on_step=show, costs=costs,
                    cv_repeats=a.cv_repeats, fairness_cols=a.fairness_col,
                    budget=Budget(max_steps=a.max_steps, max_seconds=a.max_seconds,
                                  **({"min_recall": a.min_recall} if a.min_recall is not None else {})))
    s = run.summary
    print("\n══ Summary ══")
    for item in s["out_of_scope"]:
        print(f"  NOT SUPPORTED: '{item['matched']}' – {item['reason']}")
    for c in s["conclusions"]:
        print(f"  • {c}")
    for c in s["caveats"]:
        print(f"  ! {c}")
    print(f"  Claims verified: {s['all_claims_verified']}  |  stop: {s['stop_reason']}  |  {s['elapsed_s']}s")
    print(f"  Run folder: {run.run_dir}")
    if not a.no_report:
        from trustml.llm import planner_model, try_client
        llm = None if a.no_llm_text else (planner.client if a.planner == "llm" else try_client())
        out = finalize_run(run, llm_client=llm, llm_model=getattr(planner, "model", None) or a.model or planner_model())
        print(f"  Report: {out['report']}  ({len(out['charts'])} charts; summary source: "
              f"{out['narratives']['summary_meta']['source']})")


if __name__ == "__main__":
    main()
