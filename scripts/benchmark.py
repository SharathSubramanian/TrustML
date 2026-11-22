"""TrustML vs Muse benchmark workflow (one folder per prompt).

  python -m scripts.benchmark init P1 --dataset data/benchmarks/churn_synthetic_v1.csv --target churned \
        --truth data/benchmarks/churn_synthetic_v1.truth.json --source own --prompt "Audit this dataset ..."
  # run the same prompt + file in Muse; paste its full answer into benchmarks/results/P1/muse_output.md,
  # save screenshots in benchmarks/results/P1/muse_screenshots/, fill muse_claims.csv
  python -m scripts.benchmark freeze-rubric P1     # BEFORE looking at scores
  python -m scripts.benchmark run-trustml P1
  python -m scripts.benchmark score P1
  python -m scripts.benchmark summary              # all prompts -> results/summary.md (for slides)

Objective items are scored by code against the truth file and against facts computed here with plain
pandas (independent of TrustML's own tools). Subjective items come from rubric_scores.csv, filled by a
human after the rubric was frozen. No LLM grades anything.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional  # noqa: F401

import pandas as pd

RESULTS = Path("benchmarks/results")
CLAIM_FIELDS = ["agent_claim_type", "check", "columns", "metric", "value", "quote_from_output", "note"]
CHECK_ALIASES = {  # map free-text issue names onto truth checks
    "leakage": "suspicious_predictor", "target_leakage": "suspicious_predictor", "suspicious_predictor": "suspicious_predictor",
    "missing": "missing_values", "missing_values": "missing_values", "duplicates": "duplicate_rows",
    "duplicate_rows": "duplicate_rows", "id": "id_like_column", "identifier": "id_like_column",
    "id_like_column": "id_like_column", "constant": "constant_column", "constant_column": "constant_column",
    "imbalance": "class_imbalance", "class_imbalance": "class_imbalance", "sentinel": "sentinel_value",
    "sentinel_value": "sentinel_value", "placeholder": "sentinel_value", "mixed_type": "mixed_type_column",
    "mixed_type_column": "mixed_type_column", "datetime": "datetime_present", "datetime_present": "datetime_present",
    "informative_missingness": "informative_missingness", "engineered_feature": "engineered_feature",
    "subgroup_failure": "subgroup_failure",
}

RUBRIC_TEMPLATE = """# Subjective rubric - edit BEFORE scoring, then run freeze-rubric. Scores 1-5.
criteria:
  relevance: "Answers what the prompt asked, focused on the user's objective"
  evidence: "Claims are backed by shown computations, tables or charts"
  actionability: "Recommendations are specific and could be acted on"
  honesty: "States uncertainty and limitations; no overclaiming (e.g. leakage, causation)"
  clarity: "A non-specialist can follow the answer"
scale: "1 = poor, 3 = acceptable, 5 = excellent"
"""


def _dir(pid: str) -> Path:
    return RESULTS / pid


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- init
def cmd_init(a) -> None:
    d = _dir(a.prompt_id)
    if d.exists() and not a.force:
        raise SystemExit(f"{d} exists (use --force to overwrite templates)")
    (d / "muse_screenshots").mkdir(parents=True, exist_ok=True)
    (d / "prompt.txt").write_text(a.prompt.strip() + "\n")
    meta = {"prompt_id": a.prompt_id, "source": a.source, "dataset": a.dataset, "dataset_sha256": _sha(Path(a.dataset)),
            "target": a.target, "truth": a.truth, "created": datetime.now().isoformat(timespec="seconds"),
            "costs": {"fn": a.cost_fn, "fp": a.cost_fp} if a.cost_fn else None}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    (d / "muse_output.md").write_text("<!-- Paste Muse's COMPLETE answer here, unedited. Note date/time and whether "
                                      "Muse ran code. -->\n")
    with open(d / "muse_claims.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CLAIM_FIELDS)
        w.writeheader()
        w.writerow({"agent_claim_type": "detection", "check": "leakage", "columns": "cancellation_fee_charged",
                    "quote_from_output": "EXAMPLE ROW - delete", "note": "one row per claim"})
        w.writerow({"agent_claim_type": "number", "metric": "target_rate_pct", "value": "10.4",
                    "quote_from_output": "EXAMPLE ROW - delete"})
    (d / "rubric.yaml").write_text(RUBRIC_TEMPLATE)
    with open(d / "rubric_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "relevance", "evidence", "actionability", "honesty", "clarity", "scorer", "comment"])
        w.writerow(["trustml", "", "", "", "", "", "", ""])
        w.writerow(["muse", "", "", "", "", "", "", ""])
    print(f"Created {d}. Next: run the prompt in Muse and fill muse_output.md + muse_claims.csv.")


def cmd_freeze(a) -> None:
    d = _dir(a.prompt_id)
    (d / "rubric.frozen.json").write_text(json.dumps({
        "rubric_sha256": _sha(d / "rubric.yaml"), "frozen_at": datetime.now().isoformat(timespec="seconds")}, indent=2))
    print("Rubric frozen. Any later edit to rubric.yaml will be reported by `score`.")


# --------------------------------------------------------------------------- run TrustML
def cmd_run(a) -> None:
    from trustml.agent.loop import Budget, run_agent
    from trustml.agent.planner import LLMPlanner, RulePlanner
    from trustml.report import finalize_run
    d = _dir(a.prompt_id)
    meta = json.loads((d / "meta.json").read_text())
    if _sha(Path(meta["dataset"])) != meta["dataset_sha256"]:
        raise SystemExit("Dataset changed since init - comparison would not be fair.")
    planner = LLMPlanner(model=a.model) if a.planner == "llm" else RulePlanner()
    run = run_agent(meta["dataset"], meta["target"], (d / "prompt.txt").read_text(), planner=planner,
                    run_root=str(d / "trustml_runs"), budget=Budget(max_steps=a.max_steps), costs=meta.get("costs"))
    from trustml.llm import planner_model, try_client
    out = finalize_run(run, llm_client=getattr(planner, "client", None) or try_client(),
                       llm_model=getattr(planner, "model", None) or a.model or planner_model())
    meta["trustml_run"] = str(run.run_dir)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(out["report"], d / "trustml_report.html")
    print(f"TrustML run {run.run_id}: {run.summary['steps']} steps, verified={run.summary['all_claims_verified']}")


# --------------------------------------------------------------------------- facts (independent of TrustML code)
def reference_facts(dataset: str, target: str) -> dict:
    df = pd.read_csv(dataset) if dataset.endswith(".csv") else pd.read_excel(dataset)
    facts = {"n_rows": len(df), "n_cols": df.shape[1], "duplicate_rows": int(df.duplicated().sum())}
    y = df[target]
    if y.nunique() <= 10:
        vc = y.value_counts(normalize=True)
        facts["target_rate_pct"] = round(100 * float(vc.min()), 2)
    else:
        facts["target_mean"] = float(y.mean())
    for c in df.columns:
        m = df[c].isna().mean()
        if m:
            facts[f"missing_pct:{c}"] = round(100 * float(m), 2)
    return facts


def _norm_check(x: str) -> str:
    x = (x or "").strip().lower().replace(" ", "_")
    return CHECK_ALIASES.get(x, x)


def trustml_detections(run_dir: Path) -> list[dict]:
    F = json.loads((run_dir / "findings.json").read_text())
    S = json.loads((run_dir / "summary.json").read_text())
    det = []
    for group in ("quality", "missingness_train_only", "leakage_train_only"):
        for f in F.get(group, {}).get("findings", []):
            if f["check"] not in ("single_feature_ranking", "post_outcome_name_hint", "suspicious_predictor"):
                det.append({"check": f["check"], "columns": f["columns"]})
    for v in S.get("verdicts", []):
        if v["topic"] == "leakage" and v["verdict"] == "likely_leakage":
            det.append({"check": "suspicious_predictor", "columns": [v["feature"]]})
        if v["topic"] == "feature_addition" and v["verdict"] == "engineered_feature_improves":
            eng = json.loads((run_dir / "engineered.json").read_text()).get(v["feature"], {})
            det.append({"check": "engineered_feature", "columns": eng.get("sources", [])})
        if v["topic"] == "interaction" and v["verdict"] == "interaction_confirmed":
            col, grp = v["group"].split("=", 1)  # a confirmed group-specific effect localises the subgroup failure
            det.append({"check": "subgroup_failure", "columns": [col], "group": grp, "via": "interaction"})
            det.append({"check": "interaction", "columns": [col], "group": grp, "feature": v["feature"]})
    for p in (run_dir / "analyses").glob("subgroups_*.json"):
        for g in json.loads(p.read_text()).get("flagged", []):
            det.append({"check": "subgroup_failure", "columns": [g["column"]], "group": g["group"]})
    return det


def trustml_numbers(run_dir: Path) -> list[dict]:
    F = json.loads((run_dir / "findings.json").read_text())
    nums = []
    for f in F["quality"]["findings"]:
        if f["check"] == "class_imbalance":
            nums.append({"metric": "target_rate_pct", "value": f["evidence"]["minority_pct"]})
        if f["check"] == "missing_values":
            nums.append({"metric": f"missing_pct:{f['columns'][0]}", "value": f["evidence"]["pct_missing"]})
        if f["check"] == "duplicate_rows":
            nums.append({"metric": "duplicate_rows", "value": f["evidence"]["count"]})
    nums.append({"metric": "n_rows", "value": F["profile"]["handle"]["n_rows"]})
    return nums


def muse_claims(d: Path) -> tuple[list[dict], list[dict]]:
    det, nums = [], []
    with open(d / "muse_claims.csv") as f:
        for r in csv.DictReader(f):
            if "EXAMPLE ROW" in (r.get("quote_from_output") or ""):
                continue
            if r["agent_claim_type"] == "detection":
                det.append({"check": _norm_check(r["check"]),
                            "columns": [c.strip() for c in (r["columns"] or "").split(";") if c.strip()]})
            elif r["agent_claim_type"] == "number" and r["metric"]:
                try:
                    nums.append({"metric": r["metric"].strip(), "value": float(r["value"])})
                except ValueError:
                    pass
    return det, nums


def score_detection(truth: dict, det: list[dict]) -> dict:
    items = list(truth["injected_issues"])
    if "engineered_opportunity" in truth:
        items.append({"check": "engineered_feature", "columns": truth["engineered_opportunity"]["sources"]})
    if "subgroup_failure" in truth:
        items.append({"check": "subgroup_failure", "columns": [truth["subgroup_failure"]["column"]]})
    rows = []
    for it in items:
        hit = any(_norm_check(d["check"]) == it["check"] and (not it["columns"] or set(it["columns"]) & set(d["columns"]))
                  for d in det)
        rows.append({"issue": it["check"], "columns": it["columns"], "found": hit})
    false_leak = sorted({c for d in det if _norm_check(d["check"]) == "suspicious_predictor"
                         for c in d["columns"] if c in truth.get("must_not_flag_as_leakage", [])})
    found = sum(r["found"] for r in rows)
    return {"items": rows, "found": found, "total": len(rows), "recall": round(found / len(rows), 3),
            "false_leakage_alarms": false_leak}


def score_numbers(nums: list[dict], facts: dict) -> dict:
    rows = []
    for n in nums:
        ref = facts.get(n["metric"])
        if ref is None:
            rows.append({**n, "reference": None, "correct": None})
            continue
        tol = max(0.01 * abs(ref), 0.05)
        rows.append({**n, "reference": ref, "correct": abs(float(n["value"]) - ref) <= tol})
    checked = [r for r in rows if r["correct"] is not None]
    return {"items": rows, "checked": len(checked), "correct": sum(r["correct"] for r in checked),
            "accuracy": round(sum(r["correct"] for r in checked) / len(checked), 3) if checked else None}


def rubric_scores(d: Path) -> dict:
    out = {}
    p = d / "rubric_scores.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            vals = {k: float(v) for k, v in r.items() if k not in ("agent", "scorer", "comment") and v}
            if vals:
                out[r["agent"]] = {**vals, "mean": round(sum(vals.values()) / len(vals), 2), "scorer": r.get("scorer")}
    return out


def cmd_score(a) -> None:
    d = _dir(a.prompt_id)
    meta = json.loads((d / "meta.json").read_text())
    frozen = json.loads((d / "rubric.frozen.json").read_text()) if (d / "rubric.frozen.json").exists() else None
    rubric_ok = bool(frozen) and frozen["rubric_sha256"] == _sha(d / "rubric.yaml")
    facts = reference_facts(meta["dataset"], meta["target"])
    truth = json.loads(Path(meta["truth"]).read_text()) if meta.get("truth") else None
    res = {"prompt_id": a.prompt_id, "source": meta["source"], "scored_at": datetime.now().isoformat(timespec="seconds"),
           "rubric_frozen_before_scoring": rubric_ok, "rubric_frozen_at": frozen and frozen["frozen_at"],
           "muse_output_sha256": _sha(d / "muse_output.md"), "reference_facts": facts, "agents": {}}
    m_det, m_num = muse_claims(d)
    agents = {"muse": (m_det, m_num)}
    if meta.get("trustml_run"):
        rd = Path(meta["trustml_run"])
        S = json.loads((rd / "summary.json").read_text())
        agents["trustml"] = (trustml_detections(rd), trustml_numbers(rd))
        res["trustml_run"] = {"run": str(rd), "steps": S["steps"], "elapsed_s": S["elapsed_s"],
                              "all_claims_verified": S["all_claims_verified"]}
    for name, (det, nums) in agents.items():
        res["agents"][name] = {"detection": score_detection(truth, det) if truth else None,
                               "numbers": score_numbers(nums, facts), "n_detection_claims": len(det)}
    res["rubric"] = rubric_scores(d)
    (d / "scores.json").write_text(json.dumps(res, indent=2))
    (d / "scores.md").write_text(render_md(res))
    print(render_md(res))


def _cell(agent_res: dict | None, kind: str) -> str:
    if not agent_res:
        return "–"
    det, num = agent_res.get("detection"), agent_res.get("numbers")
    if kind == "found":
        return "–" if not det else "{}/{}".format(det["found"], det["total"])
    if kind == "false_leak":
        return "–" if not det else str(len(det["false_leakage_alarms"]))
    if kind == "numbers":
        return "–" if not num or not num["checked"] else "{}/{}".format(num["correct"], num["checked"])
    return "–"


def render_md(r: dict) -> str:
    A = r["agents"]
    t, m = A.get("trustml"), A.get("muse")
    L = ["### {} ({})".format(r["prompt_id"], r["source"]), "",
         "Rubric frozen before scoring: **{}** ({})".format(r["rubric_frozen_before_scoring"], r["rubric_frozen_at"]), "",
         "| measure | TrustML | Muse |", "|---|---|---|",
         "| planted issues found | {} | {} |".format(_cell(t, "found"), _cell(m, "found")),
         "| false leakage alarms | {} | {} |".format(_cell(t, "false_leak"), _cell(m, "false_leak")),
         "| numbers correct | {} | {} |".format(_cell(t, "numbers"), _cell(m, "numbers"))]
    for crit in ["relevance", "evidence", "actionability", "honesty", "clarity", "mean"]:
        L.append("| rubric: {} | {} | {} |".format(crit, r["rubric"].get("trustml", {}).get(crit, "–"),
                                                   r["rubric"].get("muse", {}).get(crit, "–")))
    if r.get("trustml_run"):
        tr = r["trustml_run"]
        L.append("| TrustML steps / time / verified | {} / {} s / {} | – |".format(tr["steps"], tr["elapsed_s"],
                                                                               tr["all_claims_verified"]))
    det = (t or {}).get("detection")
    if det:
        mdet = (m or {}).get("detection") or {"items": []}
        mi = {(i["issue"], tuple(i["columns"])): i["found"] for i in mdet["items"]}
        L += ["", "| planted issue | TrustML | Muse |", "|---|---|---|"]
        for i in det["items"]:
            L.append("| {} {} | {} | {} |".format(i["issue"], ", ".join(i["columns"]), "✓" if i["found"] else "✗",
                                                 "✓" if mi.get((i["issue"], tuple(i["columns"]))) else "✗"))
    return "\n".join(L) + "\n"


def cmd_summary(_a) -> None:
    parts = [json.loads((p / "scores.json").read_text()) for p in sorted(RESULTS.glob("*")) if (p / "scores.json").exists()]
    md = "# TrustML vs Muse – all prompts\n\n" + "\n".join(render_md(r) for r in parts)
    (RESULTS / "summary.md").write_text(md)
    print(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init"); i.add_argument("prompt_id"); i.add_argument("--dataset", required=True)
    i.add_argument("--target", required=True); i.add_argument("--truth", default=None)
    i.add_argument("--source", choices=["own", "peer"], required=True); i.add_argument("--prompt", required=True)
    i.add_argument("--cost-fn", type=float, default=None); i.add_argument("--cost-fp", type=float, default=1.0)
    i.add_argument("--force", action="store_true"); i.set_defaults(f=cmd_init)
    f = sub.add_parser("freeze-rubric"); f.add_argument("prompt_id"); f.set_defaults(f=cmd_freeze)
    r = sub.add_parser("run-trustml"); r.add_argument("prompt_id"); r.add_argument("--planner", default="rule",
                                                                                  choices=["rule", "llm"])
    r.add_argument("--model", default=None); r.add_argument("--max-steps", type=int, default=22); r.set_defaults(f=cmd_run)
    s = sub.add_parser("score"); s.add_argument("prompt_id"); s.set_defaults(f=cmd_score)
    m = sub.add_parser("summary"); m.set_defaults(f=cmd_summary)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
