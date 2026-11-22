"""Ask TrustML: follow-up questions about a finished run, answered from evidence.

Each question is routed to an intent. Answers come either from stored, verified run data, or from a NEW
experiment run on the run's exact split (the test-set hash must match), verified independently.
The LLM may only (a) classify a question it cannot route by keywords and (b) phrase a general answer
from stored facts; numbers in LLM text are checked against the facts, as in the report.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

INTENTS = {
    "remove_feature": "What happens if a feature is removed? (runs a new experiment)",
    "threshold_for_recall": "Which decision threshold reaches a recall target? (runs a new experiment, binary)",
    "weak_spots": "Where / why does the model fail? (subgroups, missed cases)",
    "top_features": "Which features matter most? (SHAP and permutation importance)",
    "leakage": "Is there leakage? What evidence?",
    "data_quality": "What is wrong with the data?",
    "collect_data": "What additional data should be collected?",
    "model_choice": "Which model was chosen and why?",
    "time_drift": "Does the model hold up over time?",
    "fairness": "Does the model treat groups (e.g. gender, age) differently? (error rates per group)",
    "calibration": "Can the predicted probabilities be trusted? (Brier score, calibration error)",
    "interactions": "Does a feature act differently inside one group? (interaction screen and test)",
    "summary": "What did the analysis find overall? (conclusions and final model)",
}

KEYWORDS = [
    ("data_quality", r"wrong with the data|\bdata (quality|problems?|issues?)\b|problems? (with|in) the data|"
                     r"\b(missing values?|duplicates?|outliers?|placeholders?|sentinels?)\b|more than once|"
                     r"\brepeat\w*\b|same (customer|person|user|patient|entity)|multiple (applications|rows|records)"),
    ("fairness", r"\b(fair\w*|bias\w*|discriminat\w*|gender|sex|race|ethnic\w*|protected|equal\w* (treatment|error))\b"),
    ("calibration", r"calibrat\w*|\bbrier\b|\btrust\b.*probabilit|probabilit\w*.*\b(trust\w*|reliab\w*|accura\w*|realistic|mean)"),
    ("interactions", r"\binteraction\w*\b|differently (for|in|inside)|only (for|in) (one|a|the) (group|segment)"),
    ("summary", r"^(summari[sz]e|overall|what did (you|it|the agent|trustml) find|give me (a|the) summary|tl;?dr)"),
    ("threshold_for_recall", r"\bthreshold\b|\bcut-?off\b|recall (of|to|at|above|>=?)\s*0?\.\d|\d+ ?% recall"),
    ("remove_feature", r"\b(remove|drop|without|exclude|ablat\w*)\b"),
    ("leakage", r"\bleak\w*\b"),
    ("collect_data", r"\bcollect\b|\bnew data\b|additional (data|variables|features)|more data"),
    ("time_drift", r"\b(time|drift|future|over time|temporal)\b"),
    ("weak_spots", r"\b(miss\w*|why .*recall|recall .*low|low recall|wrong|fail\w*|weak\w*|worst|error\w*|subgroup\w*|"
                   r"least accurate|inaccura\w*|where .*(bad|poor)\w*|struggl\w*)\b"),
    ("top_features", r"\b(important|importance|matter\w*|drivers?|influential|shap|top features?)\b"),
    ("model_choice", r"\b(which|what|why) model\b|model (choice|chosen|selected)|random forest|gradient boost\w*|logistic"),
]


class Answer(BaseModel):
    question: str
    intent: str
    text: str
    evidence: list[dict] = []
    new_experiments: list[str] = []
    verification: dict = {}
    source: str = "stored results"
    seconds: float = 0.0


class FollowUp:
    def __init__(self, run_dir: str | Path, llm_client=None, llm_model: Optional[str] = None):
        self.dir = Path(run_dir)
        j = lambda n: json.loads((self.dir / n).read_text()) if (self.dir / n).exists() else {}  # noqa: E731
        self.run, self.summary, self.findings = j("run.json"), j("summary.json"), j("findings.json")
        self.explanations, self.narratives = j("explanations.json"), j("narratives.json")
        self.analyses = {p.stem: json.loads(p.read_text()) for p in (self.dir / "analyses").glob("*.json")} \
            if (self.dir / "analyses").exists() else {}
        self.columns = [c["name"] for c in self.findings.get("profile", {}).get("columns", [])]
        self.target = self.run.get("target")
        self.client, self.model = llm_client, llm_model
        self._ctx = None

    # ------------------------------------------------------------------ routing
    def find_feature(self, q: str) -> Optional[str]:
        ql = q.lower().replace("-", "_")
        hits = [c for c in self.columns if c != self.target and
                (c.lower() in ql or c.lower().replace("_", " ") in q.lower())]
        return max(hits, key=len) if hits else None

    @staticmethod
    def find_number(q: str) -> Optional[float]:
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
        if m:
            return float(m.group(1)) / 100
        m = re.search(r"\b(0?\.\d+)\b", q)
        return float(m.group(1)) if m else None

    def route(self, q: str) -> tuple[str, str]:
        ql = q.lower()
        for intent, pat in KEYWORDS:
            if re.search(pat, ql):
                if intent == "remove_feature" and not self.find_feature(q):
                    continue
                return intent, "keywords"
        if self.client is not None:
            try:
                from trustml.llm import chat
                r = chat(self.client, self.model, messages=[{"role": "user", "content":
                         "Classify the question into exactly one intent. Reply with JSON {\"intent\": \"...\"}.\n"
                         f"Intents: {json.dumps(INTENTS)}\nQuestion: {q}"}])
                from trustml.agent.planner import extract_json
                txt = r.choices[0].message.content or ""
                m = re.search(r"\{[^{}]*\"intent\"[^{}]*\}", txt)
                obj = json.loads(m.group(0)) if m else None
                if obj and obj.get("intent") in INTENTS:
                    return obj["intent"], f"llm:{self.model}"
            except Exception:  # noqa: BLE001
                pass
        return "general", "fallback"

    # ------------------------------------------------------------------ main
    def ask(self, question: str) -> Answer:
        t = time.time()
        intent, how = self.route(question)
        handler = getattr(self, f"_a_{intent}", self._a_general)
        ans = handler(question)
        ans.intent = intent if ans.intent == "?" else ans.intent
        ans.source = ans.source + (f" · routed by {how}" if how != "keywords" else "")
        ans.seconds = round(time.time() - t, 2)
        log = self.dir / "followups.jsonl"
        with open(log, "a") as f:
            f.write(ans.model_dump_json() + "\n")
        return ans

    def _A(self, q, intent, text, **kw) -> Answer:
        return Answer(question=q, intent=intent, text=text, **kw)

    # ------------------------------------------------------------------ stored-evidence answers
    def _final(self):
        return self.summary.get("final_model") or {}

    def _a_weak_spots(self, q):
        subs = sorted([v for k, v in self.analyses.items() if k.startswith("subgroups_")],
                      key=lambda a: a["exp_id"])
        final_id = self._final().get("exp_id")
        sub = next((a for a in subs if a["exp_id"] == final_id), subs[-1] if subs else None)
        parts, ev = [], []
        if sub and sub.get("flagged"):
            key = sub["metric"]
            for g in sub["flagged"][:3]:
                val = g.get("recall", g.get("mae"))
                n = g.get("n_positives", g.get("n_rows"))
                parts.append(f"{g['column']}={g['group']} ({key} {val:.3f} vs overall {sub['overall']:.3f}, n={n})")
                ev.append({"source": f"analyses/subgroups_{sub['exp_id']}.json", **{k: g[k] for k in ("column", "group", key)}})
            text = (f"The final model ({sub['exp_id']}) is weakest for: " + "; ".join(parts) +
                    ". These come from out-of-fold training predictions, so the test set was not used to find them.")
        elif sub:
            text = f"No subgroup of {sub['exp_id']} is clearly worse than overall {sub['metric']} {sub['overall']:.3f}."
        else:
            text = "Subgroup analysis was not run in this run."
        loc = [l for l in self.explanations.get("final", {}).get("local", []) if "missed" in l["case"] or "error" in l["case"]]
        if loc:
            top = ", ".join(f"{c['feature']}={c['value']} ({c['shap']:+.3g})" for c in loc[0]["contributions"][:3])
            text += f" Example {loc[0]['case']} (row {loc[0]['row_index']}): the strongest SHAP contributions were {top}."
            ev.append({"source": "explanations.json", "row": loc[0]["row_index"]})
        tm = self._final().get("test_metrics", {})
        if "recall" in tm:
            text += f" Overall test recall of the final model: {tm['recall']:.3f}."
        return self._A(q, "weak_spots", text, evidence=ev)

    def _a_top_features(self, q):
        ex = self.explanations.get("final", {})
        imp = self.analyses.get("importance", {})
        parts, ev = [], []
        if ex.get("supported"):
            g = ex["global"][:5]
            parts.append("SHAP (final model, mean |SHAP| in " + ex["output_space"] + "): " +
                         ", ".join(f"{r['feature']} {r['mean_abs_shap']:.3g}" for r in g))
            ev.append({"source": "explanations.json", "top": [r["feature"] for r in g]})
        if imp:
            parts.append(f"Permutation importance ({imp['exp_id']}, drop in CV {imp['scoring']}): " +
                         ", ".join(f"{r['feature']} {r['mean']:+.4f}" for r in imp["ranking"][:5]))
            ev.append({"source": "analyses/importance.json"})
        text = (". ".join(parts) + ". These describe how the model uses features, not causal effects.") if parts \
            else "No importance analysis is stored for this run."
        return self._A(q, "top_features", text, evidence=ev)

    def _a_leakage(self, q):
        vs = [v for v in self.summary.get("verdicts", []) if v["topic"] == "leakage"]
        lk = self.findings.get("leakage_train_only", {}).get("findings", [])
        sus = [f for f in lk if f["check"] == "suspicious_predictor"]
        if not sus and not vs:
            return self._A(q, "leakage", "The leakage screen flagged no feature: no single feature predicted the target "
                                         "suspiciously well on the training split.")
        parts = []
        for v in vs:
            parts.append(f"'{v['feature']}': {v['verdict']} (experiments {v['base']} → {v['new']})")
        for f in sus:
            if not any(v["feature"] in f["columns"] for v in vs):
                parts.append(f"'{f['columns'][0]}' was flagged ({f['message']}) but not ablated")
        trusted = self.run.get("trusted_features") or []
        extra = f" The user marked {trusted} as available at prediction time." if trusted else ""
        return self._A(q, "leakage", "Leakage findings: " + "; ".join(parts) + "." + extra +
                       " High predictive power alone does not prove leakage; confirm when each value is recorded.",
                       evidence=[{"source": "findings.json + summary.json verdicts"}])

    def _a_data_quality(self, q):
        fs = [f for g in ("quality", "missingness_train_only") for f in self.findings.get(g, {}).get("findings", [])]
        fs += self.findings.get("split_checks", [])
        hm = [f for f in fs if f["severity"] in ("high", "medium")]
        words = set(re.findall(r"[a-z]+", q.lower()))
        topic = {"repeated_ids": {"more", "once", "repeat", "repeated", "same", "multiple", "again", "twice"},
                 "duplicate_rows": {"duplicate", "duplicates", "copies"}, "missing_values": {"missing"},
                 "sentinel_value": {"placeholder", "placeholders", "sentinel", "sentinels", "99", "999"},
                 "class_imbalance": {"imbalance", "balanced", "rare"}}
        hm.sort(key=lambda f: 0 if topic.get(f["check"], set()) & words else 1)
        fixes = [v for v in self.summary.get("verdicts", []) if v["topic"] == "data_fix"]
        text = (f"{len(hm)} high/medium data issues: " + "; ".join(
            f"{f['check']} ({', '.join(f['columns']) or 'rows'})" + (f" – {f['message']}" if i == 0 and
                                                                    topic.get(f["check"], set()) & words else "")
            for i, f in enumerate(hm)) + ".")
        if fixes:
            text += " Fixes applied by the agent: " + "; ".join(
                f"{v['fix']}" + (f" by '{v['column']}'" if v.get("column") else "") + f" → {v['verdict']}"
                for v in fixes) + "."
        return self._A(q, "data_quality", text, evidence=[{"source": "findings.json"}])

    def _a_collect_data(self, q):
        c = self.narratives.get("collection") or []
        if not c:
            return self._A(q, "collect_data", "No data-collection suggestions were generated for this run (they need "
                                              "the NaviGator LLM).")
        text = "Suggested NEW variables (hypotheses, not tested): " + "; ".join(
            f"{s['name']} - {s['rationale']} (known at prediction time: {s['available_at_prediction_time']})" for s in c)
        return self._A(q, "collect_data", text, source="LLM suggestions stored in narratives.json (untested)")

    def _a_model_choice(self, q):
        vs = [v for v in self.summary.get("verdicts", []) if v["topic"] == "model_choice"]
        f = self._final()
        text = f"The final model is {f.get('exp_id')} ({(f.get('spec') or {}).get('model')})."
        if vs:
            text += " Model comparisons on identical CV folds: " + "; ".join(
                f"{v['model']} vs base {v['base']}: {v['verdict']}" for v in vs) + "."
        text += " A model is adopted only if it is better in every fold (binary tasks: without losing recall)."
        return self._A(q, "model_choice", text, evidence=[{"source": "summary.json verdicts"}])

    def _a_time_drift(self, q):
        ts = self.analyses.get("time_split")
        if not ts or ts.get("time_split_mean") is None:
            return self._A(q, "time_drift", "No forward-in-time check was run (no date column, or not evaluable).")
        drop = ts["random_cv_mean"] - ts["time_split_mean"]
        text = (f"Trained on earlier rows and tested on later ones (ordered by '{ts['date_col']}'), {ts['metric']} is "
                f"{ts['time_split_mean']:.3f} vs {ts['random_cv_mean']:.3f} with shuffled CV (drop {drop:+.3f}). " +
                ("That is a material drop: expect lower performance on future data." if drop > max(0.03, 2 * ts["random_cv_std"])
                 else "No material degradation."))
        return self._A(q, "time_drift", text, evidence=[{"source": "analyses/time_split.json"}])

    def _a_fairness(self, q):
        fr = {k[len("fairness_"):]: v for k, v in self.analyses.items() if k.startswith("fairness_")}
        if not fr:
            return self._A(q, "fairness", "No fairness check was run: no sensitive column (gender, sex, race, age, "
                                          "marital status, ...) was detected and none was chosen. Re-run with a "
                                          "fairness column selected to check error rates per group.")
        words = set(re.findall(r"[a-z_]+", q.lower()))
        cols = sorted(fr, key=lambda c: 0 if c.lower() in words else 1)
        parts = []
        for c in cols:
            r = fr[c]
            if r.get("metric") == "recall" and "gap" in r:
                line = (f"'{c}': recall lowest for {c}={r['worst_group']}, highest for {c}={r['best_group']}, gap {r['gap']:.3f} "
                        f"(95% CI {tuple(r['gap_ci95'])})")
                if r.get("false_alarm_gap") is not None:
                    line += f"; false-alarm rate gap {r['false_alarm_gap']:.3f}"
                    if r.get("false_alarm_highest"):
                        line += (f" (highest for {c}={r['false_alarm_highest']}, 95% CI "
                                 f"{tuple(r['false_alarm_gap_ci95'])})")
                line += " – material" if r["material"] else " – not material"
            elif r.get("metric") == "mae":
                line = (f"'{c}': absolute MAE ratio {r['ratio']}, relative-error ratio {r.get('relative_ratio')} "
                        f"({'material' if r['material'] else 'not material'}; judged on relative error)")
            else:
                line = f"'{c}': {r.get('note', 'not evaluable')}"
            parts.append(line)
        return self._A(q, "fairness", "Fairness checks on out-of-fold training predictions: " + "; ".join(parts) +
                       ". A gap is a signal for review, not a legal finding; dropping a sensitive column does not "
                       "remove proxies for it.", evidence=[{"source": f"analyses/fairness_{c}.json"} for c in cols])

    def _a_calibration(self, q):
        c = self.analyses.get("calibration")
        if not c:
            return self._A(q, "calibration", "Calibration was not checked in this run (it applies to binary "
                                             "classification).")
        text = (f"Calibration of {c['exp_id']} on out-of-fold predictions: Brier score {c['brier']:.4f}, expected "
                f"calibration error {c['ece']:.3f}; the model predicts {c['mean_predicted']:.3f} on average while the "
                f"actual rate is {c['actual_rate']:.3f}. ")
        text += ("The probabilities can be read as risk estimates." if c["well_calibrated"] else
                 "The probabilities should not be quoted as risk percentages" +
                 (" – class weighting inflates them by design; the ranking and the decision threshold are still "
                  "valid." if c["class_weighted"] else "; recalibrate first."))
        return self._A(q, "calibration", text, evidence=[{"source": "analyses/calibration.json"}])

    def _a_interactions(self, q):
        it = self.analyses.get("interactions")
        if not it:
            return self._A(q, "interactions", "The interaction screen was not run in this analysis.")
        vs = [v for v in self.summary.get("verdicts", []) if v["topic"] == "interaction"]
        if not it.get("candidates"):
            return self._A(q, "interactions", f"No feature behaved differently inside one group: {it['n_tested']} "
                                              f"feature × group pairs were screened and none reached |z| ≥ "
                                              f"{it['z_threshold']:g}.")
        c = it["candidates"][0]
        text = (f"The screen found that '{c['feature']}' relates to the errors differently when "
                f"{c['group_col']}={c['level']} (residual correlation {c['r_in']:+.3f} inside vs {c['r_out']:+.3f} "
                f"outside, z={c['z']:+.1f}).")
        for v in vs:
            text += (f" Tested as '{v['feature']}': {v['verdict']} ({v['group']} {v['group_metric']} "
                     f"{v['group_before']:.3f} → {v['group_after']:.3f}).")
        return self._A(q, "interactions", text, evidence=[{"source": "analyses/interactions.json"}])

    def _a_summary(self, q):
        cs = self.summary.get("conclusions", [])
        text = " ".join(cs[-6:]) if cs else "This run has no conclusions yet."
        cav = self.summary.get("caveats", [])
        if cav:
            text += " Main caveat: " + cav[0]
        return self._A(q, "summary", text, evidence=[{"source": "summary.json"}])

    def _a_general(self, q):
        supported = "; ".join(INTENTS.values())
        if self.client is None:
            return self._A(q, "general", "I can answer questions of these kinds: " + supported, source="help")
        from trustml.llm import chat
        from trustml.narrative import numbers_supported
        facts = {"conclusions": self.summary.get("conclusions", []), "caveats": self.summary.get("caveats", [])}
        try:
            r = chat(self.client, self.model, messages=[{"role": "user", "content":
                     "Answer the question in 2-4 sentences using ONLY these facts. If they do not contain the answer, "
                     f"say so. Do not add numbers.\nFacts: {json.dumps(facts)}\nQuestion: {q}"}])
            text = (r.choices[0].message.content or "").strip()
            ok, bad = numbers_supported(text, facts["conclusions"] + facts["caveats"])
            if text and ok:
                return self._A(q, "general", text, source=f"llm:{self.model} from stored conclusions (numbers checked)")
            return self._A(q, "general", "I could not answer that reliably from this run's results. I can answer: " +
                           supported, source=f"LLM answer rejected (unsupported numbers {bad})")
        except Exception as e:  # noqa: BLE001
            return self._A(q, "general", "I can answer questions of these kinds: " + supported,
                           source=f"help (LLM unavailable: {type(e).__name__})")

    # ------------------------------------------------------------------ new-experiment answers
    def _context(self):
        """Rebuild the run's exact data split (verified by the test-index hash)."""
        if self._ctx is not None:
            return self._ctx
        from trustml.experiments import RunContext
        from trustml.io import load_dataset
        from trustml.tools.feature_eng import apply_recipe
        from trustml.tools.profiler import profile_dataset
        sp = self.run["split"]
        df, h = load_dataset(self.run["data_path"])
        if h.sha256 != self.run["data_sha256"]:
            raise ValueError("The data file changed since this run; follow-up experiments would not be comparable.")
        prof = profile_dataset(df, h, target=self.target)
        fu_dir = self.dir / "followups" / time.strftime("%Y%m%d_%H%M%S")
        ctx = RunContext(df, prof, fu_dir, seed=sp["seed"], test_size=sp["test_size"], group_col=sp.get("group_col"),
                         costs=self.run.get("costs"), cv_repeats=self.run.get("cv_repeats", 1))
        if any(t["op"] == "deduplicate" for t in self.summary.get("transform_log", [])):
            ctx.deduplicate()
        eng = json.loads((self.dir / "engineered.json").read_text()) if (self.dir / "engineered.json").exists() else {}
        for name, rec in eng.items():
            ctx.add_engineered(name, rec, apply_recipe(ctx.data, rec))
        if ctx.split.test_index_sha != sp["test_index_sha"]:
            raise ValueError("Could not reproduce this run's train/test split; refusing to compare.")
        self._ctx, self._df = ctx, df
        return ctx

    def _verify(self, ctx, ids) -> dict:
        from trustml.verify import Verifier
        v = Verifier(ctx.run_dir, self._df, self.target, ctx.positive_class)
        res = {e: v.verify_experiment(e) for e in ids}
        return {"passed": all(r.passed for r in res.values()),
                "checks": {e: len(r.checks) for e, r in res.items()},
                "failed": {e: [c.name for c in r.checks if not c.passed] for e, r in res.items()}}

    def _base_spec(self, ctx):
        from trustml.experiments import ExperimentSpec
        spec = dict(self._final()["spec"])
        spec.update(name="final_rebuilt", parent_id=None)
        return ExperimentSpec(**spec)

    def _a_remove_feature(self, q):
        feat = self.find_feature(q)
        ctx = self._context()
        base_spec = self._base_spec(ctx)
        from trustml.experiments import ExperimentSpec, compare
        base = ctx.run(base_spec)
        if feat not in base.features_used:
            return self._A(q, "remove_feature", f"'{feat}' is not used by the final model "
                           f"(it was excluded or is not a usable feature), so removing it changes nothing.")
        if len(base.features_used) <= 1:
            return self._A(q, "remove_feature", f"'{feat}' is the only feature the final model uses, so it cannot be "
                                                "removed; the model would have no inputs.")
        d = base_spec.model_dump()
        d.update(name=f"minus_{feat}", exclude_features=d["exclude_features"] + [feat],
                 extra_features=[x for x in d["extra_features"] if feat not in ctx.engineered.get(x, {}).get("sources", [])],
                 sentinel_to_missing={k: v for k, v in d["sentinel_to_missing"].items() if k != feat})
        new = ctx.run(ExperimentSpec(**d))
        task = ctx.task.value
        m = {"binary_classification": "roc_auc", "regression": "r2"}.get(task, "balanced_accuracy")
        comp = compare(base, new, m)
        extra = ""
        if task == "binary_classification":
            rc = compare(base, new, "recall")
            extra = f"; CV recall {rc.base_mean:.3f} → {rc.new_mean:.3f}"
        direction = ("lower in every fold" if comp.consistent_direction and comp.mean_diff < 0 else
                     "higher in every fold" if comp.consistent_direction else "mixed across folds")
        text = (f"Removing '{feat}' from the final model: CV {m} {comp.base_mean:.4f} → {comp.new_mean:.4f} "
                f"({comp.mean_diff:+.4f}, {direction}){extra}. Test {m} {base.test_metrics[m]:.4f} → "
                f"{new.test_metrics[m]:.4f}. Compared on the run's identical split and folds.")
        return self._A(q, "remove_feature", text, new_experiments=[base.exp_id, new.exp_id],
                       verification=self._verify(ctx, [base.exp_id, new.exp_id]), source="new experiment",
                       evidence=[{"comparison": comp.model_dump()}])

    def _a_threshold_for_recall(self, q):
        ctx = self._context()
        if ctx.task.value != "binary_classification":
            return self._A(q, "threshold_for_recall", "Decision thresholds apply to binary classification only.")
        target = self.find_number(q) or 0.8
        if not 0 < target < 1:
            return self._A(q, "threshold_for_recall", "Give the recall target as a fraction or percentage, e.g. 0.9 or 90%.")
        from trustml.experiments import ExperimentSpec
        d = self._base_spec(ctx).model_dump()
        d.update(name=f"recall_{int(target * 100)}", threshold_strategy="target_recall", target_recall=target)
        new = ctx.run(ExperimentSpec(**d))
        cv, tm = new.cv_metrics, new.test_metrics
        text = (f"To reach recall ≥ {target:.2f} on out-of-fold training predictions, the threshold is "
                f"{new.threshold:.2f}. CV recall {cv['recall'].mean:.3f}, precision {cv['precision'].mean:.3f}. "
                f"On the untouched test set: recall {tm['recall']:.3f}, precision {tm['precision']:.3f}. "
                f"Higher recall costs precision: more false alarms per caught positive.")
        return self._A(q, "threshold_for_recall", text, new_experiments=[new.exp_id],
                       verification=self._verify(ctx, [new.exp_id]), source="new experiment")
