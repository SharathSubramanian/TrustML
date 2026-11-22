"""The TrustML agent: observe -> plan -> act -> verify, bounded by a budget.

LangGraph wires three nodes (plan, act, verify). All numbers come from tools;
all interpretations of numbers are deterministic functions in this file. The
planner (rule or LLM) only chooses which tool to run next.
"""
from __future__ import annotations

import json
import time

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from trustml.agent.actions import (AblateFeature, AblateFeatureGroup, AddEngineeredFeature, AnalyzeSubgroups,
                                   ApplyClassWeighting, CheckTimeSplit, DeduplicateRows, Finish, LogTransformTarget,
                                   RankFeatureImportance, RunBaseline, TreatSentinelsAsMissing, TryModel,
                                   TuneThreshold, UseGroupSplit, ScreenInteractions, InteractionTest, CheckFairness,
                                   CheckCalibration, signature)
from trustml.agent.planner import Decision, Planner, RulePlanner, validate_action
from trustml.agent.scope import scope_guard
from trustml.experiments import ExperimentResult, ExperimentSpec, RunContext, compare
from trustml.io import load_dataset
from trustml.schemas import Severity, TaskType
from trustml.tools.feature_eng import apply_recipe, suggest_engineered
from trustml.tools.importance import cv_permutation_importance
from trustml.tools.leakage import screen_leakage
from trustml.tools.missingness import missingness_vs_target
from trustml.tools.profiler import profile_dataset
from trustml.tools.quality import audit_quality
from trustml.tools.subgroups import subgroup_errors
from trustml.tools.diagnostics import (calibration_report, detect_sensitive, fairness_report, group_metric,
                                       screen_interactions)
from trustml.tools.feature_eng import interaction_name
from trustml.verify import Verifier


def infer_costs(objective: str):
    """Read a cost ratio such as 'a missed default costs 10 times more than ...' from the objective text."""
    import re
    t = objective.lower()
    if not re.search(r"\bcost", t):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:x|×|times)\b", t)
    if not m:
        return None
    ratio = float(m.group(1))
    if not 1 < ratio <= 1000:
        return None
    s0 = max(0, t.rfind(".", 0, m.start()) + 1)
    s1 = t.find(".", m.end())
    snippet = objective[s0: s1 if s1 > 0 else len(objective)].strip()
    # which error is the expensive one? default: missing a positive (false negative)
    fp_expensive = re.search(r"false (alarm|positive)s? costs? .{0,30}" + re.escape(m.group(0)), t)
    return ({"fn": 1.0, "fp": ratio} if fp_expensive else {"fn": ratio, "fp": 1.0}), snippet


def infer_recall_target(objective: str):
    """Read a recall goal such as 'catch at least 70% of defaulters' or 'recall of 0.8' from the objective."""
    import re
    t = objective.lower()
    pats = [r"(?:catch|detect|identify|find|flag|recall|capture)\w*\s+(?:at least|>=?|over|above|about)?\s*(\d{1,2}(?:\.\d+)?)\s*%",
            r"(\d{1,2}(?:\.\d+)?)\s*%\s+(?:recall|of (?:the )?(?:positive|churner|default|fraud|readmi|case)\w*)",
            r"recall\s+(?:of|at|above|>=?|at least)?\s*(0?\.\d+)"]
    for pat in pats:
        m = re.search(pat, t)
        if m:
            v = float(m.group(1))
            v = v / 100 if v > 1 else v
            if 0.3 <= v <= 0.99:
                s0 = max(0, t.rfind(".", 0, m.start()) + 1)
                s1 = t.find(".", m.end())
                return v, objective[s0: s1 if s1 > 0 else len(objective)].strip()
    return None


class Budget(BaseModel):
    max_steps: int = 22         # actions, excluding finish
    max_seconds: float = 600.0
    min_recall: float = 0.6     # target minority recall for binary tasks


class StepRecord(BaseModel):
    step: int
    planner: str
    planner_note: str = ""
    observation: str
    action: dict
    result: dict = {}
    verification: dict = {}
    elapsed_s: float = 0.0


def _fmt(ms) -> str:
    return f"{ms.mean:.3f} ± {ms.std:.3f}"


class AgentRun:
    def __init__(self, data_path: str, target: str, objective: str, planner: Optional[Planner] = None,
                 budget: Optional[Budget] = None, run_root: str = "runs", seed: int = 42,
                 group_col: Optional[str] = None, task_type: Optional[TaskType] = None,
                 costs: Optional[dict[str, float]] = None, on_step: Optional[Callable[[StepRecord], None]] = None,
                 run_id: Optional[str] = None, trusted_features: Optional[list[str]] = None,
                 fairness_cols: Optional[list[str]] = None, cv_repeats: int = 1):
        self.t0 = time.time()
        self.objective, self.budget = objective, budget or Budget()
        self.planner = planner or RulePlanner()
        self.on_step = on_step
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.run_dir = Path(run_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "analyses").mkdir(exist_ok=True)

        # ---- observe: deterministic audit
        self.df, self.handle = load_dataset(data_path)
        self.profile = profile_dataset(self.df, self.handle, target=target, task_type=task_type)
        self.quality = audit_quality(self.df, self.profile)
        self.out_of_scope = scope_guard(objective)
        self.costs_source = "user" if costs else None
        if not costs and self.profile.target.task_type == TaskType.BINARY:
            inferred = infer_costs(objective)
            if inferred:
                costs, self.costs_source = inferred[0], f"inferred from the objective: “{inferred[1]}”"
        self.cv_repeats = max(1, int(cv_repeats))
        self.ctx = RunContext(self.df, self.profile, self.run_dir, seed=seed, group_col=group_col, costs=costs,
                              cv_repeats=self.cv_repeats)
        self.recall_source = "default" if budget is None or "min_recall" not in budget.model_fields_set else "user"
        if self.recall_source == "default" and self.profile.target.task_type == TaskType.BINARY:
            rt = infer_recall_target(objective)
            if rt:
                self.budget = self.budget.model_copy(update={"min_recall": rt[0]})
                self.recall_source = f"inferred from the objective: “{rt[1]}”"
        self.trusted_features = [c for c in (trusted_features or []) if c in self.df.columns]
        self.facts: dict[str, Any] = {
            "baseline_id": None, "dummy_id": None, "working_id": None,
            "ablated": {}, "excluded_as_leak": [], "tried": set(),
            "verdicts": [], "done": {}, "noise_candidates": [], "subgroup_runs": {},
            "trusted_features": self.trusted_features,
            "sentinels": {c: [x.evidence["value"]] for x in self.quality.by_check("sentinel_value") for c in x.columns},
            "target_outliers": bool(self.quality.by_check("target_outliers")),
            "date_columns": [c.name for c in self.profile.columns if c.inferred_kind == "datetime"],
            "target_min": (float(self.ctx.y.min()) if self.ctx.task == TaskType.REGRESSION else None),
            "interaction_candidates": [], "fairness_runs": {},
            "fairness_user": [c for c in (fairness_cols or []) if c in self.df.columns and c != target],
        }
        auto = [c for c in detect_sensitive(self.ctx.candidate_features) if c not in self.facts["fairness_user"]
                and 2 <= self.df[c].nunique() and (self.df[c].nunique() <= 12 or pd.api.types.is_numeric_dtype(self.df[c]))]
        self.facts["fairness_columns"] = self.facts["fairness_user"] + auto
        for x in self.quality.by_check("sentinel_value"):   # several sentinels in one column
            for c in x.columns:
                if x.evidence["value"] not in self.facts["sentinels"][c]:
                    self.facts["sentinels"][c].append(x.evidence["value"])
        self.verifier = Verifier(self.run_dir, self.df, target, self.ctx.positive_class)
        self._train_split_audit()
        self.analyses: dict[str, dict] = {}
        self.steps: list[StepRecord] = []
        self.pending: Optional[Decision] = None
        self.last_observation = self._audit_observation()
        self.done, self.stop_reason = False, ""
        self.summary: dict = {}
        self._write_inputs(data_path, target)

    def _train_split_audit(self) -> None:
        """Checks that depend on the split (re-run after deduplication)."""
        self.split_findings = self.ctx.split_checks()
        train = self.ctx.data.iloc[self.ctx.train_pos]
        self.leakage = screen_leakage(train, self.profile)             # training split only
        self.missingness = missingness_vs_target(train, self.profile)  # training split only
        self.leak_scores = next(f for f in self.leakage.findings if f.check == "single_feature_ranking").evidence["ranking"]
        cands = [c for f in self.leakage.by_check("suspicious_predictor")
                 if f.severity in (Severity.HIGH, Severity.MEDIUM) for c in f.columns]
        self.facts["leak_candidates"] = [c for c in cands if c not in self.trusted_features]
        self.facts["trusted_leak_suspects"] = [c for c in cands if c in self.trusted_features]
        self.engineering = suggest_engineered(self.ctx, exclude=self.facts["leak_candidates"])
        self.facts["engineered_suggestions"] = self.engineering["suggestions"]
        n_dup = int(self.ctx.data.drop(columns=list(self.ctx.engineered)).duplicated().sum())
        dup_f = next((x for x in self.quality.by_check("duplicate_rows")), None)
        actionable = bool(dup_f and dup_f.evidence.get("actionable"))
        # only duplicates that look like copy errors are acted on (and required before the baseline)
        self.facts["n_duplicate_rows"] = n_dup if actionable else 0
        self.facts["n_identical_rows_expected"] = 0 if actionable else n_dup
        cands, ev = [], {}
        if not self.ctx.group_col and not self.facts["n_duplicate_rows"]:
            for x in sorted(self.split_findings, key=lambda x: -x.evidence.get("n_rows_involved", 0)):
                if x.check == "repeated_ids" and x.evidence["n_rows_involved"] >= 0.05 * len(self.ctx.data):
                    cands.append(x.columns[0])
                    ev[x.columns[0]] = (f"{x.evidence['n_repeated_values']} values cover "
                                        f"{x.evidence['n_rows_involved']} of {len(self.ctx.data)} rows")
        self.facts["group_candidates"], self.facts["group_evidence"] = cands, ev
        self.verifier = Verifier(self.run_dir, self.df, self.ctx.target, self.ctx.positive_class)

    # ------------------------------------------------------------------ views
    def all_findings(self):
        return self.quality.findings + self.split_findings + self.missingness.findings + self.leakage.findings

    def _audit_observation(self) -> str:
        hi = [f for f in self.all_findings() if f.severity in (Severity.HIGH, Severity.MEDIUM)]
        parts = [f"{f.check}({', '.join(f.columns) or 'rows'})" for f in hi]
        return f"Audit found {len(hi)} high/medium issues: " + "; ".join(parts)

    def planner_view(self) -> dict:
        """Compact state for the planner. Test-set metrics are intentionally excluded."""
        exps = []
        for e in self.ctx.experiments.values():
            keys = (["recall", "precision", "f1", "roc_auc", "pr_auc", "cost_per_row"]
                    if self.ctx.task == TaskType.BINARY else list(e.cv_metrics))
            exps.append({"exp_id": e.exp_id, "model": e.spec.model, "excluded": e.spec.exclude_features,
                         "extra_features": e.spec.extra_features, "class_weight": e.spec.class_weight,
                         "threshold_strategy": e.spec.threshold_strategy, "n_features": len(e.features_used),
                         "cv": {k: _fmt(e.cv_metrics[k]) for k in keys if k in e.cv_metrics}})
        findings = [{"check": f.check, "severity": f.severity.value, "columns": f.columns, "message": f.message}
                    for f in self.all_findings() if f.severity in (Severity.HIGH, Severity.MEDIUM)]
        return {
            "objective": self.objective, "target": self.ctx.target, "task_type": self.ctx.task.value,
            "positive_class": self.ctx.positive_class, "min_recall_target": self.budget.min_recall,
            "misclassification_costs": self.ctx.costs,
            "steps_used": len(self.steps), "max_steps": self.budget.max_steps,
            "findings": findings,
            "single_feature_leakage_scores_top5": dict(list(self.leak_scores.items())[:5]),
            "engineered_suggestions": [{k: s[k] for k in ("name", "recipe", "screen_gain")}
                                       for s in self.facts["engineered_suggestions"]],
            "noise_candidates": self.facts["noise_candidates"],
            "duplicate_rows_remaining": self.facts["n_duplicate_rows"],
            "group_split_candidates": self.facts["group_candidates"], "group_split_in_use": self.ctx.group_col,
            "sentinel_columns": self.facts["sentinels"], "target_has_extreme_outliers": self.facts["target_outliers"],
            "date_columns": self.facts["date_columns"],
            "user_trusted_features": self.trusted_features,
            "experiments_cv_only": exps, "working_exp_id": self.facts["working_id"],
            "verdicts_so_far": self.facts["verdicts"], "completed_investigations": sorted(self.facts["done"]),
            "actions_already_executed": [s.action for s in self.steps],  # never repeat one of these
            "subgroups_already_analyzed_for": sorted(self.facts["subgroup_runs"]),
            "interaction_candidates": self.facts["interaction_candidates"],
            "fairness_columns": self.facts["fairness_columns"], "fairness_checked": sorted(self.facts["fairness_runs"]),
            "cv_folds_per_experiment": self.ctx.cv_folds * (self.cv_repeats if not self.ctx.group_col else 1),
            "last_observation": self.last_observation,
            "out_of_scope_requests": [i.model_dump() for i in self.out_of_scope],
        }

    # ------------------------------------------------------------------ graph nodes
    def plan_node(self) -> None:
        if len(self.steps) >= self.budget.max_steps:
            self.pending = Decision(Finish(reason=f"Step budget reached ({self.budget.max_steps})."), "budget")
        elif time.time() - self.t0 > self.budget.max_seconds:
            self.pending = Decision(Finish(reason=f"Time budget reached ({self.budget.max_seconds}s)."), "budget")
        else:
            d = self.planner.decide(self)
            err = validate_action(d.action, self)
            if err:  # defence in depth: planners must not bypass validation
                d = Decision(Finish(reason=f"Planner proposed an invalid action ({err}); stopping safely."),
                             d.planner, d.note)
            self.pending = d
        if isinstance(self.pending.action, Finish):
            self.done, self.stop_reason = True, self.pending.action.reason
            self._record(StepRecord(step=len(self.steps) + 1, planner=self.pending.planner,
                                    planner_note=self.pending.note, observation=self.last_observation,
                                    action=self.pending.action.model_dump()))
            self._finalise()

    def act_node(self) -> None:
        t = time.time()
        a = self.pending.action
        self.facts["tried"].add(signature(a))
        handler = {RunBaseline: self._do_baseline, AblateFeature: self._do_ablation,
                   AblateFeatureGroup: self._do_group_ablation, ApplyClassWeighting: self._do_weighting,
                   TuneThreshold: self._do_threshold, RankFeatureImportance: self._do_importance,
                   AddEngineeredFeature: self._do_engineered, TryModel: self._do_try_model,
                   AnalyzeSubgroups: self._do_subgroups, DeduplicateRows: self._do_dedupe,
                   TreatSentinelsAsMissing: self._do_sentinels, LogTransformTarget: self._do_log_target,
                   CheckTimeSplit: self._do_time_split, UseGroupSplit: self._do_group_split,
                   ScreenInteractions: self._do_screen_interactions, InteractionTest: self._do_test_interaction,
                   CheckFairness: self._do_fairness, CheckCalibration: self._do_calibration}[type(a)]
        result, new_ids, observation = handler(a)
        self._current = StepRecord(step=len(self.steps) + 1, planner=self.pending.planner,
                                   planner_note=self.pending.note, observation=self.last_observation,
                                   action=a.model_dump(), result={**result, "new_experiments": new_ids})
        self._new_ids, self._next_obs, self._t_step = new_ids, observation, t

    def verify_node(self) -> None:
        results = [self.verifier.verify_experiment(e) for e in self._new_ids]
        exps = {r.exp_id: {"passed": r.passed, "n_checks": len(r.checks),
                           "failed": [c.model_dump() for c in r.checks if not c.passed]} for r in results}
        extra, note = [], ""
        act = self._current.action["action"]
        if act == "analyze_subgroups":
            extra = self.verifier.verify_subgroups(self._current.result, self.ctx)
        elif act == "check_time_split":
            extra = self.verifier.verify_time_split(self._current.result)
        elif act == "deduplicate_rows":
            extra = self.verifier.verify_dedupe(self._current.result, self.ctx)
        elif act == "use_group_split":
            extra = self.verifier.verify_group_split(self._current.result)
        elif act == "screen_interactions":
            extra = self.verifier.verify_interactions(self._current.result, self.ctx)
        elif act == "test_interaction":
            extra = self.verifier.verify_group_metrics(self._current.result, self.ctx)
        elif act == "check_fairness":
            extra = self.verifier.verify_fairness(self._current.result, self.ctx)
        elif act == "check_calibration":
            extra = self.verifier.verify_calibration(self._current.result)
        elif self._current.action["action"] == "rank_feature_importance":
            note = ("Permutation importance is stochastic and not recomputed; its noise candidates are only "
                    "acted on through an ablation experiment, which is verified.")
        ok = all(r.passed for r in results) and all(c.passed for c in extra)
        self._current.verification = {"passed": ok, "experiments": exps,
                                      "analysis_checks": [c.model_dump() for c in extra], "note": note}
        self._current.elapsed_s = round(time.time() - self._t_step, 2)
        self._record(self._current)
        self.last_observation = self._next_obs
        if not ok:
            self.done = True
            self.stop_reason = "Independent verification failed; stopping so no unverified result is reported."
            self._finalise()

    # ------------------------------------------------------------------ helpers
    def primary(self) -> str:
        return {TaskType.BINARY: "roc_auc", TaskType.REGRESSION: "r2", TaskType.MULTICLASS: "balanced_accuracy"}[self.ctx.task]

    def _child(self, base_id: str, name: str, rationale: str, **changes) -> ExperimentSpec:
        data = self.ctx.experiments[base_id].spec.model_dump()
        data.update(name=name, parent_id=base_id, rationale=rationale, **changes)
        # a column-level fix cannot outlive its column
        data["sentinel_to_missing"] = {k: v for k, v in (data.get("sentinel_to_missing") or {}).items()
                                       if k not in data["exclude_features"]}
        return ExperimentSpec(**data)

    def _working_note(self) -> str:
        w = self.ctx.experiments[self.facts["working_id"]]
        m = self.primary()
        s = f" Working model {w.exp_id}: CV {m} {_fmt(w.cv_metrics[m])}"
        if self.ctx.task == TaskType.BINARY:
            s += f", recall {_fmt(w.cv_metrics['recall'])}"
        return s + "."

    def _verdict(self, **v) -> None:
        self.facts["verdicts"].append(v)

    # ------------------------------------------------------------------ action handlers
    def _do_baseline(self, a: RunBaseline):
        d = self.ctx.run(ExperimentSpec(name="dummy", model="dummy", rationale="reference: predicts the prior/mean"))
        label = "linear" if a.model == "logreg" and self.ctx.task == TaskType.REGRESSION else a.model
        b = self.ctx.run(ExperimentSpec(name=f"baseline_{label}", model=a.model, rationale=a.rationale))
        self.facts.update(dummy_id=d.exp_id, baseline_id=b.exp_id, working_id=b.exp_id)
        m = self.primary()
        comp = compare(d, b, m)
        obs = f"Baseline {b.exp_id} CV {m} {_fmt(b.cv_metrics[m])} vs dummy {_fmt(d.cv_metrics[m])}."
        if self.ctx.task == TaskType.BINARY:
            obs += f" Minority recall {_fmt(b.cv_metrics['recall'])}."
        if self.facts["leak_candidates"]:
            obs += f" Leakage suspects still untested: {self.facts['leak_candidates']}."
        return {"comparison_vs_dummy": comp.model_dump(),
                "interpretation": obs}, [d.exp_id, b.exp_id], obs

    def _do_ablation(self, a: AblateFeature):
        base = self.ctx.experiments[a.base_exp_id]
        new = self.ctx.run(self._child(a.base_exp_id, f"minus_{a.feature}", a.rationale,
                                       exclude_features=base.spec.exclude_features + [a.feature]))
        m = self.primary()
        comp = compare(base, new, m)
        drop = -comp.mean_diff
        screen = self.leak_scores.get(a.feature)
        others = [v for k, v in self.leak_scores.items() if k != a.feature]
        next_best = max(others) if others else None
        high = 0.97 if self.ctx.task != TaskType.REGRESSION else 0.95
        if a.feature in self.trusted_features:
            verdict = "user_confirmed_available"
            text = (f"Removing '{a.feature}' changes CV {m} by {comp.mean_diff:+.3f}. The user confirmed this feature "
                    f"is available at prediction time, so it is kept.")
        elif screen is not None and drop >= 0.05 and comp.consistent_direction and (
                screen >= high or (screen >= 0.90 and next_best is not None and screen - next_best >= 0.25)):
            verdict = "likely_leakage"
            why = (f"on its own it scores {screen:.3f}" if screen >= high else
                   f"on its own it scores {screen:.3f}, far above the next-best feature ({next_best:.3f})")
            text = (f"Removing '{a.feature}' lowers CV {m} by {drop:.3f} (every fold), and {why}. This pattern is "
                    f"consistent with target leakage. It is NOT proof: confirm the value is unavailable at prediction "
                    f"time. Later experiments exclude it (conservative choice).")
            self.facts["working_id"] = new.exp_id
            self.facts["excluded_as_leak"].append(a.feature)
        elif drop >= 0.02 and comp.consistent_direction:
            verdict = "important_feature_leakage_uncertain"
            text = (f"Removing '{a.feature}' lowers CV {m} by {drop:.3f}. It is an important feature, but the "
                    f"evidence does not single it out as leakage; it is kept.")
        else:
            verdict = "no_material_dependence"
            text = f"Removing '{a.feature}' changes CV {m} by {comp.mean_diff:+.3f}; the model does not depend on it."
        self.facts["ablated"][a.feature] = new.exp_id
        self._verdict(topic="leakage", feature=a.feature, verdict=verdict, base=base.exp_id, new=new.exp_id, metric=m)
        return ({"comparison": comp.model_dump(), "single_feature_score": screen, "verdict": verdict,
                 "interpretation": text}, [new.exp_id], text + self._working_note())

    def _do_group_ablation(self, a: AblateFeatureGroup):
        self.facts["done"]["noise_ablation"] = True
        base = self.ctx.experiments[a.base_exp_id]
        new = self.ctx.run(self._child(a.base_exp_id, "minus_noise", a.rationale,
                                       exclude_features=base.spec.exclude_features + list(a.features),
                                       extra_features=[x for x in base.spec.extra_features if x not in a.features]))
        m = self.primary()
        comp = compare(base, new, m)
        tol = 0.005 if m != "r2" else 0.003
        if comp.mean_diff >= -tol and not (comp.consistent_direction and comp.mean_diff < 0):
            verdict = "simpler_model_no_material_loss"
            text = (f"Removing {list(a.features)} changes CV {m} by {comp.mean_diff:+.4f} (within ±{tol}, not a "
                    f"consistent loss): the simpler model with {len(new.features_used)} features is adopted.")
            self.facts["working_id"] = new.exp_id
        else:
            verdict = "removal_hurts"
            text = (f"Removing {list(a.features)} lowers CV {m} by {-comp.mean_diff:.4f}"
                    f"{' in every fold' if comp.consistent_direction else ''}; the features are kept despite low "
                    f"importance (importance is marginal; they matter jointly).")
        self._verdict(topic="feature_ablation", features=list(a.features), verdict=verdict, base=base.exp_id,
                      new=new.exp_id, metric=m)
        return ({"comparison": comp.model_dump(), "verdict": verdict, "interpretation": text},
                [new.exp_id], text + self._working_note())

    def _intervention(self, a, name: str, verdict_topic: str, **changes):
        base = self.ctx.experiments[a.base_exp_id]
        new = self.ctx.run(self._child(a.base_exp_id, name, a.rationale, **changes))
        keys = ["recall", "precision", "f1", "pr_auc"] + (["cost_per_row"] if self.ctx.costs else [])
        comps = {k: compare(base, new, k) for k in keys if k in base.cv_metrics}
        if changes.get("threshold_strategy") == "min_cost":
            c = comps["cost_per_row"]
            improved = c.mean_diff < 0 and (c.consistent_direction or c.no_fold_better)
            worsened = c.mean_diff > 0 and (c.consistent_direction or c.no_fold_worse)
            verdict = "cost_reduced" if improved else "cost_increased" if worsened else "no_consistent_cost_change"
            head = (f"{name}: CV cost per row {c.base_mean:.4f} → {c.new_mean:.4f} ({c.mean_diff:+.4f}, "
                    f"{'all folds improve' if improved else 'all folds worsen' if worsened else 'mixed across folds'})")
        else:
            r = comps["recall"]
            improved = r.mean_diff > 0 and (r.consistent_direction or r.no_fold_worse)
            worsened = r.mean_diff < 0 and (r.consistent_direction or r.no_fold_better)
            verdict = "recall_improved" if improved else "recall_worsened" if worsened else "no_consistent_recall_gain"
            how = ("all folds improve" if improved and r.consistent_direction else
                   "no fold worse, some unchanged" if improved else
                   "all folds worsen" if worsened and r.consistent_direction else
                   "no fold better" if worsened else "mixed across folds")
            head = f"{name}: CV recall {r.base_mean:.3f} → {r.new_mean:.3f} ({r.mean_diff:+.3f}, {how})"
        text = (head + f"; precision {comps['precision'].base_mean:.3f} → {comps['precision'].new_mean:.3f}; "
                f"PR-AUC {comps['pr_auc'].base_mean:.3f} → {comps['pr_auc'].new_mean:.3f}.")
        if new.threshold is not None and changes.get("threshold_strategy", "default") != "default":
            text += f" Chosen threshold {new.threshold:.2f} (from out-of-fold training predictions)."
        if improved:
            self.facts["working_id"] = new.exp_id
            text += " Adopted as working model."
        else:
            text += " Not adopted."
        self._verdict(topic=verdict_topic, intervention=name, verdict=verdict, base=base.exp_id, new=new.exp_id,
                      metric="cost_per_row" if changes.get("threshold_strategy") == "min_cost" else "recall")
        return ({"comparisons": {k: v.model_dump() for k, v in comps.items()}, "verdict": verdict,
                 "interpretation": text}, [new.exp_id], text)

    def _do_weighting(self, a: ApplyClassWeighting):
        self.facts["done"]["weighting"] = True
        return self._intervention(a, "class_weighted", "imbalance", class_weight="balanced")

    def _do_threshold(self, a: TuneThreshold):
        self.facts["done"]["threshold"] = True
        extra = {"target_recall": self.budget.min_recall} if a.strategy == "target_recall" else {}
        return self._intervention(a, f"threshold_{a.strategy}", "threshold", threshold_strategy=a.strategy, **extra)

    def _do_importance(self, a: RankFeatureImportance):
        self.facts["done"]["importance"] = True
        res = cv_permutation_importance(self.ctx, a.base_exp_id)
        self.facts["noise_candidates"] = res["noise_candidates"]
        self.analyses["importance"] = res
        (self.run_dir / "analyses" / "importance.json").write_text(json.dumps(res, indent=2))
        top = ", ".join(f"{r['feature']} {r['mean']:+.4f}" for r in res["ranking"][:4])
        text = (f"Permutation importance ({res['scoring']}, {res['n_folds']} folds) for {a.base_exp_id}: top {top}. "
                f"Noise candidates: {res['noise_candidates'] or 'none'}.")
        return {"analysis": "importance", "top": res["ranking"][:6], "noise_candidates": res["noise_candidates"],
                "interpretation": text}, [], text

    def _do_engineered(self, a: AddEngineeredFeature):
        self.facts["done"]["engineered"] = True
        sugg = next(s for s in self.facts["engineered_suggestions"] if s["name"] == a.feature_name)
        if a.feature_name not in self.ctx.engineered:
            self.ctx.add_engineered(a.feature_name, sugg["recipe"], apply_recipe(self.ctx.data, sugg["recipe"]))
        base = self.ctx.experiments[a.base_exp_id]
        new = self.ctx.run(self._child(a.base_exp_id, f"plus_{a.feature_name}", a.rationale,
                                       extra_features=base.spec.extra_features + [a.feature_name]))
        m = self.primary()
        comp = compare(base, new, m)
        if comp.mean_diff > 0 and comp.consistent_direction:
            verdict = "engineered_feature_improves"
            text = (f"Adding '{a.feature_name}' raises CV {m} by {comp.mean_diff:+.4f} in every fold; adopted. "
                    f"Derived row-wise from {sugg['recipe']['sources']}, so it is available at prediction time "
                    f"whenever its sources are.")
            self.facts["working_id"] = new.exp_id
        else:
            verdict = "no_consistent_gain"
            text = (f"Adding '{a.feature_name}' changes CV {m} by {comp.mean_diff:+.4f} "
                    f"({'consistent loss' if comp.consistent_direction else 'mixed across folds'}); not adopted, "
                    f"despite a positive screening score.")
        self._verdict(topic="feature_addition", feature=a.feature_name, verdict=verdict, base=base.exp_id,
                      new=new.exp_id, metric=m)
        return ({"comparison": comp.model_dump(), "recipe": sugg["recipe"], "verdict": verdict,
                 "interpretation": text}, [new.exp_id], text + self._working_note())

    def _do_try_model(self, a: TryModel):
        self.facts["done"]["model_switch_gb" if a.model == "gradient_boosting" else "model_switch"] = True
        if a.model == "gradient_boosting":
            self.facts["done"]["model_switch"] = True
        base = self.ctx.experiments[a.base_exp_id]
        new = self.ctx.run(self._child(a.base_exp_id, f"{a.model}", a.rationale, model=a.model))
        m = self.primary()
        comp = compare(base, new, m)
        ok = comp.mean_diff > 0 and comp.consistent_direction
        guard = ""
        if ok and self.ctx.task == TaskType.BINARY:
            rc = compare(base, new, "recall")
            if rc.mean_diff < -0.02:
                ok, guard = False, f" but recall falls {rc.mean_diff:+.3f}, which conflicts with the recall target"
        verdict = "model_improves" if ok else "no_consistent_gain"
        _lbl = lambda mm: "linear regression" if mm == "logreg" and self.ctx.task == TaskType.REGRESSION else mm  # noqa: E731
        text = (f"{a.model} vs {_lbl(base.spec.model)}: CV {m} {comp.base_mean:.4f} → {comp.new_mean:.4f} "
                f"({comp.mean_diff:+.4f}, {'every fold' if comp.consistent_direction else 'mixed across folds'}){guard}; "
                f"{'adopted' if ok else 'not adopted'}.")
        if ok:
            self.facts["working_id"] = new.exp_id
        self._verdict(topic="model_choice", model=a.model, verdict=verdict, base=base.exp_id, new=new.exp_id, metric=m)
        return ({"comparison": comp.model_dump(), "verdict": verdict, "interpretation": text},
                [new.exp_id], text + self._working_note())

    def _do_subgroups(self, a: AnalyzeSubgroups):
        res = subgroup_errors(self.ctx, a.base_exp_id)
        prev = self.analyses.get("subgroups")
        self.analyses.setdefault("subgroups_history", []).append(res)
        self.analyses["subgroups"] = res
        (self.run_dir / "analyses" / f"subgroups_{a.base_exp_id}.json").write_text(json.dumps(res, indent=2))
        self.facts["subgroup_runs"][a.base_exp_id] = [f"{g['column']}={g['group']}" for g in res.get("flagged", [])]
        key = "recall" if res.get("metric") == "recall" else "mae"

        def show(g):
            if key == "recall":
                return f"{g['column']}={g['group']} recall {g['recall']:.3f} (n_pos={g['n_positives']})"
            return f"{g['column']}={g['group']} MAE {g['mae']:.0f} ({g['mae_ratio_vs_rest']:.2f}× rest, n={g['n_rows']})"

        if not res.get("supported"):
            text = res["reason"]
        else:
            text = (f"Out-of-fold {res['metric']} overall {res['overall']:.3f}; " +
                    ("clearly worse groups: " + "; ".join(show(g) for g in res["flagged"][:3]) + "."
                     if res["flagged"] else "no group is clearly worse (|z| < 3)."))
        recheck = []
        if prev and prev.get("flagged") and prev["exp_id"] != a.base_exp_id:
            index = {(g["column"], g["group"]): g for g in res.get("all_groups", [])}
            for g in prev["flagged"][:3]:
                now = index.get((g["column"], g["group"]))
                if now:
                    recheck.append({"group": f"{g['column']}={g['group']}", "interval": now.get("interval"),
                                    "before_exp": prev["exp_id"],
                                    "after_exp": a.base_exp_id, "before": g[key], "after": now[key],
                                    "still_flagged": any((x["column"], x["group"]) == (g["column"], g["group"])
                                                         for x in res["flagged"])})
            if recheck:
                text += " Re-check of previously flagged groups: " + "; ".join(
                    f"{r['group']} {key} {r['before']:.3f} → {r['after']:.3f}"
                    f"{' (still flagged)' if r['still_flagged'] else ' (no longer flagged)'}" for r in recheck) + "."
        self._verdict(topic="subgroups", exp=a.base_exp_id, n_flagged=len(res.get("flagged", [])), recheck=recheck)
        return {"analysis": "subgroups", "exp_id": a.base_exp_id, "flagged": res.get("flagged", []),
                "overall": res.get("overall"), "metric": res.get("metric"), "recheck": recheck,
                "interpretation": text}, [], text

    def _do_dedupe(self, a: DeduplicateRows):
        self.facts["done"]["dedupe"] = True
        before = len(self.ctx.data)
        n = self.ctx.deduplicate()
        self._train_split_audit()
        text = (f"Removed {n} exact duplicate rows ({before:,} → {len(self.ctx.data):,}) before any model was trained, "
                f"and re-split: {self.ctx.split.n_train:,} train / {self.ctx.split.n_test:,} test. "
                f"Split checks, leakage screen and feature suggestions were re-run on the new training split.")
        self._verdict(topic="data_fix", fix="deduplicate", verdict="applied", rows_removed=n)
        return {"rows_removed": n, "rows_before": before, "rows_after": len(self.ctx.data),
                "interpretation": text}, [], text

    def _do_group_split(self, a: UseGroupSplit):
        self.facts["done"]["group_split"] = True
        res = self.ctx.use_group_split(a.group_col)
        self._train_split_audit()
        text = (f"Switched to a group split by '{a.group_col}': {res['n_groups']:,} groups, {res['n_train']:,} train / "
                f"{res['n_test']:,} test rows, and CV folds also keep each group together. No '{a.group_col}' value "
                f"appears on both sides, so the model is evaluated on entities it has never seen.")
        self._verdict(topic="data_fix", fix="group_split", verdict="applied", column=a.group_col)
        return {**res, "interpretation": text}, [], text

    def _do_sentinels(self, a: TreatSentinelsAsMissing):
        self.facts["done"]["sentinels"] = True
        base = self.ctx.experiments[a.base_exp_id]
        mapping = {**base.spec.sentinel_to_missing, **{c: self.facts["sentinels"][c] for c in a.columns}}
        new = self.ctx.run(self._child(a.base_exp_id, "sentinels_as_missing", a.rationale, sentinel_to_missing=mapping))
        m = self.primary()
        comp = compare(base, new, m)
        # a correctness fix is kept unless it costs MATERIAL performance (the pipeline still adds a missing
        # indicator, so a placeholder that carries signal keeps most of it)
        hurts = comp.mean_diff <= -0.005 and comp.consistent_direction
        if hurts:
            verdict = "fix_costs_performance"
            text = (f"Treating {mapping} as missing lowers CV {m} by {-comp.mean_diff:.4f} in every fold - the placeholder "
                    f"itself seems to carry signal (e.g. 'unknown' is informative). Not adopted; confirm with the data "
                    f"owner what the code means.")
        else:
            verdict = "fix_adopted"
            text = (f"Treating {mapping} as missing changes CV {m} by {comp.mean_diff:+.4f} "
                    f"({'better in every fold' if comp.consistent_direction and comp.mean_diff > 0 else 'no material loss'}). "
                    f"Adopted as a data-correctness fix: a placeholder is not a real measurement.")
            self.facts["working_id"] = new.exp_id
        self._verdict(topic="data_fix", fix="sentinels", verdict=verdict, base=base.exp_id, new=new.exp_id, metric=m,
                      columns=a.columns)
        return ({"comparison": comp.model_dump(), "mapping": mapping, "verdict": verdict, "interpretation": text},
                [new.exp_id], text + self._working_note())

    def _do_log_target(self, a: LogTransformTarget):
        self.facts["done"]["log_target"] = True
        base = self.ctx.experiments[a.base_exp_id]
        new = self.ctx.run(self._child(a.base_exp_id, "log_target", a.rationale, target_transform="log1p"))
        mae, r2 = compare(base, new, "mae"), compare(base, new, "r2")
        improved = mae.mean_diff < 0 and (mae.consistent_direction or mae.no_fold_better)
        verdict = "log_target_improves_mae" if improved else "no_consistent_gain"
        text = (f"Training on log1p(target): CV MAE {mae.base_mean:,.4g} → {mae.new_mean:,.4g} ({mae.mean_diff:+,.4g}, "
                f"{'no fold worse' if improved else 'not consistently better'}); R² {r2.base_mean:.3f} → {r2.new_mean:.3f}. "
                f"Predictions are back-transformed, so metrics stay on the original scale. "
                f"{'Adopted.' if improved else 'Not adopted.'}")
        if improved:
            self.facts["working_id"] = new.exp_id
        self._verdict(topic="data_fix", fix="log_target", verdict=verdict, base=base.exp_id, new=new.exp_id, metric="mae")
        return ({"comparison_mae": mae.model_dump(), "comparison_r2": r2.model_dump(), "verdict": verdict,
                 "interpretation": text}, [new.exp_id], text + self._working_note())

    def _do_time_split(self, a: CheckTimeSplit):
        self.facts["done"]["time_split"] = True
        res = self.ctx.time_split_eval(a.base_exp_id, a.date_col)
        self.analyses["time_split"] = res
        (self.run_dir / "analyses" / "time_split.json").write_text(json.dumps(res, indent=2))
        m = res["metric"]
        if res["time_split_mean"] is None:
            verdict, text = "not_evaluable", f"Could not build forward-in-time folds from '{a.date_col}'."
        else:
            drop = res["random_cv_mean"] - res["time_split_mean"]
            limit = max(0.03, 2 * (res["random_cv_std"] or 0))
            dr = res.get("target_drift") or {}
            if drop > limit:
                verdict = "degrades_forward_in_time"
            elif dr.get("material"):
                verdict = "target_drift"
            else:
                verdict = "stable_over_time"
            text = (f"Forward-in-time evaluation on '{a.date_col}' ({len(res['folds'])} folds, train on earlier rows, "
                    f"test on later): {m} {res['time_split_mean']:.3f} vs shuffled CV {res['random_cv_mean']:.3f} "
                    f"(drop {drop:+.3f}, threshold {limit:.3f}). ")
            if dr.get("kind") == "positive_rate":
                text += (f"Outcome rate {dr['early_rate']:.1%} in {dr['early_period']} vs {dr['late_rate']:.1%} in "
                         f"{dr['late_period']} ({dr['relative_change']:+.0%}, p={dr['p_value']:.2g}). ")
            elif dr.get("kind") == "target_mean":
                text += (f"Target mean {dr['early_mean']:,.4g} early vs {dr['late_mean']:,.4g} late "
                         f"({dr['effect_size_sd']:+.2f} sd, p={dr['p_value']:.2g}). ")
            text += {"degrades_forward_in_time": "Performance on newer rows is clearly worse: the shuffled estimate is "
                     "optimistic for future data. Report the forward-in-time number and investigate drift.",
                     "target_drift": "Ranking quality holds, but the outcome itself is drifting: a fixed threshold "
                     "will catch a different share of cases on future data, so monitor the rate and recalibrate or "
                     "retrain regularly.",
                     "stable_over_time": "No material degradation or outcome drift on newer rows."}[verdict]
        self._verdict(topic="time_split", exp=a.base_exp_id, verdict=verdict, date_col=a.date_col)
        return {"analysis": "time_split", **res, "verdict": verdict, "interpretation": text}, [], text

    def _do_screen_interactions(self, a: ScreenInteractions):
        self.facts["done"]["interaction_screen"] = True
        base = self.ctx.experiments[a.base_exp_id]
        # features removed as noise are screened too (a feature can look useless on average but matter in one group);
        # columns removed as leakage never are
        sent = {**self.facts["sentinels"], **base.spec.sentinel_to_missing}
        res = screen_interactions(self.ctx, a.base_exp_id, list(self.facts["excluded_as_leak"]), sent)
        self.facts["interaction_candidates"] = res["candidates"]
        self.analyses["interactions"] = res
        (self.run_dir / "analyses" / "interactions.json").write_text(json.dumps(res, indent=2))
        if res["candidates"]:
            c = res["candidates"][0]
            text = (f"Interaction screen on {a.base_exp_id} ({res['n_tested']} feature × group pairs, out-of-fold "
                    f"residuals): '{c['feature']}' behaves differently when {c['group_col']}={c['level']} "
                    f"(residual correlation {c['r_in']:+.3f} inside vs {c['r_out']:+.3f} outside, z={c['z']:+.1f}, "
                    f"n={c['n_in']}). This is a candidate only; test_interaction must confirm it.")
        else:
            text = (f"Interaction screen on {a.base_exp_id}: none of {res['n_tested']} feature × group pairs reached "
                    f"|z| ≥ {res['z_threshold']:g} with a residual-correlation difference ≥ {res['min_corr_diff']:g}; "
                    "no group-specific effect detected.")
        return {"analysis": "interactions", "exp_id": a.base_exp_id, "candidates": res["candidates"],
                "top": res["top10"][:5], "sentinels": sent, "excluded": list(self.facts["excluded_as_leak"]),
                "interpretation": text}, [], text

    def _do_test_interaction(self, a: InteractionTest):
        self.facts["done"]["interaction_test"] = True
        base = self.ctx.experiments[a.base_exp_id]
        sent = {**self.facts["sentinels"], **base.spec.sentinel_to_missing}
        recipe = {"op": "interaction", "sources": [a.feature, a.group_col], "level": a.level,
                  "mask": sent.get(a.feature, [])}
        name = interaction_name(a.feature, a.group_col, a.level)
        if name not in self.ctx.engineered:
            self.ctx.add_engineered(name, recipe, apply_recipe(self.ctx.data, recipe))
        excl = [x for x in base.spec.exclude_features if x not in (a.feature, a.group_col)]
        readded = [x for x in (a.feature, a.group_col) if x in base.spec.exclude_features]
        new = self.ctx.run(self._child(a.base_exp_id, f"plus_{name}", a.rationale, exclude_features=excl,
                                       extra_features=base.spec.extra_features + [name]))
        m = self.primary()
        comp = compare(base, new, m)
        g0, g1 = group_metric(self.ctx, base.exp_id, a.group_col, a.level), group_metric(self.ctx, new.exp_id, a.group_col, a.level)
        ok = comp.mean_diff > 0 and (comp.consistent_direction or comp.no_fold_worse)
        better_group = g1["value"] > g0["value"] if g1["metric"] == "recall" else g1["value"] < g0["value"]
        verdict = "interaction_confirmed" if ok else "interaction_not_confirmed"
        text = (f"Interaction '{name}'{' (re-adding ' + str(readded) + ', removed earlier as low-importance)' if readded else ''}: "
                f"CV {m} {comp.base_mean:.4f} → {comp.new_mean:.4f} ({comp.mean_diff:+.4f}, "
                f"{'every fold' if comp.consistent_direction else 'no fold worse' if comp.no_fold_worse else 'mixed across folds'}); "
                f"inside {a.group_col}={a.level} out-of-fold {g0['metric']} {g0['value']:.3f} → {g1['value']:.3f} "
                f"(n={g1['n']}). {'Adopted as working model.' if ok else 'Not adopted.'}")
        if ok:
            self.facts["working_id"] = new.exp_id
        self._verdict(topic="interaction", feature=name, verdict=verdict, base=base.exp_id, new=new.exp_id, metric=m,
                      group=f"{a.group_col}={a.level}", group_metric=g0["metric"], group_before=round(g0["value"], 4),
                      group_after=round(g1["value"], 4), readded=readded, group_improved=bool(better_group))
        return ({"comparison": comp.model_dump(), "recipe": recipe, "verdict": verdict, "interpretation": text,
                 "group_checks": [{"exp_id": base.exp_id, "column": a.group_col, "level": a.level, **g0},
                                  {"exp_id": new.exp_id, "column": a.group_col, "level": a.level, **g1}]},
                [new.exp_id], text + self._working_note())

    def _do_fairness(self, a: CheckFairness):
        res = fairness_report(self.ctx, a.base_exp_id, a.column)
        self.facts["fairness_runs"][a.column] = a.base_exp_id
        self.analyses.setdefault("fairness", {})[a.column] = res
        (self.run_dir / "analyses" / f"fairness_{a.column}.json").write_text(json.dumps(res, indent=2))
        if res.get("metric") == "recall" and "gap" in res:
            text = (f"Fairness on '{a.column}' ({a.base_exp_id}, out-of-fold): recall ranges from "
                    f"{min(g['recall'] for g in res['groups'] if g['group'] in (res['worst_group'], res['best_group'])):.3f} "
                    f"({a.column}={res['worst_group']}) to "
                    f"{max(g['recall'] for g in res['groups'] if g['group'] in (res['worst_group'], res['best_group'])):.3f} "
                    f"({a.column}={res['best_group']}); gap {res['gap']:.3f}, 95% CI {res['gap_ci95']}. ")
            text += ("The recall gap is material (≥ 0.10 and its CI excludes 0). " if res["recall_gap_material"]
                     else "No material recall gap (gap < 0.10 or its CI includes 0). ")
            if res.get("false_alarm_gap") is not None:
                text += (f"False-alarm rate is highest for {a.column}={res['false_alarm_highest']} and lowest for "
                         f"{a.column}={res['false_alarm_lowest']}: gap {res['false_alarm_gap']:.3f}, 95% CI "
                         f"{res['false_alarm_gap_ci95']}" + (" – material: this group is wrongly flagged much more "
                         "often. " if res["false_alarm_gap_material"] else " – not material. "))
            if res["material"]:
                text += "Review before deployment."
        elif res.get("metric") == "mae":
            text = (f"Fairness on '{a.column}' ({a.base_exp_id}, out-of-fold): absolute MAE differs {res['ratio']}× "
                    f"between groups, but relative error (MAE ÷ the group's mean actual value) differs "
                    f"{res['relative_ratio']}× ({a.column}={res['best_group']} best, {a.column}={res['worst_group']} worst). "
                    + ("Material (relative ratio ≥ 1.25×)." if res["material"] else
                       "Not material (relative ratio < 1.25×)" + (": larger absolute errors follow larger values."
                                                                    if (res["ratio"] or 0) >= 1.25 else ".")))
        else:
            text = f"Fairness on '{a.column}': {res.get('note', 'not evaluable')}."
        which = (", ".join(x for x, ok in [(f"recall gap {res.get('gap', 0):.3f}", res.get("recall_gap_material")),
                                           (f"false-alarm gap {res.get('false_alarm_gap') or 0:.3f}",
                                            res.get("false_alarm_gap_material"))] if ok)
                 if res.get("metric") == "recall" else f"relative MAE ratio {res.get('relative_ratio')}")
        self._verdict(topic="fairness", column=a.column, exp=a.base_exp_id, material=bool(res.get("material")),
                      gap=res.get("gap"), metric=res.get("metric"), which=which or "none",
                      false_alarm_gap=res.get("false_alarm_gap"))
        return {"analysis": "fairness", **res, "interpretation": text}, [], text

    def _do_calibration(self, a: CheckCalibration):
        self.facts["done"]["calibration"] = True
        res = calibration_report(self.ctx, a.base_exp_id)
        self.analyses["calibration"] = res
        (self.run_dir / "analyses" / "calibration.json").write_text(json.dumps(res, indent=2))
        text = (f"Calibration of {a.base_exp_id} (out-of-fold): Brier {res['brier']:.4f}, expected calibration error "
                f"{res['ece']:.3f}; mean predicted probability {res['mean_predicted']:.3f} vs actual rate "
                f"{res['actual_rate']:.3f}. ")
        if res["well_calibrated"]:
            text += "Probabilities are usable as risk estimates (ECE < 0.05)."
        else:
            text += ("Probabilities are NOT reliable risk estimates (ECE ≥ 0.05)"
                     + (" - expected, because class weighting inflates them; use the model for ranking and the chosen "
                        "threshold, or recalibrate before quoting probabilities." if res["class_weighted"] else
                        "; recalibrate before quoting probabilities."))
        self._verdict(topic="calibration", exp=a.base_exp_id, ece=res["ece"], brier=res["brier"],
                      well_calibrated=res["well_calibrated"], class_weighted=res["class_weighted"])
        return {"analysis": "calibration", **res, "interpretation": text}, [], text

    # ------------------------------------------------------------------ output
    def _record(self, step: StepRecord) -> None:
        self.steps.append(step)
        with open(self.run_dir / "steps.jsonl", "a") as f:
            f.write(step.model_dump_json() + "\n")
        if self.on_step:
            self.on_step(step)

    def _write_findings(self) -> None:
        (self.run_dir / "findings.json").write_text(json.dumps({
            "profile": self.profile.model_dump(mode="json"),
            "quality": self.quality.model_dump(mode="json"),
            "split_checks": [f.model_dump(mode="json") for f in self.split_findings],
            "missingness_train_only": self.missingness.model_dump(mode="json"),
            "leakage_train_only": self.leakage.model_dump(mode="json"),
            "engineered_screen": self.engineering,
            "out_of_scope": [i.model_dump() for i in self.out_of_scope],
            "trusted_features": self.trusted_features, "costs_source": self.costs_source}, indent=2, default=str))

    def _write_inputs(self, data_path: str, target: str) -> None:
        (self.run_dir / "run.json").write_text(json.dumps({
            "run_id": self.run_id, "data_path": str(data_path), "data_sha256": self.handle.sha256, "target": target,
            "objective": self.objective, "planner": getattr(self.planner, "name", "?"),
            "budget": self.budget.model_dump(), "costs": self.ctx.costs, "split": self.ctx.split.model_dump(),
            "trusted_features": self.trusted_features, "costs_source": self.costs_source,
            "cv_repeats": self.cv_repeats, "fairness_columns": self.facts["fairness_columns"],
            "recall_target_source": self.recall_source},
            indent=2))
        (self.run_dir / "findings.json").write_text(json.dumps({
            "profile": self.profile.model_dump(mode="json"),
            "quality": self.quality.model_dump(mode="json"),
            "split_checks": [f.model_dump(mode="json") for f in self.split_findings],
            "missingness_train_only": self.missingness.model_dump(mode="json"),
            "leakage_train_only": self.leakage.model_dump(mode="json"),
            "engineered_screen": self.engineering,
            "out_of_scope": [i.model_dump() for i in self.out_of_scope]}, indent=2, default=str))

    def _finalise(self) -> None:
        """Deterministic conclusions; every quoted number is re-checked against stored records."""
        self._write_findings()  # split-dependent findings may have changed (deduplication)
        (self.run_dir / "run.json").write_text(json.dumps({
            **json.loads((self.run_dir / "run.json").read_text()), "split": self.ctx.split.model_dump()}, indent=2))
        E, f = self.ctx.experiments, self.facts
        conclusions, claims = [], []

        def claim(exp_id, metric, source="cv", nd=3):
            e = E[exp_id]
            v = e.cv_metrics[metric].mean if source == "cv" else e.test_metrics[metric]
            claims.append((exp_id, metric, round(v, 4), source))
            return round(v, nd)

        m = self.primary()
        if f["baseline_id"]:
            b, d = f["baseline_id"], f["dummy_id"]
            conclusions.append(f"Baseline {b}: CV {m} {claim(b, m)} vs dummy {claim(d, m)}.")
        for v in f["verdicts"]:
            t = v["topic"]
            if t == "leakage":
                conclusions.append(f"Leakage check on '{v['feature']}': {v['verdict']} "
                                   f"(CV {m} {claim(v['base'], m)} → {claim(v['new'], m)}).")
            elif t in ("imbalance", "threshold"):
                met = v["metric"]
                conclusions.append(f"{v['intervention']}: {v['verdict']} (CV {met} {claim(v['base'], met, nd=4)} → "
                                   f"{claim(v['new'], met, nd=4)}).")
            elif t == "feature_ablation":
                conclusions.append(f"Removing low-importance {v['features']}: {v['verdict']} "
                                   f"(CV {m} {claim(v['base'], m, nd=4)} → {claim(v['new'], m, nd=4)}).")
            elif t == "feature_addition":
                conclusions.append(f"Engineered feature '{v['feature']}': {v['verdict']} "
                                   f"(CV {m} {claim(v['base'], m, nd=4)} → {claim(v['new'], m, nd=4)}).")
            elif t == "data_fix" and v["fix"] == "deduplicate":
                conclusions.append(f"Removed {v['rows_removed']} duplicate rows before splitting.")
            elif t == "data_fix" and v["fix"] == "group_split":
                conclusions.append(f"Evaluated with a group split by '{v['column']}' (no entity in both train and test).")
            elif t == "data_fix":
                met = v["metric"]
                conclusions.append(f"Data fix '{v['fix']}': {v['verdict']} (CV {met} {claim(v['base'], met, nd=4)} → "
                                   f"{claim(v['new'], met, nd=4)}).")
            elif t == "time_split":
                ts = self.analyses.get("time_split", {})
                if ts.get("time_split_mean") is not None:
                    line = (f"Forward-in-time check on '{v['date_col']}': {v['verdict']} "
                            f"({ts['metric']} {ts['time_split_mean']:.3f} on later rows vs shuffled CV "
                            f"{ts['random_cv_mean']:.3f})")
                    dr = ts.get("target_drift") or {}
                    if dr.get("kind") == "positive_rate":
                        line += (f"; outcome rate {100 * dr['early_rate']:.1f}% in {dr['early_period']} vs "
                                 f"{100 * dr['late_rate']:.1f}% in {dr['late_period']}")
                    elif dr.get("kind") == "target_mean":
                        line += f"; target mean {dr['early_mean']:,.4g} early vs {dr['late_mean']:,.4g} late"
                    conclusions.append(line + ".")
            elif t == "interaction":
                conclusions.append(f"Interaction '{v['feature']}': {v['verdict']} (CV {m} {claim(v['base'], m, nd=4)} → "
                                   f"{claim(v['new'], m, nd=4)}; {v['group']} {v['group_metric']} "
                                   f"{v['group_before']:.3f} → {v['group_after']:.3f}, out-of-fold).")
            elif t == "fairness" and v.get("gap") is not None:
                fa = f", false-alarm gap {v['false_alarm_gap']:.3f}" if v.get("false_alarm_gap") is not None else ""
                conclusions.append(f"Fairness on '{v['column']}' ({v['exp']}): {v['metric']} gap {v['gap']:.3f}{fa}; "
                                   f"{'material (' + v['which'] + ')' if v['material'] else 'not material'}.")
            elif t == "calibration":
                conclusions.append(f"Calibration of {v['exp']}: ECE {v['ece']:.3f}, Brier {v['brier']:.4f}; "
                                   f"{'well calibrated' if v['well_calibrated'] else 'probabilities need recalibration'}.")
            elif t == "model_choice":
                conclusions.append(f"Model {v['model']}: {v['verdict']} "
                                   f"(CV {m} {claim(v['base'], m, nd=4)} → {claim(v['new'], m, nd=4)}).")
        for v in [v for v in f["verdicts"] if v["topic"] == "subgroups"]:
            sg = next(h for h in self.analyses["subgroups_history"] if h["exp_id"] == v["exp"])
            if sg.get("flagged"):
                g = sg["flagged"][0]
                conclusions.append(f"Weakest subgroup for {v['exp']}: {g['column']}={g['group']} "
                                   f"({sg['metric']} {g.get('recall', g.get('mae')):.3f} vs overall "
                                   f"{sg['overall']:.3f}; out-of-fold).")
            for r in v.get("recheck", []):
                conclusions.append(f"Subgroup re-check {r['group']}: {r['before']:.3f} ({r['before_exp']}) → "
                                   f"{r['after']:.3f} ({r['after_exp']}); "
                                   f"{'still flagged' if r['still_flagged'] else 'no longer flagged'}.")
        final = None
        if f["working_id"]:
            w = E[f["working_id"]]
            final = {"exp_id": w.exp_id, "spec": w.spec.model_dump(), "test_metrics": w.test_metrics,
                     "test_ci95": w.test_ci95, "threshold": w.threshold, "features_used": w.features_used}
            keys = ["recall", "precision", "roc_auc"] if self.ctx.task == TaskType.BINARY else [m]
            if self.ctx.task == TaskType.REGRESSION:
                keys = ["r2", "rmse"]
            parts = [f"{k} {claim(w.exp_id, k, 'test')} (95% CI {w.test_ci95.get(k)})" for k in keys]
            conclusions.append(f"Final model {w.exp_id}, evaluated once on the untouched test set: " + ", ".join(parts) + ".")
        checks = [self.verifier.verify_claim(e, met, val, src) for e, met, val, src in claims]

        handled = set()
        if self.ctx.group_col:
            handled.add(("repeated_ids", self.ctx.group_col))
        if f["done"].get("dedupe"):
            handled |= {("train_test_overlap", None), ("repeated_ids", "*dup")}
        ts_done = bool(self.analyses.get("time_split", {}).get("time_split_mean") is not None)
        caveats = []
        for x in self.quality.findings + self.split_findings + self.missingness.findings:
            if x.check not in ("possible_missing_header", "small_sample", "target_outliers", "datetime_present",
                               "train_test_overlap", "repeated_ids", "informative_missingness"):
                continue
            col = x.columns[0] if x.columns else None
            if (x.check, col) in handled or (x.check == "train_test_overlap" and f["done"].get("dedupe")):
                continue
            if x.check == "datetime_present" and ts_done:
                continue  # replaced by the forward-in-time result below
            caveats.append(x.message)
        if self.ctx.group_col:
            caveats.append(f"Rows of the same '{self.ctx.group_col}' were kept on one side of the split, so scores "
                           f"reflect entities the model has never seen.")
        for v in f["verdicts"]:
            if v["topic"] == "time_split" and v["verdict"] in ("target_drift", "degrades_forward_in_time"):
                ts = self.analyses["time_split"]
                dr = ts.get("target_drift") or {}
                msg = (f"The model may not hold up on future data: forward-in-time {ts['metric']} "
                       f"{ts['time_split_mean']:.3f} vs {ts['random_cv_mean']:.3f} shuffled")
                if dr.get("kind") == "positive_rate" and dr.get("material"):
                    msg += (f"; the outcome rate changed from {100 * dr['early_rate']:.1f}% to "
                            f"{100 * dr['late_rate']:.1f}%, so a fixed decision threshold will flag a different share "
                            f"of cases - monitor and recalibrate")
                caveats.insert(0, msg + ".")
        for v in f["verdicts"]:
            if v["topic"] == "fairness" and v["material"]:
                caveats.append(f"Error rates differ materially across '{v['column']}' groups ({v['which']}); "
                               "review with domain and legal owners before use.")
            if v["topic"] == "calibration" and not v["well_calibrated"]:
                caveats.append(f"Predicted probabilities are not calibrated (ECE {v['ece']:.3f}); use rankings/decisions, "
                               "not the raw probabilities, unless they are recalibrated.")
        if self.cv_repeats > 1 and not self.ctx.group_col:
            caveats.append(f"Cross-validation was repeated {self.cv_repeats}× with different shuffles; fold-level "
                           "comparisons use all repeats.")
        if f["excluded_as_leak"]:
            caveats.append("Leakage verdicts are evidence-based suspicions; prediction-time availability of "
                           f"{f['excluded_as_leak']} must be confirmed with the data owner.")
        if self.recall_source.startswith("inferred"):
            caveats.append(f"Target recall {self.budget.min_recall:g} was {self.recall_source}.")
        if self.costs_source and self.costs_source != "user":
            caveats.append(f"Misclassification costs FN={self.ctx.costs['fn']:g}, FP={self.ctx.costs['fp']:g} were "
                           f"{self.costs_source}. Check they match your intent.")
        if self.trusted_features:
            caveats.append(f"The user marked {self.trusted_features} as available at prediction time; they were not "
                           "excluded as leakage.")
        caveats.append("Several experiments were each scored on the same test set for reporting; all decisions "
                       "were made on cross-validation only, but the final test score is still a single estimate.")
        self.summary = {
            "run_id": self.run_id, "objective": self.objective, "stop_reason": self.stop_reason,
            "steps": len(self.steps), "elapsed_s": round(time.time() - self.t0, 2),
            "out_of_scope": [i.model_dump() for i in self.out_of_scope],
            "conclusions": conclusions, "final_model": final,
            "baseline_test_metrics": E[f["baseline_id"]].test_metrics if f["baseline_id"] else None,
            "claim_checks": [c.model_dump() for c in checks],
            "all_claims_verified": all(c.passed for c in checks),
            "caveats": caveats, "transform_log": self.ctx.transform_log, "verdicts": f["verdicts"],
        }
        (self.run_dir / "summary.json").write_text(json.dumps(self.summary, indent=2, default=str))


# ---------------------------------------------------------------------- LangGraph wiring
class _S(TypedDict):
    run: Any


def build_graph():
    g = StateGraph(_S)
    g.add_node("plan", lambda s: (s["run"].plan_node(), s)[1])
    g.add_node("act", lambda s: (s["run"].act_node(), s)[1])
    g.add_node("verify", lambda s: (s["run"].verify_node(), s)[1])
    g.set_entry_point("plan")
    g.add_conditional_edges("plan", lambda s: END if s["run"].done else "act")
    g.add_edge("act", "verify")
    g.add_conditional_edges("verify", lambda s: END if s["run"].done else "plan")
    return g.compile()


def run_agent(data_path: str, target: str, objective: str, **kwargs) -> AgentRun:
    run = AgentRun(data_path, target, objective, **kwargs)
    build_graph().invoke({"run": run}, config={"recursion_limit": 3 * run.budget.max_steps + 10})
    return run
