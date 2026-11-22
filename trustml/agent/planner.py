"""Planners choose the next action from a compact, test-set-free state summary.

RulePlanner  - deterministic policy; always available (demo insurance).
LLMPlanner   - NaviGator model chooses via function calling; every choice is
               schema-validated and checked against the run state. Invalid or
               failed calls fall back to RulePlanner for that step (logged).
"""
from __future__ import annotations

import json
from typing import Any, Optional, Protocol

from pydantic import ValidationError

from trustml.agent.actions import (ACTION_ADAPTER, AblateFeature, AblateFeatureGroup, AddEngineeredFeature,
                                   AnalyzeSubgroups, ApplyClassWeighting, CheckTimeSplit, DeduplicateRows, Finish,
                                   LogTransformTarget, RankFeatureImportance, RunBaseline, TreatSentinelsAsMissing,
                                   TryModel, TuneThreshold, UseGroupSplit, ScreenInteractions, InteractionTest,
                                   CheckFairness, CheckCalibration, signature, tool_definitions)


class Decision:
    def __init__(self, action, planner: str, note: str = ""):
        self.action, self.planner, self.note = action, planner, note


class Planner(Protocol):
    def decide(self, run: Any) -> Decision: ...


# --------------------------------------------------------------------------- validation
def validate_action(action, run) -> Optional[str]:
    """Return an error message if the action is not allowed in the current state."""
    f, ctx = run.facts, run.ctx
    task = ctx.task.value
    if signature(action) in f["tried"]:
        return "this exact action was already executed"
    if isinstance(action, Finish):
        return None
    if isinstance(action, DeduplicateRows):
        if ctx.experiments:
            return "deduplication is only allowed before the first experiment"
        return None if f["n_duplicate_rows"] else "there are no duplicate rows"
    if isinstance(action, UseGroupSplit):
        if ctx.experiments:
            return "the split can only be changed before the first experiment"
        if ctx.group_col:
            return f"a group split by '{ctx.group_col}' is already in use"
        if action.group_col not in f["group_candidates"]:
            return f"'{action.group_col}' is not a repeated-ID column (candidates: {f['group_candidates']})"
        return None
    if isinstance(action, RunBaseline):
        if f["baseline_id"]:
            return "baseline already exists"
        if f["n_duplicate_rows"] and not f["done"].get("dedupe"):
            return "deduplicate_rows must come before run_baseline (duplicates would leak across the split)"
        if f["group_candidates"] and not ctx.group_col:
            return (f"use_group_split must come before run_baseline: {f['group_candidates']} repeat across rows, so the "
                    "same entity would appear in train and test")
        return None
    if not f["baseline_id"]:
        return "run_baseline must come first"
    if action.base_exp_id not in ctx.experiments:
        return f"unknown experiment id '{action.base_exp_id}'"
    base = ctx.experiments[action.base_exp_id]
    if base.spec.model == "dummy":
        return "the dummy model cannot be a base experiment"
    if isinstance(action, AblateFeature):
        if action.feature not in base.features_used:
            return f"feature '{action.feature}' is not used by {action.base_exp_id}"
        if len(base.features_used) <= 1:
            return "cannot remove the last feature"
    if isinstance(action, AblateFeatureGroup):
        missing = [x for x in action.features if x not in base.features_used]
        if missing:
            return f"features {missing} are not used by {action.base_exp_id}"
        if len(base.features_used) - len(set(action.features)) < 1:
            return "cannot remove every feature"
    if isinstance(action, ApplyClassWeighting):
        if task == "regression":
            return "class weighting needs a classification task"
        if base.spec.class_weight == "balanced":
            return "base experiment is already class-weighted"
    if isinstance(action, TuneThreshold):
        if task != "binary_classification":
            return "threshold tuning needs a binary task"
        if base.spec.threshold_strategy != "default":
            return "base experiment already has a tuned threshold"
        if action.strategy == "min_cost" and not ctx.costs:
            return "min_cost needs misclassification costs (none were given for this run)"
        if action.strategy == "target_recall" and base.cv_metrics["recall"].mean >= run.budget.min_recall:
            return (f"recall {base.cv_metrics['recall'].mean:.3f} already meets the target {run.budget.min_recall}; "
                    "a target_recall threshold would only lower it (use min_cost if costs are known)")
    if isinstance(action, AddEngineeredFeature):
        sugg = {s["name"]: s for s in f["engineered_suggestions"]}
        if action.feature_name not in sugg:
            return f"'{action.feature_name}' is not one of the engineered suggestions {list(sugg)}"
        if action.feature_name in base.features_used:
            return "feature already in the base experiment"
        used = set(sugg[action.feature_name]["recipe"]["sources"]) & set(base.spec.exclude_features)
        if used:
            return f"suggestion uses excluded column(s) {sorted(used)}"
    if isinstance(action, TryModel) and action.model == base.spec.model:
        return "base experiment already uses this model"
    if isinstance(action, TreatSentinelsAsMissing):
        bad = [c for c in action.columns if c not in f["sentinels"]]
        if bad:
            return f"{bad} were not flagged by the sentinel check (flagged: {sorted(f['sentinels'])})"
        missing = [c for c in action.columns if c not in base.features_used]
        if missing:
            return f"{missing} are not used by {action.base_exp_id}"
        if set(action.columns) <= set(base.spec.sentinel_to_missing):
            return "these sentinels are already treated as missing in the base experiment"
    if isinstance(action, LogTransformTarget):
        if task != "regression":
            return "log_transform_target needs a regression task"
        if base.spec.target_transform:
            return "base experiment already transforms the target"
        if f["target_min"] is not None and f["target_min"] <= -1:
            return "target has values <= -1; log1p is undefined"
    if isinstance(action, ScreenInteractions):
        if f["done"].get("interaction_screen"):
            return "the interaction screen was already run in this analysis"
        if task == "multiclass_classification":
            return "the interaction screen supports binary classification and regression"
    if isinstance(action, InteractionTest):
        cands = {(c["feature"], c["group_col"], c["level"]) for c in f["interaction_candidates"]}
        if (action.feature, action.group_col, action.level) not in cands:
            return (f"({action.feature}, {action.group_col}={action.level}) is not a screened interaction candidate "
                    f"{sorted(cands) or '(run screen_interactions first)'}")
        if action.feature in f["excluded_as_leak"] or action.group_col in f["excluded_as_leak"]:
            return "interaction would re-introduce a column removed as leakage"
    if isinstance(action, CheckFairness):
        if action.column not in f["fairness_columns"]:
            return f"'{action.column}' is not a fairness column for this run {f['fairness_columns']}"
        if action.column in f["fairness_runs"]:
            return f"fairness for '{action.column}' was already checked"
        if task == "multiclass_classification":
            return "the fairness check supports binary classification and regression"
    if isinstance(action, CheckCalibration):
        if task != "binary_classification":
            return "calibration needs a binary task (predicted probabilities)"
        if f["done"].get("calibration"):
            return "calibration was already checked in this analysis"
    if isinstance(action, CheckTimeSplit):
        if f["done"].get("time_split"):
            return "the forward-in-time check was already run in this analysis"
        if action.date_col not in f["date_columns"]:
            return f"'{action.date_col}' is not a detected date column {f['date_columns']}"
    return None


# --------------------------------------------------------------------------- rule planner
class RulePlanner:
    name = "rule"

    def _ok(self, a, run) -> bool:
        return validate_action(a, run) is None

    def decide(self, run) -> Decision:
        f, ctx = run.facts, run.ctx
        D = lambda a: Decision(a, self.name)  # noqa: E731
        if not f["baseline_id"] and f["n_duplicate_rows"] and not f["done"].get("dedupe"):
            return D(DeduplicateRows(rationale=(
                f"{f['n_duplicate_rows']} exact duplicate rows were found; copies would land on both sides of the "
                "split and inflate test scores, so remove them before any model is trained.")))
        if not f["baseline_id"] and f["group_candidates"] and not ctx.group_col:
            col = f["group_candidates"][0]
            return D(UseGroupSplit(group_col=col, rationale=(
                f"'{col}' repeats across rows ({f['group_evidence'].get(col, '')}); rows of the same entity in both train "
                "and test would make scores optimistic, so keep each entity on one side of the split.")))
        if not f["baseline_id"]:
            return D(RunBaseline(model="logreg", rationale="No model exists yet; a dummy and a baseline are needed "
                                 "before any investigation can be judged."))
        w_id = f["working_id"]
        w = ctx.experiments[w_id]

        # 1. leakage suspicions first: a leak invalidates every other metric
        for feat in f["leak_candidates"]:
            if feat not in f["ablated"] and feat in w.features_used:
                return D(AblateFeature(feature=feat, base_exp_id=w_id, rationale=(
                    f"'{feat}' was flagged by the leakage screen; removing it and retraining on identical folds "
                    "tests how much of the model's performance depends on it.")))

        # 1b. data fixes that change values, not rows
        sent = [c for c in f["sentinels"] if c in w.features_used and c not in w.spec.sentinel_to_missing]
        if sent and not f["done"].get("sentinels"):
            return D(TreatSentinelsAsMissing(columns=sent, base_exp_id=w_id, rationale=(
                f"The sentinel check found placeholder values in {sent} "
                "(" + ", ".join(f"{c}: {f['sentinels'][c]}" for c in sent) + "); "
                "as numbers they distort the model, so retrain with them treated as missing.")))
        if (ctx.task.value == "regression" and f["target_outliers"] and not f["done"].get("log_target")):
            a = LogTransformTarget(base_exp_id=w_id, rationale=(
                "The target has extreme values that dominate squared-error training; test a model trained on "
                "log1p(target), compared on MAE."))
            if self._ok(a, run):
                return D(a)

        # 2. imbalance / decision threshold (binary)
        if ctx.task.value == "binary_classification":
            recall = w.cv_metrics["recall"].mean
            if recall < run.budget.min_recall and not f["done"].get("weighting"):
                a = ApplyClassWeighting(base_exp_id=w_id, rationale=(
                    f"CV minority recall of the working model is {recall:.3f}, below the target "
                    f"{run.budget.min_recall}; class weighting is the first remedy for imbalance."))
                if self._ok(a, run):
                    return D(a)
            if not f["done"].get("threshold"):
                if ctx.costs:
                    a = TuneThreshold(base_exp_id=w_id, strategy="min_cost", rationale=(
                        f"Misclassification costs were given (FN={ctx.costs['fn']}, FP={ctx.costs['fp']}); "
                        "the 0.5 threshold ignores them, so choose the cost-minimising threshold."))
                    if self._ok(a, run):
                        return D(a)
                elif recall < run.budget.min_recall:
                    a = TuneThreshold(base_exp_id=w_id, strategy="target_recall", rationale=(
                        f"Recall {recall:.3f} is still below the target {run.budget.min_recall}; lower the "
                        "threshold to the highest value that meets the target on out-of-fold predictions."))
                    if self._ok(a, run):
                        return D(a)

        # 3. which features matter? (proposes candidates; ablation tests them)
        if not f["done"].get("importance"):
            done = "Leakage and imbalance are handled" if ctx.task.value != "regression" else "Leakage is handled"
            return D(RankFeatureImportance(base_exp_id=w_id, rationale=(
                f"{done}; measure which features the working model actually relies on "
                "to find noise features and explain performance.")))
        noise = [x for x in f["noise_candidates"] if x in w.features_used]
        if noise and not f["done"].get("noise_ablation"):
            a = AblateFeatureGroup(features=noise, base_exp_id=w_id, rationale=(
                f"Permutation importance marks {noise} as near-zero contributors; remove them together and test on "
                "identical folds whether any performance is lost (simpler model if not)."))
            if self._ok(a, run):
                return D(a)

        # 4. feature addition
        if not f["done"].get("engineered"):
            for s in f["engineered_suggestions"]:
                a = AddEngineeredFeature(feature_name=s["name"], base_exp_id=w_id, rationale=(
                    f"Screening on training rows suggests '{s['name']}' ({s['recipe']['op']} of "
                    f"{s['recipe']['sources']}) adds signal (screen gain {s['screen_gain']:+.4f}); test it in the "
                    "full model."))
                if self._ok(a, run):
                    return D(a)

        # 4b. does a feature act differently inside one group? (screen -> test)
        if ctx.task.value != "multiclass_classification" and not f["done"].get("interaction_screen"):
            return D(ScreenInteractions(base_exp_id=w_id, rationale=(
                "A single model-wide slope can hide effects that exist only inside one group; screen out-of-fold "
                "residuals for numeric features that behave differently within a category level.")))
        if f["interaction_candidates"] and not f["done"].get("interaction_test"):
            c = f["interaction_candidates"][0]
            a = InteractionTest(feature=c["feature"], group_col=c["group_col"], level=c["level"], base_exp_id=w_id,
                                rationale=(f"The screen found that '{c['feature']}' relates to the residuals differently "
                                           f"when {c['group_col']}={c['level']} (z={c['z']:+.1f}); add that interaction "
                                           "term and test it on identical folds."))
            if self._ok(a, run):
                return D(a)

        # 5. where does the working model fail?
        sub_ok = ctx.task.value in ("binary_classification", "regression")
        runs = f["subgroup_runs"]
        if sub_ok and not runs:
            return D(AnalyzeSubgroups(base_exp_id=w_id, rationale=(
                "Overall metrics can hide groups where the model fails; check out-of-fold errors per group "
                "before choosing the final model.")))

        # 6. model family (non-linear effects / interactions, e.g. a failing subgroup)
        if f["done"].get("model_switch") and not f["done"].get("model_switch_gb") and w.spec.model != "gradient_boosting":
            a = TryModel(model="gradient_boosting", base_exp_id=w_id, rationale=(
                f"The working model is {w.spec.model}; gradient boosting is usually the strongest model family for "
                "tabular data, so test it on the same features and folds."))
            if self._ok(a, run):
                return D(a)
        if not f["done"].get("model_switch") and w.spec.model == "logreg":
            flagged = [g for r in runs.values() for g in r]
            why = (f"Subgroup analysis flagged {flagged[0]}; a group-specific error pattern often means an "
                   "interaction a linear model cannot represent." if flagged else
                   "The linear model cannot capture interactions or non-linear effects.")
            a = TryModel(model="random_forest", base_exp_id=w_id, rationale=why +
                         " Test a random forest on the same features and folds.")
            if self._ok(a, run):
                return D(a)

        # 7. re-check previously flagged subgroups on a changed working model
        if sub_ok and w_id not in runs and any(runs.values()):
            return D(AnalyzeSubgroups(base_exp_id=w_id, rationale=(
                "The working model changed after subgroups were flagged; re-check whether those groups improved.")))

        # 7b. trust checks on the final model: fairness, calibration
        for col in f["fairness_columns"]:
            a = CheckFairness(column=col, base_exp_id=w_id, rationale=(
                f"'{col}' is {'a user-chosen' if col in f['fairness_user'] else 'a potentially sensitive'} column; "
                "check whether the working model's error rates differ materially between its groups."))
            if self._ok(a, run):
                return D(a)
        if ctx.task.value == "binary_classification" and not f["done"].get("calibration"):
            return D(CheckCalibration(base_exp_id=w_id, rationale=(
                "Before anyone acts on the predicted probabilities, check whether they match observed rates "
                "(Brier score, calibration error, reliability table).")))

        # 8. does the model hold up forward in time?
        if f["date_columns"] and not f["done"].get("time_split"):
            dc = f["date_columns"][0]
            return D(CheckTimeSplit(date_col=dc, base_exp_id=f["working_id"], rationale=(
                f"'{dc}' is a date column and all validation so far used shuffled folds; check whether the final "
                "model still performs when it is trained on earlier rows and tested on later ones.")))

        return D(Finish(reason="All triggered investigations are complete."))


# --------------------------------------------------------------------------- LLM planner
SYSTEM_PROMPT = """You are the planning component of TrustML, an ML analysis agent.
You choose exactly ONE next investigation step by calling one tool.
Rules:
- You never compute numbers; you read the tool results given to you.
- Decide only from cross-validation results; test-set results are hidden from you on purpose.
- run_baseline must come first. Never repeat an action listed in actions_already_executed
  (same action with the same arguments), e.g. analyze_subgroups for an experiment already analyzed.
- Investigate HIGH/MEDIUM leakage suspicions before anything else: a leaking feature makes every other
  metric meaningless. Build later experiments on working_exp_id.
- Binary tasks: if minority recall is below the target, try class weighting, then tune_threshold
  (use min_cost only when costs are given).
- Use rank_feature_importance before removing features; test noise candidates with ablate_feature_group.
- Test engineered features only from the listed suggestions.
- Use analyze_subgroups to find where the final model fails before finishing.
- If duplicate rows exist, deduplicate_rows must come BEFORE run_baseline; if an ID column repeats across rows,
  use_group_split must also come BEFORE run_baseline.
- If misclassification_costs are given, tune_threshold with strategy min_cost; use target_recall only when recall
  is below the target.
- Fix flagged placeholder values with treat_sentinels_as_missing; for regression targets with extreme
  values consider log_transform_target. If a date column exists, run check_time_split before finishing.
- After features are settled, run screen_interactions once; if it lists interaction_candidates, test the strongest
  with test_interaction (arguments copied exactly from the candidate). A screen result alone is not a finding.
- Before finishing: check_fairness for every column in fairness_columns, and (binary) check_calibration once.
- Prefer steps that address the user's objective. Call finish when no open issue remains.
- The rationale must cite the specific observation (finding or metric) that motivates the step."""


JSON_INSTRUCTIONS = """Respond with ONE JSON object and nothing else, choosing exactly one action.
Format: {{"action": "<name>", ...arguments..., "rationale": "<why>"}}  (finish uses "reason" instead of "rationale")
Available actions and their arguments:
{catalog}"""


def extract_json(text: str) -> Optional[dict]:
    """Return the last balanced {...} block in text that parses as a JSON object with an 'action' key."""
    text = text or ""
    found = None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        depth, in_str, esc = 0, False, False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[i:j + 1])
                        if isinstance(obj, dict) and "action" in obj:
                            found = obj
                    except json.JSONDecodeError:
                        pass
                    break
    return found


class LLMPlanner:
    """Chooses actions with an LLM. Tool-calling mode is tried first ("required", then "auto");
    if the server does not support it, the planner switches to JSON mode for the rest of the run.
    Either way the output is parsed into the same Pydantic action schema and state-validated."""
    name = "llm"
    MODES = ("tools_required", "tools_auto", "json")

    def __init__(self, client=None, model: Optional[str] = None, fallback: Optional[RulePlanner] = None,
                 mode: Optional[str] = None):
        from trustml.llm import get_client, planner_model
        self.client = client or get_client()
        self.model = model or planner_model()
        self.fallback = fallback or RulePlanner()
        self.tools = tool_definitions()
        self.mode = mode  # None = auto-detect on first call
        self.mode_log: list[str] = []

    def _messages(self, run, json_mode: bool) -> list[dict]:
        system = SYSTEM_PROMPT
        if json_mode:
            catalog = "\n".join(f"- {t['function']['name']}: {t['function']['description']} "
                                 f"Arguments: {json.dumps(t['function']['parameters']['properties'])}"
                                 for t in self.tools)
            system += "\n\n" + JSON_INSTRUCTIONS.format(catalog=catalog)
        return [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(run.planner_view(), default=str)}]

    def _call(self, run, mode: str) -> dict:
        from trustml.llm import chat
        if mode == "json":
            r = chat(self.client, self.model, messages=self._messages(run, True))
            msg = r.choices[0].message
            obj = extract_json(msg.content or "")
            if obj is None:
                raise ValueError(f"no JSON action in model output: {(msg.content or '')[:200]!r}")
            return obj
        r = chat(self.client, self.model, tools=self.tools, messages=self._messages(run, False),
                 tool_choice="required" if mode == "tools_required" else "auto")
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        if calls:
            return {"action": calls[0].function.name, **json.loads(calls[0].function.arguments or "{}")}
        obj = extract_json(msg.content or "")  # some servers put the call in the text
        if obj is None:
            raise ValueError("model returned no tool call")
        return obj

    def _detect(self, run) -> dict:
        """Try modes in order; keep the first that the server accepts."""
        last = None
        for mode in self.MODES:
            try:
                obj = self._call(run, mode)
                self.mode = mode
                self.mode_log.append(f"using {mode}")
                return obj
            except Exception as e:  # noqa: BLE001
                if _is_auth_or_network(e):
                    raise
                self.mode_log.append(f"{mode} failed: {type(e).__name__}: {str(e)[:160]}")
                last = e
        raise ValueError(f"no planning mode worked: {last}")

    def decide(self, run) -> Decision:
        try:
            obj = self._call(run, self.mode) if self.mode else self._detect(run)
            action = ACTION_ADAPTER.validate_python(obj)
            err = validate_action(action, run)
            if err:
                raise ValueError(f"rejected {obj.get('action')}: {err}")
            return Decision(action, f"llm:{self.model}")
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            d = self.fallback.decide(run)
            return Decision(d.action, "rule(fallback)", note=f"LLM decision rejected: {str(e)[:300]}")
        except Exception as e:  # network / API errors
            d = self.fallback.decide(run)
            return Decision(d.action, "rule(fallback)", note=f"LLM unavailable: {type(e).__name__}: {str(e)[:300]}")


def _is_auth_or_network(e: Exception) -> bool:
    name = type(e).__name__
    return name in ("AuthenticationError", "PermissionDeniedError", "APIConnectionError", "APITimeoutError",
                    "NotFoundError")
