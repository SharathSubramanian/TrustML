"""Actions the planner may choose. Every LLM decision is parsed into one of these
schemas and validated against the run state before anything executes."""
from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter


class RunBaseline(BaseModel):
    action: Literal["run_baseline"] = "run_baseline"
    model: Literal["logreg", "random_forest"] = "logreg"
    rationale: str = ""


class AblateFeature(BaseModel):
    action: Literal["ablate_feature"] = "ablate_feature"
    feature: str
    base_exp_id: str
    rationale: str = ""


class AblateFeatureGroup(BaseModel):
    action: Literal["ablate_feature_group"] = "ablate_feature_group"
    features: list[str] = Field(min_length=1)
    base_exp_id: str
    rationale: str = ""


class ApplyClassWeighting(BaseModel):
    action: Literal["apply_class_weighting"] = "apply_class_weighting"
    base_exp_id: str
    rationale: str = ""


class TuneThreshold(BaseModel):
    action: Literal["tune_threshold"] = "tune_threshold"
    base_exp_id: str
    strategy: Literal["target_recall", "min_cost", "max_f1"] = "target_recall"
    rationale: str = ""


class RankFeatureImportance(BaseModel):
    action: Literal["rank_feature_importance"] = "rank_feature_importance"
    base_exp_id: str
    rationale: str = ""


class AddEngineeredFeature(BaseModel):
    action: Literal["add_engineered_feature"] = "add_engineered_feature"
    feature_name: str
    base_exp_id: str
    rationale: str = ""


class TryModel(BaseModel):
    action: Literal["try_model"] = "try_model"
    model: Literal["logreg", "random_forest", "gradient_boosting"]
    base_exp_id: str
    rationale: str = ""


class AnalyzeSubgroups(BaseModel):
    action: Literal["analyze_subgroups"] = "analyze_subgroups"
    base_exp_id: str
    rationale: str = ""


class DeduplicateRows(BaseModel):
    action: Literal["deduplicate_rows"] = "deduplicate_rows"
    rationale: str = ""


class UseGroupSplit(BaseModel):
    action: Literal["use_group_split"] = "use_group_split"
    group_col: str
    rationale: str = ""


class TreatSentinelsAsMissing(BaseModel):
    action: Literal["treat_sentinels_as_missing"] = "treat_sentinels_as_missing"
    columns: list[str] = Field(min_length=1)
    base_exp_id: str
    rationale: str = ""


class LogTransformTarget(BaseModel):
    action: Literal["log_transform_target"] = "log_transform_target"
    base_exp_id: str
    rationale: str = ""


class CheckTimeSplit(BaseModel):
    action: Literal["check_time_split"] = "check_time_split"
    date_col: str
    base_exp_id: str
    rationale: str = ""


class ScreenInteractions(BaseModel):
    action: Literal["screen_interactions"] = "screen_interactions"
    base_exp_id: str
    rationale: str = ""


class InteractionTest(BaseModel):
    action: Literal["test_interaction"] = "test_interaction"
    feature: str
    group_col: str
    level: str
    base_exp_id: str
    rationale: str = ""


class CheckFairness(BaseModel):
    action: Literal["check_fairness"] = "check_fairness"
    column: str
    base_exp_id: str
    rationale: str = ""


class CheckCalibration(BaseModel):
    action: Literal["check_calibration"] = "check_calibration"
    base_exp_id: str
    rationale: str = ""


class Finish(BaseModel):
    action: Literal["finish"] = "finish"
    reason: str


ACTION_TYPES = [DeduplicateRows, UseGroupSplit, RunBaseline, AblateFeature, AblateFeatureGroup, ApplyClassWeighting, TuneThreshold,
                TreatSentinelsAsMissing, LogTransformTarget, RankFeatureImportance, AddEngineeredFeature, TryModel,
                AnalyzeSubgroups, ScreenInteractions, InteractionTest, CheckFairness, CheckCalibration,
                CheckTimeSplit, Finish]
Action = Annotated[Union[tuple(ACTION_TYPES)], Field(discriminator="action")]
ACTION_ADAPTER = TypeAdapter(Action)
ANALYSIS_ACTIONS = (RankFeatureImportance, AnalyzeSubgroups, CheckTimeSplit, ScreenInteractions, CheckFairness,
                    CheckCalibration)

DESCRIPTIONS = {
    "deduplicate_rows": "Remove exact duplicate rows and re-split. Only allowed BEFORE run_baseline, because it "
                        "changes the train/test split. Use when duplicate rows or train/test overlap were found.",
    "use_group_split": "Keep all rows of one entity (e.g. the same customer) on one side of the train/test split and "
                       "CV folds. Only allowed BEFORE run_baseline. Use when an ID column repeats across rows.",
    "treat_sentinels_as_missing": "Retrain an experiment with placeholder values (e.g. -999, 99999) in the given "
                                  "columns treated as missing; only columns flagged by the sentinel check.",
    "log_transform_target": "Regression only: train on log1p(target) and predict on the original scale, to reduce "
                            "the influence of extreme target values; compared on MAE.",
    "check_time_split": "Analysis: evaluate an experiment with forward-in-time folds ordered by a date column and "
                        "compare with shuffled CV, to detect drift or features that encode the future.",
    "run_baseline": "Train a dummy model and a baseline model on all candidate features. Must be first.",
    "ablate_feature": "Retrain an experiment without ONE feature, compared on identical CV folds. "
                      "Use to test a leakage suspicion.",
    "ablate_feature_group": "Retrain an experiment without several features (e.g. noise candidates from "
                            "rank_feature_importance) and test whether performance is lost.",
    "apply_class_weighting": "Retrain with balanced class weights to address weak minority-class recall.",
    "tune_threshold": "Choose the decision threshold on out-of-fold training predictions (binary only; the test "
                      "set is never used). target_recall = highest threshold meeting the recall target; "
                      "min_cost = minimise the run's misclassification cost (only if costs were given); "
                      "max_f1 = maximise F1.",
    "rank_feature_importance": "Permutation importance per CV fold for an experiment. Analysis only; it proposes "
                               "noise candidates that an ablation must then test.",
    "add_engineered_feature": "Add one of the listed engineered-feature suggestions (derived row-wise from existing "
                              "columns) to an experiment and compare on identical folds.",
    "try_model": "Retrain an experiment with a different model family (random_forest or gradient_boosting for "
                 "non-linear effects and interactions).",
    "analyze_subgroups": "Find groups (categories / numeric quartiles) where an experiment's out-of-fold errors are "
                         "clearly worse than overall. Analysis only.",
    "screen_interactions": "Analysis: look for a numeric feature whose effect differs inside one category level "
                           "(e.g. income matters only for one loan purpose), using out-of-fold residuals of an "
                           "experiment (Fisher z-test). Proposes candidates that test_interaction must then test.",
    "test_interaction": "Add ONE interaction candidate from screen_interactions (feature x indicator of group_col == "
                        "level) to an experiment and compare on identical folds, overall and inside that group.",
    "check_fairness": "Analysis: per-group recall / false-alarm rate (or MAE) of an experiment for a sensitive or "
                      "user-chosen column, with a bootstrap CI on the gap. Only listed fairness columns.",
    "check_calibration": "Analysis (binary): are the predicted probabilities trustworthy? Brier score, expected "
                         "calibration error and a reliability table on out-of-fold predictions.",
    "finish": "Stop investigating. Use when no open issue remains or further steps are unlikely to help.",
}


def tool_definitions() -> list[dict]:
    """OpenAI function-calling definitions generated from the schemas."""
    tools = []
    for cls in ACTION_TYPES:
        schema = cls.model_json_schema()
        name = schema["properties"]["action"]["default"]
        props = {k: v for k, v in schema["properties"].items() if k != "action"}
        for v in props.values():
            v.pop("title", None)
        required = [r for r in schema.get("required", []) if r != "action"]
        if "rationale" in props:
            required.append("rationale")
        tools.append({"type": "function", "function": {
            "name": name, "description": DESCRIPTIONS[name],
            "parameters": {"type": "object", "properties": props, "required": required}}})
    return tools


def signature(a) -> str:
    """Identity of an action, used to block repeats."""
    d = a.model_dump(exclude={"rationale", "reason"})
    return "|".join(f"{k}={sorted(d[k]) if isinstance(d[k], list) else d[k]}" for k in sorted(d))
