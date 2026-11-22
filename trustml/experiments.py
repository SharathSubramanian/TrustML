"""Experiment engine: one fixed train/test split per run, CV on train, a single
evaluation on the untouched test set, and everything saved to the run folder.

Guarantees (each is checked by trustml.verify):
- The test split is created once per run and never changes (hash recorded).
- All preprocessing lives inside a sklearn Pipeline, so it is fit on training
  folds only.
- Threshold tuning uses out-of-fold training predictions, never the test set.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold, RepeatedKFold, RepeatedStratifiedKFold,
                                     StratifiedGroupKFold, StratifiedKFold, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from trustml.io import save_predictions
from trustml.schemas import DatasetProfile, Finding, Severity, TaskType

EXCLUDED_KINDS = {"id_like", "constant", "text", "datetime"}
ModelName = Literal["dummy", "logreg", "random_forest", "gradient_boosting"]


# --------------------------------------------------------------------------- schemas
class SplitInfo(BaseModel):
    strategy: Literal["stratified", "random", "group"]
    seed: int
    test_size: float
    group_col: Optional[str] = None
    n_train: int
    n_test: int
    test_index_sha: str
    cv_folds: int


class ExperimentSpec(BaseModel):
    name: str
    model: ModelName = "logreg"
    exclude_features: list[str] = Field(default_factory=list)
    class_weight: Optional[Literal["balanced"]] = None
    threshold_strategy: Literal["default", "max_f1", "min_cost", "target_recall"] = "default"
    target_recall: Optional[float] = None
    extra_features: list[str] = Field(default_factory=list, description="registered engineered features to add")
    sentinel_to_missing: dict[str, list[float]] = Field(default_factory=dict,
                                                        description="column -> placeholder values treated as missing")
    target_transform: Optional[Literal["log1p"]] = None
    parent_id: Optional[str] = None
    rationale: str = ""


class MetricStat(BaseModel):
    mean: float
    std: float
    folds: list[float]


class ExperimentResult(BaseModel):
    exp_id: str
    spec: ExperimentSpec
    task_type: TaskType
    positive_class: Optional[str]
    features_used: list[str]
    n_train: int
    n_test: int
    test_index_sha: str
    threshold: Optional[float]
    cv_metrics: dict[str, MetricStat]
    test_metrics: dict[str, float]
    test_ci95: dict[str, tuple[float, float]]
    predictions_path: str
    oof_path: str
    costs: Optional[dict[str, float]] = None
    elapsed_s: float


# --------------------------------------------------------------------------- metrics
def compute_metrics(task: TaskType, y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray],
                    costs: Optional[dict[str, float]] = None) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if task == TaskType.REGRESSION:
            return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "r2": float(r2_score(y_true, y_pred))}
        if task == TaskType.BINARY:
            m = {"accuracy": accuracy_score(y_true, y_pred),
                 "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                 "precision": precision_score(y_true, y_pred, zero_division=0),
                 "recall": recall_score(y_true, y_pred, zero_division=0),
                 "f1": f1_score(y_true, y_pred, zero_division=0)}
            if y_score is not None and len(np.unique(y_true)) == 2:
                m["roc_auc"] = roc_auc_score(y_true, y_score)
                m["pr_auc"] = average_precision_score(y_true, y_score)
            if costs:
                fn = np.sum((y_pred == 0) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0))
                m["cost_per_row"] = (costs["fn"] * fn + costs["fp"] * fp) / len(y_true)
            return {k: float(v) for k, v in m.items()}
        return {"accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0))}


# --------------------------------------------------------------------------- data fixes inside the pipeline
class SentinelToMissing(BaseEstimator, TransformerMixin):
    """Stateless, row-wise: replace placeholder values (e.g. -999) by NaN in the given columns."""

    def __init__(self, mapping: Optional[dict] = None):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.mapping:
            return X
        X = X.copy()
        for col, vals in self.mapping.items():
            if col in X:
                X[col] = X[col].mask(X[col].isin(vals))
        return X


# --------------------------------------------------------------------------- context
class RunContext:
    """Holds the immutable data, the fixed split and the experiment store for one run."""

    def __init__(self, df: pd.DataFrame, profile: DatasetProfile, run_dir: Path, seed: int = 42,
                 test_size: float = 0.2, cv_folds: int = 5, group_col: Optional[str] = None,
                 costs: Optional[dict[str, float]] = None, cv_repeats: int = 1):
        if profile.target is None:
            raise ValueError("A target is required.")
        self.profile = profile
        self.target = profile.target.name
        self.task = profile.target.task_type
        self.seed, self.cv_folds, self.group_col = seed, cv_folds, group_col
        self.cv_repeats = max(1, int(cv_repeats))  # >1: repeated K-fold with different shuffles (not with groups)
        self.run_dir = Path(run_dir)
        (self.run_dir / "experiments").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "predictions").mkdir(parents=True, exist_ok=True)
        self.transform_log: list[dict] = []
        self.experiments: dict[str, ExperimentResult] = {}
        self.engineered: dict[str, dict] = {}  # name -> recipe
        self.costs = costs if costs and self.task == TaskType.BINARY else None

        labelled = df[df[self.target].notna()]
        if len(labelled) < len(df):
            self.transform_log.append({"op": "exclude_unlabelled_rows", "n_rows": int(len(df) - len(labelled))})
        self.data = self._derive(labelled)  # derived copy; the caller's df is never modified
        self.kinds = {c.name: c.inferred_kind for c in profile.columns}
        self.candidate_features = [c for c in self.data.columns
                                   if c != self.target and self.kinds.get(c) not in EXCLUDED_KINDS]
        for c in self.data.columns:
            if c != self.target and c not in self.candidate_features:
                self.transform_log.append({"op": "exclude_feature", "column": c, "reason": f"kind={self.kinds.get(c)}"})

        self.positive_class: Optional[str] = None
        self.test_size = test_size
        self._encode_y()
        self._split_and_check_folds()

    def _encode_y(self) -> None:
        y_raw = self.data[self.target]
        if self.task == TaskType.BINARY:
            self.positive_class = self.profile.target.minority_class  # minority class = class of interest
            self.y = (y_raw.astype(str) == self.positive_class).astype(int).to_numpy()
        elif self.task == TaskType.MULTICLASS:
            self.classes_ = sorted(y_raw.astype(str).unique())
            self.y = y_raw.astype(str).map({c: i for i, c in enumerate(self.classes_)}).to_numpy()
        else:
            self.y = y_raw.astype(float).to_numpy()

    def _split_and_check_folds(self) -> None:
        self.split = self._make_split(self.test_size)
        if self.task != TaskType.REGRESSION:
            counts = np.bincount(self.y[self.train_pos].astype(int))
            smallest = int(counts[counts > 0].min()) if len(counts) else 0
            if smallest < 2:
                raise ValueError(f"Too few training examples of the rarest class ({smallest}); at least 2 are needed "
                                 "for cross-validation. Provide more data or merge rare classes.")
            if smallest < self.cv_folds:
                self.transform_log.append({"op": "reduce_cv_folds", "from": self.cv_folds, "to": smallest,
                                           "reason": f"rarest class has only {smallest} training rows"})
                self.cv_folds = smallest
                self.split = self.split.model_copy(update={"cv_folds": smallest})

    def use_group_split(self, col: str) -> dict:
        """Keep all rows of one entity (e.g. a customer) on one side of the split. Before any experiment."""
        if self.experiments:
            raise ValueError("The split can only be changed before the first experiment.")
        if col not in self.data.columns:
            raise ValueError(f"Unknown column '{col}'.")
        self.group_col = col
        self._split_and_check_folds()
        self.transform_log.append({"op": "group_split", "column": col, "n_groups": int(self.data[col].nunique()),
                                   "n_train": self.split.n_train, "n_test": self.split.n_test})
        return {"column": col, "n_groups": int(self.data[col].nunique()), "n_train": self.split.n_train,
                "n_test": self.split.n_test}

    def deduplicate(self) -> int:
        """Drop exact duplicate rows BEFORE any experiment, then re-split. Logged; original file untouched."""
        if self.experiments:
            raise ValueError("Deduplication must happen before the first experiment (it changes the split).")
        cols = [c for c in self.data.columns if c not in self.engineered]
        dup = self.data.duplicated(subset=cols, keep="first")
        n = int(dup.sum())
        if n:
            self.data = self.data[~dup].copy()
            self.transform_log.append({"op": "deduplicate", "rows_removed": n,
                                       "example_row_indices": [int(i) for i in dup[dup].index[:10]]})
            self._encode_y()
            self._split_and_check_folds()
        return n

    # numbers-stored-as-text -> numeric, logged
    def _derive(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in self.profile.columns:
            if c.inferred_kind == "numeric_text" and c.name in out:
                before = out[c.name].isna().sum()
                out[c.name] = pd.to_numeric(out[c.name].astype(str).str.strip(), errors="coerce")
                self.transform_log.append({"op": "to_numeric", "column": c.name,
                                           "new_missing": int(out[c.name].isna().sum() - before)})
        return out

    def _make_split(self, test_size: float) -> SplitInfo:
        idx = np.arange(len(self.data))
        if self.group_col:
            groups = self.data[self.group_col].astype(str)
            if self.task != TaskType.REGRESSION:  # keep the class balance while keeping groups together
                k = max(2, int(round(1 / test_size)))
                tr, te = next(StratifiedGroupKFold(k, shuffle=True, random_state=self.seed).split(idx, self.y, groups))
            else:
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=self.seed)
                tr, te = next(gss.split(idx, self.y, groups=groups))
            strategy = "group"
        else:
            strat = self.y if self.task != TaskType.REGRESSION else None
            tr, te = train_test_split(idx, test_size=test_size, random_state=self.seed, stratify=strat)
            strategy = "stratified" if strat is not None else "random"
        self.train_pos, self.test_pos = np.sort(tr), np.sort(te)
        test_labels = self.data.index[self.test_pos].tolist()
        sha = hashlib.sha256(json.dumps(test_labels).encode()).hexdigest()
        info = SplitInfo(strategy=strategy, seed=self.seed, test_size=test_size, group_col=self.group_col,
                         n_train=len(tr), n_test=len(te), test_index_sha=sha, cv_folds=self.cv_folds)
        (self.run_dir / "split.json").write_text(json.dumps({
            **info.model_dump(), "train_index": self.data.index[self.train_pos].tolist(), "test_index": test_labels}))
        return info

    # ------------------------------------------------------------------ split checks (A4, A5)
    def split_checks(self) -> list[Finding]:
        findings: list[Finding] = []
        cols = [c for c in self.data.columns if c != self.target and self.kinds.get(c) != "id_like"]
        h = pd.util.hash_pandas_object(self.data[cols], index=False).to_numpy()
        train_h = set(h[self.train_pos])
        overlap = int(sum(x in train_h for x in h[self.test_pos]))
        if overlap:
            findings.append(Finding(
                check="train_test_overlap", severity=Severity.MEDIUM,
                message=f"{overlap} test rows have an identical copy (all non-ID features) in the training set.",
                evidence={"n_overlap_test_rows": overlap, "n_test": self.split.n_test,
                          "pct_of_test": round(100 * overlap / self.split.n_test, 3)},
                recommendation="Duplicates across the split make test scores optimistic. Deduplicate before "
                               "splitting, or use a group split.",
            ))
        for c in [c for c, k in self.kinds.items() if k == "id_like" and c in self.data]:
            vc = self.data[c].value_counts()
            rep = vc[vc > 1]
            if len(rep):
                findings.append(Finding(
                    check="repeated_ids", severity=Severity.MEDIUM, columns=[c],
                    message=f"{len(rep)} values of '{c}' appear in more than one row ({int(rep.sum())} rows).",
                    evidence={"n_repeated_values": int(len(rep)), "n_rows_involved": int(rep.sum()),
                              "examples": {str(k): int(v) for k, v in rep.head(5).items()}},
                    recommendation=f"If '{c}' identifies an entity, its rows must stay on one side of the split: "
                                   f"rerun with group_col='{c}'.",
                ))
        return findings

    # ------------------------------------------------------------------ experiments
    def add_engineered(self, name: str, recipe: dict, values: pd.Series) -> None:
        """Register a feature derived row-by-row from existing columns (no fitting, so no split leakage)."""
        if name in self.data.columns:
            raise ValueError(f"Column '{name}' already exists.")
        if self.target in recipe["sources"]:
            raise ValueError("Engineered features may not use the target.")
        self.data[name] = values.reindex(self.data.index)
        self.engineered[name] = recipe
        self.transform_log.append({"op": "engineered_feature", "name": name, "recipe": recipe})
        (self.run_dir / "engineered.json").write_text(json.dumps(self.engineered, indent=2))

    def _features(self, spec: ExperimentSpec) -> list[str]:
        unknown = set(spec.exclude_features) - set(self.candidate_features) - set(self.engineered)
        if unknown:
            raise ValueError(f"Cannot exclude unknown features: {sorted(unknown)}")
        bad = set(spec.extra_features) - set(self.engineered)
        if bad:
            raise ValueError(f"Unregistered engineered features: {sorted(bad)}")
        for e in spec.extra_features:
            used = set(self.engineered[e]["sources"]) & set(spec.exclude_features)
            if used:
                raise ValueError(f"Engineered feature '{e}' uses excluded column(s) {sorted(used)}.")
        feats = [c for c in self.candidate_features if c not in spec.exclude_features]
        feats += [e for e in spec.extra_features if e not in spec.exclude_features]
        if not feats:
            raise ValueError("No features left after exclusions.")
        return feats

    def _pipeline(self, spec: ExperimentSpec, feats: list[str]) -> Pipeline:
        X = self.data[feats]
        num = [c for c in feats if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]
        cat = [c for c in feats if c not in num]
        scale = spec.model == "logreg"
        num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
        if scale:
            num_steps.append(("scale", StandardScaler()))
        pre = ColumnTransformer([
            ("num", Pipeline(num_steps), num),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                                       sparse_output=False))]), cat),
        ])
        reg = self.task == TaskType.REGRESSION
        if spec.target_transform and not reg:
            raise ValueError("target_transform is only valid for regression.")
        if spec.model == "dummy":
            est = DummyRegressor() if reg else DummyClassifier(strategy="prior")
        elif spec.model == "logreg":
            est = LinearRegression() if reg else LogisticRegression(max_iter=2000, class_weight=spec.class_weight)
        elif spec.model == "gradient_boosting":
            est = (HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, random_state=self.seed) if reg
                   else HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, class_weight=spec.class_weight,
                                                       random_state=self.seed))
        else:
            est = (RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=self.seed) if reg
                   else RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight=spec.class_weight,
                                               n_jobs=-1, random_state=self.seed))
        if spec.target_transform == "log1p":
            est = TransformedTargetRegressor(regressor=est, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
        return Pipeline([("fix", SentinelToMissing(spec.sentinel_to_missing or None)), ("pre", pre), ("model", est)])

    def _cv(self, repeats: Optional[int] = None):
        r = self.cv_repeats if repeats is None else repeats
        if self.group_col:
            return (StratifiedGroupKFold(self.cv_folds, shuffle=True, random_state=self.seed)
                    if self.task != TaskType.REGRESSION else GroupKFold(self.cv_folds))
        if r > 1:
            return (RepeatedKFold(n_splits=self.cv_folds, n_repeats=r, random_state=self.seed)
                    if self.task == TaskType.REGRESSION else
                    RepeatedStratifiedKFold(n_splits=self.cv_folds, n_repeats=r, random_state=self.seed))
        return (KFold(self.cv_folds, shuffle=True, random_state=self.seed) if self.task == TaskType.REGRESSION
                else StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.seed))

    def _scores(self, model: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self.task == TaskType.REGRESSION:
            return model.predict(X), None
        if self.task == TaskType.BINARY:
            return model.predict(X), model.predict_proba(X)[:, 1]
        return model.predict(X), None

    def run(self, spec: ExperimentSpec) -> ExperimentResult:
        t0 = time.time()
        if spec.class_weight and self.task == TaskType.REGRESSION:
            raise ValueError("class_weight is only valid for classification.")
        if spec.threshold_strategy != "default" and self.task != TaskType.BINARY:
            raise ValueError("Threshold tuning is only supported for binary classification.")
        if spec.threshold_strategy == "min_cost" and not self.costs:
            raise ValueError("min_cost thresholding needs misclassification costs for this run.")
        if spec.threshold_strategy == "target_recall" and not spec.target_recall:
            raise ValueError("target_recall thresholding needs target_recall.")
        feats = self._features(spec)
        if spec.target_transform == "log1p" and np.nanmin(self.y) <= -1:
            raise ValueError("log1p target transform needs all target values > -1.")
        bad = [c for c in spec.sentinel_to_missing if c not in feats]
        if bad:
            raise ValueError(f"sentinel_to_missing refers to columns not used as features: {bad}")
        X_tr, y_tr = self.data.iloc[self.train_pos][feats], self.y[self.train_pos]
        X_te, y_te = self.data.iloc[self.test_pos][feats], self.y[self.test_pos]
        groups = self.data.iloc[self.train_pos][self.group_col].astype(str) if self.group_col else None

        # ---- cross-validation on the training split only (out-of-fold predictions)
        folds = list(self._cv().split(X_tr, y_tr, groups))
        n_first = min(len(folds), self.cv_folds)          # first repeat -> out-of-fold predictions
        oof_pred = np.zeros(len(y_tr), dtype=float)
        oof_score = np.zeros(len(y_tr), dtype=float) if self.task == TaskType.BINARY else None
        fold_id = np.zeros(len(y_tr), dtype=int)
        split_out = []                                    # (val_idx, pred, score) for every split
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, (a, b) in enumerate(folds):
                m = self._pipeline(spec, feats).fit(X_tr.iloc[a], y_tr[a])
                p, s = self._scores(m, X_tr.iloc[b])
                split_out.append((b, p, s))
                if k < n_first:
                    oof_pred[b], fold_id[b] = p, k
                    if oof_score is not None:
                        oof_score[b] = s

        threshold = None
        if self.task == TaskType.BINARY:
            threshold = 0.5
            grid = np.round(np.arange(0.01, 0.991, 0.01), 2)
            if spec.threshold_strategy == "max_f1":
                f1s = [f1_score(y_tr, (oof_score >= t).astype(int), zero_division=0) for t in grid]
                threshold = float(grid[int(np.argmax(f1s))])
            elif spec.threshold_strategy == "min_cost":
                pos = y_tr == 1
                cost = [self.costs["fn"] * np.sum(pos & (oof_score < t)) + self.costs["fp"] * np.sum(~pos & (oof_score >= t))
                        for t in grid]
                threshold = float(grid[int(np.argmin(cost))])
            elif spec.threshold_strategy == "target_recall":
                n_pos = max(int(np.sum(y_tr == 1)), 1)
                ok = [t for t in grid if np.sum((oof_score >= t) & (y_tr == 1)) / n_pos >= spec.target_recall]
                threshold = float(max(ok)) if ok else float(grid[0])  # highest threshold meeting the target
            oof_pred = (oof_score >= threshold).astype(int)

        per_fold: dict[str, list[float]] = {}
        for b, p, sc in split_out:
            pr = (sc >= threshold).astype(int) if self.task == TaskType.BINARY else p
            fm = compute_metrics(self.task, y_tr[b], pr, sc, self.costs)
            for name, v in fm.items():
                per_fold.setdefault(name, []).append(round(v, 6))
        cv_metrics = {n: MetricStat(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)), folds=v)
                      for n, v in per_fold.items()}

        # ---- final fit on full training split, single evaluation on test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final = self._pipeline(spec, feats).fit(X_tr, y_tr)
            te_pred, te_score = self._scores(final, X_te)
        if threshold is not None:
            te_pred = (te_score >= threshold).astype(int)
        test_metrics = compute_metrics(self.task, y_te, te_pred, te_score, self.costs)
        ci = self._bootstrap_ci(y_te, te_pred, te_score)

        exp_id = f"E{len(self.experiments) + 1:02d}_{spec.name}"
        pred_path = self.run_dir / "predictions" / f"{exp_id}.csv"
        save_predictions(pd.DataFrame({"row_index": self.data.index[self.test_pos], "y_true": y_te, "y_pred": te_pred,
                                       **({"y_score": te_score} if te_score is not None else {})}), pred_path)
        oof_path = self.run_dir / "predictions" / f"{exp_id}_oof.csv"
        save_predictions(pd.DataFrame({"row_index": self.data.index[self.train_pos], "fold": fold_id, "y_true": y_tr,
                                       "y_pred": oof_pred, **({"y_score": oof_score} if oof_score is not None else {})}),
                         oof_path)

        res = ExperimentResult(
            exp_id=exp_id, spec=spec, task_type=self.task, positive_class=self.positive_class, features_used=feats,
            n_train=len(y_tr), n_test=len(y_te), test_index_sha=self.split.test_index_sha, threshold=threshold,
            cv_metrics=cv_metrics, test_metrics=test_metrics, test_ci95=ci,
            predictions_path=str(pred_path.relative_to(self.run_dir)),
            oof_path=str(oof_path.relative_to(self.run_dir)), costs=self.costs, elapsed_s=round(time.time() - t0, 3))
        self._models = getattr(self, "_models", {})
        self._models[exp_id] = (final, feats)  # kept in memory for explanations
        (self.run_dir / "experiments" / f"{exp_id}.json").write_text(res.model_dump_json(indent=2))
        self.experiments[exp_id] = res
        return res

    def time_split_eval(self, exp_id: str, date_col: str, n_splits: int = 4) -> dict:
        """Forward-chaining evaluation on the TRAINING split ordered by `date_col`: always train on earlier
        rows and predict later ones. Compared with the experiment's shuffled CV, a large drop means the
        model does not generalise forward in time (drift, or features that encode the future)."""
        from sklearn.model_selection import TimeSeriesSplit
        e = self.experiments[exp_id]
        feats = e.features_used
        dates = pd.to_datetime(self.data.iloc[self.train_pos][date_col], errors="coerce", format="mixed")
        ok = dates.notna().to_numpy()
        order = np.argsort(dates.to_numpy()[ok], kind="mergesort")
        pos = self.train_pos[ok][order]
        X, y = self.data.iloc[pos][feats], self.y[pos]
        metric = {TaskType.BINARY: "roc_auc", TaskType.REGRESSION: "r2", TaskType.MULTICLASS: "balanced_accuracy"}[self.task]
        rows, folds = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k, (a, b) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X)):
                if self.task != TaskType.REGRESSION and (len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2):
                    continue
                m = self._pipeline(e.spec, feats).fit(X.iloc[a], y[a])
                pred, score = self._scores(m, X.iloc[b])
                if self.task == TaskType.BINARY:
                    pred = (score >= (e.threshold or 0.5)).astype(int)
                val = compute_metrics(self.task, y[b], pred, score)[metric]
                d = dates.to_numpy()[ok][order]
                folds.append({"fold": k, "train_until": str(pd.Timestamp(d[a[-1]]).date()),
                              "test_from": str(pd.Timestamp(d[b[0]]).date()), "test_to": str(pd.Timestamp(d[b[-1]]).date()),
                              "n_train": int(len(a)), "n_test": int(len(b)), metric: round(float(val), 6)})
                rows.append(pd.DataFrame({"row_index": self.data.index[pos[b]], "fold": k, "y_true": y[b], "y_pred": pred,
                                          **({"y_score": score} if score is not None else {})}))
        path = self.run_dir / "predictions" / f"{exp_id}_timesplit.csv"
        if rows:
            save_predictions(pd.concat(rows, ignore_index=True), path)
        vals = [f[metric] for f in folds]
        # target drift: does the outcome itself change over time? (ranking metrics such as ROC-AUC cannot see this)
        drift = None
        half = len(y) // 2
        if half >= 20:
            from scipy import stats as _st
            early, late = y[:half], y[half:]
            d_dates = dates.to_numpy()[ok][order]
            if self.task == TaskType.BINARY:
                tab = np.array([[np.sum(early == 1), np.sum(early == 0)], [np.sum(late == 1), np.sum(late == 0)]])
                pval = float(_st.chi2_contingency(tab)[1])
                e_rate, l_rate = float(early.mean()), float(late.mean())
                rel = (l_rate - e_rate) / e_rate if e_rate else 0.0
                drift = {"kind": "positive_rate", "early_rate": round(e_rate, 4), "late_rate": round(l_rate, 4),
                         "relative_change": round(rel, 3), "p_value": pval,
                         "material": bool(pval < 0.01 and abs(rel) >= 0.25)}
            elif self.task == TaskType.REGRESSION:
                t = _st.ttest_ind(early, late, equal_var=False)
                eff = float((late.mean() - early.mean()) / (y.std() or 1))
                drift = {"kind": "target_mean", "early_mean": float(early.mean()), "late_mean": float(late.mean()),
                         "effect_size_sd": round(eff, 3), "p_value": float(t.pvalue),
                         "material": bool(t.pvalue < 0.01 and abs(eff) >= 0.2)}
            if drift:
                drift.update(early_period=f"{pd.Timestamp(d_dates[0]).date()} – {pd.Timestamp(d_dates[half - 1]).date()}",
                             late_period=f"{pd.Timestamp(d_dates[half]).date()} – {pd.Timestamp(d_dates[-1]).date()}",
                             n_early=int(half), n_late=int(len(y) - half))
        for f, part in zip(folds, rows):
            yy = part["y_true"].to_numpy()
            f["target_rate" if self.task == TaskType.BINARY else "target_mean"] = round(float(yy.mean()), 6)
            if self.task == TaskType.BINARY:
                pos = yy == 1
                f["recall"] = round(float((part["y_pred"].to_numpy()[pos] == 1).mean()), 6) if pos.any() else None
        return {"exp_id": exp_id, "date_col": date_col, "metric": metric, "n_rows_with_date": int(ok.sum()),
                "target_drift": drift,
                "folds": folds, "time_split_mean": float(np.mean(vals)) if vals else None,
                "time_split_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
                "random_cv_mean": e.cv_metrics[metric].mean, "random_cv_std": e.cv_metrics[metric].std,
                "predictions_path": str(path.relative_to(self.run_dir)) if rows else None,
                "threshold": e.threshold}

    def _bootstrap_ci(self, y, pred, score, n_boot: int = 200) -> dict[str, tuple[float, float]]:
        rng = np.random.default_rng(self.seed)
        keys = {TaskType.BINARY: ["recall", "precision", "f1", "roc_auc", "pr_auc"] + (["cost_per_row"] if self.costs else []),
                TaskType.REGRESSION: ["rmse", "r2"],
                TaskType.MULTICLASS: ["balanced_accuracy", "f1_macro"]}[self.task]
        samples: dict[str, list[float]] = {k: [] for k in keys}
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if self.task != TaskType.REGRESSION and len(np.unique(y[i])) < 2:
                continue
            m = compute_metrics(self.task, y[i], pred[i], None if score is None else score[i], self.costs)
            for k in keys:
                if k in m:
                    samples[k].append(m[k])
        return {k: (round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4))
                for k, v in samples.items() if v}


# --------------------------------------------------------------------------- comparisons
class Comparison(BaseModel):
    base_id: str
    new_id: str
    metric: str
    base_mean: float
    new_mean: float
    mean_diff: float
    fold_diffs: list[float]
    std_diff: float
    consistent_direction: bool = Field(description="all fold differences have the same sign")
    no_fold_worse: bool = Field(False, description="every fold difference >= 0 and at least one > 0")
    no_fold_better: bool = Field(False, description="every fold difference <= 0 and at least one < 0")
    test_base: Optional[float] = None
    test_new: Optional[float] = None


def compare(base: ExperimentResult, new: ExperimentResult, metric: str) -> Comparison:
    """Paired comparison on identical CV folds (same seed, same training split)."""
    a, b = np.array(base.cv_metrics[metric].folds), np.array(new.cv_metrics[metric].folds)
    d = b - a
    return Comparison(base_id=base.exp_id, new_id=new.exp_id, metric=metric, base_mean=float(a.mean()),
                      new_mean=float(b.mean()), mean_diff=float(d.mean()), fold_diffs=[round(float(x), 6) for x in d],
                      std_diff=float(d.std(ddof=1)), consistent_direction=bool(np.all(d > 0) or np.all(d < 0)),
                      no_fold_worse=bool(np.all(d >= 0) and np.any(d > 0)),
                      no_fold_better=bool(np.all(d <= 0) and np.any(d < 0)),
                      test_base=base.test_metrics.get(metric), test_new=new.test_metrics.get(metric))
