"""Engineered-feature suggestions, screened on the training split.

Every candidate is a row-wise recipe over existing columns (log1p, product,
ratio), so computing it cannot leak information across the split. Candidates are
screened with a cheap paired comparison: a small linear model on the source
columns vs. the same columns plus the candidate (3-fold CV, training rows only).
Screening only ranks candidates; the agent's experiment decides.
"""
from __future__ import annotations

import itertools
import re
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trustml.schemas import TaskType


def apply_recipe(data: pd.DataFrame, recipe: dict) -> pd.Series:
    op, src = recipe["op"], recipe["sources"]
    a = pd.to_numeric(data[src[0]] if pd.api.types.is_numeric_dtype(data[src[0]]) else data[src[0]].astype(str).str.strip(),
                      errors="coerce").astype(float)
    if recipe.get("mask"):  # sentinel placeholders are missing, not numbers
        a = a.mask(a.isin(recipe["mask"]))
    if op == "interaction":  # numeric slope that applies only inside one category level
        return a * (data[src[1]].astype(str) == str(recipe["level"])).astype(float)
    if op == "log1p":
        return np.log1p(a.clip(lower=0))
    b = pd.to_numeric(data[src[1]], errors="coerce").astype(float)
    if op == "product":
        return a * b
    if op == "ratio":
        return a / b.where(b != 0)
    raise ValueError(f"unknown op {op}")


def interaction_name(feature: str, group_col: str, level: str) -> str:
    return f"{feature}_if_{group_col}_{re.sub(r'[^0-9A-Za-z]+', '_', str(level)).strip('_')}"


def _name(recipe: dict) -> str:
    s, op = recipe["sources"], recipe["op"]
    if op == "interaction":
        return interaction_name(s[0], s[1], recipe["level"])
    if op == "log1p":
        return f"log1p_{s[0]}"
    return f"{s[0]}_x_{s[1]}" if op == "product" else f"{s[0]}_per_{s[1]}"


def suggest_engineered(ctx, exclude: list[str], max_numeric: int = 8, top_k: int = 3, min_gain: float = 0.005) -> dict:
    data = ctx.data.iloc[ctx.train_pos]
    y = ctx.y[ctx.train_pos]
    numeric = [c for c in ctx.candidate_features
               if c not in exclude and ctx.kinds.get(c) in ("numeric", "numeric_text") and data[c].nunique() > 10]
    numeric = numeric[:max_numeric]
    recipes = []
    for c in numeric:
        s = data[c].dropna()
        if len(s) and s.min() >= 0 and abs(float(s.skew())) > 1:
            recipes.append({"op": "log1p", "sources": [c]})
    for a, b in itertools.combinations(numeric, 2):
        recipes.append({"op": "product", "sources": [a, b]})
        if (data[b].dropna() > 0).all():
            recipes.append({"op": "ratio", "sources": [a, b]})
        if (data[a].dropna() > 0).all():
            recipes.append({"op": "ratio", "sources": [b, a]})

    reg = ctx.task == TaskType.REGRESSION
    cv = KFold(3, shuffle=True, random_state=ctx.seed) if reg else StratifiedKFold(3, shuffle=True, random_state=ctx.seed)
    scoring = "r2" if reg else ("roc_auc" if ctx.task == TaskType.BINARY else "balanced_accuracy")

    def score(X):
        est = LinearRegression() if reg else LogisticRegression(max_iter=1000)
        pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(np.mean(cross_val_score(pipe, X.replace([np.inf, -np.inf], np.nan), y, cv=cv, scoring=scoring)))

    rows = []
    for r in recipes:
        srcs = [pd.to_numeric(data[s], errors="coerce") for s in r["sources"]]
        base_X = pd.concat(srcs, axis=1)
        new = apply_recipe(data, r).rename("new")
        b0, b1 = score(base_X), score(pd.concat([base_X, new], axis=1))
        rows.append({"name": _name(r), "recipe": r, "screen_base": round(b0, 4), "screen_with": round(b1, 4),
                     "screen_gain": round(b1 - b0, 4)})
    rows.sort(key=lambda r: r["screen_gain"], reverse=True)
    return {"scoring": scoring, "n_candidates": len(rows),
            "method": "3-fold CV linear model on source columns, with vs without the candidate (training rows only)",
            "suggestions": [r for r in rows if r["screen_gain"] >= min_gain][:top_k], "top10": rows[:10]}
