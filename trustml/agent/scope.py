"""Scope guard (A1): name unsupported requests explicitly instead of guessing."""
from __future__ import annotations

import re

from pydantic import BaseModel

UNSUPPORTED = [
    (r"\b(forecast\w*|time[- ]series|arima|seasonal\w*|how many .{0,40} next (day|week|month|quarter|year))\b",
     "Time-series forecasting (predicting future values of a series) is not supported. Checking whether a model "
     "holds up on later data IS supported (forward-in-time validation)."),
    (r"\b(image|images|photo|cnn|computer vision)\b", "Image data is not supported."),
    (r"\b(nlp|sentiment|free[- ]text|text classification|topic model\w*|embedding\w*)\b",
     "Text/NLP modelling is not supported."),
    (r"\b(deep learning|neural net\w*|lstm|transformer\w*)\b",
     "Deep learning models are not supported; TrustML uses logistic/linear regression and random forests."),
    (r"\b(cluster\w*|segmentation|unsupervised|k-?means)\b",
     "Unsupervised tasks (clustering/segmentation) are not supported; a target column is required."),
    (r"\b(deploy\w*|api endpoint|real[- ]time serving|model monitoring)\b", "Deployment and monitoring are not supported."),
    (r"\b(join|merge|multiple (files|tables|datasets))\b", "Only a single table per run is supported."),
    (r"\b(hyper-?parameter (search|tuning|optimi[sz]ation)|automl|grid search)\b",
     "Hyperparameter search/AutoML is not supported; a small fixed model set is used."),
    (r"\b(xgboost|lightgbm|catboost|svm|gradient boost\w*)\b",
     "Only dummy, logistic/linear regression and random forest models are available."),
]


class ScopeItem(BaseModel):
    matched: str
    reason: str


def scope_guard(objective: str) -> list[ScopeItem]:
    items: list[ScopeItem] = []
    for pattern, reason in UNSUPPORTED:
        m = re.search(pattern, objective, flags=re.I)
        if m:
            items.append(ScopeItem(matched=m.group(0), reason=reason))
    return items
