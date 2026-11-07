"""The XGBoost models, with early stopping on the validation fold."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

from skeptical_second_reader.split import SEED

ScoreFn = Callable[[pd.DataFrame], np.ndarray]
FLAG_PRECISION = 0.7

BASELINE_PARAMS: dict[str, Any] = {
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "max_depth": 6,
}

# Best accepted trial of the validation-only search in notebook 04.
FINAL_PARAMS: dict[str, Any] = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 1.0,
    "reg_lambda": 1,
}


def _fit(
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> ScoreFn:
    model = XGBClassifier(
        **params,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=SEED,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    def score(X: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    score.model = model
    return score


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> ScoreFn:
    """Train the baseline; returns its scoring function (`FitFn` of `cv_predict`).

    The trained model stays available as `score.model`.
    """
    return _fit(BASELINE_PARAMS, X_train, y_train, X_val, y_val)


def fit_final(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> ScoreFn:
    """Train with `FINAL_PARAMS`; otherwise the same as `fit_xgboost`."""
    return _fit(FINAL_PARAMS, X_train, y_train, X_val, y_val)


def precision_threshold(
    y_val: pd.Series, val_scores: np.ndarray, wanted: float = FLAG_PRECISION
) -> float:
    """The lowest threshold with at least `wanted` precision on the validation fold."""
    precision, _, thresholds = precision_recall_curve(y_val, val_scores)
    reached = precision[:-1] >= wanted
    if not reached.any():
        raise ValueError(f"no threshold reaches precision {wanted}")
    return float(thresholds[reached.argmax()])
