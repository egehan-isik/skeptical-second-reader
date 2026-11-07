"""SHAP explanations through XGBoost's own TreeSHAP (`pred_contribs`)."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

BASE_VALUE = "base_value"


def shap_values(model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """One column per feature plus `BASE_VALUE`, in log-odds; each row adds up to the
    model's margin for that account. Raises if it does not."""
    booster = model.get_booster()
    matrix = xgb.DMatrix(X)
    limit = (0, model.best_iteration + 1)
    values = booster.predict(matrix, pred_contribs=True, iteration_range=limit)
    margin = booster.predict(matrix, output_margin=True, iteration_range=limit)
    if not np.allclose(values.sum(axis=1), margin, atol=1e-3):
        raise ValueError("SHAP values do not add up to the model's margin")
    return pd.DataFrame(values, index=X.index, columns=[*X.columns, BASE_VALUE])


def out_of_fold_shap(
    models: Sequence[xgb.XGBClassifier], X: pd.DataFrame, folds: pd.Series
) -> pd.DataFrame:
    """Explain every account with the model of the round in which it was the test
    fold: `models[k]` must be the model of round k and explains fold k."""
    if not X.index.equals(folds.index):
        raise ValueError("X and folds must have the same index")
    if set(folds.unique()) != set(range(len(models))):
        raise ValueError("one model per fold is needed")
    parts = [shap_values(model, X[folds == k]) for k, model in enumerate(models)]
    return pd.concat(parts).loc[X.index]


def top_contributions(values: pd.Series, n: int = 5) -> pd.Series:
    """The n features with the largest absolute contribution for one account."""
    features = values.drop(BASE_VALUE)
    order = features.abs().sort_values(ascending=False, kind="stable").index
    return features.reindex(order[:n])
