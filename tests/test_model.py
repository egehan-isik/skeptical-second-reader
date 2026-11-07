"""Tests for the baseline model wrapper, on synthetic data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from skeptical_second_reader.model import FINAL_PARAMS, fit_xgboost


def test_final_parameters_are_the_chosen_ones():
    assert FINAL_PARAMS == {
        "n_estimators": 3000,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 1,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "reg_lambda": 1,
    }


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(1000, 3)), columns=["signal", "a", "b"])
    y = pd.Series((X["signal"] > 0.8).astype(int))
    return X.iloc[:600], y.iloc[:600], X.iloc[600:], y.iloc[600:]


def test_scores_are_probabilities_that_follow_the_signal(data):
    score = fit_xgboost(*data)
    X_val, y_val = data[2], data[3]
    scores = score(X_val)
    assert scores.shape == (len(X_val),)
    assert ((scores >= 0) & (scores <= 1)).all()
    assert roc_auc_score(y_val, scores) > 0.95


def test_same_data_gives_same_scores(data):
    np.testing.assert_array_equal(
        fit_xgboost(*data)(data[2]), fit_xgboost(*data)(data[2])
    )


def test_training_stops_early_on_the_validation_fold(data):
    X_train, y_train, X_val, y_val = data
    # Validation labels unrelated to the data: more trees cannot keep helping.
    noise = pd.Series(
        np.random.default_rng(1).integers(0, 2, len(y_val)), index=y_val.index
    )
    model = fit_xgboost(X_train, y_train, X_val, noise).model
    assert model.best_iteration < 200
    # On these labels nothing scores well. On the training data it would be near 1.
    assert model.best_score < 0.8
