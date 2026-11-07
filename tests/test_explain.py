"""Tests for the SHAP helpers and the final model wrapper, on synthetic data."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from skeptical_second_reader.explain import (
    BASE_VALUE,
    out_of_fold_shap,
    shap_values,
    top_contributions,
)
from skeptical_second_reader.model import FINAL_PARAMS, fit_final


def fit(X, y, y_val=None):
    model = XGBClassifier(
        n_estimators=60,
        max_depth=3,
        eval_metric="aucpr",
        early_stopping_rounds=5,
        random_state=0,
    )
    return model.fit(X, y, eval_set=[(X, y if y_val is None else y_val)], verbose=False)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 3)), columns=["signal", "a", "b"])
    X.index = [f"account{i}" for i in range(len(X))]
    return X, (X["signal"] > 0.5).astype(int)


def test_values_add_up_to_the_margin(data):
    X, y = data
    model = fit(X, y)
    values = shap_values(model, X)
    assert list(values.columns) == [*X.columns, BASE_VALUE]
    assert values.index.equals(X.index)
    margin = model.predict(X, output_margin=True)
    np.testing.assert_allclose(values.sum(axis=1), margin, atol=1e-3)


def test_the_signal_gets_the_largest_contribution(data):
    X, y = data
    values = shap_values(fit(X, y), X)
    assert values[list(X.columns)].abs().mean().idxmax() == "signal"


def test_explanation_stops_at_the_best_iteration(data):
    X, y = data
    noise = pd.Series(np.random.default_rng(1).integers(0, 2, len(y)), index=y.index)
    model = fit(X, y, noise)
    assert model.best_iteration < model.get_booster().num_boosted_rounds() - 1
    margin = model.predict(X, output_margin=True)
    np.testing.assert_allclose(shap_values(model, X).sum(axis=1), margin, atol=1e-3)


def test_every_account_is_explained_by_the_model_of_its_own_fold(data):
    X, y = data
    folds = pd.Series(np.arange(len(X)) % 2, index=X.index)
    # Model 0 learned the opposite of model 1, so a swap cannot go unnoticed.
    models = [fit(X, y), fit(X, 1 - y)]
    values = out_of_fold_shap(models, X, folds)
    assert values.index.equals(X.index)
    for k, model in enumerate(models):
        rows = X[folds == k]
        pd.testing.assert_frame_equal(values.loc[rows.index], shap_values(model, rows))


def test_fold_mistakes_are_refused(data):
    X, y = data
    folds = pd.Series(np.arange(len(X)) % 2, index=X.index)
    model = fit(X, y)
    with pytest.raises(ValueError, match="one model per fold"):
        out_of_fold_shap([model], X, folds)
    with pytest.raises(ValueError, match="same index"):
        out_of_fold_shap([model, model], X, folds.iloc[::-1])


def test_top_contributions_order_by_size_and_drop_the_base_value():
    values = pd.Series({"a": 0.1, "b": -2.0, "c": 0.5, BASE_VALUE: 9.0})
    top = top_contributions(values, n=2)
    assert list(top.index) == ["b", "c"]
    assert top["b"] == -2.0


def test_final_model_uses_the_final_parameters_and_the_validation_fold(data):
    X, y = data
    noise = pd.Series(np.random.default_rng(1).integers(0, 2, len(y)), index=y.index)
    model = fit_final(X, y, X, noise).model
    params = model.get_params()
    for name, value in FINAL_PARAMS.items():
        assert params[name] == value
    assert params["early_stopping_rounds"] == 50
    assert model.best_iteration < FINAL_PARAMS["n_estimators"] - 1
