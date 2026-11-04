"""The baseline model: XGBoost with early stopping on the validation fold."""

from collections.abc import Callable

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from skeptical_second_reader.split import SEED


def fit_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Train one model and return its scoring function (`FitFn` of `cv_predict`).

    The trained model stays available as `score.model`. Parameters are close to the
    defaults; this is the first baseline, without a hyperparameter search.
    """
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.1,
        max_depth=6,
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
