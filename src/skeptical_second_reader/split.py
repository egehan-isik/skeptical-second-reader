"""Group-disjoint cross-validation: groups, fold assignment, fold roles, CV loop.

Every account gets one fold number. All accounts of a group share that fold, so a
group is never on both sides of a split. A group is a laundering pattern for pattern
accounts and the owner for all other accounts (see `split_groups`). In round k the
roles rotate:

    test  = fold k
    val   = fold k + 1 (mod n_folds)   -- early stopping only
    train = the other folds

So each account is tested exactly once and used for validation exactly once.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 5
SEED = 42

TRAIN, VAL, TEST = "train", "val", "test"

# fit(X_train, y_train, X_val, y_val) returns a function that scores rows of X.
FitFn = Callable[
    [pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    Callable[[pd.DataFrame], np.ndarray],
]

# choose_threshold(y_val, val_scores) returns the score from which a case is flagged.
ThresholdFn = Callable[[pd.Series, np.ndarray], float]


PATTERN_PREFIX = "pattern:"


def split_groups(owners: pd.Series, members: pd.DataFrame) -> pd.Series:
    """The unit kept together by the split: laundering pattern first, then owner.

    Accounts of the same laundering attempt get one group; attempts that share an
    account are merged (connected components). Accounts outside any pattern keep
    their owner as group. Owners and patterns are not merged with each other,
    because owners with accounts in several patterns would pull these patterns into
    one giant group.

    `owners` is indexed by account key; `members` has columns attempt, account.
    """
    if owners.astype(str).str.startswith(PATTERN_PREFIX).any():
        raise ValueError(f"owner IDs must not start with {PATTERN_PREFIX!r}")
    accounts = pd.Index(members["account"].unique())
    if not accounts.isin(owners.index).all():
        raise ValueError("pattern accounts missing from the owner index")
    attempts = pd.Index(members["attempt"].unique())
    n_acc, n_att = accounts.size, attempts.size
    rows = accounts.get_indexer(members["account"])
    cols = n_acc + attempts.get_indexer(members["attempt"])
    graph = coo_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(n_acc + n_att, n_acc + n_att)
    )
    _, component = connected_components(graph, directed=False)
    groups = owners.astype(str).rename("group")
    groups[accounts] = [f"{PATTERN_PREFIX}{c}" for c in component[:n_acc]]
    return groups


def assign_folds(
    labels: pd.Series,
    groups: pd.Series,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> pd.Series:
    """Give every account a fold number in 0..n_folds-1.

    Accounts of the same group (owner) always share a fold; the positive rate is
    kept similar across folds. `labels` and `groups` must have the same index.
    """
    if not labels.index.equals(groups.index):
        raise ValueError("labels and groups must have the same index")
    if groups.isna().any():
        raise ValueError("every account needs a group")
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = np.full(len(labels), -1)
    dummy_x = np.zeros(len(labels))
    split = splitter.split(dummy_x, labels.to_numpy(), groups.to_numpy())
    for fold, (_, test_idx) in enumerate(split):
        folds[test_idx] = fold
    return pd.Series(folds, index=labels.index, name="fold")


def fold_roles(folds: pd.Series, k: int, n_folds: int = N_FOLDS) -> pd.Series:
    """Role of every account in round k: "test", "val" or "train"."""
    if not 0 <= k < n_folds:
        raise ValueError(f"round {k} is outside 0..{n_folds - 1}")
    roles = pd.Series(TRAIN, index=folds.index, name="role")
    roles[folds == k] = TEST
    roles[folds == (k + 1) % n_folds] = VAL
    return roles


@dataclass
class CVResult:
    """Out-of-fold results of `cv_predict`.

    scores: score of every account, from the round in which it was the test fold.
    thresholds: one threshold per round, chosen on that round's val fold only
        (empty if no threshold rule was given).
    flags: 1 if an account's score reaches its round's threshold (None if no rule).
    """

    scores: pd.Series
    thresholds: list[float] = field(default_factory=list)
    flags: pd.Series | None = None


def cv_predict(
    X: pd.DataFrame,
    y: pd.Series,
    folds: pd.Series,
    fit: FitFn,
    choose_threshold: ThresholdFn | None = None,
    n_folds: int = N_FOLDS,
) -> CVResult:
    """Out-of-fold scores: each account is scored by the model that never saw it.

    In every round, `fit` receives only the train and val rows, and
    `choose_threshold` only the val labels and val scores. The test rows are
    passed only to the scoring function, after the threshold is fixed.
    """
    if not (X.index.equals(y.index) and X.index.equals(folds.index)):
        raise ValueError("X, y and folds must have the same index")
    scores = pd.Series(np.nan, index=X.index, name="score")
    flags = pd.Series(-1, index=X.index, name="flag")
    thresholds = []
    for k in range(n_folds):
        roles = fold_roles(folds, k, n_folds)
        train, val, test = roles == TRAIN, roles == VAL, roles == TEST
        score = fit(X[train], y[train], X[val], y[val])
        if choose_threshold is not None:
            thresholds.append(float(choose_threshold(y[val], score(X[val]))))
        scores[test] = score(X[test])
        if choose_threshold is not None:
            flags[test] = (scores[test] >= thresholds[-1]).astype(int)
    if scores.isna().any():
        raise ValueError("some accounts were never scored")
    if choose_threshold is None:
        return CVResult(scores=scores)
    return CVResult(scores=scores, thresholds=thresholds, flags=flags)
