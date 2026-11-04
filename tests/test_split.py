"""Leakage tests for the entity-disjoint cross-validation.

Synthetic data only (no real dataset), so they run in CI. Owners have 1-6 accounts
and about 10% of the accounts are positive.
"""

import numpy as np
import pandas as pd
import pytest

from skeptical_second_reader.split import (
    N_FOLDS,
    PATTERN_PREFIX,
    TEST,
    TRAIN,
    VAL,
    assign_folds,
    cv_predict,
    fold_roles,
    split_groups,
)


@pytest.fixture
def accounts():
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 7, size=600)
    owners = np.repeat([f"E{i}" for i in range(sizes.size)], sizes)
    index = pd.Index([f"A{i}" for i in range(owners.size)], name="account")
    labels = pd.Series((rng.random(owners.size) < 0.1).astype(int), index=index)
    groups = pd.Series(owners, index=index)
    return labels, groups


def test_every_account_gets_exactly_one_valid_fold(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    assert folds.index.equals(labels.index)
    assert set(folds.unique()) == set(range(N_FOLDS))


def test_no_owner_is_in_two_folds(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    assert (folds.groupby(groups).nunique() == 1).all()


def test_positive_rate_is_similar_across_folds(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    rates = labels.groupby(folds).mean()
    assert (rates - labels.mean()).abs().max() < 0.02


def test_same_seed_gives_same_folds_and_other_seed_differs(accounts):
    labels, groups = accounts
    first = assign_folds(labels, groups, seed=1)
    assert first.equals(assign_folds(labels, groups, seed=1))
    assert not first.equals(assign_folds(labels, groups, seed=2))


def test_mismatched_or_missing_groups_are_refused(accounts):
    labels, groups = accounts
    with pytest.raises(ValueError, match="same index"):
        assign_folds(labels, groups.iloc[1:])
    with pytest.raises(ValueError, match="group"):
        assign_folds(labels, groups.where(groups != "E0"))


def test_roles_are_disjoint_cover_all_and_rotate(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    tested = pd.Series(0, index=folds.index)
    validated = pd.Series(0, index=folds.index)
    for k in range(N_FOLDS):
        roles = fold_roles(folds, k)
        # One role per account, all three roles present, owners on one side.
        assert set(roles.unique()) == {TRAIN, VAL, TEST}
        assert (roles.groupby(groups).nunique() == 1).all()
        tested += roles == TEST
        validated += roles == VAL
    # Over the rounds every account is tested once and validated once.
    assert (tested == 1).all()
    assert (validated == 1).all()


def test_round_outside_range_is_refused(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    with pytest.raises(ValueError, match="outside"):
        fold_roles(folds, N_FOLDS)


def test_cv_predict_never_trains_on_test_rows(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    X = pd.DataFrame({"x": np.arange(labels.size)}, index=labels.index)
    seen = []

    def spy_fit(x_train, y_train, x_val, y_val):
        # Records what each round's model saw, and what it was asked to score.
        record = {"train": set(x_train.index), "val": set(x_val.index)}
        assert x_train.index.equals(y_train.index)
        assert x_val.index.equals(y_val.index)

        def score(x_test):
            record["test"] = set(x_test.index)
            return np.ones(len(x_test))

        seen.append(record)
        return score

    scores = cv_predict(X, labels, folds, spy_fit).scores

    assert len(seen) == N_FOLDS
    for k, record in enumerate(seen):
        assert record["test"] == set(folds.index[folds == k])
        assert not record["test"] & (record["train"] | record["val"])
        # No owner of a test account was seen in training or validation.
        train_owners = set(groups[list(record["train"] | record["val"])])
        assert not set(groups[list(record["test"])]) & train_owners
    assert scores.notna().all()


def test_cv_predict_returns_each_rounds_scores_for_its_own_test_rows(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    X = pd.DataFrame({"x": np.zeros(labels.size)}, index=labels.index)
    round_no = iter(range(N_FOLDS))

    def fit(x_train, y_train, x_val, y_val):
        k = next(round_no)
        return lambda x_test: np.full(len(x_test), float(k))

    scores = cv_predict(X, labels, folds, fit).scores
    # The score of an account names the round in which it was the test fold.
    assert (scores == folds.astype(float)).all()


def test_cv_predict_refuses_misaligned_inputs(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    X = pd.DataFrame({"x": np.zeros(labels.size)}, index=labels.index)
    with pytest.raises(ValueError, match="same index"):
        cv_predict(X.iloc[1:], labels, folds, lambda *a: None)


def test_threshold_is_chosen_on_val_rows_only(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    X = pd.DataFrame({"x": np.arange(labels.size)}, index=labels.index)
    chosen_on = []

    def fit(x_train, y_train, x_val, y_val):
        return lambda x: np.asarray(x["x"], dtype=float)

    def spy_threshold(y_val, val_scores):
        chosen_on.append(set(y_val.index))
        assert len(val_scores) == len(y_val)
        return 0.0

    result = cv_predict(X, labels, folds, fit, choose_threshold=spy_threshold)

    assert len(chosen_on) == N_FOLDS
    for k, rows in enumerate(chosen_on):
        assert rows == set(folds.index[folds == (k + 1) % N_FOLDS])
        assert not rows & set(folds.index[folds == k])
    assert len(result.thresholds) == N_FOLDS


def test_flags_use_the_threshold_of_the_accounts_own_round(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    # Score = row number; threshold of round k = median val score of round k.
    X = pd.DataFrame({"x": np.arange(labels.size)}, index=labels.index)

    def fit(x_train, y_train, x_val, y_val):
        return lambda x: np.asarray(x["x"], dtype=float)

    def median_threshold(y_val, val_scores):
        return float(np.median(val_scores))

    result = cv_predict(X, labels, folds, fit, choose_threshold=median_threshold)

    assert set(result.flags.unique()) <= {0, 1}
    for k, threshold in enumerate(result.thresholds):
        test = folds == k
        expected = (result.scores[test] >= threshold).astype(int)
        assert result.flags[test].equals(expected)


def test_no_threshold_rule_gives_no_flags(accounts):
    labels, groups = accounts
    folds = assign_folds(labels, groups)
    X = pd.DataFrame({"x": np.zeros(labels.size)}, index=labels.index)
    result = cv_predict(X, labels, folds, lambda *a: lambda x: np.zeros(len(x)))
    assert result.thresholds == []
    assert result.flags is None


def test_pattern_accounts_share_a_group_across_owners():
    owners = pd.Series(["O1", "O2", "O3", "O4"], index=["a", "b", "c", "d"])
    members = pd.DataFrame({"attempt": [0, 0], "account": ["a", "b"]})
    groups = split_groups(owners, members)
    # a and b are in one pattern but have different owners.
    assert groups["a"] == groups["b"]
    assert groups["a"].startswith(PATTERN_PREFIX)
    # c and d are outside any pattern and keep their owners.
    assert groups["c"] == "O3"
    assert groups["d"] == "O4"
    assert groups.index.equals(owners.index)


def test_attempts_sharing_an_account_are_merged():
    owners = pd.Series(["O1", "O2", "O3", "O4"], index=["a", "b", "c", "d"])
    # Attempt 0 = {a, b}, attempt 1 = {b, c}: b links them; d is alone.
    members = pd.DataFrame(
        {"attempt": [0, 0, 1, 1, 2], "account": ["a", "b", "b", "c", "d"]}
    )
    groups = split_groups(owners, members)
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]


def test_owner_of_a_pattern_account_does_not_pull_in_other_patterns():
    # a (attempt 0) and c (attempt 1) have the same owner; they stay separate.
    owners = pd.Series(["O1", "O2", "O1", "O3"], index=["a", "b", "c", "d"])
    members = pd.DataFrame({"attempt": [0, 0, 1, 1], "account": ["a", "b", "c", "d"]})
    groups = split_groups(owners, members)
    assert groups["a"] == groups["b"]
    assert groups["c"] == groups["d"]
    assert groups["a"] != groups["c"]


def test_split_groups_refuses_unknown_accounts_and_clashing_owner_ids():
    owners = pd.Series(["O1"], index=["a"])
    with pytest.raises(ValueError, match="missing"):
        split_groups(owners, pd.DataFrame({"attempt": [0], "account": ["zz"]}))
    clash = pd.Series([f"{PATTERN_PREFIX}0"], index=["a"])
    with pytest.raises(ValueError, match="must not start"):
        split_groups(clash, pd.DataFrame({"attempt": [0], "account": ["a"]}))


def test_no_pattern_crosses_folds(accounts):
    labels, owners = accounts
    rng = np.random.default_rng(1)
    # 40 patterns of 3-8 accounts drawn across owners; some accounts in two.
    rows = [
        (attempt, account)
        for attempt in range(40)
        for account in rng.choice(labels.index, size=rng.integers(3, 9), replace=False)
    ]
    members = pd.DataFrame(rows, columns=["attempt", "account"]).drop_duplicates()
    groups = split_groups(owners, members)
    folds = assign_folds(labels, groups)
    spread = members.assign(fold=members["account"].map(folds))
    assert (spread.groupby("attempt")["fold"].nunique() == 1).all()
    # Accounts outside patterns still keep their owner on one side.
    outside = ~labels.index.isin(members["account"])
    assert (folds[outside].groupby(owners[outside]).nunique() == 1).all()
