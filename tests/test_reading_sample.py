"""Tests for the reading sample, on small hand-made data."""

import pandas as pd
import pytest

from skeptical_second_reader import reading_sample as rs


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(rs, "PER_PATTERN", 3)
    monkeypatch.setattr(rs, "OUTSIDE", 2)
    monkeypatch.setattr(rs, "MATCHED", 4)
    monkeypatch.setattr(rs, "RANDOM_CLEAN", 5)
    accounts = [f"a{i}" for i in range(200)]
    index = pd.Index(accounts, name="account")
    labels = pd.Series(0, index=index)
    labels.iloc[:40] = 1
    in_file = pd.Series(False, index=index)
    in_file.iloc[:30] = True
    in_file.iloc[100:] = True  # in the file but clean: never a clean pick
    activity = pd.Series(range(200), index=index) % 20 + 1
    patterns = pd.Series(
        ["FAN-IN"] * 10 + ["CYCLE"] * 10 + ["STACK"] * 10, index=index[:30]
    )
    # Clean accounts with a type, as the file has: never a pattern pick.
    clean_typed = pd.Series("FAN-IN", index=index[150:200])
    patterns = pd.concat([patterns, clean_typed])
    return labels, activity, patterns, in_file


def test_account_patterns_keeps_accounts_of_one_type_only():
    patterns = pd.DataFrame(
        {"attempt": [0, 0, 1, 2], "pattern": ["FAN-IN", "FAN-IN", "CYCLE", "FAN-IN"]}
    )
    members = pd.DataFrame(
        {"attempt": [0, 1, 0, 2, 1], "account": ["x", "x", "y", "y", "z"]}
    )
    kinds = rs.account_patterns(patterns, members)
    assert kinds.to_dict() == {"y": "FAN-IN", "z": "CYCLE"}


def test_every_stratum_has_its_size_and_its_kind_of_account(small):
    labels, activity, patterns, in_file = small
    sample = rs.reading_sample(labels, activity, patterns, in_file, seed=0)
    counts = sample.value_counts()
    for kind in ["FAN-IN", "CYCLE", "STACK"]:
        name = rs.pattern_stratum(kind)
        assert counts[name] == 3
        assert (patterns[sample[sample == name].index] == kind).all()
    assert counts[rs.OUTSIDE_NAME] == 2
    assert counts[rs.MATCHED_NAME] == 4
    assert counts[rs.RANDOM_NAME] == 5
    assert not sample.index.duplicated().any()
    positives = sample[~sample.isin([rs.MATCHED_NAME, rs.RANDOM_NAME])].index
    assert (labels[positives] == 1).all()
    outside = sample[sample == rs.OUTSIDE_NAME].index
    assert not in_file[outside].any()
    clean = sample[sample.isin([rs.MATCHED_NAME, rs.RANDOM_NAME])].index
    assert (labels[clean] == 0).all()
    assert not in_file[clean].any()


def test_matched_accounts_are_as_busy_as_chosen_positives(small):
    labels, activity, patterns, in_file = small
    sample = rs.reading_sample(labels, activity, patterns, in_file, seed=0)
    positives = sample[~sample.isin([rs.MATCHED_NAME, rs.RANDOM_NAME])].index
    matched = sample[sample == rs.MATCHED_NAME].index
    assert activity[matched].isin(activity[positives]).all()


def test_matching_takes_the_nearest_when_no_equal_activity_is_left():
    activity = pd.Series({"p": 10, "c1": 3, "c2": 9, "c3": 30})
    chosen = rs._matched(pd.Index(["p"]), activity, pd.Index(["c1", "c2", "c3"]), 0)
    assert chosen.tolist() == ["c2"]


def test_the_same_seed_gives_the_same_sample_and_another_seed_another(small):
    labels, activity, patterns, in_file = small
    first = rs.reading_sample(labels, activity, patterns, in_file, seed=0)
    again = rs.reading_sample(labels, activity, patterns, in_file, seed=0)
    other = rs.reading_sample(labels, activity, patterns, in_file, seed=1)
    pd.testing.assert_series_equal(first, again)
    assert set(first.index) != set(other.index)


def test_dev_and_test_samples_are_disjoint_and_skip_excluded(small):
    labels, activity, patterns, in_file = small
    excluded = pd.Index(["a0", "a1", "a35", "a100"])
    dev, test = rs.reading_samples(
        labels, activity, patterns, in_file, 0, 1, exclude=excluded
    )
    assert not set(dev.index) & set(test.index)
    assert not set(excluded) & (set(dev.index) | set(test.index))
    assert len(dev) == len(test) == 3 * 3 + 2 + 4 + 5
