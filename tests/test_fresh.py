import numpy as np
import pandas as pd
import pytest

from skeptical_second_reader import fresh
from skeptical_second_reader.pipeline import READ_ALONE, READ_SHOTS
from skeptical_second_reader.reading_sample import (
    MATCHED_NAME,
    OUTSIDE_NAME,
    RANDOM_NAME,
)

STRATA = pd.Series(
    ["pattern CYCLE", OUTSIDE_NAME, MATCHED_NAME, RANDOM_NAME],
    index=["p1", "p2", "c1", "c2"],
)


def answer(model, condition, account, decision, confidence):
    return {
        "model": model,
        "condition": condition,
        "account": account,
        "decision": decision,
        "confidence": confidence,
    }


def test_the_reading_seeds_are_refused():
    for seed in (11, 12):
        with pytest.raises(ValueError):
            fresh.check_seed(seed)
    fresh.check_seed(101)


def test_only_the_clean_strata_are_clean():
    assert fresh.truth_of(STRATA).to_dict() == {"p1": 1, "p2": 1, "c1": 0, "c2": 0}


def test_a_small_run_keeps_every_stratum():
    small = fresh.SIZES["small"]
    assert set(small) == {"n_per_pattern", "n_outside", "n_matched", "n_random_clean"}
    assert all(n > 0 for n in small.values())
    assert fresh.SIZES["full"] == {}


def test_the_llm_score_follows_the_decision():
    both = (READ_ALONE, READ_SHOTS)
    rows = [answer("m:1", c, "p1", "suspicious", 0.9) for c in both]
    rows += [answer("m:1", c, "c1", "not_suspicious", 0.9) for c in both]
    readers = fresh.llm_readers(pd.DataFrame(rows))
    score, flag = readers[f"m {READ_ALONE}"]
    assert score.to_dict() == pytest.approx({"p1": 0.9, "c1": 0.1})
    assert flag.to_dict() == {"p1": True, "c1": False}


def test_an_unusable_answer_leaves_both_conditions_of_that_model():
    rows = [
        answer("m:1", READ_ALONE, "p1", "suspicious", 0.9),
        answer("m:1", READ_SHOTS, "p1", None, None),
        answer("m:1", READ_ALONE, "c1", "not_suspicious", 0.8),
        answer("m:1", READ_SHOTS, "c1", "not_suspicious", 0.8),
        answer("n:1", READ_ALONE, "p1", "suspicious", 0.9),
        answer("n:1", READ_SHOTS, "p1", "suspicious", 0.9),
    ]
    readers = fresh.llm_readers(pd.DataFrame(rows))
    for condition in (READ_ALONE, READ_SHOTS):
        assert list(readers[f"m {condition}"][0].index) == ["c1"]
        assert list(readers[f"n {condition}"][0].index) == ["p1"]


def test_the_table_measures_each_reader_on_its_own_accounts():
    readers = {
        "perfect": (
            pd.Series([0.9, 0.8, 0.2, 0.1], index=STRATA.index),
            pd.Series([True, True, False, False], index=STRATA.index),
        ),
        "alarmed": (
            pd.Series([0.1, 0.9, 0.8, 0.2], index=STRATA.index),
            pd.Series([False, True, True, False], index=STRATA.index),
        ),
        "positives only": (
            pd.Series([0.9, 0.8], index=["p1", "p2"]),
            pd.Series([True, False], index=["p1", "p2"]),
        ),
    }
    table = fresh.reader_table(STRATA, readers).set_index("reader")
    assert table.loc["perfect", "AUC-ROC"] == 1.0
    assert table.loc["perfect", "recall"] == 1.0
    assert table.loc["perfect", "false alarm rate"] == 0.0
    assert table.loc["alarmed", "AUC-ROC"] == 0.5
    assert table.loc["alarmed", "recall"] == 0.5
    assert table.loc["alarmed", "false alarm rate"] == 0.5
    assert table.loc["positives only", "accounts"] == 2
    assert np.isnan(table.loc["positives only", "AUC-ROC"])
    assert np.isnan(table.loc["positives only", "false alarm rate"])


def test_the_ml_readers_use_the_stored_scores_and_flags():
    accounts = pd.Index(["p1", "c1"])
    reading = pd.DataFrame(
        {"score": [0.7, 0.2], "flag": [1, 0]}, index=["c1", "p1"]
    ).iloc[::-1]
    ceiling = pd.DataFrame(
        {"scores evidence and facts": [0.6, 0.3], "flags evidence and facts": [1, 0]},
        index=["p1", "c1"],
    )
    readers = fresh.ml_readers(accounts, reading, ceiling)
    score, flag = readers["XGBoost transaction model"]
    assert score.to_dict() == {"p1": 0.2, "c1": 0.7}
    assert flag.to_dict() == {"p1": False, "c1": True}
    score, flag = readers["XGBoost ceiling (evidence and facts)"]
    assert flag.to_dict() == {"p1": True, "c1": False}
