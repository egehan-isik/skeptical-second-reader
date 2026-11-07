"""Tests for the batch pipeline: what is skipped, what runs again, what is reused."""

import numpy as np
import pandas as pd
import pytest

from skeptical_second_reader import pipeline as pl
from skeptical_second_reader import second_reader as sr
from skeptical_second_reader.facts import FACTS
from skeptical_second_reader.transactions import TX_FEATURES


def test_file_digest_follows_the_bytes(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"one")
    b.write_bytes(b"one")
    assert pl.file_digest(a) == pl.file_digest(b)
    b.write_bytes(b"two")
    assert pl.file_digest(a) != pl.file_digest(b)


def test_stage_key_follows_the_inputs_not_their_order():
    assert pl.stage_key(a=1, b=2) == pl.stage_key(b=2, a=1)
    assert pl.stage_key(a=1, b=2) != pl.stage_key(a=1, b=3)
    assert pl.stage_key(a=1) != pl.stage_key(b=1)


def test_saving_leaves_no_temporary_file(tmp_path):
    path = tmp_path / "x.parquet"
    pl.save_parquet(pd.DataFrame({"a": [1]}), path)
    assert [p.name for p in tmp_path.iterdir()] == ["x.parquet"]
    assert pd.read_parquet(path)["a"].tolist() == [1]


def writer(store, calls, name="out"):
    def compute():
        calls.append(1)
        store.path(name).write_text("x")

    return compute


def test_a_stage_with_the_same_key_is_skipped(tmp_path):
    store, calls = pl.StageStore(tmp_path), []
    assert store.run("s", "k1", ["out"], writer(store, calls)) is True
    assert store.run("s", "k1", ["out"], writer(store, calls)) is False
    assert len(calls) == 1


def test_a_changed_key_runs_the_stage_again(tmp_path):
    store, calls = pl.StageStore(tmp_path), []
    store.run("s", "k1", ["out"], writer(store, calls))
    assert store.run("s", "k2", ["out"], writer(store, calls)) is True
    assert len(calls) == 2


def test_a_missing_file_runs_the_stage_again(tmp_path):
    store, calls = pl.StageStore(tmp_path), []
    store.run("s", "k1", ["out"], writer(store, calls))
    store.path("out").unlink()
    assert store.run("s", "k1", ["out"], writer(store, calls)) is True


def test_finished_stages_survive_a_restart(tmp_path):
    store, calls = pl.StageStore(tmp_path), []
    store.run("s", "k1", ["out"], writer(store, calls))
    again = pl.StageStore(tmp_path)
    assert again.run("s", "k1", ["out"], writer(again, calls)) is False


def test_a_failed_stage_is_not_recorded(tmp_path):
    store = pl.StageStore(tmp_path)

    def fail():
        store.path("out").write_text("half")
        raise OSError("stopped")

    with pytest.raises(OSError):
        store.run("s", "k1", ["out"], fail)
    calls = []
    assert store.run("s", "k1", ["out"], writer(store, calls)) is True
    assert len(calls) == 1


def test_a_stage_that_writes_nothing_is_an_error(tmp_path):
    store = pl.StageStore(tmp_path)
    with pytest.raises(RuntimeError, match="did not write"):
        store.run("s", "k1", ["out"], lambda: None)
    assert store.manifest == {}


def test_an_answer_key_follows_everything_that_changes_the_call():
    key = pl.answer_key("p", "d", False)
    assert key == pl.answer_key("p", "d", False)
    assert key != pl.answer_key("p ", "d", False)
    assert key != pl.answer_key("p", "e", False)
    assert key != pl.answer_key("p", "d", True)


REPLY = {
    "content": '{"decision": "suspicious", "confidence": 0.5, "evidence": [1], '
    '"reason": "r"}',
    "prompt_tokens": 3,
    "answer_tokens": 2,
    "seconds": 1.5,
}


def test_the_cache_keeps_answers_across_a_restart(tmp_path):
    cache = pl.AnswerCache(tmp_path / "a.sqlite")
    assert cache.get("k") is None
    cache.put("k", "m", "d", REPLY)
    cache.close()
    cache = pl.AnswerCache(tmp_path / "a.sqlite")
    assert cache.get("k") == REPLY
    assert len(cache) == 1
    cache.close()


@pytest.fixture
def prompts():
    return pd.DataFrame(
        {
            "account": ["a", "a", "b"],
            "group": ["g", "g", "h"],
            "condition": [sr.K1, sr.K3_D100, sr.K2],
            "prompt": ["p1", "p2", "p3"],
            "shown_level": [None, sr.NOT_FLAGGED, sr.STRONG],
            "shown_rate": [np.nan, 0.01, 0.9],
            "flipped": [None, True, False],
        }
    )


def counting_ask(calls):
    def ask(prompt, model, think):
        calls.append((prompt, model, think))
        return REPLY

    return ask


def test_only_new_prompts_reach_the_model(tmp_path, prompts):
    cache, calls = pl.AnswerCache(tmp_path / "a.sqlite"), []
    first = pl.answer_prompts(
        prompts.iloc[:2], "m", "d", False, cache, counting_ask(calls)
    )
    assert len(calls) == 2 and not first["cached"].any()
    second = pl.answer_prompts(prompts, "m", "d", False, cache, counting_ask(calls))
    assert [c[0] for c in calls] == ["p1", "p2", "p3"]
    assert second["cached"].tolist() == [True, True, False]
    cache.close()


def test_another_model_or_version_is_not_answered_from_the_cache(tmp_path, prompts):
    cache, calls = pl.AnswerCache(tmp_path / "a.sqlite"), []
    pl.answer_prompts(prompts, "m", "d", False, cache, counting_ask(calls))
    pl.answer_prompts(prompts, "m", "new digest", False, cache, counting_ask(calls))
    assert len(calls) == 6
    assert calls[0][2] is False
    cache.close()


def test_a_stopped_run_keeps_its_answers(tmp_path, prompts):
    cache, calls = pl.AnswerCache(tmp_path / "a.sqlite"), []

    def stops(prompt, model, think):
        if len(calls) == 1:
            raise OSError("stopped")
        calls.append(prompt)
        return REPLY

    with pytest.raises(OSError):
        pl.answer_prompts(prompts, "m", "d", False, cache, stops)
    assert len(cache) == 1
    cache.close()


def test_results_compute_following_from_what_was_shown(tmp_path, prompts):
    cache = pl.AnswerCache(tmp_path / "a.sqlite")
    results = pl.answer_prompts(prompts, "m", "d", False, cache, counting_ask([]))
    assert "prompt" not in results.columns
    assert results["decision"].tolist() == ["suspicious"] * 3
    # K1 shows nothing; suspicious against NOT FLAGGED; suspicious with a STRONG ALERT.
    assert results["follows_model"].tolist() == [None, False, True]
    assert results["evidence"].iloc[0] == "[1]"
    assert results["digest"].iloc[0] == "d"
    cache.close()


def test_the_experiment_sample_leaves_out_the_trial_accounts():
    names = sr.GROUP_NAMES
    index = [f"a{i}" for i in range(70 * len(names))]
    groups = pd.Series(names * 70, index=index)
    activity = pd.Series(50, index=index)
    trial = pl.trial_accounts(groups, activity)
    assert len(trial) == len(names) + 1
    sample = pl.experiment_accounts(groups, activity, 29, seed=1)
    assert len(sample) == 29 * len(names)
    assert not set(sample.index) & set(trial.index)
    pd.testing.assert_series_equal(
        sample, pl.experiment_accounts(groups, activity, 29, seed=1)
    )


def test_the_busy_trial_account_is_busy():
    names = sr.GROUP_NAMES
    index = [f"a{i}" for i in range(10 * len(names))]
    groups = pd.Series(names * 10, index=index)
    activity = pd.Series(1, index=index)
    busy_one = groups[groups == "uncertain, model wrong"].index[3]
    activity[busy_one] = sr.MAX_ROWS + 1
    trial = pl.trial_accounts(groups, activity)
    assert trial.index[-1] == busy_one


def two_accounts():
    table = pd.DataFrame(
        {
            "sender": ["1:A", "2:B"],
            "receiver": ["2:B", "1:A"],
            "timestamp": pd.to_datetime(["2022-09-01 10:00", "2022-09-01 11:00"]),
            "payment_format": ["ACH", "Wire"],
            "paid_currency": ["USD", "USD"],
            "received_currency": ["USD", "USD"],
            "usd": [10.0, 20.0],
            "usd_received": [10.0, 20.0],
            "same_bank": [False, False],
        }
    )
    features = pd.DataFrame(1.0, index=table.index, columns=TX_FEATURES)
    reading = pd.DataFrame(
        {
            "flag": [1, 0],
            "score": [0.95, 0.1],
            "calibrated": [0.9, 0.02],
            "n_high": [1, 0],
            "strongest_row": [0, 1],
        },
        index=["1:A", "2:B"],
    )
    shap = pd.DataFrame(0.1, index=reading.index, columns=TX_FEATURES)
    cases = pd.Series(["g1", "g2"], index=["1:A", "2:B"])
    return table, features, reading, shap, cases


def test_prompts_cover_every_account_and_condition():
    table, features, reading, shap, cases = two_accounts()
    prompts = pl.build_prompts(table, features, reading, shap, cases)
    assert len(prompts) == 2 * len(sr.CONDITIONS)
    assert prompts["prompt"].is_unique
    k1 = prompts[prompts["condition"] == sr.K1]
    assert k1["shown_level"].isna().all()
    assert k1["prompt"].str.contains("UPSTREAM").sum() == 0
    strong = prompts[(prompts["account"] == "1:A") & (prompts["condition"] == sr.K2)]
    assert strong["shown_level"].iloc[0] == sr.STRONG
    reversed_quiet = prompts[
        (prompts["account"] == "2:B") & (prompts["condition"] == sr.K3_D100)
    ]
    assert bool(reversed_quiet["flipped"].iloc[0]) is True
    # A strongest row that belongs to the account: never another account's.
    assert "strongest transaction is number" in strong["prompt"].iloc[0]


def test_the_reading_test_alone_is_the_first_experiment_k1_prompt():
    table, features, reading, shap, cases = two_accounts()
    first = pl.build_prompts(table, features, reading, shap, cases)
    facts = pd.DataFrame({name: [1.0, 2.0] for name in FACTS}, index=["1:A", "2:B"])
    reading_test = pl.reading_prompts(
        table, features["pair_count"].to_numpy(), cases, facts
    )
    assert len(reading_test) == 2 * 2
    alone = reading_test[reading_test["condition"] == pl.READ_ALONE]
    old = first[first["condition"] == sr.K1].set_index("account")["prompt"]
    assert (alone.set_index("account")["prompt"] == old).all()
    with_facts = reading_test[reading_test["condition"] == pl.READ_FACTS]
    text = with_facts.set_index("account").loc["2:B", "prompt"]
    assert "paid by 2 different accounts" in text
    assert text.index("FACTS COMPUTED") > text.index("ACCOUNT A, 1-10 September")
    assert text.index("FACTS COMPUTED") < text.index("Answer ONLY with JSON")
    assert "UPSTREAM" not in text
    assert reading_test["shown_level"].isna().all()
