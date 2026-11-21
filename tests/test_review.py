"""Tests for the review flow: the frozen model, the two steps and what they may see."""

import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from skeptical_second_reader import review as rv
from skeptical_second_reader import second_reader as sr
from skeptical_second_reader.pipeline import AnswerCache
from skeptical_second_reader.transactions import TX_FEATURES

ACCOUNTS = ["001:A", "001:B", "002:C", "002:D"]


def window(n=40, seed=0):
    """A small window: transactions between four accounts, and random features."""
    rng = np.random.default_rng(seed)
    sender = rng.choice(ACCOUNTS, n)
    receiver = rng.choice(ACCOUNTS, n)
    table = pd.DataFrame(
        {
            "sender": sender,
            "receiver": receiver,
            "timestamp": pd.Timestamp("2022-09-01")
            + pd.to_timedelta(np.sort(rng.integers(0, 9 * 24 * 60, n)), unit="min"),
            "payment_format": rng.choice(["ACH", "Cheque", "Wire"], n),
            "paid_currency": "US Dollar",
            "received_currency": "US Dollar",
            "usd": rng.uniform(10, 5000, n).round(2),
            "same_bank": [
                s[:3] == r[:3] for s, r in zip(sender, receiver, strict=True)
            ],
        }
    )
    table["usd_received"] = table["usd"]
    features = pd.DataFrame(
        rng.normal(size=(n, len(TX_FEATURES))), columns=TX_FEATURES
    ).astype("float32")
    features["pair_count"] = (
        table.groupby(["sender", "receiver"])["usd"].transform("size").astype("float32")
    )
    return table, features


def small_model(features, seed=0):
    rng = np.random.default_rng(seed)
    y = (features["log_usd"] + rng.normal(scale=0.5, size=len(features)) > 0).astype(
        int
    )
    model = xgb.XGBClassifier(
        n_estimators=30, max_depth=2, early_stopping_rounds=5, eval_metric="aucpr"
    )
    model.fit(features, y, eval_set=[(features, y)], verbose=False)
    return model


def frozen_for(features, reviewable=ACCOUNTS[:2]):
    return rv.FrozenModel(
        small_model(features),
        0.5,
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0]),
        0,
        pd.Index(reviewable, name="account"),
    )


class FakeAsk:
    """Answers suspicious first and not suspicious second; counts the calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, messages, model, think=None):
        self.calls.append(messages)
        decision = sr.SUSPICIOUS if len(messages) == 1 else sr.NOT_SUSPICIOUS
        content = json.dumps(
            {"decision": decision, "confidence": 0.7, "evidence": [1], "reason": "r"}
        )
        return {
            "content": content,
            "prompt_tokens": 10,
            "answer_tokens": 5,
            "seconds": 0.1,
        }


def test_calibrate_matches_the_isotonic_curve():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=200)
    y = (rng.uniform(size=200) < x).astype(int)
    curve = IsotonicRegression(out_of_bounds="clip").fit(x, y)
    frozen = rv.FrozenModel(
        None, 0.5, curve.X_thresholds_, curve.y_thresholds_, 0, pd.Index([])
    )
    grid = np.linspace(-0.5, 1.5, 101)
    assert np.allclose(frozen.calibrate(grid), curve.predict(grid))


def test_a_saved_model_scores_as_before(tmp_path):
    _, features = window()
    frozen = frozen_for(features)
    rv.save_frozen(frozen, tmp_path)
    loaded = rv.load_frozen(tmp_path)
    assert np.allclose(
        loaded.model.predict_proba(features)[:, 1],
        frozen.model.predict_proba(features)[:, 1],
    )
    assert loaded.model.best_iteration == frozen.model.best_iteration
    assert loaded.threshold == frozen.threshold
    assert list(loaded.reviewable) == list(frozen.reviewable)


def test_accounts_seen_in_training_are_refused(tmp_path):
    table, features = window()
    with pytest.raises(ValueError, match="outside the test fold"):
        rv.review(
            [ACCOUNTS[3]],
            table,
            features,
            frozen_for(features),
            {"m": "d"},
            AnswerCache(tmp_path / "a.sqlite"),
            ask=FakeAsk(),
        )


def test_score_window_follows_the_noisy_or_of_the_transactions():
    table, features = window()
    frozen = frozen_for(features)
    scored = rv.score_window(frozen, table, features)
    tx = frozen.model.predict_proba(features)[:, 1]
    for account in ACCOUNTS:
        mine = (table["sender"] == account) | (table["receiver"] == account)
        expected = 1 - np.prod(1 - np.clip(tx[mine.to_numpy()], 0, 1 - 1e-7))
        assert scored.loc[account, "score"] == pytest.approx(expected, abs=1e-6)
    assert (scored["flag"] == (scored["score"] >= 0.5)).all()


def test_the_first_step_is_the_k1_prompt_and_the_second_adds_the_ml_output():
    first = rv.first_messages("EVIDENCE")
    assert first == [{"role": "user", "content": sr.build_prompt("EVIDENCE", None)}]
    second = rv.second_messages(first, '{"decision": "suspicious"}', "ML BLOCK")
    assert second[:1] == first
    assert second[1] == {"role": "assistant", "content": '{"decision": "suspicious"}'}
    assert "ML BLOCK" in second[2]["content"]
    assert "UPSTREAM ML MODEL OUTPUT" not in first[0]["content"]


def test_review_asks_twice_per_model_and_reports_the_change(tmp_path):
    table, features = window()
    ask = FakeAsk()
    cache = AnswerCache(tmp_path / "a.sqlite")
    result = rv.review(
        ACCOUNTS[:2],
        table,
        features,
        frozen_for(features),
        {"m1": "d1", "m2": "d2"},
        cache,
        ask=ask,
    )
    assert len(ask.calls) == 2 * 2 * 2
    assert all("UPSTREAM ML MODEL OUTPUT" not in c[0]["content"] for c in ask.calls)
    assert all("UPSTREAM ML MODEL OUTPUT" in c[-1]["content"] for c in ask.calls[1::2])
    reader = result[0]["readers"]["m1"]
    assert reader["first"]["decision"] == sr.SUSPICIOUS
    assert reader["final"]["decision"] == sr.NOT_SUSPICIOUS
    assert reader["changed"] is True
    assert len(result[0]["trace"]) == 4


def test_a_second_review_comes_from_the_cache(tmp_path):
    table, features = window()
    cache = AnswerCache(tmp_path / "a.sqlite")
    frozen = frozen_for(features)
    rv.review(ACCOUNTS[:1], table, features, frozen, {"m": "d"}, cache, ask=FakeAsk())
    again = FakeAsk()
    result = rv.review(
        ACCOUNTS[:1], table, features, frozen, {"m": "d"}, cache, ask=again
    )
    assert again.calls == []
    assert all(step["cached"] for step in result[0]["trace"])


def test_the_result_carries_no_label(tmp_path):
    table, features = window()
    result = rv.review(
        ACCOUNTS[:1],
        table,
        features,
        frozen_for(features),
        {"m": "d"},
        AnswerCache(tmp_path / "a.sqlite"),
        ask=FakeAsk(),
    )
    text = json.dumps(result)
    assert "label" not in text
    assert "is_laundering" not in text
