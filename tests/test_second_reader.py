"""Tests for what the second reader is shown, on small hand-made data."""

import json

import numpy as np
import pandas as pd
import pytest

from skeptical_second_reader import second_reader as sr
from skeptical_second_reader.transactions import TX_FEATURES


def transactions():
    def row(sender, receiver, minute, usd, fmt="ACH", paid="USD", received="USD"):
        return {
            "sender": sender,
            "receiver": receiver,
            "timestamp": pd.Timestamp("2022-09-02 09:00")
            + pd.Timedelta(minutes=minute),
            "payment_format": fmt,
            "paid_currency": paid,
            "received_currency": received,
            "usd": usd,
            "usd_received": usd,
            "same_bank": False,
        }

    return pd.DataFrame(
        [
            row("1:A", "2:B", 0, 100),
            row("3:C", "1:A", 1, 50, fmt="Wire"),
            row("1:A", "1:A", 2, 70, paid="Euro"),
            row("4:D", "5:E", 3, 999),
            row("1:A", "2:B", 4, 200, fmt="Cheque"),
        ]
    )


@pytest.fixture
def rows():
    return sr.account_rows(transactions(), np.array([2, 1, 1, 1, 2]), "1:A")


def test_account_rows_are_those_of_the_account(rows):
    assert rows["row"].tolist() == [0, 1, 2, 4]
    assert rows["pair_txs"].tolist() == [2, 1, 1, 2]


def test_evidence_masks_identities_and_summarises(rows):
    text, numbers = sr.account_evidence(rows, "1:A")
    for hidden in ["1:A", "2:B", "3:C", "4:D"]:
        assert hidden not in text
    assert numbers == {0: 1, 1: 2, 2: 3, 4: 4}
    assert "4 transactions: 2 sent, 1 received, 1 to itself" in text
    assert "1 currency conversions" in text
    assert "2 counterparties" in text
    assert "Sent 300.00 USD, received 50.00 USD" in text
    assert "ACH 2, Cheque 1, Wire 1" in text
    lines = text.splitlines()
    assert lines[2] == "1,Sep-02 09:00,sent,K1,ACH,100.00,USD,USD,2"
    assert lines[3] == "2,Sep-02 09:01,received,K2,Wire,50.00,USD,USD,1"
    assert lines[4] == "3,Sep-02 09:02,self,A,ACH,70.00,Euro,USD,1"
    assert lines[5] == "4,Sep-02 09:04,sent,K1,Cheque,200.00,USD,USD,2"


def test_a_busy_account_shows_its_largest_transactions_in_time_order():
    base = transactions().iloc[[0]]
    many = pd.concat([base] * 40, ignore_index=True)
    many["usd"] = np.arange(40.0)
    many["timestamp"] = many["timestamp"] + pd.to_timedelta(39 - np.arange(40), "min")
    rows = sr.account_rows(many, np.ones(40), "1:A")
    text, numbers = sr.account_evidence(rows, "1:A")
    assert len(numbers) == sr.MAX_ROWS
    assert set(numbers) == set(range(10, 40))
    assert "Showing the 30 largest of 40 transactions." in text
    # The largest is the earliest here, so it comes first.
    assert numbers[39] == 1
    assert text.splitlines()[2].split(",")[5] == "39.00"


def test_the_evidence_does_not_mention_the_model(rows):
    text, _ = sr.account_evidence(rows, "1:A")
    prompt = sr.build_prompt(text, None)
    for word in ["model", "score", "alert", "strongest", "UPSTREAM"]:
        assert word not in prompt


def test_true_levels():
    assert sr.true_level(False, 0.95) == sr.NOT_FLAGGED
    assert sr.true_level(True, 0.89) == sr.UNCERTAIN
    assert sr.true_level(True, 0.9) == sr.STRONG


def test_k1_hides_the_model_and_k2_tells_the_truth():
    assert sr.shown_output(True, 0.95, 0.9, 2, sr.K1) is None
    shown = sr.shown_output(True, 0.95, 0.9, 2, sr.K2)
    assert shown == sr.ShownOutput(sr.STRONG, 0.9, 2, False)
    assert shown.alert
    assert not sr.shown_output(False, 0.2, 0.05, 0, sr.K2).alert


@pytest.mark.parametrize("condition", [sr.K3_D100, sr.SCORE_ONLY])
@pytest.mark.parametrize(
    ("flagged", "score", "rate", "n_high"),
    [(True, 0.99, 0.98, 3), (True, 0.4, 0.38, 1), (False, 0.01, 0.002, 0)],
)
def test_full_dose_always_reverses_the_decision(
    condition, flagged, score, rate, n_high
):
    shown = sr.shown_output(flagged, score, rate, n_high, condition)
    assert shown.alert != flagged
    assert shown.flipped
    assert shown.rate == pytest.approx(sr.LOW_TARGET if flagged else sr.HIGH_TARGET)
    assert shown.n_high == (0 if flagged else 1)


def test_doses_move_the_rate_and_the_level_follows():
    moved = [
        sr.shown_output(False, 0.01, 0.0, 0, condition)
        for condition in [sr.K3_D33, sr.K3_D66, sr.K3_D100]
    ]
    assert [m.rate for m in moved] == pytest.approx([0.3267, 0.6534, 0.99])
    assert [m.level for m in moved] == [sr.NOT_FLAGGED, sr.UNCERTAIN, sr.STRONG]
    assert [m.flipped for m in moved] == [False, True, True]
    assert moved[0].n_high == 0


def test_a_partial_dose_on_a_strong_alert():
    moved = [
        sr.shown_output(True, 0.99, 1.0, 2, condition)
        for condition in [sr.K3_D33, sr.K3_D66, sr.K3_D100]
    ]
    assert [m.level for m in moved] == [sr.UNCERTAIN, sr.NOT_FLAGGED, sr.NOT_FLAGGED]
    assert [m.flipped for m in moved] == [False, True, True]
    assert [m.n_high for m in moved] == [2, 0, 0]


def test_every_feature_can_be_described_also_when_missing():
    for name in TX_FEATURES:
        assert sr._describe(name, 1.0, "ACH", "Euro")
    for name in [
        "hours_since_money_in",
        "ratio_to_money_in",
        "hours_to_next_send",
        "next_send_ratio",
    ]:
        assert "nan" not in sr._describe(name, float("nan"), "ACH", "Euro")
    with pytest.raises(ValueError, match="no description"):
        sr._describe("unknown", 1.0, "ACH", "Euro")


def test_descriptions_undo_the_logs():
    assert "99 USD" in sr._describe("log_usd", np.log1p(99), "ACH", "Euro")
    text = sr._describe("log_usd_over_sender_median", np.log(2.0), "ACH", "Euro")
    assert "2.00 times" in text


def test_reasons_are_the_largest_contributions_with_their_direction():
    contributions = pd.Series(
        {"pair_count": 3.0, "hour": 0.1, "payment_format": -2.0, "log_usd": 0.5}
    )
    values = pd.Series(
        {"pair_count": 1.0, "hour": 9.0, "payment_format": 0.0, "log_usd": np.log1p(99)}
    )
    lines = sr.reason_lines(contributions, values, "ACH", "Euro", flipped=False)
    assert len(lines) == sr.TOP_REASONS
    assert (
        lines[0] == "- this sender paid this receiver 1 times in the window: "
        "pushes towards suspicious"
    )
    assert lines[1] == "- the payment format is ACH: pushes towards normal"
    assert "99 USD" in lines[2]
    reversed_lines = sr.reason_lines(contributions, values, "ACH", "Euro", flipped=True)
    assert reversed_lines[0].endswith("pushes towards normal")
    assert reversed_lines[1].endswith("pushes towards suspicious")


def test_ml_block_with_and_without_reasons():
    shown = sr.ShownOutput(sr.UNCERTAIN, 0.384, 1, False)
    with_reasons = sr.ml_block(shown, 2, "", ["- a: pushes towards normal"])
    assert "Level: UNCERTAIN ALERT." in with_reasons
    assert "about 38%" in with_reasons
    assert "strongest transaction is number 2" in with_reasons
    assert "- a: pushes towards normal" in with_reasons
    assert "alert threshold: 1" in with_reasons
    only_score = sr.ml_block(shown, 2, "", None)
    assert "strongest" not in only_score and "threshold" not in only_score
    assert "under 1%" in sr.ml_block(
        sr.ShownOutput(sr.NOT_FLAGGED, 0.002, 0, False), 1, "", None
    )
    outside = sr.ml_block(shown, None, "a Wire of 90 USD", ["- a"])
    assert "not in the list above: a Wire of 90 USD" in outside


def test_the_instruction_is_the_same_with_and_without_the_model():
    without = sr.build_prompt("EVIDENCE", None)
    with_model = sr.build_prompt("EVIDENCE", "UPSTREAM")
    assert with_model.replace("UPSTREAM" + chr(10) + chr(10), "") == without
    assert with_model.index("EVIDENCE") < with_model.index("UPSTREAM")
    assert with_model.index("UPSTREAM") < with_model.index("Answer ONLY")


@pytest.mark.parametrize(
    ("content", "decision", "confidence"),
    [
        (
            '{"decision": "suspicious", "confidence": 0.8, '
            '"evidence": [1], "reason": "r"}',
            "suspicious",
            0.8,
        ),
        ('{"decision": "not_suspicious", "confidence": 1}', "not_suspicious", 1),
        ('{"decision": "maybe", "confidence": 2}', None, None),
        ('{"decision": "suspicious", "confidence": true}', "suspicious", None),
        ("[1, 2]", None, None),
        ("not json", None, None),
    ],
)
def test_parse_answer(content, decision, confidence):
    answer = sr.parse_answer(content)
    assert answer["decision"] == decision
    assert answer["confidence"] == confidence
    assert isinstance(answer["evidence"], list)


def test_following_the_model_is_computed_from_what_was_shown():
    alert = sr.ShownOutput(sr.UNCERTAIN, 0.6, 0, True)
    quiet = sr.ShownOutput(sr.NOT_FLAGGED, 0.1, 0, False)
    assert sr.follows_model(sr.SUSPICIOUS, alert) is True
    assert sr.follows_model(sr.NOT_SUSPICIOUS, alert) is False
    assert sr.follows_model(sr.NOT_SUSPICIOUS, quiet) is True
    assert sr.follows_model(sr.SUSPICIOUS, quiet) is False
    assert sr.follows_model(sr.SUSPICIOUS, None) is None
    assert sr.follows_model(None, alert) is None


def test_groups_cover_every_kind_of_account():
    index = [f"a{i}" for i in range(7)]
    flags = pd.Series([1, 1, 1, 1, 0, 0, 0], index=index)
    scores = pd.Series([0.95, 0.95, 0.5, 0.5, 0.1, 0.1, 0.1], index=index)
    labels = pd.Series([1, 0, 1, 0, 1, 1, 0], index=index)
    in_file = pd.Series([1, 0, 1, 0, 1, 0, 0], index=index).astype(bool)
    groups = sr.case_groups(flags, scores, labels, in_file)
    assert groups.tolist() == sr.GROUP_NAMES


def test_sampling_takes_each_group_and_can_exclude():
    groups = pd.Series(
        sr.GROUP_NAMES * 3,
        index=[f"a{i}" for i in range(3 * len(sr.GROUP_NAMES))],
    )
    first = sr.sample_cases(groups, 2, seed=0)
    assert (first.value_counts() == 2).all() and len(first) == 14
    pd.testing.assert_series_equal(first, sr.sample_cases(groups, 2, seed=0))
    second = sr.sample_cases(groups, 1, seed=0, exclude=first.index)
    assert not set(second.index) & set(first.index)
    assert sr.sample_cases(groups, 9, seed=0).value_counts().eq(3).all()


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        reply = {
            "message": {"content": "{}"},
            "prompt_eval_count": 7,
            "eval_count": 3,
        }
        return json.dumps(reply).encode()


def test_the_call_is_deterministic_and_asks_for_json(monkeypatch):
    sent = {}

    def fake_open(request, timeout):
        sent.update(json.loads(request.data.decode()))
        return FakeResponse()

    monkeypatch.setattr(sr.urllib.request, "urlopen", fake_open)
    reply = sr.ask_ollama("hello", "m", think=False)
    assert sent["options"]["temperature"] == 0
    assert sent["options"]["num_ctx"] == sr.NUM_CTX
    assert sent["format"] == "json"
    assert sent["think"] is False
    assert sent["messages"][0]["content"] == "hello"
    assert reply["prompt_tokens"] == 7 and reply["content"] == "{}"
    sent.clear()
    sr.ask_ollama("hello", "m")
    assert "think" not in sent


@pytest.fixture
def case(rows):
    reading = pd.Series(
        {
            "flag": 1,
            "score": 0.95,
            "calibrated": 0.9,
            "n_high": 2,
            "strongest_row": 4,
        }
    )
    contributions = pd.Series(0.0, index=TX_FEATURES)
    contributions["pair_count"] = 2.0
    values = pd.Series(1.0, index=TX_FEATURES)
    return rows, reading, contributions, values


def test_a_case_without_the_model_says_nothing_of_it(case):
    rows, reading, contributions, values = case
    prompt, shown = sr.case_prompt(rows, "1:A", reading, contributions, values, sr.K1)
    assert shown is None
    assert "UPSTREAM" not in prompt and "strongest" not in prompt


def test_a_case_with_the_true_output(case):
    rows, reading, contributions, values = case
    prompt, shown = sr.case_prompt(rows, "1:A", reading, contributions, values, sr.K2)
    assert shown.level == sr.STRONG
    assert "Level: STRONG ALERT." in prompt
    assert "about 90%" in prompt
    assert "strongest transaction is number 4" in prompt
    assert (
        "paid this receiver 1 times in the window: pushes towards suspicious" in prompt
    )


def test_the_evidence_is_the_same_in_every_condition(case):
    rows, reading, contributions, values = case
    evidence, _ = sr.account_evidence(rows, "1:A")
    for condition in sr.CONDITIONS:
        prompt, _ = sr.case_prompt(
            rows, "1:A", reading, contributions, values, condition
        )
        assert evidence in prompt


def test_score_only_gives_no_reasons_at_full_dose(case):
    rows, reading, contributions, values = case
    prompt, shown = sr.case_prompt(
        rows, "1:A", reading, contributions, values, sr.SCORE_ONLY
    )
    assert shown.flipped and shown.level == sr.NOT_FLAGGED
    assert "strongest" not in prompt and "pushes towards" not in prompt


def test_full_dose_reverses_the_stated_reasons(case):
    rows, reading, contributions, values = case
    prompt, shown = sr.case_prompt(
        rows, "1:A", reading, contributions, values, sr.K3_D100
    )
    assert shown.flipped
    assert "paid this receiver 1 times in the window: pushes towards normal" in prompt


def test_a_strongest_transaction_outside_the_list_is_described(rows):
    base = rows.iloc[[0]]
    many = pd.concat([base] * 40, ignore_index=True)
    many["usd"] = np.arange(40.0)
    many["row"] = np.arange(40)
    reading = pd.Series(
        {"flag": 1, "score": 0.5, "calibrated": 0.3, "n_high": 1, "strongest_row": 0}
    )
    contributions = pd.Series(0.0, index=TX_FEATURES)
    values = pd.Series(1.0, index=TX_FEATURES)
    prompt, _ = sr.case_prompt(many, "1:A", reading, contributions, values, sr.K2)
    assert "not in the list above: a ACH of 0.00 USD" in prompt


def test_a_certain_rate_is_not_shown_as_certainty():
    shown = sr.ShownOutput(sr.STRONG, 1.0, 1, False)
    assert "over 99%" in sr.ml_block(shown, 1, "", None)
    assert "100" not in sr.ml_block(shown, 1, "", None)


def test_a_conversion_is_described_as_the_senders():
    assert "the sender converted" in sr._describe("after_conversion", 1.0, "ACH", "E")
    assert "the sender did not" in sr._describe("after_conversion", 0.0, "ACH", "E")
