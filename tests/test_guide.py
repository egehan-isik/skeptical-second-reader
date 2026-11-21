"""Tests for the choice of the solved examples."""

import pandas as pd
import pytest

from skeptical_second_reader import guide as g
from skeptical_second_reader.facts import FACTS


def frame():
    base = {name: 0.0 for name in FACTS}
    rows = {
        # strong sign: positive, one conversion-linked payment, one payee
        "s1": {**base, "sends_with_conversion": 1, "payees": 1},
        "s2": {**base, "sends_with_conversion": 1, "payees": 1},
        "s_many_payees": {**base, "sends_with_conversion": 1, "payees": 2},
        # weaker sign: positive, both ways and 9+ payers
        "w1": {**base, "both_ways": 1, "payers": 9},
        "w_few_payers": {**base, "both_ways": 1, "payers": 8},
        # busy clean
        "b1": {**base, "most_sends_same_amount": 5, "share_sent_on_within_day": 0.8},
        "b_quiet": {
            **base,
            "most_sends_same_amount": 5,
            "share_sent_on_within_day": 0.8,
        },
        # many conversions, clean
        "m1": {**base, "sends_with_conversion": 3},
        "m_two": {**base, "sends_with_conversion": 2},
    }
    facts = pd.DataFrame.from_dict(rows, orient="index")[FACTS]
    labels = pd.Series({name: int(name[0] in "sw") for name in rows}, name="label")
    activity = pd.Series({name: 10 for name in rows})
    activity["b_quiet"] = 9
    return facts, labels, activity


def test_each_rule_takes_only_accounts_that_satisfy_it():
    facts, labels, activity = frame()
    masks = g.shot_candidates(facts, labels, activity)
    assert set(facts.index[masks["strong sign"]]) == {"s1", "s2"}
    assert set(facts.index[masks["weaker sign"]]) == {"w1"}
    assert set(facts.index[masks["busy clean"]]) == {"b1"}
    assert set(facts.index[masks["many conversions"]]) == {"m1"}


def test_examples_follow_the_rules_in_order_and_never_come_from_exclude():
    facts, labels, activity = frame()
    shots = g.choose_shots(facts, labels, activity, pd.Index(["s1"]), seed=0)
    assert shots.tolist() == [name for name, _, _ in g.SHOT_RULES]
    assert shots.index.tolist() == ["s2", "w1", "b1", "m1"]


def test_an_excluded_account_is_never_an_example_whatever_the_seed():
    facts, labels, activity = frame()
    for seed in range(20):
        shots = g.choose_shots(facts, labels, activity, pd.Index(["s1"]), seed=seed)
        assert "s1" not in shots.index


def test_a_rule_without_accounts_is_an_error():
    facts, labels, activity = frame()
    with pytest.raises(ValueError, match="weaker sign"):
        g.choose_shots(facts, labels, activity, pd.Index(["w1"]), seed=0)


def test_the_answers_match_the_rules():
    involved = {name: answer for name, answer, _ in g.SHOT_RULES}
    assert involved == {
        "strong sign": True,
        "weaker sign": True,
        "busy clean": False,
        "many conversions": False,
    }


def test_the_examples_block_shows_facts_answer_and_why():
    facts, labels, activity = frame()
    shots = g.choose_shots(facts, labels, activity, pd.Index([]), seed=0)
    text = g.shots_text(facts, shots, activity)
    assert text.startswith("SOLVED EXAMPLES")
    assert text.count("10 transactions; paid by") == 4
    assert "1 payments in the same minute as a conversion" in text
    assert "Example 1: involved in money laundering" in text
    assert "Example 3: not involved" in text
    assert text.index("Example 2") < text.index("a counterparty on both sides")


def test_the_summary_line_gives_the_facts_of_the_guide():
    facts, _, _ = frame()
    row = facts.loc["s1"].copy()
    row["payers"], row["share_sent_on_within_day"] = 3, 0.25
    line = g.shot_summary(row, 7)
    assert line.startswith("7 transactions; paid by 3 accounts, pays 1;")
    assert "0 counterparties on both sides" in line
    assert "at most 0 payments of the same amount" in line
    assert "25% of received payments sent on within 24 hours." in line
    row["share_sent_on_within_day"] = float("nan")
    assert g.shot_summary(row, 7).endswith("receives nothing.")


def test_guided_inputs_keep_the_samples_out_of_every_reference():
    facts, labels, activity = frame()
    # a sample account with an extreme value would move every reference value
    facts.loc["x"] = facts.loc["b1"]
    facts.loc["x", "payers"] = 1000.0
    labels["x"], activity["x"] = 0, 10
    # a clean reference account and a positive one, apart from the examples' rules
    facts.loc["c"] = facts.loc["m_two"]
    labels["c"], activity["c"] = 0, 10
    inputs = g.guided_inputs(
        facts, labels, activity, pd.Index(["x"]), pd.Index(["s1"]), seed=0
    )
    lower, same = inputs["percentiles"]
    assert lower.loc["x", "payers"] == 1.0
    low, high = inputs["typical"]
    clean_ref = facts.drop(index="x")[labels.drop(index="x") == 0]
    assert high.loc["x", "payers"] == clean_ref["payers"].quantile(0.9)
    assert high.loc["x", "sends_with_conversion"] == pytest.approx(
        clean_ref["sends_with_conversion"].quantile(0.9)
    )
    assert inputs["guide"] == g.GUIDE
    assert "SOLVED EXAMPLES" in inputs["shots"]


def test_guided_inputs_never_take_an_example_from_the_earlier_experiments():
    facts, labels, activity = frame()
    activity["s1"] = 11  # only s1 would show "11 transactions"
    for seed in range(20):
        inputs = g.guided_inputs(
            facts, labels, activity, pd.Index([]), pd.Index(["s1"]), seed=seed
        )
        assert "11 transactions" not in inputs["shots"]
