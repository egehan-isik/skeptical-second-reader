"""Tests for the account facts, on a small hand-made table."""

import numpy as np
import pandas as pd
import pytest

from skeptical_second_reader import facts as fx
from skeptical_second_reader.transactions import account_positions


def table():
    def row(sender, receiver, when, usd, paid="USD", received="USD"):
        return {
            "sender": sender,
            "receiver": receiver,
            "timestamp": pd.Timestamp(when),
            "paid_currency": paid,
            "received_currency": received,
            "usd": usd,
        }

    return pd.DataFrame(
        [
            row("B", "A", "2022-09-01 10:00", 100),  # A receives
            row("C", "A", "2022-09-01 11:00", 100),  # A receives
            row("A", "A", "2022-09-01 12:00", 90, "USD", "Euro"),  # conversion
            row("A", "D", "2022-09-01 12:00", 90),  # send, same minute
            row("A", "D", "2022-09-03 12:00", 90),  # send, same amount
            row("A", "B", "2022-09-03 13:00", 30),  # B both ways
            row("B", "E", "2022-09-02 09:00", 5),  # B pays E as well
            row("F", "D", "2022-09-02 09:00", 5),  # D has two payers
        ]
    )


@pytest.fixture
def facts():
    t = table()
    accounts = pd.Index(list("ABCDEF"))
    sender, receiver = account_positions(t, accounts)
    return fx.account_facts(t, sender, receiver, accounts)


def test_counterparty_counts(facts):
    assert facts.loc["A", "payers"] == 2
    assert facts.loc["A", "payees"] == 2
    assert facts.loc["A", "both_ways"] == 1
    assert facts.loc["B", "both_ways"] == 1
    assert facts.loc["C", "both_ways"] == 0


def test_neighbours_degrees(facts):
    # payers of A: B pays A, E (2); C pays A (1) -> median 1.5
    assert facts.loc["A", "payers_median_payees"] == 1.5
    # payees of A: D paid by A, F (2); B paid by A (1) -> median 1.5
    assert facts.loc["A", "payees_median_payers"] == 1.5
    assert np.isnan(facts.loc["C", "payers_median_payees"])


def test_sending_on_after_receiving(facts):
    # receipts 10:00 and 11:00, next send 12:00 -> 2 h and 1 h, both within a day
    assert facts.loc["A", "share_sent_on_within_day"] == 1.0
    assert facts.loc["A", "median_hours_to_send_on"] == 1.5
    # B receives at 09-03 13:00 and never sends after it
    assert facts.loc["B", "share_sent_on_within_day"] == 0.0
    assert np.isnan(facts.loc["B", "median_hours_to_send_on"])
    assert np.isnan(facts.loc["C", "share_sent_on_within_day"])


def test_conversion_amounts_and_days(facts):
    assert facts.loc["A", "sends_with_conversion"] == 1
    assert facts.loc["A", "most_sends_same_amount"] == 2
    assert facts.loc["A", "active_days"] == 2
    assert facts.loc["A", "busiest_day"] == 4
    assert facts.loc["E", "most_sends_same_amount"] == 0


def test_the_block_says_each_fact(facts):
    text = fx.fact_lines(facts.loc["A"])
    assert "paid by 2 different accounts and pays 2 different accounts" in text
    assert "1 counterparty both pays A and is paid by A." in text
    plural = fx.fact_lines(facts.loc["C"])
    assert "0 counterparties both pay A and are paid by A." in plural
    assert "After 100% of the payments" in text and "1.5 hours" in text
    assert "1 payment of A" in text
    assert "exactly the same amount: 2" in text
    assert "active on 2 days, with at most 4 transactions" in text


def test_the_block_handles_accounts_without_receipts_or_sends(facts):
    assert "receives no payments" in fx.fact_lines(facts.loc["C"])
    assert "sends nothing to other accounts after" in fx.fact_lines(facts.loc["B"])
    assert "pay a median" not in fx.fact_lines(facts.loc["C"])


def test_percentiles_count_lower_and_equal_reference_values():
    facts = pd.DataFrame({"x": [2.0, np.nan, 10.0]}, index=list("abc"))
    reference = pd.DataFrame({"x": [1.0, 2.0, 2.0, 3.0, np.nan]})
    lower, same = fx.fact_percentiles(facts, reference)
    assert lower.loc["a", "x"] == 0.25
    assert same.loc["a", "x"] == 0.5
    assert lower.loc["c", "x"] == 1.0 and same.loc["c", "x"] == 0.0
    assert np.isnan(lower.loc["b", "x"]) and np.isnan(same.loc["b", "x"])


def test_percentile_lines_say_value_and_place_and_skip_missing(facts):
    reference = facts
    lower, same = fx.fact_percentiles(facts, reference)
    text = fx.percentile_lines(facts.loc["A"], lower.loc["A"], same.loc["A"])
    assert text.startswith("HOW THESE FACTS COMPARE")
    # A has 2 payers; 4 of the 6 accounts have fewer (B and E 1, C and F 0).
    assert "- different accounts that pay A: 2; higher than" in text
    share = lower.loc["A", "payers"]
    assert f"higher than {share * 100:.0f}% of accounts" in text
    assert "share of received payments followed by a send within 24 hours: 100%" in (
        text
    )
    c = fx.percentile_lines(facts.loc["C"], lower.loc["C"], same.loc["C"])
    assert "followed by a send" not in c
    assert len(c.splitlines()) == 1 + facts.loc["C"].notna().sum()


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.0, "0%"),
        (0.004, "less than 1%"),
        (0.5, "50%"),
        (0.997, "99.7%"),
        (1.0, "100%"),
    ],
)
def test_shares_are_written_without_hiding_small_or_large_ones(value, text):
    assert fx._share(value) == text


def test_activity_bins_group_by_powers_of_two():
    activity = pd.Series([1, 2, 3, 4, 5, 8, 9, 64, 65, 20_000])
    assert fx.activity_bins(activity).tolist() == [0, 1, 2, 2, 3, 3, 4, 6, 7, 7]


def test_typical_ranges_use_only_the_reference_of_the_same_activity():
    reference = pd.DataFrame({"x": [0.0, 10.0, 20.0, 1000.0, np.nan]})
    reference_activity = pd.Series([3, 3, 4, 50, 3])
    facts = pd.DataFrame({"x": [5.0, 7.0, 1.0]}, index=list("abc"))
    activity = pd.Series([4, 40, 1], index=list("abc"))
    low, high = fx.typical_ranges(facts, activity, reference, reference_activity)
    # a: bin 3-4 holds 0, 10, 20 (the NaN is left out) -> 10th 2, 90th 18
    assert low.loc["a", "x"] == pytest.approx(2.0)
    assert high.loc["a", "x"] == pytest.approx(18.0)
    # b: bin 33-64 holds only 1000
    assert low.loc["b", "x"] == 1000.0 and high.loc["b", "x"] == 1000.0
    # c: no reference account with one transaction
    assert np.isnan(low.loc["c", "x"]) and np.isnan(high.loc["c", "x"])
    assert low.index.equals(facts.index)


def test_percentile_lines_add_the_typical_range_when_given(facts):
    lower, same = fx.fact_percentiles(facts, facts)
    low = facts.loc["A"] * 0
    high = facts.loc["A"] * 0 + 3
    low["both_ways"] = np.nan
    text = fx.percentile_lines(facts.loc["A"], lower.loc["A"], same.loc["A"], low, high)
    assert "typical for clean accounts as busy as A: 0 to 3." in text
    line = next(
        x for x in text.splitlines() if x.startswith("- counterparties that both")
    )
    assert "typical" not in line
    plain = fx.percentile_lines(facts.loc["A"], lower.loc["A"], same.loc["A"])
    assert "typical" not in plain
