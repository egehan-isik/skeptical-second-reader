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
