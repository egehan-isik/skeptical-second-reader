"""Tests for the account features: exact counts, the window, and no use of the label."""

import pandas as pd
import pytest

from skeptical_second_reader.features import FEATURES, account_features


def make_tx(*rows):
    """Rows: timestamp, sender, receiver, format; optional currency pair and label."""
    records = []
    for row in rows:
        when, sender, receiver, payment_format, *rest = row
        received, paid = rest[0] if rest else ("Euro", "Euro")
        records.append(
            {
                "timestamp": pd.Timestamp(when),
                "from_bank": "001",
                "from_account": sender,
                "to_bank": "001",
                "to_account": receiver,
                "receiving_currency": received,
                "payment_currency": paid,
                "payment_format": payment_format,
                "is_laundering": 0,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def table():
    return account_features(
        make_tx(
            ("2022-09-01 10:00", "A", "B", "ACH"),
            ("2022-09-01 11:00", "A", "B", "ACH"),
            ("2022-09-02 10:00", "A", "C", "Wire"),
            ("2022-09-03 10:00", "C", "A", "Cash", ("Euro", "Yuan")),
            ("2022-09-04 10:00", "A", "A", "Credit Card"),
        )
    )


def row(table, account):
    return table.loc[f"001:{account}"]


def test_columns_are_exactly_the_feature_list(table):
    assert list(table.columns) == FEATURES
    assert len(FEATURES) == 15


def test_direction_and_counterparty_counts(table):
    a = row(table, "A")
    # Out: B, B, C, A. In: C, A.
    assert (a["out_tx"], a["in_tx"]) == (4, 2)
    assert (a["n_receivers"], a["n_senders"]) == (3, 2)
    b = row(table, "B")
    assert (b["out_tx"], b["in_tx"], b["n_receivers"], b["n_senders"]) == (0, 2, 0, 1)


def test_ach_is_counted_by_direction_and_other_formats_by_account(table):
    a, b, c = row(table, "A"), row(table, "B"), row(table, "C")
    assert (a["ach_out_tx"], a["ach_in_tx"]) == (2, 0)
    assert (b["ach_out_tx"], b["ach_in_tx"]) == (0, 2)
    assert (a["wire_tx"], a["cash_tx"], a["credit_card_tx"]) == (1, 1, 1)
    assert (c["wire_tx"], c["cash_tx"], c["credit_card_tx"]) == (1, 1, 0)


def test_self_transaction_counts_once_per_account(table):
    a = row(table, "A")
    assert a["self_tx"] == 1
    assert a["credit_card_tx"] == 1
    # 1, 2, 3 and 4 September.
    assert a["active_days"] == 4


def test_currency_change_counts_for_both_sides(table):
    assert row(table, "A")["currency_change_tx"] == 1
    assert row(table, "C")["currency_change_tx"] == 1
    assert row(table, "B")["currency_change_tx"] == 0


def test_transactions_after_the_window_are_ignored():
    inside = ("2022-09-10 23:59", "A", "B", "ACH")
    only_inside = account_features(make_tx(inside))
    with_tail = account_features(
        make_tx(
            inside,
            ("2022-09-11 00:00", "A", "B", "ACH"),
            ("2022-09-12 10:00", "A", "B", "ACH"),
        )
    )
    pd.testing.assert_frame_equal(only_inside, with_tail)


def test_accounts_active_only_after_the_window_get_no_row():
    table = account_features(
        make_tx(
            ("2022-09-01 10:00", "A", "B", "ACH"),
            ("2022-09-11 00:00", "A", "TAIL", "ACH"),
        )
    )
    assert "001:TAIL" not in table.index
    assert set(table.index) == {"001:A", "001:B"}


def test_features_do_not_depend_on_the_label():
    tx = make_tx(
        ("2022-09-01 10:00", "A", "B", "ACH"),
        ("2022-09-02 10:00", "B", "C", "Wire"),
    )
    flipped = tx.assign(is_laundering=1 - tx["is_laundering"])
    pd.testing.assert_frame_equal(account_features(tx), account_features(flipped))


def test_unknown_payment_format_is_refused():
    with pytest.raises(ValueError, match="unknown payment formats"):
        account_features(make_tx(("2022-09-01 10:00", "A", "B", "Gold")))
