"""Tests for the extended features, with values worked out by hand."""

from math import log1p

import numpy as np
import pandas as pd
import pytest

from skeptical_second_reader.features import (
    ALL_FEATURES,
    FEATURES,
    account_features,
    exchange_rates,
    extended_features,
)


def tx(
    when,
    sender,
    receiver,
    payment_format="Wire",
    paid=(100.0, "US Dollar"),
    received=None,
    label=0,
):
    """One transaction; accounts are written as bank:account."""
    received = paid if received is None else received
    from_bank, from_account = sender.split(":")
    to_bank, to_account = receiver.split(":")
    return {
        "timestamp": pd.Timestamp(when),
        "from_bank": from_bank,
        "from_account": from_account,
        "to_bank": to_bank,
        "to_account": to_account,
        "amount_received": received[0],
        "receiving_currency": received[1],
        "amount_paid": paid[0],
        "payment_currency": paid[1],
        "payment_format": payment_format,
        "is_laundering": label,
    }


def frame(*rows):
    return pd.DataFrame(list(rows))


A, B, C, D = "1:A", "1:B", "2:C", "3:D"
DAY = "2022-09-01"


def test_rate_is_the_median_of_received_over_paid():
    rates = exchange_rates(
        frame(
            tx(DAY, A, B, paid=(100.0, "Euro"), received=(120.0, "US Dollar")),
            tx(DAY, A, B, paid=(100.0, "Euro"), received=(140.0, "US Dollar")),
            tx(DAY, A, B, paid=(100.0, "Euro"), received=(190.0, "US Dollar")),
        )
    )
    assert rates["Euro"] == pytest.approx(1.4)
    assert rates["US Dollar"] == 1.0


def test_rate_falls_back_on_the_opposite_direction_only_if_needed():
    rates = exchange_rates(
        frame(
            tx(DAY, A, B, paid=(100.0, "US Dollar"), received=(500.0, "Yuan")),
            tx(DAY, A, B, paid=(100.0, "Euro"), received=(120.0, "US Dollar")),
            tx(DAY, A, B, paid=(100.0, "US Dollar"), received=(10.0, "Euro")),
        )
    )
    assert rates["Yuan"] == pytest.approx(0.2)
    assert rates["Euro"] == pytest.approx(1.2)


def test_rates_ignore_the_tail():
    rows = [tx(DAY, A, B, paid=(100.0, "Euro"), received=(120.0, "US Dollar"))]
    tail = tx("2022-09-11", A, B, paid=(100.0, "Euro"), received=(900.0, "US Dollar"))
    assert exchange_rates(frame(*rows, tail))["Euro"] == pytest.approx(1.2)


def test_rates_do_not_depend_on_the_label():
    rows = frame(
        tx(DAY, A, B, paid=(100.0, "Euro"), received=(120.0, "US Dollar"), label=1),
        tx(DAY, A, B, paid=(100.0, "Euro"), received=(160.0, "US Dollar")),
    )
    flipped = rows.assign(is_laundering=1 - rows["is_laundering"])
    pd.testing.assert_series_equal(exchange_rates(rows), exchange_rates(flipped))
    assert exchange_rates(rows)["Euro"] == pytest.approx(1.4)


def test_a_currency_without_a_rate_is_refused():
    with pytest.raises(ValueError, match="without a rate"):
        extended_features(frame(tx(DAY, A, B, paid=(100.0, "Euro"))))


def test_columns_and_accounts_follow_the_baseline():
    rows = frame(tx(DAY, A, B), tx(DAY, B, C), tx("2022-09-11", A, D))
    table = extended_features(rows)
    assert list(table.columns) == ALL_FEATURES
    assert len(ALL_FEATURES) == 33
    assert len(set(ALL_FEATURES)) == 33
    assert D not in table.index
    pd.testing.assert_frame_equal(table[FEATURES], account_features(rows))
    assert not np.isinf(table.drop(columns="median_gap_hours")).any().any()


def test_amounts_are_converted_and_leave_out_self_transactions():
    table = extended_features(
        frame(
            tx(DAY, A, B, paid=(100.0, "Euro"), received=(120.0, "US Dollar")),
            tx(DAY, A, C, paid=(300.0, "US Dollar")),
            tx(DAY, C, A, paid=(40.0, "US Dollar")),
            tx(DAY, A, A, paid=(9000.0, "US Dollar")),
        )
    )
    a = table.loc[A]
    # Sent: 100 Euro at 1.2 = 120, and 300. Received: 40.
    assert a["log_sent_usd"] == pytest.approx(log1p(420))
    assert a["log_received_usd"] == pytest.approx(log1p(40))
    assert a["log_median_sent_usd"] == pytest.approx(log1p(210))
    assert a["log_median_received_usd"] == pytest.approx(log1p(40))
    assert a["log_max_sent_usd"] == pytest.approx(log1p(300))
    assert a["log_max_received_usd"] == pytest.approx(log1p(40))
    assert a["flow_balance"] == pytest.approx((40 - 420) / (40 + 420 + 1))
    assert table.loc[B, "log_sent_usd"] == 0
    assert table.loc[B, "flow_balance"] == pytest.approx(120 / 121)


def test_the_median_is_not_the_mean():
    table = extended_features(
        frame(
            tx(DAY, A, B, paid=(10.0, "US Dollar")),
            tx(DAY, A, B, paid=(20.0, "US Dollar")),
            tx(DAY, A, B, paid=(900.0, "US Dollar")),
        )
    )
    assert table.loc[A, "log_median_sent_usd"] == pytest.approx(log1p(20))
    assert table.loc[B, "log_median_received_usd"] == pytest.approx(log1p(20))


def test_given_rates_replace_the_window_rates():
    rows = frame(tx(DAY, A, B, paid=(100.0, "Euro"), received=(120.0, "US Dollar")))
    rates = pd.Series({"Euro": 2.0, "US Dollar": 1.0})
    table = extended_features(rows, rates=rates)
    assert table.loc[A, "log_sent_usd"] == pytest.approx(log1p(200))


def test_timing_counts_each_side_once_and_uses_the_median_gap():
    table = extended_features(
        frame(
            tx("2022-09-01 10:00", A, B),
            tx("2022-09-01 10:30", A, A),
            tx("2022-09-01 11:00", C, A),
            tx("2022-09-02 11:00", A, C),
        )
    )
    a = table.loc[A]
    assert a["max_tx_per_day"] == 3
    assert a["max_tx_per_hour"] == 2
    # Gaps: 0.5, 0.5 and 24 hours.
    assert a["median_gap_hours"] == pytest.approx(0.5)
    assert np.isnan(table.loc[B, "median_gap_hours"])


@pytest.fixture
def graph():
    return extended_features(
        frame(
            tx(DAY, A, B, "ACH"),
            tx(DAY, A, B, "ACH"),
            tx(DAY, A, C, "Wire"),
            tx(DAY, C, A, "Cash"),
            tx(DAY, D, B, "ACH"),
            tx(DAY, A, A, "ACH"),
        )
    )


def test_reciprocal_counterparties(graph):
    assert graph.loc[A, "reciprocal_counterparties"] == 1
    assert graph.loc[C, "reciprocal_counterparties"] == 1
    assert graph.loc[B, "reciprocal_counterparties"] == 0


def test_counterparty_degrees(graph):
    # A sends to B (in-degree 2) and C (in-degree 1).
    assert graph.loc[A, "mean_receiver_in_degree"] == pytest.approx(1.5)
    # B receives from A (out-degree 2) and D (out-degree 1).
    assert graph.loc[B, "mean_sender_out_degree"] == pytest.approx(1.5)
    assert graph.loc[B, "mean_receiver_in_degree"] == 0


def test_given_degree_transactions_replace_the_window_graph():
    rows = frame(tx(DAY, A, B), tx(DAY, C, B), tx(DAY, D, A))
    limited = extended_features(rows, degree_transactions=frame(tx(DAY, A, B)))
    # B's in-degree is 1 in the limited graph, 2 in the full one.
    assert limited.loc[A, "mean_receiver_in_degree"] == 1
    # D sends nothing in the limited graph, so A's only sender has degree 0.
    assert limited.loc[A, "mean_sender_out_degree"] == 0
    assert extended_features(rows).loc[A, "mean_receiver_in_degree"] == 2


def test_ach_counterparties_are_distinct_and_leave_out_self(graph):
    assert graph.loc[A, "ach_receivers"] == 1
    assert graph.loc[B, "ach_senders"] == 2
    assert graph.loc[C, "ach_senders"] == 0


def test_counterparty_banks(graph):
    # A sends to bank 1 (B) and bank 2 (C); B receives from banks 1 (A) and 3 (D).
    assert graph.loc[A, "receiver_banks"] == 2
    assert graph.loc[B, "sender_banks"] == 2
    assert graph.loc[C, "sender_banks"] == 1


def test_two_counterparties_in_one_bank_count_as_one_bank():
    other_in_bank_1 = "1:E"
    table = extended_features(
        frame(
            tx(DAY, A, B),
            tx(DAY, A, other_in_bank_1),
            tx(DAY, B, C),
            tx(DAY, other_in_bank_1, C),
        )
    )
    assert table.loc[A, "receiver_banks"] == 1
    assert table.loc[C, "sender_banks"] == 1


def test_currencies_cover_both_sides_of_each_account():
    table = extended_features(
        frame(
            tx(DAY, A, A, paid=(100.0, "Euro"), received=(120.0, "US Dollar")),
            tx(DAY, A, B, paid=(120.0, "US Dollar")),
        )
    )
    assert table.loc[A, "n_currencies"] == 2
    assert table.loc[B, "n_currencies"] == 1


def test_the_tail_changes_nothing():
    rows = [
        tx(DAY, A, B, "ACH"),
        tx("2022-09-02 10:00", B, C, "Wire"),
        tx("2022-09-10 23:59", C, A, "Cash"),
    ]
    tail = [
        tx("2022-09-11 00:00", A, C, "ACH"),
        tx("2022-09-12 10:00", B, A, "Wire", paid=(1e6, "US Dollar")),
    ]
    plain = extended_features(frame(*rows))
    pd.testing.assert_frame_equal(plain, extended_features(frame(*rows, *tail)))


def test_the_label_changes_nothing():
    rows = frame(tx(DAY, A, B, "ACH"), tx(DAY, B, C, "Wire"))
    flipped = rows.assign(is_laundering=1)
    pd.testing.assert_frame_equal(extended_features(rows), extended_features(flipped))
