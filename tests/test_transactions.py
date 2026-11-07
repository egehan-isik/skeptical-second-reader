"""Tests for the transaction model, on small hand-made data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from skeptical_second_reader.explain import BASE_VALUE, shap_values
from skeptical_second_reader.model import fit_xgboost, precision_threshold
from skeptical_second_reader.split import fold_roles
from skeptical_second_reader.transactions import (
    TX_FEATURES,
    account_positions,
    explain_strongest,
    fit_rows,
    noisy_or,
    strongest_transaction,
    transaction_cv,
    transaction_features,
    transaction_labels,
    transaction_table,
)

START = pd.Timestamp("2022-09-01 00:00")
USD, EUR = "US Dollar", "Euro"
BANK = {"A": "1", "B": "1", "C": "2", "D": "3", "E": "4"}


def row(sender, receiver, hours, paid, received=None, fmt="ACH", cur=USD, rec_cur=USD):
    return {
        "timestamp": START + pd.Timedelta(hours=hours),
        "from_bank": BANK[sender],
        "from_account": sender,
        "to_bank": BANK[receiver],
        "to_account": receiver,
        "amount_received": paid if received is None else received,
        "receiving_currency": rec_cur,
        "amount_paid": paid,
        "payment_currency": cur,
        "payment_format": fmt,
        "is_laundering": 0,
    }


@pytest.fixture
def raw():
    return pd.DataFrame(
        [
            row("A", "B", 0, 100),
            row("A", "B", 1, 100, fmt="Wire"),
            # A conversion and, in the same minute, a send of the money it produced.
            row("C", "C", 2, 50, 100, cur=EUR),
            row("C", "D", 2, 100),
            # 90 of the 100 that reached D an hour before.
            row("D", "E", 3, 90),
            row("A", "D", 30, 400),
            # D received 400 thirty hours before: too long ago to count.
            row("D", "B", 60, 50),
            # After the window.
            row("E", "A", 24 * 11, 10, fmt="Cheque"),
        ]
    )


@pytest.fixture
def table(raw):
    return transaction_table(raw)


@pytest.fixture
def features(table):
    return transaction_features(table)


def test_the_table_has_the_window_only_and_no_label(raw, table):
    assert len(table) == 7
    assert "is_laundering" not in table.columns
    assert table["timestamp"].max() < pd.Timestamp("2022-09-11")
    assert table["same_bank"].tolist() == [True, True, True, False, False, False, False]


def test_labels_follow_the_rows_of_the_table(raw, table):
    marked = raw.assign(is_laundering=[0, 0, 0, 1, 0, 0, 0, 1])
    labels = transaction_labels(marked)
    assert labels.tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert len(labels) == len(table)


def test_amounts_are_in_dollars_at_the_rate_of_the_window(table):
    assert table.loc[2, "usd"] == 100
    assert table.loc[2, "usd_received"] == 100


def test_a_currency_without_a_rate_is_refused(raw):
    raw.loc[0, ["payment_currency", "receiving_currency"]] = "Yen"
    with pytest.raises(ValueError, match="without a rate"):
        transaction_table(raw)


def test_features_do_not_depend_on_the_label(raw):
    other = raw.assign(is_laundering=1)
    pd.testing.assert_frame_equal(
        transaction_features(transaction_table(raw)),
        transaction_features(transaction_table(other)),
    )


def test_columns_are_the_listed_ones(features):
    assert list(features.columns) == TX_FEATURES
    assert (features.dtypes == "float32").all()


def test_pair_count_counts_one_direction_of_one_pair(raw):
    # Add B to A: it must not count towards A to B.
    back = pd.concat([raw, pd.DataFrame([row("B", "A", 5, 10)])], ignore_index=True)
    counted = transaction_features(transaction_table(back))["pair_count"].tolist()
    assert counted[:2] == [2, 2]
    assert counted[-1] == 1


def test_pair_count_and_activity(features):
    assert features["pair_count"].tolist() == [2, 2, 1, 1, 1, 1, 1]
    assert features["sender_tx"].tolist() == [3, 3, 2, 2, 2, 3, 2]
    assert features["receiver_tx"].tolist() == [3, 3, 1, 2, 1, 2, 3]
    assert features["receiver_in_degree"].tolist() == [2, 2, 1, 2, 1, 2, 2]
    assert features["sender_out_degree"].tolist() == [2] * 7


def test_a_self_transaction_in_one_currency_is_not_a_conversion(raw):
    same = raw[raw["from_account"] != "C"].copy()
    extra = pd.DataFrame([row("C", "C", 2, 100), row("C", "D", 2, 100)])
    moved = transaction_features(transaction_table(pd.concat([same, extra])))
    assert moved["after_conversion"].sum() == 0
    assert moved["sender_conversions"].sum() == 0


def test_a_send_after_a_conversion_needs_the_converted_currency(raw, features):
    assert features["after_conversion"].tolist() == [0, 0, 0, 1, 0, 0, 0]
    assert features["sender_conversions"].tolist() == [0, 0, 1, 1, 0, 0, 0]
    assert features["changes_currency"].tolist() == [0, 0, 1, 0, 0, 0, 0]
    assert features["is_self"].tolist() == [0, 0, 1, 0, 0, 0, 0]
    # The same send, a minute later: no longer after the conversion.
    later = raw.copy()
    later.loc[3, "timestamp"] += pd.Timedelta(minutes=1)
    moved = transaction_features(transaction_table(later))
    assert moved["after_conversion"].sum() == 0


def test_money_lookups_look_back_and_forward_within_a_day(features):
    since = features["hours_since_money_in"]
    assert since[4] == pytest.approx(1)
    assert features["ratio_to_money_in"][4] == pytest.approx(0.9)
    # Nothing received before; a self-transaction; received too long ago.
    assert since[[0, 2, 6]].isna().all()
    to_next = features["hours_to_next_send"]
    assert to_next[3] == pytest.approx(1)
    assert features["next_send_ratio"][3] == pytest.approx(0.9)
    # No later send; a self-transaction; the next send is too far ahead.
    assert to_next[[0, 2, 5]].isna().all()


def test_a_later_send_does_not_count_as_money_received_before(features):
    # D receives at hour 2 and sends at hour 3; C has received nothing at hour 2.
    assert np.isnan(features["hours_since_money_in"][3])


def test_amount_against_the_usual_amount_of_the_account(features):
    usual = np.log1p(100)
    assert features["log_usd"][5] == pytest.approx(np.log1p(400))
    assert features["log_usd_over_sender_median"][5] == pytest.approx(
        np.log1p(400) - usual
    )
    median_d = (np.log1p(100) + np.log1p(400)) / 2
    assert features["log_usd_over_receiver_median"][3] == pytest.approx(
        np.log1p(100) - median_d
    )


def test_an_account_missing_from_the_list_is_refused(table):
    with pytest.raises(ValueError, match="outside"):
        account_positions(table, pd.Index(["1:A"]))


def test_a_transaction_is_fitted_only_when_both_ends_have_the_role():
    roles = np.array(["train", "train", "val", "val", "test"])
    sender = np.array([0, 0, 2, 2, 4, 0])
    receiver = np.array([1, 2, 3, 4, 0, 0])
    train_rows, val_rows = fit_rows(roles, sender, receiver)
    assert train_rows.tolist() == [True, False, False, False, False, True]
    assert val_rows.tolist() == [False, False, True, False, False, False]


def test_noisy_or_by_hand():
    # Account 0 sends 0.5 and receives 0.5; account 1 receives 0.5 and has a self 0.2.
    scores = np.array([0.5, 0.5, 0.2])
    sender = np.array([0, 2, 1])
    receiver = np.array([1, 0, 1])
    values = noisy_or(scores, sender, receiver, 3)
    assert values[0] == pytest.approx(0.75)
    assert values[1] == pytest.approx(1 - 0.5 * 0.8)
    assert values[2] == pytest.approx(0.5)


def test_noisy_or_stays_below_one_and_finite():
    values = noisy_or(np.array([1.0, 1.0]), np.array([0, 0]), np.array([1, 1]), 2)
    assert np.isfinite(values).all()
    assert (values < 1).all()


def test_strongest_transaction_sees_both_sides():
    scores = np.array([0.1, 0.9, 0.5])
    sender = np.array([0, 1, 0])
    receiver = np.array([1, 0, 2])
    assert strongest_transaction(scores, sender, receiver, 4).tolist() == [1, 1, 2, -1]


def test_precision_threshold_is_the_lowest_that_reaches_it():
    y = pd.Series([0, 0, 1, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    # From 0.3: 4 of 5 are positive, 0.8. From 0.2: 4 of 6, below 0.7.
    assert precision_threshold(y, scores, wanted=0.7) == pytest.approx(0.3)


def test_a_precision_nobody_reaches_is_refused():
    with pytest.raises(ValueError, match="no threshold"):
        precision_threshold(pd.Series([1, 0]), np.array([0.1, 0.9]), wanted=0.9)


@pytest.fixture
def small():
    rng = np.random.default_rng(0)
    accounts = pd.Index([f"a{i:02d}" for i in range(50)], name="account")
    n = 3000
    table = pd.DataFrame(
        {
            "sender": rng.choice(accounts, n),
            "receiver": rng.choice(accounts, n),
        }
    )
    signal = rng.normal(size=n)
    features = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    tx_labels = pd.Series((signal > 1.5).astype(int))
    bad = set(table.loc[tx_labels == 1, "sender"]) | set(
        table.loc[tx_labels == 1, "receiver"]
    )
    y = pd.Series(accounts.isin(bad).astype(int), index=accounts)
    folds = pd.Series(np.arange(50) % 5, index=accounts)
    return features, tx_labels, table, y, folds


def spy_fit(log):
    """A fake model: the score rises with the round, so a mixed-up round shows."""

    def fit(X_train, y_train, X_val, y_val):
        k = len(log)
        log.append((X_train.index, y_train.index, X_val.index, y_val.index))
        return lambda X: (k + 1) / 10 * np.ones(len(X))

    return fit


def test_no_transaction_of_a_test_account_reaches_the_fit(small):
    features, tx_labels, table, y, folds = small
    log = []
    transaction_cv(features, tx_labels, table, y, folds, lambda *_: 0.5, spy_fit(log))
    assert len(log) == 5
    for k, (train_x, train_y, val_x, val_y) in enumerate(log):
        roles = fold_roles(folds, k)
        send, take = table["sender"].map(roles), table["receiver"].map(roles)
        expected_train = table.index[(send == "train") & (take == "train")]
        expected_val = table.index[(send == "val") & (take == "val")]
        assert train_x.equals(expected_train) and train_y.equals(expected_train)
        assert val_x.equals(expected_val) and val_y.equals(expected_val)


def test_the_threshold_sees_only_validation_accounts(small):
    features, tx_labels, table, y, folds = small
    seen = []

    def spy(y_val, scores):
        seen.append((y_val.index, scores))
        return 0.5

    transaction_cv(features, tx_labels, table, y, folds, spy, spy_fit([]))
    sender, receiver = account_positions(table, folds.index)
    for k, (index, scores) in enumerate(seen):
        assert index.equals(folds.index[folds == (k + 1) % 5])
        expected = noisy_or(
            (k + 1) / 10 * np.ones(len(table)), sender, receiver, len(folds)
        )
        np.testing.assert_allclose(scores, expected[folds == (k + 1) % 5])


def test_every_account_is_scored_in_its_own_test_round(small):
    features, tx_labels, table, y, folds = small
    result = transaction_cv(
        features, tx_labels, table, y, folds, lambda *_: 0.5, spy_fit([])
    )
    sender, receiver = account_positions(table, folds.index)
    for k in range(5):
        expected = noisy_or(
            (k + 1) / 10 * np.ones(len(table)), sender, receiver, len(folds)
        )
        np.testing.assert_allclose(result.scores[folds == k], expected[folds == k])
    assert set(result.flags.unique()) <= {0, 1}
    assert (result.flags == (result.scores >= 0.5).astype(int)).all()


def test_flags_use_the_threshold_of_the_round(small):
    features, tx_labels, table, y, folds = small
    thresholds = iter([0.2, 2.0, 0.2, 2.0, 0.2])
    result = transaction_cv(
        features, tx_labels, table, y, folds, lambda *_: next(thresholds), spy_fit([])
    )
    assert result.thresholds == [0.2, 2.0, 0.2, 2.0, 0.2]
    assert (result.flags[folds == 1] == 0).all()
    assert (result.flags[folds == 2] == 1).all()


def test_calibration_uses_the_validation_accounts_of_the_round(small):
    features, tx_labels, table, _, folds = small
    # Nearly every account of `small` is positive: a curve on it would be flat.
    y = pd.Series(
        np.random.default_rng(3).integers(0, 2, len(folds)), index=folds.index
    )
    assert 0 < y.mean() < 1
    calls = []

    def scored(k):
        return np.clip(np.abs(features["signal"].to_numpy()) * (k + 1) / 30, 0, 0.9)

    def fit(*_):
        k = len(calls)
        calls.append(k)
        return lambda X: scored(k)[: len(X)]

    result = transaction_cv(features, tx_labels, table, y, folds, lambda *_: 0.5, fit)
    sender, receiver = account_positions(table, folds.index)
    for k in range(5):
        values = noisy_or(scored(k), sender, receiver, len(folds))
        val, test = folds == (k + 1) % 5, folds == k
        curve = IsotonicRegression(out_of_bounds="clip").fit(values[val], y[val])
        np.testing.assert_allclose(result.calibrated[test], curve.predict(values[test]))
    assert result.calibrated.between(0, 1).all()
    assert result.calibrated.nunique() > 1


def test_high_transactions_are_counted_on_both_sides_once():
    accounts = pd.Index(["a", "b", "c", "d", "e"], name="account")
    table = pd.DataFrame(
        {
            "sender": ["a", "a", "b", "c", "d"],
            "receiver": ["b", "a", "c", "c", "e"],
        }
    )
    # High: a to b, and the self-transactions of a and c. Low: b to c, d to e.
    features = pd.DataFrame({"x": [0.9, 0.9, 0.1, 0.9, 0.1]})
    tx_labels = pd.Series([0] * 5)
    y = pd.Series(0, index=accounts)
    folds = pd.Series([0, 1, 2, 3, 4], index=accounts)

    def fit(*_):
        return lambda X: X["x"].to_numpy()

    result = transaction_cv(features, tx_labels, table, y, folds, lambda *_: 0.5, fit)
    assert result.n_high.tolist() == [2, 1, 1, 0, 0]


def test_misaligned_inputs_are_refused(small):
    features, tx_labels, table, y, folds = small
    with pytest.raises(ValueError, match="same index"):
        transaction_cv(features, tx_labels, table, y, folds.iloc[::-1], lambda *_: 0.5)
    with pytest.raises(ValueError, match="same rows"):
        transaction_cv(features[:-1], tx_labels, table, y, folds, lambda *_: 0.5)


def test_the_model_finds_the_signal_and_the_explanations_add_up(small):
    features, tx_labels, table, y, folds = small
    result = transaction_cv(
        features, tx_labels, table, y, folds, lambda *_: 0.5, fit_xgboost
    )
    values = explain_strongest(result, features, folds)
    assert values.index.equals(folds.index)
    assert list(values.columns) == [*features.columns, BASE_VALUE]
    assert values[["signal", "noise"]].abs().mean().idxmax() == "signal"
    for k, model in enumerate(result.models):
        members = folds.index[folds == k]
        rows = features.iloc[result.strongest[members].to_numpy()]
        margin = model.predict(rows, output_margin=True)
        np.testing.assert_allclose(values.loc[members].sum(axis=1), margin, atol=1e-3)
        expected = shap_values(model, rows)
        np.testing.assert_allclose(values.loc[members].to_numpy(), expected.to_numpy())


def test_an_account_is_explained_by_its_strongest_transaction(small):
    features, tx_labels, table, y, folds = small
    result = transaction_cv(
        features, tx_labels, table, y, folds, lambda *_: 0.5, fit_xgboost
    )
    sender, receiver = account_positions(table, folds.index)
    for k, model in enumerate(result.models):
        tx_scores = model.predict_proba(features)[:, 1]
        top = strongest_transaction(tx_scores, sender, receiver, len(folds))
        members = folds == k
        assert (result.strongest[members].to_numpy() == top[members]).all()
        best = pd.Series(tx_scores).groupby(pd.Series(sender)).max()
        for position in np.flatnonzero(members)[:5]:
            assert tx_scores[top[position]] >= best.get(position, 0)


def test_explaining_a_subset_keeps_its_order(small):
    features, tx_labels, table, y, folds = small
    result = transaction_cv(
        features, tx_labels, table, y, folds, lambda *_: 0.5, fit_xgboost
    )
    subset = folds.index[[7, 3, 21]]
    part = explain_strongest(result, features, folds, subset)
    assert part.index.equals(subset)
    pd.testing.assert_frame_equal(
        part, explain_strongest(result, features, folds).loc[subset]
    )
