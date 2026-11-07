"""The transaction model: every transaction is scored, accounts follow from them.

Labels exist per transaction and most transactions of a positive account are ordinary,
so the model learns on transactions. An account's score is the noisy-or of the scores
of its transactions, as sender or receiver, within the feature window.

Only transactions with both ends in training accounts are trained on, and only those
with both ends in validation accounts are used for early stopping. A transaction that
touches a test account is scored, never learned from.

The features are computed across all accounts of the window, without labels, and look
at the whole window, so `pair_count` also counts transactions after the one it
describes. The account features of `features.py` share this property.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from skeptical_second_reader.data import account_key
from skeptical_second_reader.explain import shap_values
from skeptical_second_reader.features import WINDOW_END, exchange_rates
from skeptical_second_reader.model import ScoreFn, fit_xgboost
from skeptical_second_reader.split import N_FOLDS, TEST, TRAIN, VAL, fold_roles

DAY = pd.Timedelta("24h")

TX_FEATURES = [
    "log_usd",
    "payment_format",
    "paid_currency",
    "changes_currency",
    "is_self",
    "same_bank",
    "hour",
    "weekday",
    "sender_tx",
    "receiver_tx",
    "sender_out_degree",
    "receiver_in_degree",
    "pair_count",
    "log_usd_over_sender_median",
    "log_usd_over_receiver_median",
    "after_conversion",
    "sender_conversions",
    "hours_since_money_in",
    "ratio_to_money_in",
    "hours_to_next_send",
    "next_send_ratio",
]

ThresholdFn = Callable[[pd.Series, np.ndarray], float]
FitFn = Callable[[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series], ScoreFn]


def transaction_table(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per transaction of the window, in file order, without the label."""
    early = transactions[transactions["timestamp"] < WINDOW_END]
    rates = exchange_rates(transactions)
    used = set(early["payment_currency"]) | set(early["receiving_currency"])
    missing = used - set(rates.index)
    if missing:
        raise ValueError(f"currencies without a rate: {sorted(missing)}")
    return pd.DataFrame(
        {
            "sender": account_key(early["from_bank"], early["from_account"]).to_numpy(),
            "receiver": account_key(early["to_bank"], early["to_account"]).to_numpy(),
            "timestamp": early["timestamp"].to_numpy(),
            "payment_format": early["payment_format"].to_numpy(),
            "paid_currency": early["payment_currency"].to_numpy(),
            "received_currency": early["receiving_currency"].to_numpy(),
            "usd": (
                early["amount_paid"] * early["payment_currency"].map(rates)
            ).to_numpy(),
            "usd_received": (
                early["amount_received"] * early["receiving_currency"].map(rates)
            ).to_numpy(),
            "same_bank": (early["from_bank"] == early["to_bank"]).to_numpy(),
        }
    )


def transaction_labels(transactions: pd.DataFrame) -> pd.Series:
    """The label of every row of `transaction_table`, in the same order."""
    in_window = transactions["timestamp"] < WINDOW_END
    return transactions.loc[in_window, "is_laundering"].reset_index(drop=True)


def _money_lookups(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each move between two accounts: the latest money received by its sender in
    the 24 hours before, and the next send of its receiver in the 24 hours after."""
    is_self = (table["sender"] == table["receiver"]).to_numpy()
    moves = table.assign(row=np.arange(len(table)))[~is_self]
    ordered = moves.sort_values("timestamp", kind="stable")
    incoming = (
        ordered[["receiver", "timestamp", "usd_received"]]
        .rename(columns={"receiver": "account", "usd_received": "in_usd"})
        .assign(in_time=lambda t: t["timestamp"])
    )
    before = pd.merge_asof(
        ordered[["row", "sender", "timestamp", "usd"]].rename(
            columns={"sender": "account"}
        ),
        incoming,
        on="timestamp",
        by="account",
        direction="backward",
        tolerance=DAY,
    )
    before["hold_hours"] = (
        before["timestamp"] - before["in_time"]
    ).dt.total_seconds() / 3600
    before["ratio"] = before["usd"] / before["in_usd"]
    outgoing = (
        ordered[["sender", "timestamp", "usd"]]
        .rename(columns={"sender": "account", "usd": "next_usd"})
        .assign(next_time=lambda t: t["timestamp"])
    )
    after = pd.merge_asof(
        ordered[["row", "receiver", "timestamp", "usd_received"]].rename(
            columns={"receiver": "account"}
        ),
        outgoing,
        on="timestamp",
        by="account",
        direction="forward",
        tolerance=DAY,
    )
    after["wait_hours"] = (
        after["next_time"] - after["timestamp"]
    ).dt.total_seconds() / 3600
    after["next_ratio"] = after["next_usd"] / after["usd_received"]
    return before, after


def transaction_features(table: pd.DataFrame) -> pd.DataFrame:
    """The columns of `TX_FEATURES` for every row of `transaction_table`.

    The four money lookups are NaN when nothing is found and for self-transactions.
    """
    is_self = (table["sender"] == table["receiver"]).to_numpy()
    before, after = _money_lookups(table)

    def code_of(values: pd.Series) -> np.ndarray:
        return pd.factorize(values, sort=True)[0].astype("float32")

    def per_row(frame: pd.DataFrame, column: str) -> np.ndarray:
        values = np.full(len(table), np.nan, dtype="float32")
        values[frame["row"].to_numpy()] = frame[column].to_numpy(dtype="float32")
        return values

    links = table[["sender", "receiver"]].drop_duplicates()
    log_usd = np.log1p(table["usd"])
    log_received = np.log1p(table["usd_received"])
    conversion = is_self & (table["paid_currency"] != table["received_currency"])
    conversion_keys = (
        table.loc[conversion, ["sender", "timestamp", "received_currency"]]
        .drop_duplicates()
        .rename(columns={"received_currency": "paid_currency"})
        .assign(after_conversion=1.0)
    )
    after_conversion = (
        table[["sender", "timestamp", "paid_currency"]]
        .merge(
            conversion_keys, on=["sender", "timestamp", "paid_currency"], how="left"
        )["after_conversion"]
        .fillna(0)
        .to_numpy()
    )
    columns = {
        "log_usd": log_usd,
        "payment_format": code_of(table["payment_format"]),
        "paid_currency": code_of(table["paid_currency"]),
        "changes_currency": table["paid_currency"] != table["received_currency"],
        "is_self": is_self,
        "same_bank": table["same_bank"],
        "hour": table["timestamp"].dt.hour,
        "weekday": table["timestamp"].dt.day_of_week,
        "sender_tx": table["sender"].map(table["sender"].value_counts()),
        "receiver_tx": table["receiver"].map(table["receiver"].value_counts()),
        "sender_out_degree": table["sender"].map(links["sender"].value_counts()),
        "receiver_in_degree": table["receiver"].map(links["receiver"].value_counts()),
        "pair_count": table.groupby(["sender", "receiver"])["usd"].transform("size"),
        "log_usd_over_sender_median": log_usd
        - log_usd.groupby(table["sender"]).transform("median"),
        "log_usd_over_receiver_median": log_received
        - log_received.groupby(table["receiver"]).transform("median"),
        "after_conversion": after_conversion,
        "sender_conversions": table["sender"]
        .map(table.loc[conversion, "sender"].value_counts())
        .fillna(0),
        "hours_since_money_in": per_row(before, "hold_hours"),
        "ratio_to_money_in": per_row(before, "ratio"),
        "hours_to_next_send": per_row(after, "wait_hours"),
        "next_send_ratio": per_row(after, "next_ratio"),
    }
    return pd.DataFrame(columns)[TX_FEATURES].astype("float32")


def account_positions(
    table: pd.DataFrame, accounts: pd.Index
) -> tuple[np.ndarray, np.ndarray]:
    """Position in `accounts` of the sender and the receiver of every transaction."""
    sender = accounts.get_indexer(table["sender"])
    receiver = accounts.get_indexer(table["receiver"])
    if (sender < 0).any() or (receiver < 0).any():
        raise ValueError("a transaction touches an account outside `accounts`")
    return sender, receiver


def fit_rows(
    roles: np.ndarray, sender: np.ndarray, receiver: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Transactions to train on, and those for early stopping, in one round."""
    train_rows = (roles[sender] == TRAIN) & (roles[receiver] == TRAIN)
    val_rows = (roles[sender] == VAL) & (roles[receiver] == VAL)
    return train_rows, val_rows


def noisy_or(
    tx_scores: np.ndarray, sender: np.ndarray, receiver: np.ndarray, n_accounts: int
) -> np.ndarray:
    """1 - product(1 - score) over an account's transactions; self-transactions once."""
    log_miss = np.zeros(n_accounts)
    miss = np.log1p(-np.clip(tx_scores, 0, 1 - 1e-7))
    other_side = sender != receiver
    np.add.at(log_miss, sender, miss)
    np.add.at(log_miss, receiver[other_side], miss[other_side])
    return 1 - np.exp(log_miss)


def strongest_transaction(
    tx_scores: np.ndarray, sender: np.ndarray, receiver: np.ndarray, n_accounts: int
) -> np.ndarray:
    """Row of each account's highest-scored transaction, -1 if it has none.

    Among equal scores the later entry wins: the later row within one role, and a
    row as receiver over a row as sender.
    """
    rows = np.arange(len(tx_scores))
    position = np.concatenate([sender, receiver])
    row = np.concatenate([rows, rows])
    order = np.argsort(tx_scores[row], kind="stable")
    strongest = np.full(n_accounts, -1, dtype="int64")
    # With repeated positions, numpy keeps the last value assigned: the highest score.
    strongest[position[order]] = row[order]
    return strongest


@dataclass
class TransactionCV:
    """Out-of-fold results of `transaction_cv`, from each account's own test round.

    scores: noisy-or score of every account.
    thresholds: one per round, from that round's validation accounts only.
    flags: 1 if an account's score reaches its round's threshold.
    strongest: row of the account's highest-scored transaction.
    models: the model of every round.
    calibrated: the score through an isotonic curve fitted on that round's validation
        accounts, for reading it as a rate. Ranking and flags use `scores`, because the
        curve creates ties.
    n_high: the account's transactions at or above its round's threshold.
    """

    scores: pd.Series
    thresholds: list[float]
    flags: pd.Series
    strongest: pd.Series
    models: list[xgb.XGBClassifier]
    calibrated: pd.Series
    n_high: pd.Series


def transaction_cv(
    features: pd.DataFrame,
    tx_labels: pd.Series,
    table: pd.DataFrame,
    y: pd.Series,
    folds: pd.Series,
    choose_threshold: ThresholdFn,
    fit: FitFn = fit_xgboost,
    n_folds: int = N_FOLDS,
) -> TransactionCV:
    """Run the rounds of the group-disjoint split on the transaction model."""
    if not y.index.equals(folds.index):
        raise ValueError("y and folds must have the same index")
    if not (len(features) == len(tx_labels) == len(table)):
        raise ValueError("features, tx_labels and table must have the same rows")
    accounts = folds.index
    sender, receiver = account_positions(table, accounts)
    scores = pd.Series(np.nan, index=accounts, name="score")
    flags = pd.Series(-1, index=accounts, name="flag")
    strongest = pd.Series(-1, index=accounts, name="strongest_row")
    calibrated = pd.Series(np.nan, index=accounts, name="calibrated")
    n_high = pd.Series(-1, index=accounts, name="n_high")
    thresholds, models = [], []
    for k in range(n_folds):
        roles = fold_roles(folds, k, n_folds).to_numpy()
        train_rows, val_rows = fit_rows(roles, sender, receiver)
        score = fit(
            features[train_rows],
            tx_labels[train_rows],
            features[val_rows],
            tx_labels[val_rows],
        )
        tx_scores = score(features)
        values = noisy_or(tx_scores, sender, receiver, len(accounts))
        val, test = roles == VAL, roles == TEST
        threshold = float(choose_threshold(y[val], values[val]))
        thresholds.append(threshold)
        models.append(getattr(score, "model", None))
        scores[test] = values[test]
        flags[test] = (values[test] >= threshold).astype(int)
        top = strongest_transaction(tx_scores, sender, receiver, len(accounts))
        strongest[test] = top[test]
        curve = IsotonicRegression(out_of_bounds="clip").fit(values[val], y[val])
        calibrated[test] = curve.predict(values[test])
        high = tx_scores >= threshold
        counted = np.zeros(len(accounts), dtype="int64")
        np.add.at(counted, sender[high], 1)
        np.add.at(counted, receiver[high & (sender != receiver)], 1)
        n_high[test] = counted[test]
    if scores.isna().any():
        raise ValueError("some accounts were never scored")
    return TransactionCV(
        scores, thresholds, flags, strongest, models, calibrated, n_high
    )


def explain_strongest(
    result: TransactionCV,
    features: pd.DataFrame,
    folds: pd.Series,
    accounts: pd.Index | None = None,
) -> pd.DataFrame:
    """SHAP values of each account's strongest transaction, by the model of the round
    in which that account was the test fold. They explain the transaction's score, in
    log-odds, not the noisy-or score of the account."""
    if accounts is None:
        accounts = result.scores.index
    chosen = folds.loc[accounts]
    parts = []
    for k in sorted(chosen.unique()):
        model = result.models[k]
        members = chosen.index[chosen == k]
        rows = result.strongest.loc[members].to_numpy()
        values = shap_values(model, features.iloc[rows])
        values.index = members
        parts.append(values)
    return pd.concat(parts).loc[accounts]
