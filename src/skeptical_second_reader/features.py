"""Account features from the transactions of the feature window.

Only 1-10 September is used. After that day the file holds a tail in which every
active account is positive, so a feature from the tail would reveal the label. The
label, counterparty identities and pattern information are not features.

`account_features` gives the 15 counts of the baseline. `extended_features` adds
amounts, timing and network features. Two of its inputs are computed across accounts,
without labels: the exchange rates and the counterparty degrees.
"""

import numpy as np
import pandas as pd

from skeptical_second_reader.data import account_key

WINDOW_END = pd.Timestamp("2022-09-11")

ACH = "ACH"
OTHER_FORMATS = ["Cheque", "Credit Card", "Cash", "Bitcoin", "Wire", "Reinvestment"]
FORMAT_FEATURES = {
    name: f"{name.lower().replace(' ', '_')}_tx" for name in OTHER_FORMATS
}

FEATURES = [
    "out_tx",
    "in_tx",
    "n_receivers",
    "n_senders",
    "active_days",
    "self_tx",
    "currency_change_tx",
    "ach_out_tx",
    "ach_in_tx",
    *FORMAT_FEATURES.values(),
]

USD = "US Dollar"
AMOUNT_FEATURES = [
    "log_sent_usd",
    "log_received_usd",
    "log_median_sent_usd",
    "log_median_received_usd",
    "log_max_sent_usd",
    "log_max_received_usd",
    "flow_balance",
]
TIMING_FEATURES = ["max_tx_per_day", "max_tx_per_hour", "median_gap_hours"]
NETWORK_FEATURES = [
    "reciprocal_counterparties",
    "mean_receiver_in_degree",
    "mean_sender_out_degree",
    "ach_receivers",
    "ach_senders",
    "receiver_banks",
    "sender_banks",
    "n_currencies",
]
ALL_FEATURES = [*FEATURES, *AMOUNT_FEATURES, *TIMING_FEATURES, *NETWORK_FEATURES]


def account_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """One row per account active in the window, one column per name in `FEATURES`.

    A self-transaction counts once in the features that count an account's
    transactions (active days, currency changes, formats), and once in each direction
    in out_tx and in_tx. Accounts without a transaction in the window get no row: an
    all-zero row would itself reveal that the account is only active in the tail.
    """
    window = transactions[transactions["timestamp"] < WINDOW_END]
    unknown = set(window["payment_format"]) - {ACH, *OTHER_FORMATS}
    if unknown:
        raise ValueError(f"unknown payment formats: {sorted(unknown)}")

    tx = pd.DataFrame(
        {
            "tx_id": np.arange(len(window)),
            "sender": account_key(
                window["from_bank"], window["from_account"]
            ).to_numpy(),
            "receiver": account_key(window["to_bank"], window["to_account"]).to_numpy(),
            "payment_format": window["payment_format"].to_numpy(),
            "day": window["timestamp"].dt.normalize().to_numpy(),
            "changes_currency": (
                window["receiving_currency"] != window["payment_currency"]
            ).to_numpy(),
        }
    )
    # One row per (transaction, account): both sides, a self-transaction only once.
    context = ["tx_id", "payment_format", "day", "changes_currency"]
    touches = pd.concat(
        [
            tx[[*context, "sender"]].rename(columns={"sender": "account"}),
            tx[[*context, "receiver"]].rename(columns={"receiver": "account"}),
        ]
    ).drop_duplicates(["tx_id", "account"])

    is_ach = tx["payment_format"] == ACH
    pairs = tx[["sender", "receiver"]].drop_duplicates()
    counts = {
        "out_tx": tx["sender"].value_counts(),
        "in_tx": tx["receiver"].value_counts(),
        "n_receivers": pairs["sender"].value_counts(),
        "n_senders": pairs["receiver"].value_counts(),
        "active_days": touches[["account", "day"]]
        .drop_duplicates()["account"]
        .value_counts(),
        "self_tx": tx.loc[tx["sender"] == tx["receiver"], "sender"].value_counts(),
        "currency_change_tx": touches.loc[
            touches["changes_currency"], "account"
        ].value_counts(),
        "ach_out_tx": tx.loc[is_ach, "sender"].value_counts(),
        "ach_in_tx": tx.loc[is_ach, "receiver"].value_counts(),
    }
    for name, column in FORMAT_FEATURES.items():
        counts[column] = touches.loc[
            touches["payment_format"] == name, "account"
        ].value_counts()

    index = pd.Index(sorted(touches["account"].unique()), name="account")
    table = {name: counts[name].reindex(index, fill_value=0) for name in FEATURES}
    return pd.DataFrame(table, index=index).astype("int64")


def _in_window(transactions: pd.DataFrame) -> pd.DataFrame:
    """The window's transactions, with sender and receiver as account keys."""
    early = transactions[transactions["timestamp"] < WINDOW_END]
    return early.assign(
        sender=account_key(early["from_bank"], early["from_account"]),
        receiver=account_key(early["to_bank"], early["to_account"]),
    )


def exchange_rates(transactions: pd.DataFrame) -> pd.Series:
    """One rate to US dollars per currency, from the window's own transactions.

    The file has no rate table. A rate is the median of received/paid over the
    transactions from that currency to US dollars; if there are none, the median of
    paid/received over those from US dollars to it. Labels are not used.
    """
    early = transactions[transactions["timestamp"] < WINDOW_END]
    changes = early[early["receiving_currency"] != early["payment_currency"]]
    to_usd = changes[changes["receiving_currency"] == USD]
    from_usd = changes[changes["payment_currency"] == USD]
    direct = (
        (to_usd["amount_received"] / to_usd["amount_paid"])
        .groupby(to_usd["payment_currency"])
        .median()
    )
    inverse = (
        (from_usd["amount_paid"] / from_usd["amount_received"])
        .groupby(from_usd["receiving_currency"])
        .median()
    )
    rates = direct.combine_first(inverse)
    rates[USD] = 1.0
    return rates.rename("rate_to_usd")


def extended_features(
    transactions: pd.DataFrame,
    rates: pd.Series | None = None,
    degree_transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The columns of `ALL_FEATURES`, for the accounts of `account_features`.

    Self-transactions are left out of the flows and of the counterparty relations.
    median_gap_hours is NaN for an account with a single transaction. `rates` and
    `degree_transactions` default to the whole window; they can be limited to study
    what these two cross-account inputs contribute.
    """
    base = account_features(transactions)
    index = base.index
    early = _in_window(transactions)
    if rates is None:
        rates = exchange_rates(transactions)
    used = set(early["payment_currency"]) | set(early["receiving_currency"])
    missing = used - set(rates.index)
    if missing:
        raise ValueError(f"currencies without a rate: {sorted(missing)}")

    def aligned(series: pd.Series) -> pd.Series:
        return series.reindex(index, fill_value=0.0).astype("float64")

    is_self = early["sender"] == early["receiver"]
    flows = early.loc[~is_self].assign(
        paid_usd=lambda t: t["amount_paid"] * t["payment_currency"].map(rates),
        received_usd=lambda t: (
            t["amount_received"] * t["receiving_currency"].map(rates)
        ),
    )
    sent = flows.groupby("sender")["paid_usd"]
    received = flows.groupby("receiver")["received_usd"]
    sent_sum, received_sum = aligned(sent.sum()), aligned(received.sum())
    amounts = {
        "log_sent_usd": np.log1p(sent_sum),
        "log_received_usd": np.log1p(received_sum),
        "log_median_sent_usd": np.log1p(aligned(sent.median())),
        "log_median_received_usd": np.log1p(aligned(received.median())),
        "log_max_sent_usd": np.log1p(aligned(sent.max())),
        "log_max_received_usd": np.log1p(aligned(received.max())),
        "flow_balance": (received_sum - sent_sum) / (received_sum + sent_sum + 1.0),
    }

    # One row per side of a transaction; a self-transaction only once.
    touches = pd.concat(
        [
            early[["sender", "timestamp"]].rename(columns={"sender": "account"}),
            early.loc[~is_self, ["receiver", "timestamp"]].rename(
                columns={"receiver": "account"}
            ),
        ]
    ).sort_values(["account", "timestamp"])
    per_day = touches.groupby(["account", touches["timestamp"].dt.floor("D")]).size()
    per_hour = touches.groupby(["account", touches["timestamp"].dt.floor("h")]).size()
    gaps = touches.groupby("account")["timestamp"].diff().dt.total_seconds() / 3600
    timing = {
        "max_tx_per_day": aligned(per_day.groupby(level=0).max()),
        "max_tx_per_hour": aligned(per_hour.groupby(level=0).max()),
        "median_gap_hours": gaps.groupby(touches["account"]).median().reindex(index),
    }

    pairs = flows[["sender", "receiver"]].drop_duplicates()
    if degree_transactions is None:
        degree_pairs = pairs
    else:
        degree_early = _in_window(degree_transactions)
        degree_pairs = degree_early.loc[
            degree_early["sender"] != degree_early["receiver"], ["sender", "receiver"]
        ].drop_duplicates()
    out_degree = degree_pairs["sender"].value_counts()
    in_degree = degree_pairs["receiver"].value_counts()
    reversed_pairs = pairs.rename(columns={"sender": "receiver", "receiver": "sender"})
    reciprocal = pairs.merge(reversed_pairs, on=["sender", "receiver"])
    ach = flows.loc[flows["payment_format"] == ACH, ["sender", "receiver"]]
    ach = ach.drop_duplicates()
    currencies = pd.concat(
        [
            early[["sender", "payment_currency"]].set_axis(
                ["account", "currency"], axis=1
            ),
            early[["receiver", "receiving_currency"]].set_axis(
                ["account", "currency"], axis=1
            ),
        ]
    ).drop_duplicates()

    def bank_of(keys: pd.Series) -> pd.Series:
        return keys.str.split(":", n=1).str[0]

    network = {
        "reciprocal_counterparties": aligned(reciprocal["sender"].value_counts()),
        "mean_receiver_in_degree": aligned(
            pairs["receiver"].map(in_degree).fillna(0).groupby(pairs["sender"]).mean()
        ),
        "mean_sender_out_degree": aligned(
            pairs["sender"].map(out_degree).fillna(0).groupby(pairs["receiver"]).mean()
        ),
        "ach_receivers": aligned(ach["sender"].value_counts()),
        "ach_senders": aligned(ach["receiver"].value_counts()),
        "receiver_banks": aligned(
            bank_of(pairs["receiver"]).groupby(pairs["sender"]).nunique()
        ),
        "sender_banks": aligned(
            bank_of(pairs["sender"]).groupby(pairs["receiver"]).nunique()
        ),
        "n_currencies": aligned(currencies["account"].value_counts()),
    }
    extra = pd.DataFrame({**amounts, **timing, **network}, index=index)
    return pd.concat([base, extra], axis=1)[ALL_FEATURES]
