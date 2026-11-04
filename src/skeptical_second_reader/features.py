"""Account features: counts from the transactions of the feature window.

Only 1-10 September is used. After that day the file holds a tail in which every
active account is positive, so a feature from the tail would reveal the label. The
label, counterparty identities and pattern information are not features; amounts are
left out because the data has 15 currencies and no exchange rates.
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
