"""Facts about an account that its table of transactions does not show directly.

Computed from all transactions of the window, 1-10 September, without labels, pattern
names or model scores. A small reader cannot count these reliably from 30 rows, and
some need the counterparties' transactions, which the prompt never shows. The same
functions are meant to become the tools of the agent.
"""

import numpy as np
import pandas as pd

DAY_HOURS = 24
FACTS = [
    "payers",
    "payees",
    "both_ways",
    "share_sent_on_within_day",
    "median_hours_to_send_on",
    "sends_with_conversion",
    "most_sends_same_amount",
    "payers_median_payees",
    "payees_median_payers",
    "active_days",
    "busiest_day",
]


def _per_account(values: pd.Series, n_accounts: int, fill: float) -> np.ndarray:
    out = np.full(n_accounts, fill, dtype=float)
    out[values.index.to_numpy()] = values.to_numpy()
    return out


def account_facts(
    table: pd.DataFrame, sender: np.ndarray, receiver: np.ndarray, accounts: pd.Index
) -> pd.DataFrame:
    """Every fact for every account.

    `sender` and `receiver` are the positions in `accounts` of each transaction's
    ends (`transactions.account_positions`). A fact that does not apply is NaN, for
    example the time to send on for an account that receives nothing.
    """
    n = len(accounts)
    own = sender == receiver
    times = table["timestamp"].to_numpy()
    usd = table["usd"].to_numpy()
    facts = {}

    edges = pd.DataFrame({"s": sender[~own], "r": receiver[~own]}).drop_duplicates()
    out_degree = np.bincount(edges["s"], minlength=n)
    in_degree = np.bincount(edges["r"], minlength=n)
    facts["payers"] = in_degree.astype(float)
    facts["payees"] = out_degree.astype(float)
    back = edges.merge(edges, left_on=["s", "r"], right_on=["r", "s"])
    facts["both_ways"] = np.bincount(back["s_x"], minlength=n).astype(float)
    facts["payers_median_payees"] = _per_account(
        pd.Series(out_degree[edges["s"]]).groupby(edges["r"].to_numpy()).median(),
        n,
        np.nan,
    )
    facts["payees_median_payers"] = _per_account(
        pd.Series(in_degree[edges["r"]]).groupby(edges["s"].to_numpy()).median(),
        n,
        np.nan,
    )

    receipts = pd.DataFrame(
        {"account": receiver[~own], "time": times[~own]}
    ).sort_values("time", kind="stable")
    sends = pd.DataFrame(
        {"account": sender[~own], "next_send": times[~own]}
    ).sort_values("next_send", kind="stable")
    following = pd.merge_asof(
        receipts,
        sends,
        left_on="time",
        right_on="next_send",
        by="account",
        direction="forward",
    )
    hours = (following["next_send"] - following["time"]).dt.total_seconds() / 3600
    within = (hours <= DAY_HOURS).groupby(following["account"]).mean()
    facts["share_sent_on_within_day"] = _per_account(within, n, np.nan)
    facts["median_hours_to_send_on"] = _per_account(
        hours.groupby(following["account"]).median(), n, np.nan
    )

    converts = own & (
        table["paid_currency"].to_numpy() != table["received_currency"].to_numpy()
    )
    conversion_minutes = pd.DataFrame(
        {"account": sender[converts], "time": times[converts]}
    ).drop_duplicates()
    sent = pd.DataFrame({"account": sender[~own], "time": times[~own]})
    with_conversion = sent.merge(conversion_minutes, on=["account", "time"])
    facts["sends_with_conversion"] = np.bincount(
        with_conversion["account"], minlength=n
    ).astype(float)

    amounts = pd.DataFrame({"account": sender[~own], "cents": np.round(usd[~own], 2)})
    same = amounts.groupby(["account", "cents"]).size().groupby(level=0).max()
    facts["most_sends_same_amount"] = _per_account(same, n, 0.0)

    sides = pd.DataFrame(
        {
            "account": np.concatenate([sender, receiver[~own]]),
            "day": pd.to_datetime(np.concatenate([times, times[~own]])).floor("D"),
        }
    )
    per_day = sides.groupby(["account", "day"]).size()
    facts["active_days"] = _per_account(per_day.groupby(level=0).size(), n, 0.0)
    facts["busiest_day"] = _per_account(per_day.groupby(level=0).max(), n, 0.0)
    return pd.DataFrame(facts, index=accounts)[FACTS]


def _count(value: float, one: str, many: str) -> str:
    return f"{value:.0f} {one if value == 1 else many}"


def fact_lines(facts: pd.Series) -> str:
    """The facts of one account as the block the reader is shown."""
    both = _count(
        facts["both_ways"],
        "counterparty both pays A and is paid by A",
        "counterparties both pay A and are paid by A",
    )
    lines = [
        "FACTS COMPUTED FROM ALL TRANSACTIONS OF THE WINDOW, INCLUDING THOSE OF THE "
        "COUNTERPARTIES",
        f"- A is paid by {_count(facts['payers'], 'account', 'different accounts')} "
        f"and pays {_count(facts['payees'], 'account', 'different accounts')}.",
        f"- {both}.",
    ]
    if np.isnan(facts["share_sent_on_within_day"]):
        lines.append("- A receives no payments from other accounts.")
    elif np.isnan(facts["median_hours_to_send_on"]):
        lines.append(
            "- A sends nothing to other accounts after any payment it receives."
        )
    else:
        lines.append(
            f"- After {facts['share_sent_on_within_day'] * 100:.0f}% of the payments A "
            "receives, A sends money to another account within 24 hours; the median "
            f"time to its next send is {facts['median_hours_to_send_on']:.1f} hours."
        )
    lines.append(
        f"- {_count(facts['sends_with_conversion'], 'payment', 'payments')} of A to "
        "other accounts are made in the same minute as a currency conversion by A."
    )
    lines.append(
        f"- The most payments A sends with exactly the same amount: "
        f"{facts['most_sends_same_amount']:.0f}."
    )
    if not np.isnan(facts["payers_median_payees"]):
        lines.append(
            "- The accounts that pay A pay a median of "
            f"{facts['payers_median_payees']:.0f} different accounts each."
        )
    if not np.isnan(facts["payees_median_payers"]):
        lines.append(
            "- The accounts A pays are paid by a median of "
            f"{facts['payees_median_payers']:.0f} different accounts each."
        )
    lines.append(
        f"- A is active on {_count(facts['active_days'], 'day', 'days')}, with at "
        f"most {facts['busiest_day']:.0f} transactions in one day."
    )
    return "\n".join(lines)


# How each fact is named when it is compared with other accounts.
FACT_NAMES = {
    "payers": "different accounts that pay A",
    "payees": "different accounts A pays",
    "both_ways": "counterparties that both pay A and are paid by A",
    "share_sent_on_within_day": "share of received payments followed by a send "
    "within 24 hours",
    "median_hours_to_send_on": "median hours from a received payment to the next send",
    "sends_with_conversion": "payments sent in the same minute as a conversion",
    "most_sends_same_amount": "most payments sent with exactly the same amount",
    "payers_median_payees": "median number of accounts paid by each payer of A",
    "payees_median_payers": "median number of payers of each account A pays",
    "active_days": "active days",
    "busiest_day": "most transactions in one day",
}


def fact_percentiles(
    facts: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Where each fact of each account stands among the reference accounts.

    Two tables with the index and columns of `facts`: the share of reference accounts
    with a lower value, and the share with the same value. A reference account for
    which a fact does not apply (NaN) is left out of that fact; a NaN fact stays NaN.
    No label is used.
    """
    lower = pd.DataFrame(np.nan, index=facts.index, columns=facts.columns)
    same = lower.copy()
    for name in facts.columns:
        ref = np.sort(reference[name].dropna().to_numpy())
        values = facts[name].to_numpy()
        known = ~np.isnan(values)
        left = np.searchsorted(ref, values[known], side="left")
        right = np.searchsorted(ref, values[known], side="right")
        lower.loc[known, name] = left / len(ref)
        same.loc[known, name] = (right - left) / len(ref)
    return lower, same


def _share(value: float) -> str:
    percent = value * 100
    if 0 < percent < 1:
        return "less than 1%"
    if 99 < percent < 100:
        return f"{percent:.1f}%"
    return f"{percent:.0f}%"


def _value(name: str, value: float) -> str:
    if name == "share_sent_on_within_day":
        return f"{value * 100:.0f}%"
    if name == "median_hours_to_send_on":
        return f"{value:.1f}"
    return f"{value:.0f}"


def percentile_lines(
    facts: pd.Series,
    lower: pd.Series,
    same: pd.Series,
    low: pd.Series | None = None,
    high: pd.Series | None = None,
) -> str:
    """The facts of one account beside the other accounts, as the reader is shown.

    With `low` and `high` (from `typical_ranges`) each line also gives the typical
    range of clean accounts as busy as this one.
    """
    lines = [
        "HOW THESE FACTS COMPARE WITH THE OTHER ACCOUNTS OF THE BANK",
    ]
    for name in FACTS:
        if np.isnan(facts[name]):
            continue
        line = (
            f"- {FACT_NAMES[name]}: {_value(name, facts[name])}; higher than "
            f"{_share(lower[name])} of accounts, the same as {_share(same[name])}"
        )
        if low is not None and not np.isnan(low[name]):
            line += (
                f"; typical for clean accounts as busy as A: "
                f"{_value(name, low[name])} to {_value(name, high[name])}"
            )
        lines.append(line + ".")
    return "\n".join(lines)


# Bins of the number of transactions in the window, for "clean accounts as busy as A".
ACTIVITY_EDGES = [0, 1, 2, 4, 8, 16, 32, 64, np.inf]
TYPICAL_LOW, TYPICAL_HIGH = 0.1, 0.9


def activity_bins(activity: pd.Series) -> pd.Series:
    """The bin of each account's number of transactions (1, 2, 3-4, 5-8, ...)."""
    bins = pd.cut(activity, ACTIVITY_EDGES, right=True, labels=False)
    return bins.astype(int)


def typical_ranges(
    facts: pd.DataFrame,
    activity: pd.Series,
    reference_facts: pd.DataFrame,
    reference_activity: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 10th and 90th percentile of each fact among the reference accounts of the
    same activity bin, for every account of `facts`.

    The reference must hold only clean accounts outside the samples the reader is
    asked about: this is the one place where labels go into a prompt. A fact that does
    not apply to a reference account is left out of its range; a bin without any
    value gives NaN.
    """
    reference_bins = activity_bins(reference_activity)
    low = reference_facts.groupby(reference_bins).quantile(TYPICAL_LOW)
    high = reference_facts.groupby(reference_bins).quantile(TYPICAL_HIGH)
    bins = activity_bins(activity)
    low = low.reindex(bins.to_numpy()).set_axis(facts.index)[facts.columns]
    high = high.reindex(bins.to_numpy()).set_axis(facts.index)[facts.columns]
    return low, high
