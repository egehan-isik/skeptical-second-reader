"""The guide and the solved examples of the guided reading conditions.

Every rate in the guide was measured on the reference accounts: all accounts minus the
two reading samples (notebook 15, section 1, recomputes each one). The examples are
reference accounts too, chosen by a rule on their facts, so no account the reader is
asked about is shown as an example and no example is picked by how a model answered.
"""

import numpy as np
import pandas as pd

from skeptical_second_reader.facts import fact_percentiles, typical_ranges
from skeptical_second_reader.second_reader import shot_block

GUIDE = """GUIDE: WHAT SEPARATES LAUNDERING ACCOUNTS IN THIS BANK
These rates were measured on other accounts of this bank, never on the account below.
- About 1 account in 85 (1.2%) is involved in money laundering. Most accounts are \
clean, so answer suspicious only when you see one of the signs below.
- Not signs in this bank, because clean accounts show them just as often (among \
accounts with 10 or more transactions, about 2% are involved with or without them): \
many transactions, sending the same amount many times, sending money on within 24 \
hours of receiving it, large amounts, cash, cheques or credit cards.
- Strongest sign: a payment to another account in the same minute as a currency \
conversion by A (in the table, a "self" row whose paid and received currencies differ \
and a "sent" row at the same time). Of accounts with exactly one such payment, 36% are \
involved; of those that also pay at most one account, 69%. With two or more such \
payments the rate falls to 3-9%.
- Weaker signs, each about 9% involved (7 times the usual rate): a counterparty that \
both pays A and is paid by A; 9 or more different accounts paying A.
- A value inside the typical range of clean accounts as busy as A is not a sign."""

# The rule each example is chosen by, whether it is involved, and the line on why.
SHOT_RULES = [
    (
        "strong sign",
        True,
        "Exactly one payment to another account in the same minute as a currency "
        "conversion, and few payees: the strongest sign in this bank.",
    ),
    (
        "weaker sign",
        True,
        "No conversion-linked payment, but a counterparty on both sides and many "
        "payers: two weaker signs together.",
    ),
    (
        "busy clean",
        False,
        "Busy, with repeated amounts and money sent on within a day, but these are "
        "as common in clean accounts; none of the signs is present.",
    ),
    (
        "many conversions",
        False,
        "Several payments in the same minute as a conversion: the strong sign is "
        "exactly one such payment, and with three or more the rate is low.",
    ),
]


def shot_candidates(facts: pd.DataFrame, labels: pd.Series, activity: pd.Series):
    """For each rule, which accounts satisfy it (a boolean Series per rule name)."""
    f = facts
    conv = f["sends_with_conversion"]
    no_weak = (f["both_ways"] == 0) & (f["payers"] < 9)
    positive, clean = labels == 1, labels == 0
    return {
        "strong sign": positive & (conv == 1) & (f["payees"] <= 1),
        "weaker sign": positive
        & (conv == 0)
        & (f["both_ways"] >= 1)
        & (f["payers"] >= 9),
        "busy clean": clean
        & (activity >= 10)
        & (conv == 0)
        & no_weak
        & (f["most_sends_same_amount"] >= 5)
        & (f["share_sent_on_within_day"] >= 0.8),
        "many conversions": clean & (conv >= 3) & no_weak,
    }


def choose_shots(
    facts: pd.DataFrame,
    labels: pd.Series,
    activity: pd.Series,
    exclude: pd.Index,
    seed: int,
) -> pd.Series:
    """One account per rule of `SHOT_RULES`, never one in `exclude`.

    `exclude` must hold both reading samples and the accounts of earlier experiments.
    Returns the rule name of each chosen account, in the order of the rules.
    """
    pool = facts.index.difference(exclude)
    masks = shot_candidates(facts.loc[pool], labels.loc[pool], activity.loc[pool])
    rng = np.random.default_rng(seed)
    chosen = {}
    for name, _, _ in SHOT_RULES:
        candidates = pool[masks[name].to_numpy()]
        if len(candidates) == 0:
            raise ValueError(f"no account satisfies the rule {name}")
        chosen[candidates[rng.integers(len(candidates))]] = name
    return pd.Series(chosen, name="rule")


def shot_summary(facts: pd.Series, transactions: int) -> str:
    """One line with the facts the guide talks about; the prompt limit leaves no room
    for the whole facts block of four other accounts."""
    within = facts["share_sent_on_within_day"]
    sent_on = (
        "receives nothing"
        if np.isnan(within)
        else f"{within * 100:.0f}% of received payments sent on within 24 hours"
    )
    return (
        f"{transactions} transactions; paid by {facts['payers']:.0f} accounts, pays "
        f"{facts['payees']:.0f}; {facts['both_ways']:.0f} counterparties on both "
        f"sides; {facts['sends_with_conversion']:.0f} payments in the same minute as "
        f"a conversion; at most {facts['most_sends_same_amount']:.0f} payments of the "
        f"same amount; {sent_on}."
    )


def shots_text(facts: pd.DataFrame, shots: pd.Series, activity: pd.Series) -> str:
    """The solved-examples block: a summary of each chosen account, answer and why."""
    answers = {name: (involved, why) for name, involved, why in SHOT_RULES}
    examples = [
        (shot_summary(facts.loc[account], int(activity[account])), *answers[rule])
        for account, rule in shots.items()
    ]
    return shot_block(examples)


def guided_inputs(
    facts: pd.DataFrame,
    labels: pd.Series,
    activity: pd.Series,
    samples: pd.Index,
    used: pd.Index,
    seed: int,
) -> dict:
    """The keyword arguments of `pipeline.reading_prompts` for the guided conditions.

    The reference is every account outside the reading `samples`: all of them for the
    percentiles (no label), only their clean accounts for the typical range. The
    examples come from neither the samples nor the earlier experiments (`used`).
    """
    reference = facts.index.difference(samples)
    clean = reference[labels.loc[reference].to_numpy() == 0]
    shots = choose_shots(facts, labels, activity, samples.union(used), seed=seed)
    return {
        "guide": GUIDE,
        "shots": shots_text(facts, shots, activity),
        "percentiles": fact_percentiles(facts, facts.loc[reference]),
        "typical": typical_ranges(
            facts, activity.loc[facts.index], facts.loc[clean], activity.loc[clean]
        ),
    }
