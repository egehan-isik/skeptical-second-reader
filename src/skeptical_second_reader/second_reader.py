"""What the LLM second reader is shown, how it is asked, and how it answers.

Between the conditions of the first experiment the evidence block never changes: only
what the ML model is said to have concluded does. The evidence is chosen by a rule that
does not use the model, so the condition without ML output learns nothing from the
model. The reading test adds a block of computed facts (`facts.py`) to some prompts.
Labels, pattern names, folds and real account identities are never shown.
"""

import json
import math
import time
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

K1, K2 = "K1", "K2"
K3_D33, K3_D66, K3_D100 = "K3_d33", "K3_d66", "K3_d100"
SCORE_ONLY = "K3_score_only"
CONDITIONS = [K1, K2, K3_D33, K3_D66, K3_D100, SCORE_ONLY]
DOSES = {K2: 0.0, K3_D33: 0.33, K3_D66: 0.66, K3_D100: 1.0, SCORE_ONLY: 1.0}

STRONG, UNCERTAIN, NOT_FLAGGED = "STRONG ALERT", "UNCERTAIN ALERT", "NOT FLAGGED"
STRONG_SCORE = 0.9
SHOWN_ALERT_RATE = 0.5
# Where a corrupted rate moves to: the far end of the opposite decision.
LOW_TARGET, HIGH_TARGET = 0.01, 0.99
MAX_ROWS = 30
TOP_REASONS = 3

SUSPICIOUS, NOT_SUSPICIOUS = "suspicious", "not_suspicious"
OLLAMA_URL = "http://localhost:11434/api/chat"
NUM_CTX = 4096

GROUP_NAMES = [
    "strong, model right",
    "strong, model wrong",
    "uncertain, model right",
    "uncertain, model wrong",
    "missed, in the pattern file",
    "missed, outside the file",
    "not flagged, clean",
]


def true_level(flagged: bool, score: float) -> str:
    """The level of an account, from the flag and the raw score."""
    if not flagged:
        return NOT_FLAGGED
    return STRONG if score >= STRONG_SCORE else UNCERTAIN


def case_groups(
    flags: pd.Series, scores: pd.Series, labels: pd.Series, in_file: pd.Series
) -> pd.Series:
    """The experiment group of every account."""
    flagged, positive = flags == 1, labels == 1
    strong = flagged & (scores >= STRONG_SCORE)
    choices = [
        strong & positive,
        strong & ~positive,
        flagged & ~strong & positive,
        flagged & ~strong & ~positive,
        ~flagged & positive & in_file,
        ~flagged & positive & ~in_file,
        ~flagged & ~positive,
    ]
    groups = np.select(choices, GROUP_NAMES, default="")
    return pd.Series(groups, index=flags.index, name="group")


def sample_cases(
    groups: pd.Series,
    per_group: int,
    seed: int,
    exclude: pd.Index | None = None,
) -> pd.Series:
    """`per_group` accounts of every group, or all of a smaller group."""
    pool = groups if exclude is None else groups.drop(exclude, errors="ignore")
    parts = []
    for name in GROUP_NAMES:
        members = pool[pool == name]
        parts.append(members.sample(min(per_group, len(members)), random_state=seed))
    return pd.concat(parts)


@dataclass(frozen=True)
class ShownOutput:
    """The ML output as the reader sees it in one condition."""

    level: str
    rate: float
    n_high: int
    flipped: bool

    @property
    def alert(self) -> bool:
        return self.level != NOT_FLAGGED


def shown_output(
    flagged: bool, score: float, rate: float, n_high: int, condition: str
) -> ShownOutput | None:
    """The output shown in a condition; None if the condition hides the model.

    A dose moves the calibrated rate towards the far end of the opposite decision, so
    full dose always reverses the decision. The level follows the moved rate. A reversed
    output also reverses its count of high transactions and, in `reason_lines`, the
    direction of its reasons, so that the output stays consistent with itself.
    """
    if condition == K1:
        return None
    dose = DOSES[condition]
    if dose == 0:
        return ShownOutput(true_level(flagged, score), rate, n_high, False)
    target = LOW_TARGET if flagged else HIGH_TARGET
    moved = rate + dose * (target - rate)
    if moved >= STRONG_SCORE:
        level = STRONG
    elif moved >= SHOWN_ALERT_RATE:
        level = UNCERTAIN
    else:
        level = NOT_FLAGGED
    flipped = (level != NOT_FLAGGED) != flagged
    if flipped:
        n_high = 0 if flagged else max(n_high, 1)
    return ShownOutput(level, moved, n_high, flipped)


def account_rows(
    table: pd.DataFrame, pair_count: np.ndarray, account: str
) -> pd.DataFrame:
    """All transactions of one account, with their row in `table` and pair count."""
    mine = (table["sender"] == account) | (table["receiver"] == account)
    rows = table[mine].copy()
    rows["row"] = np.flatnonzero(mine.to_numpy())
    rows["pair_txs"] = pair_count[rows["row"].to_numpy()].astype(int)
    return rows


def account_evidence(rows: pd.DataFrame, account: str) -> tuple[str, dict[int, int]]:
    """The evidence block, and the number each shown table row gets.

    A busy account shows its largest transactions by dollar amount, a rule that does not
    use the model, in time order. Counterparties get names in order of appearance.
    """
    shown = rows.nlargest(MAX_ROWS, "usd").sort_values(["timestamp", "row"])
    names: dict[str, str] = {}

    def masked(other: str) -> str:
        if other not in names:
            names[other] = f"K{len(names) + 1}"
        return names[other]

    lines = []
    numbers = {}
    for number, row in enumerate(shown.itertuples(), start=1):
        numbers[int(row.row)] = number
        if row.sender == row.receiver:
            direction, other = "self", "A"
        elif row.sender == account:
            direction, other = "sent", masked(row.receiver)
        else:
            direction, other = "received", masked(row.sender)
        lines.append(
            f"{number},{row.timestamp:%b-%d %H:%M},{direction},{other},"
            f"{row.payment_format},{row.usd:.2f},{row.paid_currency},"
            f"{row.received_currency},{row.pair_txs}"
        )
    is_self = rows["sender"] == rows["receiver"]
    sent = rows[(rows["sender"] == account) & ~is_self]
    received = rows[(rows["receiver"] == account) & ~is_self]
    converted = is_self & (rows["paid_currency"] != rows["received_currency"])
    others = set(sent["receiver"]) | set(received["sender"])
    formats = rows["payment_format"].value_counts().sort_index()
    formats_text = ", ".join(f"{name} {count}" for name, count in formats.items())
    summary = (
        f"ACCOUNT A, 1-10 September: {len(rows)} transactions: {len(sent)} sent, "
        f"{len(received)} received, {int(is_self.sum())} to itself, of which "
        f"{int(converted.sum())} currency conversions. {len(others)} counterparties. "
        f"Sent {sent['usd'].sum():.2f} USD, received "
        f"{received['usd_received'].sum():.2f} USD. Formats: {formats_text}."
    )
    if len(rows) > len(shown):
        summary += f" Showing the {len(shown)} largest of {len(rows)} transactions."
    header = (
        "number,time,direction,counterparty,format,amount_usd,paid_currency,"
        "received_currency,transactions_in_this_direction_with_this_counterparty"
    )
    return "\n".join([summary, header, *lines]), numbers


def _describe(name: str, value: float, tx_format: str, paid: str) -> str:
    """One feature of a transaction in plain words."""
    missing = value is None or math.isnan(value)
    if name == "log_usd":
        return f"the amount is about {math.expm1(value):.0f} USD"
    if name == "payment_format":
        return f"the payment format is {tx_format}"
    if name == "paid_currency":
        return f"it is paid in {paid}"
    if name == "changes_currency":
        return "the currency changes" if value else "the currency does not change"
    if name == "is_self":
        return "the account pays itself" if value else "it goes to another account"
    if name == "same_bank":
        return "both ends are in one bank" if value else "the ends are in two banks"
    if name == "hour":
        return f"it is made at hour {value:.0f} of the day"
    if name == "weekday":
        return f"it is made on weekday {value:.0f} (0 is Monday)"
    if name == "sender_tx":
        return f"the sender has {value:.0f} sent transactions in the window"
    if name == "receiver_tx":
        return f"the receiver has {value:.0f} received transactions in the window"
    if name == "sender_out_degree":
        return f"the sender pays {value:.0f} different accounts"
    if name == "receiver_in_degree":
        return f"the receiver is paid by {value:.0f} different accounts"
    if name == "pair_count":
        return f"this sender paid this receiver {value:.0f} times in the window"
    if name == "log_usd_over_sender_median":
        return f"the amount is {math.exp(value):.2f} times the sender's usual amount"
    if name == "log_usd_over_receiver_median":
        return f"it is {math.exp(value):.2f} times the receiver's usual amount"
    if name == "after_conversion":
        if value:
            return (
                "the sender converted currency in the same minute, before this payment"
            )
        return "the sender did not convert currency in the same minute before it"
    if name == "sender_conversions":
        return f"the sender made {value:.0f} currency conversions in the window"
    if name == "hours_since_money_in":
        if missing:
            return "the sender received no money in the 24 hours before"
        return f"the sender received money {value:.1f} hours before"
    if name == "ratio_to_money_in":
        if missing:
            return "there is no money received before to compare the amount with"
        return f"the amount is {value:.2f} times the money received before"
    if name == "hours_to_next_send":
        if missing:
            return "the receiver sends nothing in the 24 hours after"
        return f"the receiver sends money {value:.1f} hours after"
    if name == "next_send_ratio":
        if missing:
            return "there is no later send of the receiver to compare the amount with"
        return f"the receiver's next send is {value:.2f} times this amount"
    raise ValueError(f"no description for feature {name}")


def reason_lines(
    contributions: pd.Series,
    values: pd.Series,
    tx_format: str,
    paid: str,
    flipped: bool,
) -> list[str]:
    """The largest SHAP contributions of a transaction, in plain words."""
    order = contributions.abs().sort_values(ascending=False, kind="stable").index
    lines = []
    for name in order[:TOP_REASONS]:
        towards_suspicious = (contributions[name] > 0) != flipped
        side = "suspicious" if towards_suspicious else "normal"
        text = _describe(name, float(values[name]), tx_format, paid)
        lines.append(f"- {text}: pushes towards {side}")
    return lines


def ml_block(
    shown: ShownOutput,
    strongest_number: int | None,
    strongest_text: str,
    reasons: list[str] | None,
) -> str:
    """What the upstream model is said to have concluded. No reasons, no second part."""
    if shown.rate < 0.01:
        percent = "under 1"
    elif shown.rate > 0.99:
        percent = "over 99"
    else:
        percent = f"about {shown.rate * 100:.0f}"
    parts = [
        "UPSTREAM ML MODEL OUTPUT",
        f"Level: {shown.level}.",
        f"Of similar accounts, {percent}% were truly suspicious.",
    ]
    if reasons is not None:
        where = (
            f"Its strongest transaction is number {strongest_number}."
            if strongest_number is not None
            else f"Its strongest transaction, not in the list above: {strongest_text}."
        )
        parts += [
            where,
            "Why the model scored that transaction as it did:",
            *reasons,
            f"Transactions of this account above the alert threshold: {shown.n_high}.",
        ]
    return "\n".join(parts)


def build_prompt(evidence: str, ml_text: str | None, facts: str | None = None) -> str:
    """The whole prompt. The instruction is the same in every condition.

    Without `facts` the prompt is exactly that of the first experiment, so its
    cached answers stay valid.
    """
    parts = [
        "You are assisting an anti-money-laundering analyst. Review one bank account "
        "and give your own assessment of whether it is involved in money laundering. "
        "Account names are masked.",
        evidence,
    ]
    if facts is not None:
        parts.append(facts)
    if ml_text is not None:
        parts.append(ml_text)
    parts.append(
        "Answer ONLY with JSON:\n"
        '{"decision": "suspicious" or "not_suspicious",\n'
        ' "confidence": a number from 0 to 1, how sure you are of your decision,\n'
        ' "evidence": a list of the transaction numbers you relied on,\n'
        ' "reason": "2 to 4 sentences for the analyst"}'
    )
    return "\n\n".join(parts)


def case_prompt(
    rows: pd.DataFrame,
    account: str,
    reading: pd.Series,
    contributions: pd.Series,
    values: pd.Series,
    condition: str,
) -> tuple[str, ShownOutput | None]:
    """The prompt for one account in one condition, and the output shown in it.

    `reading` holds flag, score, calibrated, n_high and strongest_row of the account.
    `contributions` and `values` are the SHAP values and features of that transaction.
    """
    evidence, numbers = account_evidence(rows, account)
    shown = shown_output(
        bool(reading["flag"] == 1),
        float(reading["score"]),
        float(reading["calibrated"]),
        int(reading["n_high"]),
        condition,
    )
    if shown is None:
        return build_prompt(evidence, None), None
    reasons, number, text = None, None, ""
    if condition != SCORE_ONLY:
        strongest_row = int(reading["strongest_row"])
        tx = rows[rows["row"] == strongest_row].iloc[0]
        reasons = reason_lines(
            contributions,
            values,
            tx["payment_format"],
            tx["paid_currency"],
            shown.flipped,
        )
        number = numbers.get(strongest_row)
        text = (
            f"a {tx['payment_format']} of {tx['usd']:.2f} USD "
            f"on {tx['timestamp']:%b-%d %H:%M}"
        )
    return build_prompt(evidence, ml_block(shown, number, text, reasons)), shown


def parse_answer(content: str) -> dict:
    """The fields of an answer; `decision` is None if it is not usable."""
    try:
        answer = json.loads(content)
    except json.JSONDecodeError:
        return {"decision": None, "confidence": None, "evidence": [], "reason": None}
    if not isinstance(answer, dict):
        return {"decision": None, "confidence": None, "evidence": [], "reason": None}
    decision = answer.get("decision")
    if decision not in (SUSPICIOUS, NOT_SUSPICIOUS):
        decision = None
    confidence = answer.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int | float)
        or not 0 <= confidence <= 1
    ):
        confidence = None
    evidence = answer.get("evidence")
    return {
        "decision": decision,
        "confidence": confidence,
        "evidence": evidence if isinstance(evidence, list) else [],
        "reason": answer.get("reason"),
    }


def follows_model(decision: str | None, shown: ShownOutput | None) -> bool | None:
    """Whether the decision matches the output shown; never self-reported."""
    if shown is None or decision is None:
        return None
    return (decision == SUSPICIOUS) == shown.alert


def ask_ollama(
    prompt: str, model: str, think: bool | None = None, timeout: int = 1800
) -> dict:
    """One call to the local model at temperature 0: JSON text and token counts.

    `think` switches a model's thinking mode; None leaves the model's default.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "keep_alive": "60m",
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }
    if think is not None:
        body["think"] = think
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.loads(response.read().decode("utf-8"))
    return {
        "content": reply["message"]["content"],
        "prompt_tokens": reply.get("prompt_eval_count"),
        "answer_tokens": reply.get("eval_count"),
        "seconds": time.time() - started,
    }
