"""One review flow: from a window of transactions to the reader's two-step opinion.

A monitoring system calls `review` with the transactions of a window and the accounts
it wants reviewed. The transaction features look at the whole window (counts, degrees,
pair counts, money in and out), so the window is passed whole, not one account's rows.

The frozen model is the model of one cross-validation round, trained exactly as in that
round and saved with the round's threshold and calibration curve. It may only review the
accounts of that round's test fold: every other account was trained on or used for early
stopping. The experiments do not use this module; their scores stay out of fold.

The reader decides twice in one conversation: first from the evidence alone, then after
it is shown the ML output, knowing its first answer.
"""

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from skeptical_second_reader import second_reader as sr
from skeptical_second_reader.explain import shap_values
from skeptical_second_reader.model import fit_xgboost, precision_threshold
from skeptical_second_reader.pipeline import AnswerCache, stage_key
from skeptical_second_reader.split import TEST, VAL, fold_roles
from skeptical_second_reader.transactions import (
    TX_FEATURES,
    FitFn,
    ThresholdFn,
    account_positions,
    fit_rows,
    noisy_or,
    strongest_transaction,
)

MODEL_FILE = "model.json"
META_FILE = "meta.json"
REVIEWABLE_FILE = "reviewable.parquet"

ANSWER_FORMAT = (
    "Answer ONLY with JSON:\n"
    '{"decision": "suspicious" or "not_suspicious",\n'
    ' "confidence": a number from 0 to 1, how sure you are of your decision,\n'
    ' "evidence": a list of the transaction numbers you relied on,\n'
    ' "reason": "2 to 4 sentences for the analyst"}'
)
SECOND_INTRO = (
    "You gave the assessment above before seeing the output of the bank's ML model. "
    "Here it is."
)
SECOND_ASK = (
    "Give your final assessment of the same account. You may keep or change your "
    "decision."
)

AskFn = Callable[..., dict]


@dataclass
class FrozenModel:
    """The model of one round, with what is needed to read its scores.

    curve_x, curve_y: the isotonic curve of that round's validation accounts, as its
        thresholds; between them the curve is linear, beyond them it is clipped.
    reviewable: the accounts of that round's test fold.
    """

    model: xgb.XGBClassifier
    threshold: float
    curve_x: np.ndarray
    curve_y: np.ndarray
    round: int
    reviewable: pd.Index

    def calibrate(self, values: np.ndarray) -> np.ndarray:
        return np.interp(values, self.curve_x, self.curve_y)


def freeze_round(
    features: pd.DataFrame,
    tx_labels: pd.Series,
    table: pd.DataFrame,
    y: pd.Series,
    folds: pd.Series,
    k: int,
    fit: FitFn = fit_xgboost,
    choose_threshold: ThresholdFn = precision_threshold,
) -> FrozenModel:
    """Train round k as `transaction_cv` does; keep its model, threshold and curve."""
    accounts = folds.index
    sender, receiver = account_positions(table, accounts)
    roles = fold_roles(folds, k).to_numpy()
    train_rows, val_rows = fit_rows(roles, sender, receiver)
    score = fit(
        features[train_rows],
        tx_labels[train_rows],
        features[val_rows],
        tx_labels[val_rows],
    )
    values = noisy_or(score(features), sender, receiver, len(accounts))
    val = roles == VAL
    threshold = float(choose_threshold(y[val], values[val]))
    curve = IsotonicRegression(out_of_bounds="clip").fit(values[val], y[val])
    return FrozenModel(
        score.model,
        threshold,
        np.asarray(curve.X_thresholds_, dtype=float),
        np.asarray(curve.y_thresholds_, dtype=float),
        k,
        accounts[roles == TEST],
    )


def save_frozen(frozen: FrozenModel, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    frozen.model.save_model(folder / MODEL_FILE)
    meta = {
        "threshold": frozen.threshold,
        "curve_x": frozen.curve_x.tolist(),
        "curve_y": frozen.curve_y.tolist(),
        "round": frozen.round,
        "best_iteration": int(frozen.model.best_iteration),
    }
    (folder / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    frozen.reviewable.to_frame(index=False, name="account").to_parquet(
        folder / REVIEWABLE_FILE
    )


def load_frozen(folder: Path) -> FrozenModel:
    meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
    model = xgb.XGBClassifier()
    model.load_model(folder / MODEL_FILE)
    if int(model.best_iteration) != meta["best_iteration"]:
        raise ValueError("the saved model lost its best iteration")
    reviewable = pd.Index(
        pd.read_parquet(folder / REVIEWABLE_FILE)["account"], name="account"
    )
    return FrozenModel(
        model,
        float(meta["threshold"]),
        np.asarray(meta["curve_x"], dtype=float),
        np.asarray(meta["curve_y"], dtype=float),
        int(meta["round"]),
        reviewable,
    )


def score_window(
    frozen: FrozenModel, table: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Score, calibrated rate, flag, strongest row and high count of every account of
    the window, with the formulas of `transaction_cv`."""
    accounts = pd.Index(
        sorted(set(table["sender"]) | set(table["receiver"])), name="account"
    )
    sender, receiver = account_positions(table, accounts)
    tx_scores = frozen.model.predict_proba(features[TX_FEATURES])[:, 1]
    values = noisy_or(tx_scores, sender, receiver, len(accounts))
    high = tx_scores >= frozen.threshold
    counted = np.zeros(len(accounts), dtype="int64")
    np.add.at(counted, sender[high], 1)
    np.add.at(counted, receiver[high & (sender != receiver)], 1)
    return pd.DataFrame(
        {
            "score": values,
            "calibrated": frozen.calibrate(values),
            "flag": (values >= frozen.threshold).astype(int),
            "strongest_row": strongest_transaction(
                tx_scores, sender, receiver, len(accounts)
            ),
            "n_high": counted,
        },
        index=accounts,
    )


def check_reviewable(frozen: FrozenModel, accounts: list[str]) -> None:
    outside = [a for a in accounts if a not in frozen.reviewable]
    if outside:
        raise ValueError(
            f"{len(outside)} account(s) are outside the test fold of round "
            f"{frozen.round} and were seen in training: {outside[:5]}"
        )


def ml_text(
    rows: pd.DataFrame,
    account: str,
    reading: pd.Series,
    contributions: pd.Series,
    values: pd.Series,
) -> tuple[str, sr.ShownOutput, list[str]]:
    """The true ML output as the reader is shown it, as in condition K2."""
    _, numbers = sr.account_evidence(rows, account)
    shown = sr.shown_output(
        bool(reading["flag"] == 1),
        float(reading["score"]),
        float(reading["calibrated"]),
        int(reading["n_high"]),
        sr.K2,
    )
    strongest_row = int(reading["strongest_row"])
    tx = rows[rows["row"] == strongest_row].iloc[0]
    reasons = sr.reason_lines(
        contributions, values, tx["payment_format"], tx["paid_currency"], False
    )
    text = (
        f"a {tx['payment_format']} of {tx['usd']:.2f} USD "
        f"on {tx['timestamp']:%b-%d %H:%M}"
    )
    block = sr.ml_block(shown, numbers.get(strongest_row), text, reasons)
    return block, shown, reasons


def first_messages(evidence: str) -> list[dict]:
    """Step one: the prompt of condition K1, unchanged."""
    return [{"role": "user", "content": sr.build_prompt(evidence, None)}]


def second_messages(first: list[dict], first_answer: str, ml: str) -> list[dict]:
    """Step two: the same conversation, the first answer, then the ML output."""
    return [
        *first,
        {"role": "assistant", "content": first_answer},
        {
            "role": "user",
            "content": "\n\n".join([SECOND_INTRO, ml, SECOND_ASK, ANSWER_FORMAT]),
        },
    ]


def ask_chat(
    messages: list[dict], model: str, think: bool | None = None, timeout: int = 1800
) -> dict:
    """One call to the local model with a whole conversation, as `sr.ask_ollama`."""
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": "json",
        "keep_alive": "60m",
        "options": {"temperature": 0, "num_ctx": sr.NUM_CTX},
    }
    if think is not None:
        body["think"] = think
    request = urllib.request.Request(
        sr.OLLAMA_URL,
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


def chat_key(messages: list[dict], digest: str, think: bool | None) -> str:
    return stage_key(
        messages=json.dumps(messages, sort_keys=True),
        digest=digest,
        think=think,
        num_ctx=sr.NUM_CTX,
        temperature=0,
    )


def cached_ask(
    messages: list[dict],
    model: str,
    digest: str,
    think: bool | None,
    cache: AnswerCache,
    ask: AskFn,
) -> tuple[dict, bool]:
    key = chat_key(messages, digest, think)
    reply = cache.get(key)
    if reply is not None:
        return reply, True
    reply = ask(messages, model, think=think)
    cache.put(key, model, digest, reply)
    return reply, False


def review(
    accounts: list[str],
    table: pd.DataFrame,
    features: pd.DataFrame,
    frozen: FrozenModel,
    models: dict[str, str],
    cache: AnswerCache,
    ask: AskFn = ask_chat,
    think: bool | None = False,
) -> list[dict]:
    """The opinion on every account: ML output, and each reader's two decisions.

    `models` maps a model name to its digest. Nothing in the result comes from a label.
    """
    check_reviewable(frozen, accounts)
    scored = score_window(frozen, table, features)
    strongest = scored.loc[accounts, "strongest_row"].to_numpy()
    contributions = shap_values(frozen.model, features.iloc[strongest][TX_FEATURES])
    contributions.index = accounts
    pair_count = features["pair_count"].to_numpy()
    results = []
    for account in accounts:
        reading = scored.loc[account]
        rows = sr.account_rows(table, pair_count, account)
        evidence, _ = sr.account_evidence(rows, account)
        row = int(reading["strongest_row"])
        ml, shown, reasons = ml_text(
            rows,
            account,
            reading,
            contributions.loc[account, TX_FEATURES],
            features.iloc[row],
        )
        readers, trace = {}, []
        for name, digest in models.items():
            first = first_messages(evidence)
            reply1, cached1 = cached_ask(first, name, digest, think, cache, ask)
            second = second_messages(first, reply1["content"], ml)
            reply2, cached2 = cached_ask(second, name, digest, think, cache, ask)
            before = sr.parse_answer(reply1["content"])
            after = sr.parse_answer(reply2["content"])
            readers[name] = {
                "first": before,
                "final": after,
                "changed": None
                if None in (before["decision"], after["decision"])
                else before["decision"] != after["decision"],
                "final_follows_model": sr.follows_model(after["decision"], shown),
            }
            for step, reply, cached in (
                ("first", reply1, cached1),
                ("final", reply2, cached2),
            ):
                trace.append(
                    {
                        "model": name,
                        "step": step,
                        "cached": cached,
                        "prompt_tokens": reply["prompt_tokens"],
                        "answer_tokens": reply["answer_tokens"],
                        "seconds": reply["seconds"],
                    }
                )
        results.append(
            {
                "account": account,
                "ml": {
                    "level": shown.level,
                    "rate": float(reading["calibrated"]),
                    "score": float(reading["score"]),
                    "flag": int(reading["flag"]),
                    "threshold": frozen.threshold,
                    "high_transactions": int(reading["n_high"]),
                    "reasons": reasons,
                },
                "readers": readers,
                "trace": trace,
            }
        )
    return results
