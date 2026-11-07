"""What the Streamlit demo shows, without Streamlit, so that it can be tested.

The demo reads the files of the batch pipeline. Accounts appear as numbered cases with
masked counterparties, and labels stay out of the analyst's view. Answers come from the
pipeline's cache, so the demo also works with the local model switched off.
"""

import json
import os
import re
import sqlite3
import time
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd

from skeptical_second_reader import pipeline as pl
from skeptical_second_reader import second_reader as sr
from skeptical_second_reader.paths import DATA_INTERIM

DIRECTORY_VARIABLE = "SSR_PIPELINE_DIR"
PROMPTS, READING, ANSWERS = "prompts.parquet", "reading.parquet", "answers.sqlite"
DECISIONS = "analyst_decisions.sqlite"
RESULTS = "results.parquet"
TAGS_URL = "http://localhost:11434/api/tags"
CASE_ORDER_SEED = 7
DOSE_ORDER = [sr.K2, sr.K3_D33, sr.K3_D66, sr.K3_D100, sr.SCORE_ONLY]
LEVEL_COLORS = {sr.STRONG: "red", sr.UNCERTAIN: "orange", sr.NOT_FLAGGED: "green"}
# Beside the words, so that the level never depends on colour alone.
LEVEL_MARKS = {sr.STRONG: "🔴", sr.UNCERTAIN: "🟠", sr.NOT_FLAGGED: "🟢"}
TABLE_LABELS = {
    "number": "No.",
    "time": "Time",
    "direction": "Direction",
    "counterparty": "Counterparty",
    "format": "Format",
    "amount_usd": "Amount (USD)",
    "paid_currency": "Paid in",
    "received_currency": "Received in",
    "transactions_in_this_direction_with_this_counterparty": "Same-pair transactions",
}
GLOSSARY = {
    "Counterparty": "the account on the other side of a transaction. Real identities "
    "are hidden: K1, K2 and so on are names used in this case only.",
    "To itself": "the account pays itself. If the two currencies differ, this is a "
    "currency conversion.",
    "Same-pair transactions": "how many transactions in the window went the same way "
    "between these two accounts. 1 means a one-time payment.",
    "ACH": "an automated bank transfer. Cash, Cheque, Wire, Credit Card, Bitcoin and "
    "Reinvestment are the file's other payment formats.",
    "Amount (USD)": "converted at a rate derived from the window's own transactions, "
    "so amounts can be compared across currencies.",
    "Level": "NOT FLAGGED below the ML model's threshold; above it, STRONG ALERT for a "
    "score of at least 0.9 and UNCERTAIN ALERT otherwise.",
}
SUMMARY_PATTERN = re.compile(
    r"(?P<transactions>\d+) transactions: (?P<sent>\d+) sent, (?P<received>\d+) "
    r"received, (?P<to_itself>\d+) to itself, of which (?P<conversions>\d+) currency "
    r"conversions\. (?P<counterparties>\d+) counterparties\. Sent "
    r"(?P<sent_usd>[\d.]+) USD, received (?P<received_usd>[\d.]+) USD\. Formats: "
    r"(?P<formats>.*?)\.(?: Showing the (?P<shown>\d+) largest of \d+ transactions\.)?$"
)


def data_directory() -> Path:
    """The pipeline's folder; the variable lets tests point somewhere else."""
    return Path(os.environ.get(DIRECTORY_VARIABLE, DATA_INTERIM / "pipeline"))


def load_cases(directory: Path) -> pd.DataFrame:
    """One row per case: its name and the level of the true ML output."""
    prompts = pd.read_parquet(directory / PROMPTS)
    true_output = prompts[prompts["condition"] == sr.K2]
    # A fixed random order: the sample is sorted by group, and a case number must
    # not reveal whether the model was right.
    cases = true_output[["account", "shown_level"]].sample(
        frac=1, random_state=CASE_ORDER_SEED
    )
    cases = cases.reset_index(drop=True)
    cases.insert(0, "case", [f"Case {i:03d}" for i in range(1, len(cases) + 1)])
    return cases.rename(columns={"shown_level": "level"})


def split_prompt(prompt: str) -> dict:
    """The parts of a prompt: summary, transactions and the ML text, if there is one."""
    parts = prompt.split("\n\n")
    lines = parts[1].split("\n")
    transactions = pd.read_csv(StringIO("\n".join(lines[1:])))
    has_model = len(parts) == 4
    return {
        "summary": lines[0],
        "transactions": transactions,
        "ml_text": parts[2] if has_model else None,
    }


def parse_summary(summary: str) -> dict:
    """The numbers of the evidence's first line, for showing them one by one."""
    match = SUMMARY_PATTERN.search(summary)
    if match is None:
        raise ValueError("the summary line has an unexpected form")
    values = match.groupdict()
    parsed = {
        name: int(values[name])
        for name in [
            "transactions",
            "sent",
            "received",
            "to_itself",
            "conversions",
            "counterparties",
        ]
    }
    parsed["sent_usd"] = float(values["sent_usd"])
    parsed["received_usd"] = float(values["received_usd"])
    parsed["formats"] = values["formats"]
    parsed["shown"] = (
        int(values["shown"]) if values["shown"] else parsed["transactions"]
    )
    return parsed


def ml_details(ml_text: str) -> str:
    """The ML text without its heading and level, which the page shows on its own."""
    lines = ml_text.split(chr(10))
    return chr(10).join(line for line in lines[1:] if not line.startswith("Level: "))


def case_prompt(prompts: pd.DataFrame, account: str, condition: str) -> pd.Series:
    """The row of one account in one condition."""
    chosen = prompts[
        (prompts["account"] == account) & (prompts["condition"] == condition)
    ]
    if len(chosen) != 1:
        raise ValueError(f"expected one prompt, found {len(chosen)}")
    return chosen.iloc[0]


def known_digest(cache: pl.AnswerCache, model: str) -> str | None:
    """The version of a model in the cache, so answers can be found with Ollama off."""
    row = cache.connection.execute(
        "SELECT digest FROM answers WHERE model = ? LIMIT 1", (model,)
    ).fetchone()
    return None if row is None else row[0]


def cached_opinion(directory: Path, prompt: str, model: str) -> dict | None:
    """The stored answer to this exact prompt, parsed, or None."""
    path = directory / ANSWERS
    if not path.exists():
        return None
    cache = pl.AnswerCache(path)
    try:
        digest = known_digest(cache, model)
        if digest is None:
            return None
        reply = cache.get(pl.answer_key(prompt, digest, False))
    finally:
        cache.close()
    if reply is None:
        return None
    return {**sr.parse_answer(reply["content"]), "seconds": reply["seconds"]}


def installed_digest(model: str) -> str:
    with urllib.request.urlopen(TAGS_URL, timeout=10) as response:
        models = json.loads(response.read().decode("utf-8"))["models"]
    return next(entry["digest"] for entry in models if entry["name"] == model)


def live_opinion(directory: Path, prompt: str, model: str, ask=sr.ask_ollama) -> dict:
    """Ask the local model and keep the answer in the same cache as the pipeline."""
    digest = installed_digest(model)
    cache = pl.AnswerCache(directory / ANSWERS)
    try:
        key = pl.answer_key(prompt, digest, False)
        reply = cache.get(key)
        if reply is None:
            reply = ask(prompt, model, think=False)
            cache.put(key, model, digest, reply)
    finally:
        cache.close()
    return {**sr.parse_answer(reply["content"]), "seconds": reply["seconds"]}


def agreement_text(decision: str | None, level: str | None) -> str | None:
    """How the LLM's decision relates to the ML level shown beside it."""
    if decision is None or level is None:
        return None
    shown = sr.ShownOutput(level, 0.0, 0, False)
    if sr.follows_model(decision, shown):
        return "The second reader agrees with the ML model."
    return "The second reader disagrees with the ML model."


def models_comparison(decisions: list[str | None]) -> str | None:
    """Whether the models decided alike; None until every model has a decision."""
    if not decisions or any(decision is None for decision in decisions):
        return None
    if len(set(decisions)) == 1:
        return "The two models reach the same decision."
    return "The two models reach different decisions."


def save_decision(directory: Path, case: str, decision: str, note: str) -> None:
    """The analyst's decision; a new decision on a case replaces the old one."""
    connection = sqlite3.connect(directory / DECISIONS)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS decisions "
            "(case_name TEXT PRIMARY KEY, decision TEXT, note TEXT, saved_at TEXT)"
        )
        connection.execute(
            "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?)",
            (case, decision, note, time.strftime("%Y-%m-%d %H:%M")),
        )
        connection.commit()
    finally:
        connection.close()


def load_decisions(directory: Path) -> pd.DataFrame:
    columns = ["case_name", "decision", "note", "saved_at"]
    path = directory / DECISIONS
    if not path.exists():
        return pd.DataFrame(columns=columns)
    connection = sqlite3.connect(path)
    try:
        return pd.read_sql_query(
            f"SELECT {', '.join(columns)} FROM decisions", connection
        )
    finally:
        connection.close()


def progress_table(directory: Path) -> pd.DataFrame:
    """Finished answers per model, from the results file."""
    path = directory / RESULTS
    if not path.exists():
        return pd.DataFrame(columns=["model", "answers"])
    results = pd.read_parquet(path, columns=["model"])
    counts = results.groupby("model", sort=False).size()
    return counts.rename("answers").reset_index()


def dose_table(results: pd.DataFrame, reading: pd.DataFrame) -> pd.DataFrame:
    """Among accounts decided correctly in K1: the share wrong in each other condition.

    An account leaves a comparison if either of its two answers is invalid.
    """
    truth = reading["label"].map({1: sr.SUSPICIOUS, 0: sr.NOT_SUSPICIOUS})
    frame = results.assign(truth=results["account"].map(truth))
    rows = []
    for model, part in frame.groupby("model", sort=False):
        alone = part[part["condition"] == sr.K1].set_index("account")
        correct = alone.index[alone["decision"] == alone["truth"]]
        for condition in DOSE_ORDER:
            other = part[part["condition"] == condition].set_index("account")
            shared = other.index.intersection(correct)
            shared = shared[other.loc[shared, "decision"].notna()]
            wrong = other.loc[shared, "decision"] != other.loc[shared, "truth"]
            rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "accounts": len(shared),
                    "turned_wrong": int(wrong.sum()),
                    "share": wrong.mean() if len(shared) else float("nan"),
                }
            )
    return pd.DataFrame(rows)
