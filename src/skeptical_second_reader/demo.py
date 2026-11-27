"""What the Streamlit demo shows, without Streamlit, so that it can be tested.

The demo reads the files of the batch pipeline. Accounts appear as numbered cases with
masked counterparties, and labels stay out of the analyst's view. Answers come from the
pipeline's cache, so the demo also works with the local model switched off.
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from skeptical_second_reader import fresh
from skeptical_second_reader import pipeline as pl
from skeptical_second_reader import second_reader as sr
from skeptical_second_reader.paths import DATA_INTERIM, PROJECT_ROOT

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


def reversal_table(results: pd.DataFrame, reading: pd.DataFrame) -> pd.DataFrame:
    """Full reversal per ML outcome group, as in the report (notebook 11).

    Counted are the accounts a model decides correctly alone (K1) and whose reversed
    output (K3 at 100%) is wrong, with a valid answer there; each cell says how many
    of them changed to the wrong answer. In a group where the ML model is wrong the
    reversed output is right, so nothing is counted there.
    """
    truth = reading["label"].map({1: sr.SUSPICIOUS, 0: sr.NOT_SUSPICIOUS})
    frame = results.assign(truth=results["account"].map(truth))
    models = list(frame["model"].drop_duplicates())
    totals = {model: [0, 0] for model in models}
    rows = []
    for group in sr.GROUP_NAMES:
        part = frame[frame["group"] == group]
        if part.empty:
            continue
        full = part[part["condition"] == sr.K3_D100]
        shown = np.where(
            full["shown_level"] == sr.NOT_FLAGGED, sr.NOT_SUSPICIOUS, sr.SUSPICIOUS
        )
        wrong_output = full[shown != full["truth"].to_numpy()]
        row = {
            "ML outcome group": group,
            "shown after reversal": (
                "not flagged"
                if (full["shown_level"] == sr.NOT_FLAGGED).all()
                else "alert"
            ),
        }
        for model in models:
            alone = part[(part["model"] == model) & (part["condition"] == sr.K1)]
            right = set(alone.loc[alone["decision"] == alone["truth"], "account"])
            counted = wrong_output[
                (wrong_output["model"] == model)
                & wrong_output["account"].isin(right)
                & wrong_output["decision"].notna()
            ]
            if wrong_output.empty:
                row[model] = "output right: not counted"
                continue
            k = int((counted["decision"] != counted["truth"]).sum())
            row[model] = f"{k} of {len(counted)}"
            totals[model][0] += k
            totals[model][1] += len(counted)
        rows.append(row)
    total = {"ML outcome group": "Total", "shown after reversal": ""}
    total |= {model: f"{k} of {n}" for model, (k, n) in totals.items()}
    return pd.DataFrame([*rows, total])


def ml_overview(reading: pd.DataFrame) -> pd.DataFrame:
    """The ML model on all accounts, out-of-fold, at its own threshold: AUC-PR as mean
    and standard deviation over the five test folds, precision and recall pooled."""
    y, flag = reading["label"].astype(bool), reading["flag"].astype(bool)
    per_fold = [
        average_precision_score(part["label"], part["score"])
        for _, part in reading.groupby("fold")
    ]
    return pd.DataFrame(
        [
            {
                "accounts": len(reading),
                "positives": int(y.sum()),
                "positive rate": y.mean(),
                "flagged": int(flag.sum()),
                "AUC-PR (mean of folds)": float(np.mean(per_fold)),
                "AUC-PR (SD of folds)": float(np.std(per_fold, ddof=1)),
                "precision": (flag & y).sum() / flag.sum(),
                "recall": (flag & y).sum() / y.sum(),
            }
        ]
    )


def reversal_by_direction(results: pd.DataFrame, reading: pd.DataFrame) -> pd.DataFrame:
    """The counts of `reversal_table`, by the direction of the wrong output shown."""
    truth = reading["label"].map({1: sr.SUSPICIOUS, 0: sr.NOT_SUSPICIOUS})
    frame = results.assign(truth=results["account"].map(truth))
    full = frame[frame["condition"] == sr.K3_D100]
    shown = np.where(
        full["shown_level"] == sr.NOT_FLAGGED, sr.NOT_SUSPICIOUS, sr.SUSPICIOUS
    )
    wrong_output = full[(shown != full["truth"].to_numpy()) & full["decision"].notna()]
    directions = {
        sr.NOT_SUSPICIOUS: "'not flagged' shown for a positive account",
        sr.SUSPICIOUS: "'alert' shown for a clean account",
    }
    rows = {name: {"wrong output shown": name} for name in directions.values()}
    total = {"wrong output shown": "Total"}
    for model, part in frame.groupby("model", sort=False):
        alone = part[part["condition"] == sr.K1]
        right = set(alone.loc[alone["decision"] == alone["truth"], "account"])
        counted = wrong_output[
            (wrong_output["model"] == model) & wrong_output["account"].isin(right)
        ]
        turned = counted["decision"] != counted["truth"]
        side = counted["truth"].map(
            {sr.SUSPICIOUS: sr.NOT_SUSPICIOUS, sr.NOT_SUSPICIOUS: sr.SUSPICIOUS}
        )
        for key, name in directions.items():
            mask = side == key
            rows[name][model] = f"{int(turned[mask].sum())} of {int(mask.sum())}"
        total[model] = f"{int(turned.sum())} of {len(counted)}"
    return pd.DataFrame([*rows.values(), total])


def reader_metrics(strata: pd.Series, readers: dict) -> pd.DataFrame:
    """`fresh.reader_table` with precision and the number of accounts called
    suspicious. Precision depends on the sample's share of positives."""
    table = fresh.reader_table(strata, readers)
    truth = fresh.truth_of(strata)
    flagged, precision = [], []
    for _, flag in readers.values():
        y = truth.loc[flag.index].to_numpy() == 1
        said = flag.to_numpy()
        flagged.append(int(said.sum()))
        precision.append((said & y).sum() / said.sum() if said.any() else np.nan)
    table.insert(3, "called suspicious", flagged)
    table.insert(5, "precision", precision)
    return table


def suspicious_share(results: pd.DataFrame, strata: pd.Series) -> pd.DataFrame:
    """Per stratum, the share of accounts each model and condition calls suspicious."""
    frame = results.assign(
        stratum=results["account"].map(strata),
        says=results["decision"] == sr.SUSPICIOUS,
    )
    table = frame.pivot_table(
        index="stratum", columns=["model", "condition"], values="says", aggfunc="mean"
    )
    order = [s for s in strata.drop_duplicates() if s in table.index]
    return table.loc[order]


# The reading test: each account read alone and with the guide, two models.

READING_VARIABLE = "SSR_READING_DIR"
SCORES_VARIABLE = "SSR_SCORES_DIR"
TEST_SET = "Test sample (notebook 16)"
LIVE_ANSWERS = "live_answers.sqlite"
READING_ORDER_SEED = 8
DEMO_SEED = 16
DEMO_PER_RULE = 2
OUTSIDE_GROUP = "positive, outside the file"
CLEAN_GROUPS = ("clean, matched activity", "clean, random")
GUIDE_HEAD, EXAMPLES_HEAD = "GUIDE:", "SOLVED EXAMPLES"
EVIDENCE_HEAD, FACTS_HEAD = "ACCOUNT A", "FACTS COMPUTED"
COMPARISON_HEAD = "HOW THESE FACTS COMPARE"


def reading_directory() -> Path:
    """The reading test's folder; the variable lets tests point somewhere else."""
    return Path(os.environ.get(READING_VARIABLE, DATA_INTERIM / "reading"))


def scores_directory() -> Path:
    """Where the stored XGBoost scores are."""
    return Path(os.environ.get(SCORES_VARIABLE, DATA_INTERIM))


STORED_RESULTS, LIVE_RESULTS = "results.parquet", "results_live.parquet"


def reading_sets(directory: Path) -> dict:
    """The samples the page can show, each as (folder, results file): the test
    sample (folder None), every fresh run with stored answers, and every live run."""
    sets = {}
    if (directory / "prompts_test_guided.parquet").exists():
        sets[TEST_SET] = (None, None)
    for path in sorted(directory.glob("fresh_seed*/prompts.parquet")):
        seed, size = path.parent.name.removeprefix("fresh_seed").split("_")
        if (path.parent / STORED_RESULTS).exists():
            sets[f"Fresh sample, seed {seed} ({size})"] = (path.parent, STORED_RESULTS)
        if (path.parent / LIVE_RESULTS).exists():
            sets[f"New accounts, seed {seed} ({size})"] = (path.parent, LIVE_RESULTS)
    return sets


def _alone_and_guided(alone: pd.DataFrame, guided: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [
            alone[alone["condition"] == pl.READ_ALONE],
            guided[guided["condition"] == pl.READ_SHOTS],
        ],
        ignore_index=True,
    )


def load_reading_set(directory: Path, folder: Path | None, results_file=None):
    """Prompts and results (empty until a run has answers) of one sample."""
    if folder is None:
        prompts = _alone_and_guided(
            pd.read_parquet(directory / "prompts_test.parquet"),
            pd.read_parquet(directory / "prompts_test_guided.parquet"),
        )
        results = _alone_and_guided(
            pd.read_parquet(directory / "results_test.parquet"),
            pd.read_parquet(directory / "results_test_guided.parquet"),
        )
    else:
        prompts = pd.read_parquet(folder / "prompts.parquet")
        path = folder / (results_file or STORED_RESULTS)
        results = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return prompts[["account", "group", "condition", "prompt"]], results


def reading_cases(prompts: pd.DataFrame) -> pd.DataFrame:
    """One row per account, in a fixed random order that hides the strata."""
    alone = prompts[prompts["condition"] == pl.READ_ALONE][["account", "group"]]
    cases = alone.sample(frac=1, random_state=READING_ORDER_SEED)
    cases = cases.reset_index(drop=True)
    cases.insert(0, "case", [f"Reading {i:03d}" for i in range(1, len(cases) + 1)])
    return cases


def is_involved(group: str) -> bool:
    return group not in CLEAN_GROUPS


def guided_sections(prompt: str) -> dict:
    """The parts of a guided prompt: guide, examples, evidence, facts, comparison."""
    parts = prompt.split("\n\n")

    def first(head: str) -> int:
        found = [i for i, part in enumerate(parts) if part.startswith(head)]
        if len(found) != 1:
            raise ValueError(f"expected one part starting with {head!r}")
        return found[0]

    evidence = first(EVIDENCE_HEAD)
    examples = []
    if any(part.startswith(EXAMPLES_HEAD) for part in parts):
        examples = parts[first(EXAMPLES_HEAD) + 1 : evidence]
    return {
        "guide": parts[first(GUIDE_HEAD)],
        "examples": examples,
        "evidence": parts[evidence],
        "facts": parts[first(FACTS_HEAD)].split("\n")[1:],
        "comparison": parts[first(COMPARISON_HEAD)].split("\n")[1:],
    }


DEMO_RULES = [
    "a false alarm the guide removes",
    "a positive caught with the guide",
    "a positive the guide misses",
    "a positive outside the pattern file",
]


def demo_cases(results: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """A few accounts for showing the page, chosen by a fixed rule on both models'
    decisions (an account counts only if every model did the same). Illustrative
    only: the rule looks at the results."""
    if results.empty:
        return pd.Series(dtype=str, name="rule")
    says = results.assign(says=results["decision"] == sr.SUSPICIOUS)
    wide = says.pivot_table(
        index="account", columns="condition", values="says", aggfunc="min"
    ).astype(bool)
    every = says.pivot_table(
        index="account", columns="condition", values="says", aggfunc="max"
    ).astype(bool)
    alone, guided = wide[pl.READ_ALONE], wide[pl.READ_SHOTS]
    never_guided = ~every[pl.READ_SHOTS]
    group = groups.reindex(wide.index)
    involved = group.map(is_involved).astype(bool)
    in_file = involved & (group != OUTSIDE_GROUP)
    masks = {
        DEMO_RULES[0]: ~involved & alone & never_guided,
        DEMO_RULES[1]: in_file & guided,
        DEMO_RULES[2]: in_file & alone & never_guided,
        DEMO_RULES[3]: group == OUTSIDE_GROUP,
    }
    rng = np.random.default_rng(DEMO_SEED)
    chosen = {}
    for rule in DEMO_RULES:
        pool = [a for a in wide.index[masks[rule].to_numpy()] if a not in chosen]
        for account in rng.permutation(pool)[:DEMO_PER_RULE]:
            chosen[str(account)] = rule
    return pd.Series(chosen, name="rule", dtype=str)


def cache_gaps(directory: Path, prompts: pd.DataFrame, accounts, models) -> int:
    """How many stored answers are missing for these accounts and models."""
    chosen = prompts[prompts["account"].isin(list(accounts))]
    return sum(
        cached_opinion(directory, prompt, model) is None
        for prompt in chosen["prompt"]
        for model in models
    )


def live_check(directory: Path, prompt: str, model: str, ask=sr.ask_ollama) -> dict:
    """Ask the local model now, whatever is stored, and keep the answer apart from
    the experiment's cache. Says whether the installed model is the version that
    gave the stored answers."""
    digest = installed_digest(model)
    stored = pl.AnswerCache(directory / ANSWERS)
    try:
        same_version = known_digest(stored, model) == digest
    finally:
        stored.close()
    reply = ask(prompt, model, think=False)
    live = pl.AnswerCache(directory / LIVE_ANSWERS)
    try:
        live.put(pl.answer_key(prompt, digest, False), model, digest, reply)
    finally:
        live.close()
    return {
        **sr.parse_answer(reply["content"]),
        "seconds": reply["seconds"],
        "same_version": same_version,
    }


def same_answer_text(stored: dict | None, live: dict) -> str:
    """Whether the live answer repeats the stored one."""
    if stored is None:
        return "No stored answer to compare with."
    fields = ["decision", "confidence", "score"]
    changed = [f for f in fields if stored.get(f) != live.get(f)]
    if not changed:
        return "The live answer repeats the stored one."
    return f"The live answer differs from the stored one in: {', '.join(changed)}."


def guide_change_text(alone: str | None, guided: str | None) -> str | None:
    """How the guide changed one model's decision."""
    if alone is None or guided is None:
        return None
    if alone == guided:
        return "The guide did not change the decision."
    before, after = alone.replace("_", " "), guided.replace("_", " ")
    return f"The guide changed the decision from {before} to {after}."


def ml_view(accounts, directory: Path) -> pd.DataFrame:
    """The stored XGBoost score and flag of each account, without its label."""
    reading = pd.read_parquet(
        directory / "transaction_reading.parquet", columns=["score", "flag"]
    )
    return reading.loc[list(accounts)]


FRESH_SCRIPT = PROJECT_ROOT / "scripts" / "run_fresh.py"
FRESH_LOG, LIVE_LOG = "run.log", "run_live.log"
LIVE_PLAN, LIVE_SUMMARY = "run_live_plan.json", "summary_live.csv"
POOL_SEED, POOL_SIZE = 102, "full"
# Measured on this machine (15 Nov 2026): seconds per call, and to load a model.
CALL_SECONDS = {"gemma4:latest": 5, "qwen3.5:9b-q4_K_M": 21}
LOAD_SECONDS = 15


def live_minutes(accounts: int, models: list[str]) -> float:
    """About how long a live run takes: two prompts per account and model."""
    seconds = sum(2 * accounts * CALL_SECONDS[m] + LOAD_SECONDS for m in models)
    return seconds / 60


def start_fresh_run(
    directory: Path,
    seed: int,
    size: str,
    models: list[str],
    accounts: int | None = None,
    live: bool = False,
    popen=subprocess.Popen,
) -> Path:
    """Start `scripts/run_fresh.py` in the background; returns its folder.

    With `live` it asks the models again for the next `accounts` accounts of the
    run, which must have its prompts already. The log is the folder's run.log
    (run_live.log when live); the page only reads it, so closing the page does not
    stop the run.
    """
    fresh.check_seed(seed)
    folder = directory / fresh.run_name(seed, size)
    folder.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(FRESH_SCRIPT),
        "--seed",
        str(seed),
        "--size",
        size,
        "--models",
        ",".join(models),
    ]
    if live:
        command.append("--live")
        if accounts is not None:
            command += ["--accounts", str(accounts)]
        # The answers file shows one run; the plan says how many answers to expect.
        (folder / fresh.LIVE_ANSWERS).unlink(missing_ok=True)
        (folder / LIVE_PLAN).write_text(
            json.dumps({"models": models, "accounts": accounts}), encoding="utf-8"
        )
    log = (folder / (LIVE_LOG if live else FRESH_LOG)).open("w", encoding="utf-8")
    popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    return folder


def fresh_run_status(folder: Path, log: str = FRESH_LOG) -> tuple[str, str]:
    """`finished`, `failed`, `running` or `not started`, and the log's last line."""
    path = folder / log
    if not path.exists():
        return "not started", ""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    last = lines[-1] if lines else ""
    if any(line == "finished" for line in lines):
        return "finished", last
    if any(line.startswith("Traceback") for line in lines):
        return "failed", last
    return "running", last


def live_plan(folder: Path) -> dict | None:
    """The models and the number of accounts of the last live run, if started here."""
    path = folder / LIVE_PLAN
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def expected_answers(plan: dict | None) -> int | None:
    """Two prompts per account and model; None when the number of accounts is open."""
    if not plan or not plan.get("accounts"):
        return None
    return 2 * int(plan["accounts"]) * len(plan["models"])


def log_tail(folder: Path, lines: int = 12, log: str = LIVE_LOG) -> str:
    """The last lines of a run's log."""
    path = folder / log
    if not path.exists():
        return ""
    text = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return "\n".join(text[-lines:])


def live_answer_table(
    answers: pd.DataFrame, cases: pd.DataFrame, models: dict
) -> pd.DataFrame:
    """The answers of a running live run as the page shows them: case names, not
    accounts, and readable models and conditions. No label, no stratum."""
    names = cases.set_index("account")["case"]
    conditions = {pl.READ_ALONE: "alone", pl.READ_SHOTS: "with the guide"}
    reading = answers["condition"].map(conditions).fillna(answers["condition"])
    table = pd.DataFrame(
        {
            "time": answers["time"],
            "case": answers["account"].map(names).fillna(answers["account"]),
            "model": answers["model"].map(models).fillna(answers["model"]),
            "reading": reading,
            "decision": answers["decision"]
            .fillna("no usable answer")
            .str.replace("_", " "),
            "confidence": answers["confidence"],
            "seconds": answers["seconds"].round(1),
            "source": answers["source"],
        }
    )
    return table.iloc[::-1].reset_index(drop=True)


def reader_name(name: str, models: dict) -> str:
    """'gemma4 K1_guide_shots' as 'Gemma 4, with the guide'; other names as they are."""
    conditions = {pl.READ_ALONE: "alone", pl.READ_SHOTS: "with the guide"}
    for model, label in models.items():
        for condition, text in conditions.items():
            if name == f"{model.split(':')[0]} {condition}":
                return f"{label}, {text}"
    return name


LIVE_OUTPUTS = [
    (fresh.LIVE_ANSWERS, "this run's answers, one line each, as they come"),
    (LIVE_RESULTS, "every live answer so far; the Guided reading page reads it"),
    ("../live_answers.sqlite", "the raw replies of the local models"),
    (LIVE_SUMMARY, "every reader on all live accounts: AUC-ROC, recall, false alarms"),
    (LIVE_LOG, "the log of the last run"),
]


def live_outputs(folder: Path) -> pd.DataFrame:
    """Where a live run writes, and what each file holds."""
    rows = []
    for name, holds in LIVE_OUTPUTS:
        path = (folder / name).resolve()
        exists = path.exists()
        rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT))
                if path.is_relative_to(PROJECT_ROOT)
                else str(path),
                "holds": holds,
                "last written": time.strftime(
                    "%d %b %H:%M", time.localtime(path.stat().st_mtime)
                )
                if exists
                else "not yet",
            }
        )
    return pd.DataFrame(rows)


def result_opinion(
    results: pd.DataFrame, account: str, condition: str, model: str
) -> dict | None:
    """One answer of a results file, in the form of `cached_opinion`."""
    rows = results[
        (results["account"] == account)
        & (results["condition"] == condition)
        & (results["model"] == model)
    ]
    if rows.empty:
        return None
    row = rows.iloc[-1]
    return {
        "decision": row["decision"],
        "confidence": row["confidence"],
        "score": None if pd.isna(row.get("score")) else row.get("score"),
        "reason": row.get("reason"),
        "seconds": row.get("seconds"),
        "source": row.get("source", "cache"),
    }


# How the numbers are checked.

MUTATION_REPORT = "mutation_report.json"
PIPELINE_DOT = """digraph {
  rankdir=LR; node [shape=box, style=rounded, fontname="sans-serif"];
  raw [label="IBM AML HI-Small\\n5.08 M transactions\\n1-10 September"];
  xgb [label="XGBoost\\ntransaction model\\nout-of-fold scores"];
  facts [label="Facts per account\\n(no labels)"];
  prompt [label="Prompt\\nevidence, facts, guide\\n(no label, no pattern name)"];
  llm [label="Local LLM\\nGemma 4, Qwen 3.5"];
  cache [label="Answer cache\\nkey = prompt + model version"];
  page [label="This demo"];
  raw -> xgb; raw -> facts; facts -> prompt; raw -> prompt;
  prompt -> llm -> cache -> page; xgb -> page;
}"""
ASSURANCE = [
    (
        "Accounts of one owner and of one laundering pattern kept in one fold",
        "the model learning an owner or a pattern instead of laundering",
        "split.py, notebook 2",
    ),
    (
        "The analysis plan of each experiment written before its results",
        "choosing the analysis after seeing the numbers",
        "notebooks 10, 14, 15, 16 (dated plan cells); 12 and 13 are exploratory",
    ),
    (
        "The test sample read once, in a condition chosen on another sample",
        "tuning the prompt on the accounts it is judged on",
        "notebooks 15 and 16",
    ),
    (
        "The key numbers of the main notebooks recomputed from the raw CSV",
        "a mistake in the project's own code going unnoticed",
        "separate check scripts, outside the project code",
    ),
    (
        "No label, pattern name or real identity in any prompt",
        "the reader being told the answer",
        "tests of the prompt builders",
    ),
    (
        "Automatic tests on every change",
        "silent mistakes in the split, features, prompts and demo",
        "tests/, CI on every push",
    ),
    (
        "Mistakes planted on purpose, one at a time",
        "tests that pass whatever the code does",
        "scripts/mutation_check.py",
    ),
]


def assurance_table() -> pd.DataFrame:
    return pd.DataFrame(ASSURANCE, columns=["What we did", "Guards against", "Where"])


def test_count(root: Path = PROJECT_ROOT) -> int:
    """The number of test functions in the repository."""
    return sum(
        line.lstrip().startswith("def test_")
        for path in (root / "tests").glob("test_*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def collected_tests(root: Path = PROJECT_ROOT, run=subprocess.run) -> int | None:
    """The number of tests pytest collects (a parametrised function counts once per
    case), or None if pytest cannot say."""
    try:
        done = run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    found = re.search(r"(\d+) tests? collected", done.stdout)
    return int(found.group(1)) if found else None


def mutation_summary(directory: Path) -> dict | None:
    """The last full mutation run, or None if it has not been saved."""
    path = directory / MUTATION_REPORT
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
