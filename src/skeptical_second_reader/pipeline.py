"""The batch pipeline: from the raw files to the LLM's answers, doing only what changed.

Stages run in order: reading, cases, prompts, answers. A stage's key is a hash of
everything its result depends on: the raw files, the source code it uses, its
parameters and the key of the stage before it. A stage with an unchanged key and all its
files in place is skipped.

LLM answers are cached one by one, keyed on the exact prompt, the model's digest and the
call's settings. If a prompt changes by one character, its old answer is not reused, and
a stopped run continues where it stopped.
"""

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from skeptical_second_reader import second_reader as sr
from skeptical_second_reader.facts import fact_lines
from skeptical_second_reader.transactions import TX_FEATURES

MANIFEST = "manifest.json"
READING_COLUMNS = [
    "score",
    "calibrated",
    "threshold",
    "flag",
    "label",
    "fold",
    "strongest_row",
    "n_high",
    "group",
]


def file_digest(path: Path) -> str:
    """SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stage_key(**inputs: object) -> str:
    """A hash of what a stage depends on; the order of the arguments does not matter."""
    text = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write through a temporary file, so a stopped run leaves no half-written file."""
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


class StageStore:
    """Remembers the key each stage last finished with."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = directory / MANIFEST
        self.manifest = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.exists()
            else {}
        )

    def path(self, name: str) -> Path:
        return self.directory / name

    def run(
        self, name: str, key: str, files: list[str], compute: Callable[[], None]
    ) -> bool:
        """Run `compute` unless this key already finished. True if it ran.

        The key is recorded only after `compute` returned and its files exist, so a
        failed stage runs again.
        """
        finished = self.manifest.get(name) == key
        if finished and all(self.path(file).exists() for file in files):
            return False
        compute()
        missing = [file for file in files if not self.path(file).exists()]
        if missing:
            raise RuntimeError(f"stage {name} did not write {missing}")
        self.manifest[name] = key
        temporary = self.manifest_path.with_name(MANIFEST + ".tmp")
        temporary.write_text(json.dumps(self.manifest, indent=1), encoding="utf-8")
        os.replace(temporary, self.manifest_path)
        return True


def answer_key(prompt: str, digest: str, think: bool | None) -> str:
    """What makes two calls the same call."""
    return stage_key(
        prompt=prompt, digest=digest, think=think, num_ctx=sr.NUM_CTX, temperature=0
    )


class AnswerCache:
    """LLM answers in a SQLite file. Every answer is committed when it arrives."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS answers ("
            "key TEXT PRIMARY KEY, model TEXT, digest TEXT, content TEXT, "
            "prompt_tokens INTEGER, answer_tokens INTEGER, seconds REAL)"
        )
        self.connection.commit()

    def get(self, key: str) -> dict | None:
        row = self.connection.execute(
            "SELECT content, prompt_tokens, answer_tokens, seconds "
            "FROM answers WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return dict(
            zip(
                ["content", "prompt_tokens", "answer_tokens", "seconds"],
                row,
                strict=True,
            )
        )

    def put(self, key: str, model: str, digest: str, reply: dict) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                model,
                digest,
                reply["content"],
                reply["prompt_tokens"],
                reply["answer_tokens"],
                reply["seconds"],
            ),
        )
        self.connection.commit()

    def __len__(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM answers").fetchone()[0]

    def close(self) -> None:
        self.connection.close()


def trial_accounts(groups: pd.Series, activity: pd.Series) -> pd.Series:
    """The accounts of the model check: one per group, and one busy account."""
    trial = sr.sample_cases(groups, 1, seed=0)
    busy = activity > sr.MAX_ROWS
    pool = groups[busy & (groups == "uncertain, model wrong")]
    pool = pool.drop(trial.index, errors="ignore")
    return pd.concat([trial, pool.sample(1, random_state=0)])


def experiment_accounts(
    groups: pd.Series, activity: pd.Series, per_group: int, seed: int
) -> pd.Series:
    """The experiment sample, without the accounts of the model check."""
    trial = trial_accounts(groups, activity)
    return sr.sample_cases(groups, per_group, seed=seed, exclude=trial.index)


def build_prompts(
    table: pd.DataFrame,
    features: pd.DataFrame,
    reading: pd.DataFrame,
    shap: pd.DataFrame,
    cases: pd.Series,
    conditions: list[str] = sr.CONDITIONS,
) -> pd.DataFrame:
    """One row per account and condition: the prompt and the ML output it shows."""
    pair_count = features["pair_count"].to_numpy()
    records = []
    for account, group in cases.items():
        rows = sr.account_rows(table, pair_count, account)
        strongest = int(reading.loc[account, "strongest_row"])
        for condition in conditions:
            prompt, shown = sr.case_prompt(
                rows,
                account,
                reading.loc[account],
                shap.loc[account, TX_FEATURES],
                features.iloc[strongest],
                condition,
            )
            records.append(
                {
                    "account": account,
                    "group": group,
                    "condition": condition,
                    "prompt": prompt,
                    "shown_level": None if shown is None else shown.level,
                    "shown_rate": np.nan if shown is None else shown.rate,
                    "flipped": None if shown is None else shown.flipped,
                }
            )
    return pd.DataFrame(records)


READ_ALONE, READ_FACTS = "K1", "K1_facts"


def reading_prompts(
    table: pd.DataFrame,
    pair_count: np.ndarray,
    cases: pd.Series,
    facts: pd.DataFrame,
) -> pd.DataFrame:
    """The reading test: each account alone, and alone with its facts block.

    No ML output in either. `cases` holds the stratum of each account.
    """
    records = []
    for account, stratum in cases.items():
        rows = sr.account_rows(table, pair_count, account)
        evidence, _ = sr.account_evidence(rows, account)
        variants = {
            READ_ALONE: sr.build_prompt(evidence, None),
            READ_FACTS: sr.build_prompt(
                evidence, None, facts=fact_lines(facts.loc[account])
            ),
        }
        for condition, prompt in variants.items():
            records.append(
                {
                    "account": account,
                    "group": stratum,
                    "condition": condition,
                    "prompt": prompt,
                    "shown_level": None,
                    "shown_rate": np.nan,
                    "flipped": None,
                }
            )
    return pd.DataFrame(records)


def answer_prompts(
    prompts: pd.DataFrame,
    model: str,
    digest: str,
    think: bool | None,
    cache: AnswerCache,
    ask: Callable[..., dict] = sr.ask_ollama,
    progress: Callable[[int, int, bool, dict], None] | None = None,
) -> pd.DataFrame:
    """Every prompt's answer, from the cache if it has it, without the prompt text."""
    records = []
    for number, row in enumerate(prompts.itertuples(), start=1):
        key = answer_key(row.prompt, digest, think)
        reply = cache.get(key)
        cached = reply is not None
        if reply is None:
            reply = ask(row.prompt, model, think=think)
            cache.put(key, model, digest, reply)
        answer = sr.parse_answer(reply["content"])
        shown = None
        if row.shown_level is not None and not pd.isna(row.shown_level):
            shown = sr.ShownOutput(
                row.shown_level, row.shown_rate, 0, bool(row.flipped)
            )
        records.append(
            {
                "model": model,
                "digest": digest,
                "think": think,
                "account": row.account,
                "group": row.group,
                "condition": row.condition,
                "shown_level": row.shown_level,
                "shown_rate": row.shown_rate,
                "flipped": row.flipped,
                "decision": answer["decision"],
                "confidence": answer["confidence"],
                "evidence": json.dumps(answer["evidence"]),
                "reason": answer["reason"],
                "follows_model": sr.follows_model(answer["decision"], shown),
                "prompt_tokens": reply["prompt_tokens"],
                "answer_tokens": reply["answer_tokens"],
                "seconds": reply["seconds"],
                "cached": cached,
            }
        )
        if progress is not None:
            progress(number, len(prompts), cached, reply)
    return pd.DataFrame(records)
