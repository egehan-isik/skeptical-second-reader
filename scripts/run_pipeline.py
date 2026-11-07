"""Run the batch pipeline from the raw files to the LLM's answers.

Run from the project root:

    uv run python scripts/run_pipeline.py

Only what changed runs again. Outputs go to data/interim/pipeline, which is not in the
repository. Decisions are not printed, so a long run can be watched without seeing its
results; notebook 11 analyses them with the plan fixed in notebook 10.
"""

import argparse
import gc
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from skeptical_second_reader import (
    data,
    explain,
    features,
    model,
    pipeline,
    second_reader,
    split,
    transactions,
)
from skeptical_second_reader.data import (
    ACCOUNTS_CSV,
    PATTERNS_TXT,
    TRANSACTIONS_CSV,
    account_entities,
    account_labels,
    load_transactions,
    pattern_members,
    read_accounts,
    read_patterns,
)
from skeptical_second_reader.model import precision_threshold
from skeptical_second_reader.paths import DATA_INTERIM
from skeptical_second_reader.split import cached_folds, split_groups
from skeptical_second_reader.transactions import (
    account_positions,
    explain_strongest,
    transaction_cv,
    transaction_features,
    transaction_labels,
    transaction_table,
)

BASE = "http://localhost:11434"
THINK = {"on": True, "off": False, "default": None}
DEFAULT_MODELS = "gemma4:latest,qwen3.5:9b-q4_K_M"


def source_digest(*modules: object) -> str:
    return pipeline.stage_key(
        **{m.__name__: pipeline.file_digest(Path(m.__file__)) for m in modules}
    )


def tables() -> tuple:
    df = load_transactions()
    table = transaction_table(df)
    return df, table, transaction_features(table)


def compute_reading(store: pipeline.StageStore) -> None:
    df, table, tx_features = tables()
    labels = account_labels(df)
    owners = account_entities(labels.index, read_accounts(ACCOUNTS_CSV))
    members = pattern_members(read_patterns(PATTERNS_TXT))
    groups = split_groups(owners, members)
    accounts = pd.Index(
        sorted(set(table["sender"]) | set(table["receiver"])), name="account"
    )
    y = labels.loc[accounts]
    fold_of = cached_folds(labels, groups, DATA_INTERIM).loc[accounts]
    result = transaction_cv(
        tx_features, transaction_labels(df), table, y, fold_of, precision_threshold
    )
    in_file = pd.Series(accounts.isin(members["account"]), index=accounts)
    reading = pd.DataFrame(
        {
            "score": result.scores,
            "calibrated": result.calibrated,
            "threshold": fold_of.map(dict(enumerate(result.thresholds))),
            "flag": result.flags,
            "label": y,
            "fold": fold_of,
            "strongest_row": result.strongest,
            "n_high": result.n_high,
            "group": second_reader.case_groups(result.flags, result.scores, y, in_file),
        }
    )[pipeline.READING_COLUMNS]
    pipeline.save_parquet(reading, store.path("reading.parquet"))
    pipeline.save_parquet(
        explain_strongest(result, tx_features, fold_of), store.path("shap.parquet")
    )


def compute_cases(store: pipeline.StageStore, per_group: int, seed: int) -> None:
    _, table, _ = tables()
    reading = pd.read_parquet(store.path("reading.parquet"))
    sender, receiver = account_positions(table, reading.index)
    activity = np.bincount(sender, minlength=len(reading))
    activity += np.bincount(receiver[sender != receiver], minlength=len(reading))
    cases = pipeline.experiment_accounts(
        reading["group"], pd.Series(activity, index=reading.index), per_group, seed
    )
    pipeline.save_parquet(cases.to_frame("group"), store.path("cases.parquet"))


def compute_prompts(store: pipeline.StageStore) -> None:
    _, table, tx_features = tables()
    prompts = pipeline.build_prompts(
        table,
        tx_features,
        pd.read_parquet(store.path("reading.parquet")),
        pd.read_parquet(store.path("shap.parquet")),
        pd.read_parquet(store.path("cases.parquet"))["group"],
    )
    pipeline.save_parquet(prompts, store.path("prompts.parquet"))


def model_digest(name: str) -> str:
    with urllib.request.urlopen(BASE + "/api/tags", timeout=30) as response:
        models = json.loads(response.read().decode("utf-8"))["models"]
    return next(entry["digest"] for entry in models if entry["name"] == name)


def unload(name: str) -> None:
    body = json.dumps({"model": name, "keep_alive": 0}).encode("utf-8")
    request = urllib.request.Request(
        BASE + "/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300):
        pass


def report(number: int, total: int, cached: bool, reply: dict) -> None:
    if not cached and (number % 25 == 0 or number == total):
        print(
            f"  {number}/{total}  {reply['seconds']:.1f}s  {time.strftime('%H:%M')}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--think", choices=list(THINK), default="off")
    parser.add_argument("--per-group", type=int, default=29)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="prompts per model")
    args = parser.parse_args()

    store = pipeline.StageStore(DATA_INTERIM / "pipeline")
    raw = {
        path.name: pipeline.file_digest(path)
        for path in (TRANSACTIONS_CSV, ACCOUNTS_CSV, PATTERNS_TXT)
    }
    reading_key = pipeline.stage_key(
        raw=raw,
        source=source_digest(data, explain, features, model, split, transactions),
    )
    ran = store.run(
        "reading",
        reading_key,
        ["reading.parquet", "shap.parquet"],
        lambda: compute_reading(store),
    )
    print("reading:", "ran" if ran else "unchanged", flush=True)

    cases_key = pipeline.stage_key(
        before=reading_key,
        per_group=args.per_group,
        seed=args.seed,
        source=source_digest(pipeline, second_reader),
    )
    ran = store.run(
        "cases",
        cases_key,
        ["cases.parquet"],
        lambda: compute_cases(store, args.per_group, args.seed),
    )
    print("cases:", "ran" if ran else "unchanged", flush=True)

    prompts_key = pipeline.stage_key(
        before=cases_key, source=source_digest(pipeline, second_reader)
    )
    ran = store.run(
        "prompts", prompts_key, ["prompts.parquet"], lambda: compute_prompts(store)
    )
    print("prompts:", "ran" if ran else "unchanged", flush=True)
    gc.collect()

    prompts = pd.read_parquet(store.path("prompts.parquet"))
    if args.limit is not None:
        prompts = prompts.head(args.limit)
    cache = pipeline.AnswerCache(store.path("answers.sqlite"))
    parts = []
    try:
        for name in args.models.split(","):
            before = len(cache)
            print(f"answers: {name}, {len(prompts)} prompts", flush=True)
            parts.append(
                pipeline.answer_prompts(
                    prompts,
                    name,
                    model_digest(name),
                    THINK[args.think],
                    cache,
                    progress=report,
                )
            )
            print(
                f"  new calls {len(cache) - before}, "
                f"from the cache {len(prompts) - (len(cache) - before)}",
                flush=True,
            )
            if len(cache) > before:
                unload(name)
            results = pd.concat(parts, ignore_index=True)
            pipeline.save_parquet(results, store.path("results.parquet"))
    finally:
        cache.close()
    print("finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
