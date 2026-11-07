"""Ask the LLM about the reading samples of notebook 13, alone and with the facts.

Run from the project root, after notebook 13:

    uv run python scripts/run_reading.py --sample dev
    uv run python scripts/run_reading.py --sample test

Prompts and answers go to data/interim/reading, which is not in the repository.
Answers are cached one by one, so a stopped run continues where it stopped, and the
alone prompts reuse the cache key rules of the first experiment. Decisions are not
printed.
"""

import argparse

import pandas as pd
from run_pipeline import DEFAULT_MODELS, THINK, model_digest, report, unload

from skeptical_second_reader import pipeline
from skeptical_second_reader.data import load_transactions
from skeptical_second_reader.paths import DATA_INTERIM
from skeptical_second_reader.transactions import transaction_table

SAMPLES = {
    "dev": "reading_dev_cases.parquet",
    "test": "reading_test_cases.parquet",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", choices=list(SAMPLES), required=True)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--think", choices=list(THINK), default="off")
    parser.add_argument("--limit", type=int, default=None, help="prompts per model")
    parser.add_argument(
        "--conditions",
        default=f"{pipeline.READ_ALONE},{pipeline.READ_FACTS}",
        help="comma-separated; the alone prompt never changes, the facts prompt may",
    )
    args = parser.parse_args()

    out = DATA_INTERIM / "reading"
    out.mkdir(exist_ok=True)
    prompts_path = out / f"prompts_{args.sample}.parquet"
    if not prompts_path.exists():
        table = transaction_table(load_transactions())
        pair_count = (
            table.groupby(["sender", "receiver"])["usd"].transform("size").to_numpy()
        )
        cases = pd.read_parquet(DATA_INTERIM / SAMPLES[args.sample])["stratum"]
        facts = pd.read_parquet(DATA_INTERIM / "account_facts.parquet")
        prompts = pipeline.reading_prompts(table, pair_count, cases, facts)
        pipeline.save_parquet(prompts, prompts_path)
        print(f"prompts: {len(prompts)} written", flush=True)
    prompts = pd.read_parquet(prompts_path)
    prompts = prompts[prompts["condition"].isin(args.conditions.split(","))]
    if args.limit is not None:
        prompts = prompts.head(args.limit)

    cache = pipeline.AnswerCache(out / "answers.sqlite")
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
            print(f"  new calls {len(cache) - before}", flush=True)
            if len(cache) > before:
                unload(name)
            results = pd.concat(parts, ignore_index=True)
            pipeline.save_parquet(results, out / f"results_{args.sample}.parquet")
    finally:
        cache.close()
    print("finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
