"""Test the frozen system on fresh accounts: XGBoost scores and the local LLMs.

Run from the project root, after notebooks 13 and 15:

    uv run python scripts/run_fresh.py --seed 101                 # 30 accounts
    uv run python scripts/run_fresh.py --seed 102 --size full --prepare
    uv run python scripts/run_fresh.py --seed 102 --size full     # store answers
    uv run python scripts/run_fresh.py --seed 102 --size full --live --accounts 20

A run draws new accounts with the strata of the reading samples, outside both reading
samples, the earlier LLM experiments and the examples: `mini` 13 accounts, `small` 30,
`full` 255. Every account is read alone (K1) and with the guide and examples
(K1_guide_shots); the prompts are built exactly as for notebooks 15 and 16. The
XGBoost scores are the out-of-fold scores already stored for every account.

`--prepare` only draws the accounts and builds their prompts (1-2 minutes, the
transaction table is loaded once). Without `--live` the answers go to the shared
cache, so a run asked twice costs nothing. `--live` asks the models again whatever is
stored, `--accounts N` takes the next N accounts of the run not yet asked live, and
an answer that cannot be had from the local model is taken from the cache and marked
so. Results go to data/interim/reading/fresh_seed<seed>_<size>/.
"""

import argparse
import gc
import time

import pandas as pd
from run_pipeline import DEFAULT_MODELS, THINK, model_digest, report, unload
from run_reading import SAMPLES, guided_inputs

from skeptical_second_reader import fresh, pipeline
from skeptical_second_reader import second_reader as sr
from skeptical_second_reader.data import (
    PATTERNS_TXT,
    load_transactions,
    pattern_members,
    read_patterns,
)
from skeptical_second_reader.guide import choose_shots
from skeptical_second_reader.paths import DATA_INTERIM
from skeptical_second_reader.reading_sample import (
    account_patterns,
    reading_sample,
    reading_samples,
)
from skeptical_second_reader.transactions import transaction_table

DEV_SEED, TEST_SEED = 11, 12  # notebook 13
SHOT_SEED = 15  # scripts/run_reading.py
LIVE_RESULTS = "results_live.parquet"


def build(seed: int, size: str, out) -> pd.DataFrame:
    """Draw the sample and write its cases and prompts; returns the prompts."""
    started = time.time()
    reading = pd.read_parquet(DATA_INTERIM / "transaction_reading.parquet")
    labels = reading["label"]
    activity = pd.read_parquet(DATA_INTERIM / "ceiling_features.parquet")[
        "transactions"
    ].astype(int)
    facts = pd.read_parquet(DATA_INTERIM / "account_facts.parquet")
    patterns = read_patterns(PATTERNS_TXT)
    members = pattern_members(patterns)
    accounts = labels.index
    in_file = pd.Series(accounts.isin(members["account"]), index=accounts)
    kind_of = account_patterns(patterns, members).reindex(accounts).dropna()
    used = pd.read_parquet(DATA_INTERIM / "llm_experiment_cases.parquet").index.union(
        pd.read_parquet(DATA_INTERIM / "llm_trial_cases.parquet").index
    )
    saved = {
        name: pd.read_parquet(DATA_INTERIM / file)["stratum"]
        for name, file in SAMPLES.items()
    }
    # The frozen samples must come out of the same code again, or the sizes changed.
    dev, test = reading_samples(
        labels, activity, kind_of, in_file, DEV_SEED, TEST_SEED, exclude=used
    )
    if not (dev.equals(saved["dev"]) and test.equals(saved["test"])):
        raise RuntimeError("the reading samples no longer come out the same")
    samples = dev.index.union(test.index)
    shots = choose_shots(facts, labels, activity, samples.union(used), seed=SHOT_SEED)
    exclude = used.union(samples).union(shots.index)
    cases = reading_sample(
        labels, activity, kind_of, in_file, seed, exclude, **fresh.SIZES[size]
    )
    assert not cases.index.intersection(exclude).size
    cases.to_frame().to_parquet(out / "cases.parquet")
    print(f"cases: {len(cases)} new accounts", flush=True)
    print(cases.value_counts().to_string(), flush=True)

    table = transaction_table(load_transactions())
    pair_count = (
        table.groupby(["sender", "receiver"])["usd"].transform("size").to_numpy()
    )
    alone = pipeline.reading_prompts(table, pair_count, cases, facts)
    guided = pipeline.reading_prompts(
        table, pair_count, cases, facts, **guided_inputs(facts)
    )
    prompts = pd.concat(
        [
            alone[alone["condition"] == pipeline.READ_ALONE],
            guided[guided["condition"] == pipeline.READ_SHOTS],
        ],
        ignore_index=True,
    )
    pipeline.save_parquet(prompts, out / "prompts.parquet")
    print(
        f"prompts: {len(prompts)} written in {time.time() - started:.0f} s", flush=True
    )
    return prompts


def ask_live(prompts, model, digest, stored, live) -> pd.DataFrame:
    """Every prompt asked now; an answer the model cannot give comes from the cache."""
    rows = []
    for number, row in enumerate(prompts.itertuples(), start=1):
        key = pipeline.answer_key(row.prompt, digest, THINK["off"])
        try:
            reply = sr.ask_ollama(row.prompt, model, think=THINK["off"])
            live.put(key, model, digest, reply)
            source = "live"
        except OSError as error:
            reply = stored.get(key)
            source = "cache"
            print(f"  {row.account}: no live answer ({error}), from cache", flush=True)
            if reply is None:
                source = "none"
                reply = {"content": "", "seconds": float("nan")}
        answer = sr.parse_answer(reply["content"])
        rows.append(
            {
                "model": model,
                "digest": digest,
                "account": row.account,
                "group": row.group,
                "condition": row.condition,
                "decision": answer["decision"],
                "confidence": answer["confidence"],
                "score": answer["score"],
                "reason": answer["reason"],
                "seconds": reply["seconds"],
                "source": source,
            }
        )
        print(f"  {number}/{len(prompts)}  {source}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--size", choices=list(fresh.SIZES), default="small")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--prepare", action="store_true", help="prompts only")
    parser.add_argument("--no-llm", action="store_true", help="XGBoost only")
    parser.add_argument("--live", action="store_true", help="ask again, now")
    parser.add_argument("--accounts", type=int, default=None, help="with --live")
    args = parser.parse_args()
    fresh.check_seed(args.seed)

    reading_dir = DATA_INTERIM / "reading"
    out = reading_dir / fresh.run_name(args.seed, args.size)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prompts.parquet").exists():
        prompts = pd.read_parquet(out / "prompts.parquet")
    else:
        prompts = build(args.seed, args.size, out)
        gc.collect()
    if args.prepare:
        print("finished", flush=True)
        return 0
    cases = pd.read_parquet(out / "cases.parquet")["stratum"]
    results_name = LIVE_RESULTS if args.live else "results.parquet"
    earlier = (
        pd.read_parquet(out / results_name)
        if (out / results_name).exists() and args.live
        else pd.DataFrame(columns=["account"])
    )
    if args.live:
        order = fresh.live_order(cases)
        waiting = [a for a in order if a not in set(earlier["account"])]
        chosen = waiting[: args.accounts] if args.accounts else waiting
        prompts = prompts[prompts["account"].isin(chosen)]
        print(f"live: {len(chosen)} accounts not asked live before", flush=True)

    readers = {}
    if not args.no_llm:
        stored = pipeline.AnswerCache(reading_dir / "answers.sqlite")
        live = pipeline.AnswerCache(reading_dir / "live_answers.sqlite")
        parts = [earlier] if len(earlier) else []
        try:
            for name in args.models.split(","):
                print(f"answers: {name}, {len(prompts)} prompts", flush=True)
                digest = model_digest(name)
                if args.live:
                    parts.append(ask_live(prompts, name, digest, stored, live))
                else:
                    before = len(stored)
                    parts.append(
                        pipeline.answer_prompts(
                            prompts, name, digest, THINK["off"], stored, progress=report
                        )
                    )
                    print(f"  new calls {len(stored) - before}", flush=True)
                    if len(stored) > before:
                        unload(name)
                results = pd.concat(parts, ignore_index=True)
                pipeline.save_parquet(results, out / results_name)
        finally:
            stored.close()
            live.close()
        readers = fresh.llm_readers(results)
        cases = cases.loc[cases.index.intersection(results["account"].unique())]

    readers = (
        fresh.ml_readers(
            cases.index,
            pd.read_parquet(
                DATA_INTERIM / "transaction_reading.parquet", columns=["score", "flag"]
            ),
            pd.read_parquet(DATA_INTERIM / "ceiling_facts_scores.parquet"),
        )
        | readers
    )
    summary = fresh.reader_table(cases, readers)
    summary.to_csv(
        out / ("summary_live.csv" if args.live else "summary.csv"), index=False
    )
    print()
    print(summary.round(3).to_string(index=False))
    print("\nSmall samples give wide intervals.")
    print("finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
