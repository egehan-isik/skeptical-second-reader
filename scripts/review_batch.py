"""Review accounts end to end: a window of transactions in, one JSON opinion out.

Run from the project root, after `scripts/freeze_model.py`:

    uv run python scripts/review_batch.py --accounts 5            # 5 random accounts
    uv run python scripts/review_batch.py --account "0012:80ABC1230"

The window is the transaction file of the dataset (1-10 September), loaded whole,
because the features look at all of it. Accounts are drawn from the test fold of the
frozen round, the only accounts the frozen model has not seen. Every account is read by
each local model twice: alone, then with the ML output. Answers are cached in
data/interim/review/answers.sqlite, so a run asked twice costs nothing; the result is
written to data/interim/review/review_<name>.json.
"""

import argparse
import json
import time

import numpy as np
from freeze_model import REVIEW_DIR, frozen_dir
from run_pipeline import DEFAULT_MODELS, THINK, model_digest, tables, unload

from skeptical_second_reader import review
from skeptical_second_reader.pipeline import AnswerCache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=0)
    parser.add_argument("--accounts", type=int, default=3, help="random accounts")
    parser.add_argument("--account", action="append", help="a given account")
    parser.add_argument(
        "--flagged",
        type=int,
        default=None,
        help="the N flagged accounts scored highest",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--think", choices=list(THINK), default="off")
    args = parser.parse_args()

    frozen = review.load_frozen(frozen_dir(args.round))
    started = time.time()
    _, table, features = tables()
    print(f"window loaded: {len(table):,} transactions", flush=True)
    if args.account:
        accounts, name = args.account, "given"
    elif args.flagged:
        # As a monitoring system would: its alerts, highest score first; no label.
        scored = review.score_window(frozen, table, features).loc[frozen.reviewable]
        alerts = scored[scored["flag"] == 1].sort_values("score", ascending=False)
        accounts = list(alerts.index[: args.flagged])
        name = f"flagged_n{args.flagged}"
    else:
        rng = np.random.default_rng(args.seed)
        chosen = rng.choice(len(frozen.reviewable), args.accounts, replace=False)
        accounts = [frozen.reviewable[i] for i in sorted(chosen)]
        name = f"seed{args.seed}_n{args.accounts}"
    review.check_reviewable(frozen, accounts)
    models = {m: model_digest(m) for m in args.models.split(",")}
    cache = AnswerCache(REVIEW_DIR / "answers.sqlite")
    try:
        results = review.review(
            accounts, table, features, frozen, models, cache, think=THINK[args.think]
        )
    finally:
        cache.close()
    for model in models:
        unload(model)
    out = REVIEW_DIR / f"review_{name}.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")

    for item in results:
        line = [f"{item['account']}: ML {item['ml']['level']}"]
        for model, reader in item["readers"].items():
            first, final = reader["first"]["decision"], reader["final"]["decision"]
            line.append(f"{model.split(':')[0]} {first} -> {final}")
        print("  " + " | ".join(line))
    print(f"written {out.name} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
