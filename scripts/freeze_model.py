"""Freeze the transaction model of one round for the review flow, and check it.

Run from the project root, after `scripts/run_pipeline.py`:

    uv run python scripts/freeze_model.py            # round 0

The round is trained again exactly as in the pipeline and saved with its threshold
and calibration curve to data/interim/review/frozen_round<k>/. The check reloads the
saved model, scores the whole window with it and compares the accounts of the round's
test fold with the out-of-fold reading of the pipeline: scores, calibrated rates,
flags, strongest transactions and high counts must be the same.
"""

import argparse
import gc

import numpy as np
import pandas as pd
from run_pipeline import tables

from skeptical_second_reader import review
from skeptical_second_reader.data import (
    ACCOUNTS_CSV,
    PATTERNS_TXT,
    account_entities,
    account_labels,
    pattern_members,
    read_accounts,
    read_patterns,
)
from skeptical_second_reader.paths import DATA_INTERIM
from skeptical_second_reader.split import cached_folds, split_groups
from skeptical_second_reader.transactions import transaction_labels

REVIEW_DIR = DATA_INTERIM / "review"


def frozen_dir(k: int):
    return REVIEW_DIR / f"frozen_round{k}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()

    df, table, tx_features = tables()
    labels = account_labels(df)
    owners = account_entities(labels.index, read_accounts(ACCOUNTS_CSV))
    members = pattern_members(read_patterns(PATTERNS_TXT))
    groups = split_groups(owners, members)
    accounts = pd.Index(
        sorted(set(table["sender"]) | set(table["receiver"])), name="account"
    )
    y = labels.loc[accounts]
    folds = cached_folds(labels, groups, DATA_INTERIM).loc[accounts]
    tx_labels = transaction_labels(df)
    del df
    gc.collect()

    print(f"training round {args.round} ...", flush=True)
    frozen = review.freeze_round(tx_features, tx_labels, table, y, folds, args.round)
    folder = frozen_dir(args.round)
    review.save_frozen(frozen, folder)
    print(
        f"saved to {folder.relative_to(DATA_INTERIM.parent.parent)}: threshold "
        f"{frozen.threshold:.6f}, {len(frozen.reviewable)} reviewable accounts",
        flush=True,
    )

    loaded = review.load_frozen(folder)
    scored = review.score_window(loaded, table, tx_features)
    reading = pd.read_parquet(DATA_INTERIM / "pipeline" / "reading.parquet")
    test = loaded.reviewable
    mine, theirs = scored.loc[test], reading.loc[test]
    checks = {
        "threshold": abs(loaded.threshold - theirs["threshold"].iloc[0]) < 1e-9,
        "score": np.allclose(mine["score"], theirs["score"], atol=1e-6),
        "calibrated": np.allclose(mine["calibrated"], theirs["calibrated"], atol=1e-6),
        "flag": (mine["flag"].to_numpy() == theirs["flag"].to_numpy()).all(),
        "strongest_row": (
            mine["strongest_row"].to_numpy() == theirs["strongest_row"].to_numpy()
        ).all(),
        "n_high": (mine["n_high"].to_numpy() == theirs["n_high"].to_numpy()).all(),
    }
    worst = float(np.abs(mine["score"] - theirs["score"]).max())
    for name, ok in checks.items():
        print(f"  {name:14s} {'same' if ok else 'DIFFERENT'}")
    print(f"  largest score difference {worst:.2e}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
