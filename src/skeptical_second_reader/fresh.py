"""A fresh sample to test the frozen system on accounts it has never seen.

The guide, the four solved examples and the reference values stay as in notebooks 15
and 16; only the accounts are new. They come from outside both reading samples, the
earlier LLM experiments and the examples, drawn with the strata of the reading samples
and a seed the reading samples did not use. Every reader is compared on the same
accounts: the transaction model and the ceiling classifier (both XGBoost, with
out-of-fold scores, so no account is scored by a model that trained on it) and the
local LLMs, alone and with the guide.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from skeptical_second_reader.pipeline import READ_ALONE, READ_SHOTS
from skeptical_second_reader.reading_sample import MATCHED_NAME, RANDOM_NAME
from skeptical_second_reader.second_reader import SUSPICIOUS

CONDITIONS = [READ_ALONE, READ_SHOTS]
# Per stratum: 8 pattern types, positives outside the file, two clean strata.
SIZES = {
    "small": {"n_per_pattern": 2, "n_outside": 2, "n_matched": 6, "n_random_clean": 6},
    "mini": {"n_per_pattern": 1, "n_outside": 1, "n_matched": 2, "n_random_clean": 2},
    "full": {},
}
CLEAN = [MATCHED_NAME, RANDOM_NAME]
RESERVED_SEEDS = {11, 12}  # the development and test samples of notebook 13


def run_name(seed: int, size: str) -> str:
    """The folder name of one fresh run."""
    return f"fresh_seed{seed}_{size}"


def check_seed(seed: int) -> None:
    if seed in RESERVED_SEEDS:
        raise ValueError(f"seed {seed} drew a reading sample; choose another")


def truth_of(strata: pd.Series) -> pd.Series:
    """1 for every stratum but the two clean ones."""
    return (~strata.isin([MATCHED_NAME, RANDOM_NAME])).astype(int)


def llm_readers(results: pd.DataFrame) -> dict:
    """Each model and condition as a reader: (score, flag) on the accounts usable in
    every condition for that model. Score = confidence when it says suspicious, one
    minus it otherwise, as in notebooks 14 to 16."""
    frame = results.assign(
        usable=results["decision"].notna() & results["confidence"].notna()
    )
    readers = {}
    for model, part in frame.groupby("model", sort=False):
        ok = part.groupby("account")["usable"].all()
        accounts = ok.index[ok.to_numpy()]
        for condition in CONDITIONS:
            rows = part[part["condition"] == condition].set_index("account")
            rows = rows.loc[rows.index.intersection(accounts)]
            if rows.empty:
                continue
            says = rows["decision"] == SUSPICIOUS
            score = np.where(says, rows["confidence"], 1 - rows["confidence"])
            readers[f"{model.split(':')[0]} {condition}"] = (
                pd.Series(score, index=rows.index, dtype=float),
                says.astype(bool),
            )
    return readers


def ml_readers(accounts: pd.Index, reading: pd.DataFrame, ceiling: pd.DataFrame):
    """The two XGBoost readers at their own thresholds, out-of-fold scores."""
    return {
        "XGBoost transaction model": (
            reading.loc[accounts, "score"].astype(float),
            reading.loc[accounts, "flag"].astype(bool),
        ),
        "XGBoost ceiling (evidence and facts)": (
            ceiling.loc[accounts, "scores evidence and facts"].astype(float),
            ceiling.loc[accounts, "flags evidence and facts"] == 1,
        ),
    }


def reader_table(strata: pd.Series, readers: dict) -> pd.DataFrame:
    """AUC-ROC, recall and false alarm rate of every reader on its accounts.

    Descriptive: a small sample gives wide intervals, which this table does not show.
    """
    truth = truth_of(strata)
    rows = []
    for name, (score, flag) in readers.items():
        y = truth.loc[score.index].to_numpy()
        both = len(np.unique(y)) == 2
        rows.append(
            {
                "reader": name,
                "accounts": len(y),
                "positives": int(y.sum()),
                "AUC-ROC": roc_auc_score(y, score.to_numpy()) if both else np.nan,
                "recall": flag.to_numpy()[y == 1].mean() if y.any() else np.nan,
                "false alarm rate": (
                    flag.to_numpy()[y == 0].mean() if (y == 0).any() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


LIVE_ORDER_SEED = 0


def live_order(strata: pd.Series) -> list:
    """The order in which a live run takes accounts: one from each stratum in turn,
    each stratum shuffled, so that even a few accounts mix positives and clean ones.
    The two clean strata come first in every round."""
    rng = np.random.default_rng(LIVE_ORDER_SEED)
    names = sorted(strata.unique(), key=lambda s: (s not in CLEAN, s))
    queues = {
        name: list(rng.permutation(strata.index[strata == name].to_numpy()))
        for name in names
    }
    order = []
    while any(queues.values()):
        for name in names:
            if queues[name]:
                order.append(str(queues[name].pop(0)))
    return order
