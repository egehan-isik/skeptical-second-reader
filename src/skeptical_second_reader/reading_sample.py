"""The sample for the reading test: chosen by what an account is, not by model errors.

The experiment sample of notebook 10 is chosen by the ML model's mistakes, which suits
the question of following a wrong output. Whether the reader can read the evidence at
all needs accounts of every laundering pattern type, positives outside the pattern
file, and clean accounts, half of them as busy as the positives. No model score is
used to choose an account.
"""

import numpy as np
import pandas as pd

PER_PATTERN = 15
OUTSIDE = 15
MATCHED = 60
RANDOM_CLEAN = 60
OUTSIDE_NAME = "positive, outside the file"
MATCHED_NAME = "clean, matched activity"
RANDOM_NAME = "clean, random"


def pattern_stratum(pattern: str) -> str:
    return f"pattern {pattern}"


def account_patterns(patterns: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """The pattern type of every account in the file.

    An account in attempts of two or more types has no single type and is left out.
    `patterns` has columns attempt and pattern, `members` attempt and account.
    """
    kinds = patterns.drop_duplicates("attempt").set_index("attempt")["pattern"]
    typed = members.assign(pattern=members["attempt"].map(kinds))
    per_account = typed.groupby("account")["pattern"].agg(["nunique", "first"])
    single = per_account[per_account["nunique"] == 1]["first"]
    return single.rename("pattern")


def _take(pool: pd.Index, n: int, seed: int) -> pd.Index:
    if len(pool) <= n:
        return pool
    return pd.Index(pool.to_series().sample(n, random_state=seed))


def _matched(
    targets: pd.Index, activity: pd.Series, clean: pd.Index, seed: int
) -> pd.Index:
    """One clean account per target with the same activity, or the nearest left."""
    order = activity[clean].sort_values(kind="stable")
    keys = order.index.to_numpy()
    values = order.to_numpy()
    free = np.ones(len(keys), dtype=bool)
    rng = np.random.default_rng(seed)
    chosen = []
    for target in targets:
        wanted = activity[target]
        same = np.flatnonzero(free & (values == wanted))
        if same.size:
            pick = rng.choice(same)
        else:
            left = np.flatnonzero(free)
            pick = left[np.argmin(np.abs(values[left] - wanted))]
        free[pick] = False
        chosen.append(keys[pick])
    return pd.Index(chosen)


def reading_sample(
    labels: pd.Series,
    activity: pd.Series,
    patterns: pd.Series,
    in_file: pd.Series,
    seed: int,
    exclude: pd.Index | None = None,
) -> pd.Series:
    """The stratum of every chosen account.

    `labels`, `activity` (transactions per account) and `in_file` share one index;
    `patterns` is `account_patterns`. Accounts in `exclude` are never chosen.
    """
    pool = labels.index if exclude is None else labels.index.difference(exclude)
    positive = labels[pool] == 1
    known = patterns.reindex(pool)
    parts = []
    for kind in sorted(patterns.unique()):
        members = pool[positive & (known == kind)]
        parts.append(
            pd.Series(pattern_stratum(kind), index=_take(members, PER_PATTERN, seed))
        )
    outside = pool[positive & ~in_file[pool]]
    parts.append(pd.Series(OUTSIDE_NAME, index=_take(outside, OUTSIDE, seed)))
    chosen = pd.concat(parts)
    clean = pool[~positive & ~in_file[pool]]
    targets = _take(chosen.index, MATCHED, seed)
    matched = _matched(targets, activity, clean, seed)
    parts.append(pd.Series(MATCHED_NAME, index=matched))
    rest = clean.difference(matched)
    parts.append(pd.Series(RANDOM_NAME, index=_take(rest, RANDOM_CLEAN, seed)))
    return pd.concat(parts).rename("stratum")


def reading_samples(
    labels: pd.Series,
    activity: pd.Series,
    patterns: pd.Series,
    in_file: pd.Series,
    dev_seed: int,
    test_seed: int,
    exclude: pd.Index | None = None,
) -> tuple[pd.Series, pd.Series]:
    """A development sample for writing the prompt and a disjoint one for the test."""
    dev = reading_sample(labels, activity, patterns, in_file, dev_seed, exclude)
    used = dev.index if exclude is None else dev.index.union(exclude)
    test = reading_sample(labels, activity, patterns, in_file, test_seed, used)
    return dev, test
