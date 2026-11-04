"""Check that the split tests catch real mistakes (a small mutation test).

A test suite that always passes proves nothing. This script inserts one deliberate
mistake at a time into the source code, runs the test suite, and records whether
the tests failed (mistake caught) or passed (mistake missed). The original source
is restored byte for byte after every mutation, also when the script is
interrupted.

Run from the project root:

    uv run python scripts/mutation_check.py

Exit code 0 if every mistake is caught.
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "src" / "skeptical_second_reader" / "split.py"
FEATURES = ROOT / "src" / "skeptical_second_reader" / "features.py"
WINDOW = 'window = transactions[transactions["timestamp"] < WINDOW_END]'


@dataclass
class Mutation:
    name: str
    file: Path
    old: str
    new: str


FIT = "score = fit(X[train], y[train], X[val], y[val])"
THRESHOLD = "thresholds.append(float(choose_threshold(y[val], score(X[val]))))"
VAL_ROLE = "roles[folds == (k + 1) % n_folds] = VAL"
PATTERN_GROUPS = (
    'groups[accounts] = [f"{PATTERN_PREFIX}{c}" for c in component[:n_acc]]'
)

MUTATIONS = [
    Mutation(
        "folds ignore owners and patterns (account-disjoint split)",
        SPLIT,
        "splitter.split(dummy_x, labels.to_numpy(), groups.to_numpy())",
        "splitter.split(dummy_x, labels.to_numpy(), np.arange(len(labels)))",
    ),
    Mutation(
        "test accounts leak into training",
        SPLIT,
        FIT,
        "score = fit(X[train | test], y[train | test], X[val], y[val])",
    ),
    Mutation(
        "early stopping uses the test fold",
        SPLIT,
        FIT,
        "score = fit(X[train], y[train], X[test], y[test])",
    ),
    Mutation(
        "validation fold is the test fold",
        SPLIT,
        VAL_ROLE,
        "roles[folds == k] = VAL",
    ),
    Mutation(
        "validation fold is not the next fold (design says k+1)",
        SPLIT,
        VAL_ROLE,
        "roles[folds == (k + 2) % n_folds] = VAL",
    ),
    Mutation(
        "threshold chosen on the test fold",
        SPLIT,
        THRESHOLD,
        "thresholds.append(float(choose_threshold(y[test], score(X[test]))))",
    ),
    Mutation(
        "threshold chosen on validation plus test",
        SPLIT,
        THRESHOLD,
        "thresholds.append("
        "float(choose_threshold(y[val | test], score(X[val | test]))))",
    ),
    Mutation(
        "flags use another round's threshold",
        SPLIT,
        "flags[test] = (scores[test] >= thresholds[-1]).astype(int)",
        "flags[test] = (scores[test] >= thresholds[0]).astype(int)",
    ),
    Mutation(
        "pattern accounts keep their owner (patterns not grouped)",
        SPLIT,
        PATTERN_GROUPS,
        "pass",
    ),
    Mutation(
        "patterns sharing an account are not merged",
        SPLIT,
        PATTERN_GROUPS,
        "groups[accounts] = [f'{PATTERN_PREFIX}{a}' for a in "
        "members.drop_duplicates('account')['attempt']]",
    ),
    Mutation(
        "pattern graph edges are wrong",
        SPLIT,
        'cols = n_acc + attempts.get_indexer(members["attempt"])',
        "cols = rows",
    ),
    Mutation(
        "features use the tail after the window",
        FEATURES,
        WINDOW,
        'window = transactions[transactions["timestamp"].notna()]',
    ),
    Mutation(
        "features include the first moment after the window",
        FEATURES,
        WINDOW,
        'window = transactions[transactions["timestamp"] <= WINDOW_END]',
    ),
    Mutation(
        "features depend on the label",
        FEATURES,
        WINDOW,
        WINDOW + '\n    window = window[window["is_laundering"] == 0]',
    ),
    Mutation(
        "accounts only active in the tail get an all-zero row",
        FEATURES,
        'index = pd.Index(sorted(touches["account"].unique()), name="account")',
        'index = pd.Index(sorted(set(touches["account"]) | set('
        'account_key(transactions["from_bank"], transactions["from_account"])) | set('
        'account_key(transactions["to_bank"], transactions["to_account"]))), '
        'name="account")',
    ),
]


def tests_pass() -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def run(mutation: Mutation) -> bool:
    """Apply one mutation, run the tests, restore. Returns True if caught."""
    # Bytes, not text: text mode on Windows would turn LF line endings into CRLF.
    original = mutation.file.read_bytes()
    old, new = mutation.old.encode(), mutation.new.encode()
    if original.count(old) != 1:
        raise RuntimeError(f"anchor not found exactly once: {mutation.name}")
    try:
        mutation.file.write_bytes(original.replace(old, new))
        return not tests_pass()
    finally:
        mutation.file.write_bytes(original)


def main() -> int:
    if not tests_pass():
        print("The unmodified code fails its tests; fix that first.")
        return 1
    print(f"{'result':<8} mutation")
    n_caught = 0
    for mutation in MUTATIONS:
        caught = run(mutation)
        n_caught += caught
        print(f"{'caught' if caught else 'MISSED':<8} {mutation.name}")
    print(f"\n{n_caught} of {len(MUTATIONS)} deliberate mistakes caught")
    return 0 if n_caught == len(MUTATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
