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
MODEL = ROOT / "src" / "skeptical_second_reader" / "model.py"
EXPLAIN = ROOT / "src" / "skeptical_second_reader" / "explain.py"
TRANSACTIONS = ROOT / "src" / "skeptical_second_reader" / "transactions.py"
READER = ROOT / "src" / "skeptical_second_reader" / "second_reader.py"
PIPELINE = ROOT / "src" / "skeptical_second_reader" / "pipeline.py"
DEMO = ROOT / "src" / "skeptical_second_reader" / "demo.py"
FACTS = ROOT / "src" / "skeptical_second_reader" / "facts.py"
READING_SAMPLE = ROOT / "src" / "skeptical_second_reader" / "reading_sample.py"
WINDOW = 'window = transactions[transactions["timestamp"] < WINDOW_END]'
EARLY = 'early = transactions[transactions["timestamp"] < WINDOW_END]'
ALL_ROWS = 'early = transactions[transactions["timestamp"].notna()]'


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
    Mutation(
        "extended features use the tail",
        FEATURES,
        EARLY + "\n    return early.assign(",
        ALL_ROWS + "\n    return early.assign(",
    ),
    Mutation(
        "exchange rates use the tail",
        FEATURES,
        EARLY + "\n    changes = early[",
        ALL_ROWS + "\n    changes = early[",
    ),
    Mutation(
        "exchange rates depend on the label",
        FEATURES,
        'changes = early[early["receiving_currency"] != early["payment_currency"]]',
        'changes = early[early["receiving_currency"] != early["payment_currency"]]\n'
        '    changes = changes[changes["is_laundering"] == 0]',
    ),
    Mutation(
        "the opposite direction wins over the direct rate",
        FEATURES,
        "rates = direct.combine_first(inverse)",
        "rates = inverse.combine_first(direct)",
    ),
    Mutation(
        "exchange rate is a mean, not a median",
        FEATURES,
        '.groupby(to_usd["payment_currency"])\n        .median()',
        '.groupby(to_usd["payment_currency"])\n        .mean()',
    ),
    Mutation(
        "given rates are ignored",
        FEATURES,
        "if rates is None:",
        "if True:",
    ),
    Mutation(
        "flows include self-transactions",
        FEATURES,
        "flows = early.loc[~is_self].assign(",
        "flows = early.assign(",
    ),
    Mutation(
        "amounts are not converted",
        FEATURES,
        'paid_usd=lambda t: t["amount_paid"] * t["payment_currency"].map(rates),',
        'paid_usd=lambda t: t["amount_paid"],',
    ),
    Mutation(
        "median amount is a mean",
        FEATURES,
        "aligned(sent.median())",
        "aligned(sent.mean())",
    ),
    Mutation(
        "flow balance has the wrong sign",
        FEATURES,
        "(received_sum - sent_sum) /",
        "(sent_sum - received_sum) /",
    ),
    Mutation(
        "a self-transaction counts twice in timing",
        FEATURES,
        'early.loc[~is_self, ["receiver", "timestamp"]]',
        'early.loc[:, ["receiver", "timestamp"]]',
    ),
    Mutation(
        "median gap is a mean",
        FEATURES,
        'gaps.groupby(touches["account"]).median()',
        'gaps.groupby(touches["account"]).mean()',
    ),
    Mutation(
        "a single transaction gets a gap of 0",
        FEATURES,
        ".median().reindex(index),",
        ".median().reindex(index).fillna(0),",
    ),
    Mutation(
        "out-degree counts the receivers",
        FEATURES,
        'out_degree = degree_pairs["sender"].value_counts()',
        'out_degree = degree_pairs["receiver"].value_counts()',
    ),
    Mutation(
        "given degree transactions are ignored",
        FEATURES,
        "if degree_transactions is None:",
        "if True:",
    ),
    Mutation(
        "every counterparty counts as reciprocal",
        FEATURES,
        'reciprocal = pairs.merge(reversed_pairs, on=["sender", "receiver"])',
        "reciprocal = pairs",
    ),
    Mutation(
        "ACH counterparties include other formats",
        FEATURES,
        'ach = flows.loc[flows["payment_format"] == ACH, ["sender", "receiver"]]',
        'ach = flows[["sender", "receiver"]]',
    ),
    Mutation(
        "ACH counterparties are not distinct",
        FEATURES,
        "ach = ach.drop_duplicates()",
        "ach = ach.copy()",
    ),
    Mutation(
        "counterparty banks are not distinct",
        FEATURES,
        'bank_of(pairs["receiver"]).groupby(pairs["sender"]).nunique()',
        'bank_of(pairs["receiver"]).groupby(pairs["sender"]).count()',
    ),
    Mutation(
        "sender banks are not distinct",
        FEATURES,
        'bank_of(pairs["sender"]).groupby(pairs["receiver"]).nunique()',
        'bank_of(pairs["sender"]).groupby(pairs["receiver"]).count()',
    ),
    Mutation(
        "currencies are not distinct",
        FEATURES,
        ").drop_duplicates()\n\n    def bank_of",
        ")\n\n    def bank_of",
    ),
    Mutation(
        "final parameters changed",
        MODEL,
        '"learning_rate": 0.03,',
        '"learning_rate": 0.1,',
    ),
    Mutation(
        "the final model trains with the baseline parameters",
        MODEL,
        "return _fit(FINAL_PARAMS,",
        "return _fit(BASELINE_PARAMS,",
    ),
    Mutation(
        "early stopping watches the training data",
        MODEL,
        "eval_set=[(X_val, y_val)]",
        "eval_set=[(X_train, y_train)]",
    ),
    Mutation(
        "SHAP uses all trees, not those up to the best iteration",
        EXPLAIN,
        "limit = (0, model.best_iteration + 1)",
        "limit = (0, 0)",
    ),
    Mutation(
        "an account is explained by another round's model",
        EXPLAIN,
        "for k, model in enumerate(models)]",
        "for k, model in enumerate(reversed(models))]",
    ),
    Mutation(
        "the base value counts as a top feature",
        EXPLAIN,
        "features = values.drop(BASE_VALUE)",
        "features = values",
    ),
    Mutation(
        "the precision threshold takes the highest score that reaches it",
        MODEL,
        "thresholds[reached.argmax()]",
        "thresholds[len(reached) - 1 - reached[::-1].argmax()]",
    ),
    Mutation(
        "an unreachable precision is not refused",
        MODEL,
        "if not reached.any():",
        "if False:",
    ),
    Mutation(
        "transactions after the window are used",
        TRANSACTIONS,
        'early = transactions[transactions["timestamp"] < WINDOW_END]',
        'early = transactions[transactions["timestamp"].notna()]',
    ),
    Mutation(
        "the label rows do not follow the window",
        TRANSACTIONS,
        'return transactions.loc[in_window, "is_laundering"]',
        'return transactions.loc[in_window.notna(), "is_laundering"]',
    ),
    Mutation(
        "the table carries the label",
        TRANSACTIONS,
        '"same_bank": (early["from_bank"] == early["to_bank"]).to_numpy(),',
        '"same_bank": (early["from_bank"] == early["to_bank"]).to_numpy(),\n'
        '            "is_laundering": early["is_laundering"].to_numpy(),',
    ),
    Mutation(
        "amounts are not converted to dollars",
        TRANSACTIONS,
        'early["amount_paid"] * early["payment_currency"].map(rates)',
        'early["amount_paid"] * 1.0',
    ),
    Mutation(
        "self-transactions count as moves",
        TRANSACTIONS,
        "moves = table.assign(row=np.arange(len(table)))[~is_self]",
        "moves = table.assign(row=np.arange(len(table)))",
    ),
    Mutation(
        "money received is looked up in the future",
        TRANSACTIONS,
        'direction="backward"',
        'direction="forward"',
    ),
    Mutation(
        "the next send is looked up in the past",
        TRANSACTIONS,
        'direction="forward"',
        'direction="backward"',
    ),
    Mutation(
        "money received has no time limit",
        TRANSACTIONS,
        'direction="backward",\n        tolerance=DAY,',
        'direction="backward",',
    ),
    Mutation(
        "the next send has no time limit",
        TRANSACTIONS,
        'direction="forward",\n        tolerance=DAY,',
        'direction="forward",',
    ),
    Mutation(
        "pair count ignores the receiver",
        TRANSACTIONS,
        'table.groupby(["sender", "receiver"])["usd"]',
        'table.groupby(["sender"])["usd"]',
    ),
    Mutation(
        "a send after a conversion may be in the paid currency",
        TRANSACTIONS,
        'table.loc[conversion, ["sender", "timestamp", "received_currency"]]',
        'table.loc[conversion, ["sender", "timestamp", "paid_currency"]]'
        '.assign(received_currency=lambda t: t["paid_currency"])'
        '[["sender", "timestamp", "received_currency"]]',
    ),
    Mutation(
        "a self-transaction without a currency change counts as a conversion",
        TRANSACTIONS,
        'conversion = is_self & (table["paid_currency"] != table["received_currency"])',
        "conversion = pd.Series(is_self, index=table.index)",
    ),
    Mutation(
        "training accepts a transaction with one training end",
        TRANSACTIONS,
        "train_rows = (roles[sender] == TRAIN) & (roles[receiver] == TRAIN)",
        "train_rows = (roles[sender] == TRAIN) | (roles[receiver] == TRAIN)",
    ),
    Mutation(
        "early stopping accepts a transaction with one validation end",
        TRANSACTIONS,
        "val_rows = (roles[sender] == VAL) & (roles[receiver] == VAL)",
        "val_rows = (roles[sender] == VAL) | (roles[receiver] == VAL)",
    ),
    Mutation(
        "the model trains on the labels of validation transactions too",
        TRANSACTIONS,
        "features[train_rows],\n            tx_labels[train_rows],",
        "features[train_rows | val_rows],\n"
        "            tx_labels[train_rows | val_rows],",
    ),
    Mutation(
        "a self-transaction counts twice in the noisy-or",
        TRANSACTIONS,
        "other_side = sender != receiver",
        "other_side = np.ones(len(sender), dtype=bool)",
    ),
    Mutation(
        "the noisy-or ignores what an account receives",
        TRANSACTIONS,
        "np.add.at(log_miss, receiver[other_side], miss[other_side])\n",
        "",
    ),
    Mutation(
        "the noisy-or is not protected from a score of one",
        TRANSACTIONS,
        "miss = np.log1p(-np.clip(tx_scores, 0, 1 - 1e-7))",
        "miss = np.log1p(-tx_scores)",
    ),
    Mutation(
        "the strongest transaction ignores what an account receives",
        TRANSACTIONS,
        "position = np.concatenate([sender, receiver])",
        "position = np.concatenate([sender, sender])",
    ),
    Mutation(
        "the strongest transaction is the weakest",
        TRANSACTIONS,
        'order = np.argsort(tx_scores[row], kind="stable")',
        'order = np.argsort(-tx_scores[row], kind="stable")',
    ),
    Mutation(
        "the threshold is chosen on the test accounts",
        TRANSACTIONS,
        "choose_threshold(y[val], values[val])",
        "choose_threshold(y[test], values[test])",
    ),
    Mutation(
        "flags use a fixed threshold",
        TRANSACTIONS,
        "flags[test] = (values[test] >= threshold).astype(int)",
        "flags[test] = (values[test] >= 0.5).astype(int)",
    ),
    Mutation(
        "accounts are scored on the validation fold",
        TRANSACTIONS,
        "scores[test] = values[test]",
        "scores[val] = values[val]",
    ),
    Mutation(
        "an account keeps the strongest transaction of another round",
        TRANSACTIONS,
        "strongest[test] = top[test]",
        "strongest[val] = top[val]",
    ),
    Mutation(
        "a transaction is explained by another round's model",
        TRANSACTIONS,
        "model = result.models[k]",
        "model = result.models[-1 - k]",
    ),
    Mutation(
        "the calibration curve is fitted on the test accounts",
        TRANSACTIONS,
        "fit(values[val], y[val])",
        "fit(values[test], y[test])",
    ),
    Mutation(
        "an account keeps the calibrated score of another round",
        TRANSACTIONS,
        "calibrated[test] = curve.predict(values[test])",
        "calibrated[val] = curve.predict(values[val])",
    ),
    Mutation(
        "a self-transaction counts twice among high transactions",
        TRANSACTIONS,
        "receiver[high & (sender != receiver)]",
        "receiver[high]",
    ),
    Mutation(
        "high transactions ignore the receiver",
        TRANSACTIONS,
        "np.add.at(counted, receiver[high & (sender != receiver)], 1)\n",
        "",
    ),
    Mutation(
        "K1 is shown the model",
        READER,
        "if condition == K1:\n        return None",
        "if condition == K1:\n        condition = K2",
    ),
    Mutation(
        "K2 is corrupted",
        READER,
        "K2: 0.0,",
        "K2: 0.33,",
    ),
    Mutation(
        "full dose moves to the same side",
        READER,
        "target = LOW_TARGET if flagged else HIGH_TARGET",
        "target = HIGH_TARGET if flagged else LOW_TARGET",
    ),
    Mutation(
        "the old corruption: one minus the rate",
        READER,
        "target = LOW_TARGET if flagged else HIGH_TARGET",
        "target = 1 - rate",
    ),
    Mutation(
        "the corrupted level ignores the moved rate",
        READER,
        "if moved >= STRONG_SCORE:",
        "if rate >= STRONG_SCORE:",
    ),
    Mutation(
        "the alert level starts too low",
        READER,
        "SHOWN_ALERT_RATE = 0.5",
        "SHOWN_ALERT_RATE = 0.1",
    ),
    Mutation(
        "a reversed output keeps its high transactions",
        READER,
        "n_high = 0 if flagged else max(n_high, 1)",
        "pass",
    ),
    Mutation(
        "the evidence keeps real counterparty names",
        READER,
        'names[other] = f"K{len(names) + 1}"',
        "names[other] = other",
    ),
    Mutation(
        "the evidence shows the smallest transactions",
        READER,
        'rows.nlargest(MAX_ROWS, "usd")',
        'rows.nsmallest(MAX_ROWS, "usd")',
    ),
    Mutation(
        "the evidence is not in time order",
        READER,
        '.sort_values(["timestamp", "row"])',
        '.sort_values(["usd", "row"])',
    ),
    Mutation(
        "the summary counts a self-transaction as sent",
        READER,
        'sent = rows[(rows["sender"] == account) & ~is_self]',
        'sent = rows[rows["sender"] == account]',
    ),
    Mutation(
        "reason directions ignore the reversal",
        READER,
        "(contributions[name] > 0) != flipped",
        "contributions[name] > 0",
    ),
    Mutation(
        "reasons are the smallest contributions",
        READER,
        "contributions.abs().sort_values(ascending=False,",
        "contributions.abs().sort_values(ascending=True,",
    ),
    Mutation(
        "following the model trusts the reversed side",
        READER,
        "(decision == SUSPICIOUS) == shown.alert",
        "(decision == SUSPICIOUS) != shown.alert",
    ),
    Mutation(
        "score-only shows reasons",
        READER,
        "if reasons is not None:",
        "if True:\n        reasons = reasons or []",
    ),
    Mutation(
        "an invalid decision is accepted",
        READER,
        "if decision not in (SUSPICIOUS, NOT_SUSPICIOUS):\n        decision = None",
        "pass",
    ),
    Mutation(
        "the call is not deterministic",
        READER,
        '"temperature": 0,',
        '"temperature": 0.7,',
    ),
    Mutation(
        "the groups swap right and wrong strong alerts",
        READER,
        "strong & positive,\n        strong & ~positive,",
        "strong & ~positive,\n        strong & positive,",
    ),
    Mutation(
        "score-only is given the reasons in a case",
        READER,
        "if condition != SCORE_ONLY:",
        "if True:",
    ),
    Mutation(
        "a stage is skipped whatever its key",
        PIPELINE,
        "finished = self.manifest.get(name) == key",
        "finished = name in self.manifest",
    ),
    Mutation(
        "a stage with a missing file is skipped",
        PIPELINE,
        "if finished and all(self.path(file).exists() for file in files):",
        "if finished:",
    ),
    Mutation(
        "a failed stage counts as finished",
        PIPELINE,
        "        compute()\n",
        "        self.manifest[name] = key\n        compute()\n",
    ),
    Mutation(
        "a stage that wrote nothing counts as finished",
        PIPELINE,
        "if missing:",
        "if False:",
    ),
    Mutation(
        "finished stages are not saved",
        PIPELINE,
        "os.replace(temporary, self.manifest_path)",
        "temporary.unlink()",
    ),
    Mutation(
        "the answer key ignores the prompt",
        PIPELINE,
        "prompt=prompt, digest=digest,",
        "prompt='', digest=digest,",
    ),
    Mutation(
        "the answer key ignores the model version",
        PIPELINE,
        "prompt=prompt, digest=digest,",
        "prompt=prompt, digest='',",
    ),
    Mutation(
        "the answer key ignores thinking",
        PIPELINE,
        "think=think, num_ctx",
        "think=None, num_ctx",
    ),
    Mutation(
        "answers are not committed",
        PIPELINE,
        "            ),\n        )\n        self.connection.commit()\n"
        "\n    def __len__",
        "            ),\n        )\n\n    def __len__",
    ),
    Mutation(
        "the cache is never read",
        PIPELINE,
        "reply = cache.get(key)\n",
        "reply = None\n",
    ),
    Mutation(
        "a new answer is not stored",
        PIPELINE,
        "cache.put(key, model, digest, reply)\n",
        "pass\n",
    ),
    Mutation(
        "the experiment sample may contain trial accounts",
        PIPELINE,
        "exclude=trial.index)",
        "exclude=None)",
    ),
    Mutation(
        "the busy trial account need not be busy",
        PIPELINE,
        "pool = groups[busy & (groups ==",
        "pool = groups[(groups ==",
    ),
    Mutation(
        "the prompt text goes into the results",
        PIPELINE,
        '"condition": row.condition,\n',
        '"condition": row.condition,\n                "prompt": row.prompt,\n',
    ),
    Mutation(
        "the case list carries the group",
        DEMO,
        'cases = true_output[["account", "shown_level"]]',
        'cases = true_output[["account", "shown_level", "group"]]',
    ),
    Mutation(
        "a stored answer is reused for another prompt",
        DEMO,
        "reply = cache.get(pl.answer_key(prompt, digest, False))",
        'reply = cache.get(pl.answer_key("", digest, False))',
    ),
    Mutation(
        "a live answer is not kept",
        DEMO,
        "cache.put(key, model, digest, reply)",
        "pass",
    ),
    Mutation(
        "agreement is reversed",
        DEMO,
        "if sr.follows_model(decision, shown):",
        "if not sr.follows_model(decision, shown):",
    ),
    Mutation(
        "an old analyst decision is not replaced",
        DEMO,
        '"INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?)"',
        '"INSERT OR IGNORE INTO decisions VALUES (?, ?, ?, ?)"',
    ),
    Mutation(
        "the dose table includes accounts wrong alone",
        DEMO,
        'correct = alone.index[alone["decision"] == alone["truth"]]',
        "correct = alone.index",
    ),
    Mutation(
        "an invalid answer counts as wrong",
        DEMO,
        'shared = shared[other.loc[shared, "decision"].notna()]',
        "pass",
    ),
    Mutation(
        "counterparties are counted per transaction, not per account",
        FACTS,
        '"r": receiver[~own]}).drop_duplicates()',
        '"r": receiver[~own]})',
    ),
    Mutation(
        "both ways matches an account with itself",
        FACTS,
        'left_on=["s", "r"], right_on=["r", "s"]',
        'left_on=["s", "r"], right_on=["s", "r"]',
    ),
    Mutation(
        "sending on looks back in time",
        FACTS,
        'direction="forward"',
        'direction="backward"',
    ),
    Mutation(
        "the day limit of sending on is wrong",
        FACTS,
        "DAY_HOURS = 24",
        "DAY_HOURS = 1",
    ),
    Mutation(
        "a conversion is a self-transaction in one currency",
        FACTS,
        'table["paid_currency"].to_numpy() != table["received_currency"].to_numpy()',
        'table["paid_currency"].to_numpy() == table["received_currency"].to_numpy()',
    ),
    Mutation(
        "the busiest day is the quietest",
        FACTS,
        'facts["busiest_day"] = _per_account(per_day.groupby(level=0).max(), n, 0.0)',
        'facts["busiest_day"] = _per_account(per_day.groupby(level=0).min(), n, 0.0)',
    ),
    Mutation(
        "a clean pick may come from the pattern file",
        READING_SAMPLE,
        "clean = pool[~positive & ~in_file[pool]]",
        "clean = pool[~positive]",
    ),
    Mutation(
        "an outside positive may come from the pattern file",
        READING_SAMPLE,
        "outside = pool[positive & ~in_file[pool]]",
        "outside = pool[positive]",
    ),
    Mutation(
        "the test sample may repeat the development sample",
        READING_SAMPLE,
        "used = dev.index if exclude is None else dev.index.union(exclude)",
        "used = exclude",
    ),
    Mutation(
        "matching ignores the activity of the positive",
        READING_SAMPLE,
        "wanted = activity[target]",
        "wanted = 0",
    ),
    Mutation(
        "a pattern stratum takes clean accounts",
        READING_SAMPLE,
        "members = pool[positive & (known == kind)]",
        "members = pool[known == kind]",
    ),
    Mutation(
        "the facts prompt has no facts",
        PIPELINE,
        "evidence, None, facts=fact_lines(facts.loc[account])",
        "evidence, None, facts=None",
    ),
    Mutation(
        "the alone prompt differs from the first experiment",
        PIPELINE,
        "READ_ALONE: sr.build_prompt(evidence, None),",
        'READ_ALONE: sr.build_prompt(evidence, None, facts=""),',
    ),
    Mutation(
        "the facts block comes before the evidence",
        READER,
        "    if facts is not None:\n        parts.append(facts)",
        "    if facts is not None:\n        parts.insert(0, facts)",
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
