"""Load the IBM AML HI-Small transactions with stable column names and string IDs."""

from pathlib import Path

import pandas as pd

from skeptical_second_reader.paths import DATA_INTERIM, DATA_RAW

TRANSACTIONS_CSV = DATA_RAW / "HI-Small_Trans.csv"
TRANSACTIONS_PARQUET = DATA_INTERIM / "HI-Small_Trans.parquet"
PATTERNS_TXT = DATA_RAW / "HI-Small_Patterns.txt"
ACCOUNTS_CSV = DATA_RAW / "HI-Small_accounts.csv"

ACCOUNT_COLUMNS = ["bank_name", "bank", "account", "entity_id", "entity_name"]

# The raw header names both account columns "Account", so columns are renamed by
# position instead of by name.
COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

# Identifiers are labels, not numbers: bank codes such as "010" must keep their
# leading zeros.
ID_COLUMNS = ["from_bank", "from_account", "to_bank", "to_account"]


# Separator for account keys; it must not occur in bank codes or account numbers.
KEY_SEPARATOR = ":"


def account_key(bank: pd.Series, account: pd.Series) -> pd.Series:
    """Identify an account by its (bank, account number) pair.

    The same account number can appear at different banks, so the number alone is
    not an identity. The separator keeps pairs such as ("01", "0A") and ("010", "A")
    apart, which plain concatenation would merge into "010A".
    """
    return bank + KEY_SEPARATOR + account


def account_labels(df: pd.DataFrame) -> pd.Series:
    """Label every account: 1 if it sends or receives any laundering transaction.

    Returns a 0/1 series indexed by account key, covering all accounts in `df`.
    Receivers count too: collecting accounts are part of a laundering pattern.
    """
    senders = account_key(df["from_bank"], df["from_account"])
    receivers = account_key(df["to_bank"], df["to_account"])
    laundering = df["is_laundering"] == 1
    positive = pd.concat([senders[laundering], receivers[laundering]]).unique()
    accounts = pd.Index(pd.concat([senders, receivers]).unique(), name="account")
    return pd.Series(accounts.isin(positive).astype(int), index=accounts, name="label")


def read_patterns(path: Path) -> pd.DataFrame:
    """Read the laundering-attempt file: one row per transaction, with its pattern.

    Columns: attempt (running number), pattern (e.g. FAN-OUT), timestamp and the ID
    columns, kept as strings so bank codes keep their leading zeros.
    """
    rows = []
    attempt, pattern = -1, None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("BEGIN"):
            attempt += 1
            # "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out"
            pattern = line.split(" - ", 1)[1].split(":", 1)[0].strip()
        elif line and not line.startswith("END"):
            fields = line.split(",")
            rows.append([attempt, pattern, fields[0], *fields[1:5]])
    patterns = pd.DataFrame(
        rows, columns=["attempt", "pattern", "timestamp", *ID_COLUMNS]
    )
    patterns["timestamp"] = pd.to_datetime(
        patterns["timestamp"], format="%Y/%m/%d %H:%M"
    )
    return patterns


def pattern_members(patterns: pd.DataFrame) -> pd.DataFrame:
    """One row per (attempt, account): every account that sends or receives in a
    laundering attempt of the pattern file."""
    sides = [
        pd.DataFrame(
            {
                "attempt": patterns["attempt"],
                "account": account_key(
                    patterns[f"{side}_bank"], patterns[f"{side}_account"]
                ),
            }
        )
        for side in ("from", "to")
    ]
    return pd.concat(sides).drop_duplicates().reset_index(drop=True)


def read_accounts(path: Path) -> pd.DataFrame:
    """Read the accounts file: one row per account, with the entity that owns it."""
    df = pd.read_csv(path, engine="c", dtype=str)
    df.columns = ACCOUNT_COLUMNS
    return df


def account_entities(keys: pd.Index, accounts: pd.DataFrame) -> pd.Series:
    """Map transaction account keys to the entity that owns each account.

    The accounts file writes bank codes without leading zeros ("10"), while the
    transactions keep them ("010", "0010"), so the join uses the unpadded code.
    Raises ValueError if dropping the zeros would merge two accounts, or if an
    account has no entity.
    """
    parts = keys.to_series().str.split(KEY_SEPARATOR, n=1, expand=True)
    unpadded_bank = parts[0].str.lstrip("0").replace("", "0")
    join_key = account_key(unpadded_bank, parts[1])
    if not join_key.is_unique:
        raise ValueError("accounts merge when bank codes lose their leading zeros")
    file_key = account_key(accounts["bank"], accounts["account"])
    if not file_key.is_unique:
        raise ValueError("duplicate accounts in the accounts file")
    owner = pd.Series(accounts["entity_id"].to_numpy(), index=file_key)
    entities = pd.Series(join_key.map(owner).to_numpy(), index=keys, name="entity_id")
    if entities.isna().any():
        raise ValueError(f"{entities.isna().sum()} accounts have no entity")
    return entities


def read_transactions_csv(path: Path) -> pd.DataFrame:
    """Read the raw transactions CSV.

    Uses the C engine: the pyarrow engine parses the bank codes as integers before
    applying dtype=str, which silently drops leading zeros ("010" -> "10").
    """
    df = pd.read_csv(
        path,
        engine="c",
        header=0,
        names=COLUMNS,
        dtype={col: "str" for col in ID_COLUMNS},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    return df


def load_transactions(refresh: bool = False) -> pd.DataFrame:
    """Return all transactions; the CSV is read once and cached as parquet."""
    if TRANSACTIONS_PARQUET.exists() and not refresh:
        return pd.read_parquet(TRANSACTIONS_PARQUET)
    df = read_transactions_csv(TRANSACTIONS_CSV)
    TRANSACTIONS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TRANSACTIONS_PARQUET, index=False)
    return df
