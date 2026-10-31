"""Load the IBM AML HI-Small transactions with stable column names and string IDs."""

from pathlib import Path

import pandas as pd

from skeptical_second_reader.paths import DATA_INTERIM, DATA_RAW

TRANSACTIONS_CSV = DATA_RAW / "HI-Small_Trans.csv"
TRANSACTIONS_PARQUET = DATA_INTERIM / "HI-Small_Trans.parquet"

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
