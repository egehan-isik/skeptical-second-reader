"""Tests for loading the raw transactions.

The fixture is a hand-written CSV in the raw format (fake IDs), so these tests run
in CI without the real dataset.
"""

import pandas as pd
import pytest

from skeptical_second_reader.data import (
    COLUMNS,
    ID_COLUMNS,
    account_key,
    account_labels,
    read_patterns,
    read_transactions_csv,
)

RAW_CSV = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
    "2022/09/01 00:20,010,AAA000001,001,BBB000002,100.00,US Dollar,100.00,US Dollar,"
    "Cheque,0\n"
    "2022/09/01 00:25,03208,CCC000003,010,AAA000001,5.50,Euro,5.50,Euro,ACH,1\n"
)


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "trans.csv"
    path.write_text(RAW_CSV)
    return path


def test_duplicate_account_headers_are_renamed_by_position(raw_csv):
    df = read_transactions_csv(raw_csv)
    assert list(df.columns) == COLUMNS
    assert df["from_account"].tolist() == ["AAA000001", "CCC000003"]
    assert df["to_account"].tolist() == ["BBB000002", "AAA000001"]


def test_bank_codes_keep_leading_zeros(raw_csv):
    df = read_transactions_csv(raw_csv)
    assert df["from_bank"].tolist() == ["010", "03208"]
    assert df["to_bank"].tolist() == ["001", "010"]


def test_parquet_cache_keeps_ids_and_timestamps(raw_csv, tmp_path):
    df = read_transactions_csv(raw_csv)
    cached = tmp_path / "trans.parquet"
    df.to_parquet(cached, index=False)
    back = pd.read_parquet(cached)
    pd.testing.assert_frame_equal(back[ID_COLUMNS], df[ID_COLUMNS])
    assert back["timestamp"].tolist() == df["timestamp"].tolist()


def test_account_key_is_the_same_in_both_roles(raw_csv):
    df = read_transactions_csv(raw_csv)
    senders = account_key(df["from_bank"], df["from_account"])
    receivers = account_key(df["to_bank"], df["to_account"])
    # AAA000001 at bank 010 sends in row 0 and receives in row 1.
    assert senders[0] == receivers[1] == "010:AAA000001"


def test_account_key_separates_banks_and_ambiguous_pairs():
    bank = pd.Series(["010", "020", "01", "010"])
    account = pd.Series(["A", "A", "0A", "A"])
    keys = account_key(bank, account)
    # Same number at another bank is another account.
    assert keys[0] != keys[1]
    # Without a separator both would read "010A".
    assert keys[2] != keys[3]
    assert keys[0] == keys[3]


def test_account_labels_mark_both_roles_of_laundering(raw_csv):
    labels = account_labels(read_transactions_csv(raw_csv))
    # Row 1 is laundering: CCC000003 sends, AAA000001 receives.
    assert labels["03208:CCC000003"] == 1
    assert labels["010:AAA000001"] == 1
    # BBB000002 only appears in the clean row 0.
    assert labels["001:BBB000002"] == 0
    assert labels.size == 3


PATTERNS_TXT = (
    "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out\n"
    "2022/09/01 00:06,010,AAA000001,001,BBB000002,1.00,Euro,1.00,Euro,ACH,1\n"
    "2022/09/01 00:07,010,AAA000001,0010,CCC000003,1.00,Euro,1.00,Euro,ACH,1\n"
    "END LAUNDERING ATTEMPT - FAN-OUT\n"
    "\n"
    "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops\n"
    "2022/09/02 00:01,020,DDD000004,020,EEE000005,1.00,Euro,1.00,Euro,ACH,1\n"
    "END LAUNDERING ATTEMPT - CYCLE\n"
)


def test_read_patterns_keeps_attempts_types_and_leading_zeros(tmp_path):
    path = tmp_path / "patterns.txt"
    path.write_text(PATTERNS_TXT)
    p = read_patterns(path)
    assert p["attempt"].tolist() == [0, 0, 1]
    assert p["pattern"].tolist() == ["FAN-OUT", "FAN-OUT", "CYCLE"]
    # 010 and 0010 are different banks; numeric parsing would merge them.
    assert p["to_bank"].tolist() == ["001", "0010", "020"]
