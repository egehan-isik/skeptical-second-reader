"""Tests for loading the raw transactions.

The fixture is a hand-written CSV in the raw format (fake IDs), so these tests run
in CI without the real dataset.
"""

import pandas as pd
import pytest

from skeptical_second_reader.data import COLUMNS, ID_COLUMNS, read_transactions_csv

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
