
import pandas as pd
import pytest

from sentinel.data import inspect_dataset, load_csv


def test_load_csv_reads_valid_dataset(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("age,income\n21,30000\n25,45000\n", encoding="utf-8")

    df = load_csv(csv_path)

    assert df.shape == (2, 2)
    assert list(df.columns) == ["age", "income"]
    assert df["age"].tolist() == [21, 25]


def test_load_csv_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_csv(tmp_path / "missing.csv")


def test_load_csv_rejects_empty_dataset(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("age,income\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no data rows"):
        load_csv(csv_path)


def test_load_csv_rejects_duplicate_column_names(tmp_path):
    csv_path = tmp_path / "duplicate.csv"
    csv_path.write_text("age,age\n21,25\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate column names"):
        load_csv(csv_path)


def test_inspect_dataset_reports_data_quality():
    df = pd.DataFrame(
        {
            "age": [21, 25, None],
            "income": [30000, 45000, 45000],
        }
    )

    summary = inspect_dataset(df)

    assert summary["rows"] == 3
    assert summary["columns"] == 2
    assert summary["duplicate_rows"] == 0
    assert summary["columns_info"]["age"]["missing_count"] == 1
    assert summary["columns_info"]["age"]["missing_percent"] == 33.33
    assert summary["columns_info"]["income"]["unique_values"] == 2


def test_inspect_dataset_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="Cannot inspect an empty dataset"):
        inspect_dataset(pd.DataFrame())



def test_load_csv_rejects_blank_column_name(tmp_path):
    csv_path = tmp_path / "blank_header.csv"
    csv_path.write_text("age,,income\n21,5,30000\n", encoding="utf-8")

    with pytest.raises(ValueError, match="empty column name"):
        load_csv(csv_path)


def test_inspect_dataset_counts_duplicate_rows():
    df = pd.DataFrame(
        {
            "age": [21, 25, 21],
            "income": [30000, 45000, 30000],
        }
    )

    summary = inspect_dataset(df)

    assert summary["duplicate_rows"] == 1
