
import pandas as pd
import pytest

from sentinel.data import (
    class_distribution,
    inspect_dataset,
    load_csv,
    split_dataset,
    validate_classification_target,
    validate_dataset,
)


def test_split_dataset_returns_expected_shapes():
    df = pd.DataFrame(
        {
            "ID": range(100),
            "age": range(20, 120),
            "income": range(30000, 30100),
            "target": [0] * 80 + [1] * 20,
        }
    )

    X_train, X_test, y_train, y_test = split_dataset(
        df,
        "target",
        exclude_columns=["ID"],
    )

    assert X_train.shape == (80, 2)
    assert X_test.shape == (20, 2)
    assert len(y_train) == 80
    assert len(y_test) == 20


def test_split_dataset_excludes_identifier_column():
    df = pd.DataFrame(
        {
            "ID": range(100),
            "age": range(100),
            "target": [0] * 80 + [1] * 20,
        }
    )

    X_train, X_test, _, _ = split_dataset(
        df,
        "target",
        exclude_columns=["ID"],
    )

    assert "ID" not in X_train.columns
    assert "ID" not in X_test.columns


def test_split_dataset_preserves_class_proportions():
    df = pd.DataFrame(
        {
            "feature": range(1000),
            "target": [0] * 800 + [1] * 200,
        }
    )

    _, _, y_train, y_test = split_dataset(df, "target")

    assert y_train.value_counts(normalize=True).to_dict() == {
        0: 0.8,
        1: 0.2,
    }
    assert y_test.value_counts(normalize=True).to_dict() == {
        0: 0.8,
        1: 0.2,
    }


def test_split_dataset_is_reproducible():
    df = pd.DataFrame(
        {
            "feature": range(100),
            "target": [0] * 80 + [1] * 20,
        }
    )

    first = split_dataset(df, "target", random_state=42)
    second = split_dataset(df, "target", random_state=42)

    for first_part, second_part in zip(first, second):
        if isinstance(first_part, pd.DataFrame):
            pd.testing.assert_frame_equal(first_part, second_part)
        else:
            pd.testing.assert_series_equal(first_part, second_part)


def test_split_dataset_rejects_missing_excluded_column():
    df = pd.DataFrame(
        {
            "feature": range(10),
            "target": [0] * 5 + [1] * 5,
        }
    )

    with pytest.raises(ValueError, match="Excluded columns not found"):
        split_dataset(df, "target", exclude_columns=["ID"])


def test_split_dataset_rejects_invalid_test_size():
    df = pd.DataFrame(
        {
            "feature": range(10),
            "target": [0] * 5 + [1] * 5,
        }
    )

    with pytest.raises(ValueError, match="test_size"):
        split_dataset(df, "target", test_size=1.0)


def test_split_dataset_rejects_target_as_excluded_feature():
    df = pd.DataFrame(
        {
            "feature": range(10),
            "target": [0] * 5 + [1] * 5,
        }
    )

    with pytest.raises(ValueError, match="target column cannot be"):
        split_dataset(df, "target", exclude_columns=["target"])

def test_validate_classification_target_accepts_multiple_classes():
    df = pd.DataFrame({"target": [0, 1, 0, 1]})

    validate_classification_target(df, "target")


def test_validate_classification_target_rejects_single_class():
    df = pd.DataFrame({"target": [0, 0, 0, 0]})

    with pytest.raises(ValueError, match="at least two distinct classes"):
        validate_classification_target(df, "target")


def test_class_distribution_returns_counts_and_percentages():
    df = pd.DataFrame({"target": [0, 0, 0, 1]})

    result = class_distribution(df, "target")

    assert result["class"].tolist() == [0, 1]
    assert result["count"].tolist() == [3, 1]
    assert result["percentage"].tolist() == [75.0, 25.0]


def test_inspect_dataset_identifies_all_missing_columns():
    df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [None, None, None],
            "target": [0, 1, 0],
        }
    )

    result = inspect_dataset(df)

    assert result["all_missing_columns"] == ["feature_b"]

def test_validate_dataset_accepts_valid_binary_target():
    df = pd.DataFrame(
        {
            "age": [21, 25, 30],
            "income": [30000, 45000, 50000],
            "default": [0, 1, 0],
        }
    )

    validate_dataset(
        df,
        target_column="default",
        required_columns=["age", "income"],
        allowed_target_values=[0, 1],
    )


def test_validate_dataset_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="no data rows"):
        validate_dataset(pd.DataFrame(), target_column="default")


def test_validate_dataset_rejects_missing_target_column():
    df = pd.DataFrame({"age": [21, 25]})

    with pytest.raises(ValueError, match="Target column"):
        validate_dataset(df, target_column="default")


def test_validate_dataset_rejects_missing_required_columns():
    df = pd.DataFrame(
        {
            "age": [21, 25],
            "default": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(
            df,
            target_column="default",
            required_columns=["income"],
        )


def test_validate_dataset_rejects_missing_target_values():
    df = pd.DataFrame(
        {
            "age": [21, 25],
            "default": [0, None],
        }
    )

    with pytest.raises(ValueError, match="missing values"):
        validate_dataset(df, target_column="default")


def test_validate_dataset_rejects_invalid_target_values():
    df = pd.DataFrame(
        {
            "age": [21, 25, 30],
            "default": [0, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="invalid values"):
        validate_dataset(
            df,
            target_column="default",
            allowed_target_values=[0, 1],
        )

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
