
"""Utilities for loading and inspecting tabular CSV datasets."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file and reject empty or structurally invalid datasets."""
    import csv

    csv_path = Path(path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        try:
            header = next(csv.reader(file))
        except StopIteration:
            raise ValueError("Dataset contains no data rows.") from None

    if not header or any(not column.strip() for column in header):
        raise ValueError("Dataset contains an empty column name.")

    if len(header) != len(set(header)):
        raise ValueError("Dataset contains duplicate column names.")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Dataset contains no data rows.")

    if len(df.columns) == 0:
        raise ValueError("Dataset contains no columns.")

    return df


def validate_dataset(
    df: pd.DataFrame,
    target_column: str,
    *,
    required_columns: list[str] | None = None,
    allowed_target_values: tuple | list | set | None = None,
) -> None:
    """Validate dataset structure and target-column integrity."""

    if df.empty:
        raise ValueError("Dataset contains no data rows.")

    if df.columns.has_duplicates:
        raise ValueError("Dataset contains duplicate column names.")

    if any(not str(column).strip() for column in df.columns):
        raise ValueError("Dataset contains an empty column name.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in dataset."
        )

    if required_columns:
        missing_columns = [
            column for column in required_columns if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {missing_columns}"
            )

    if df[target_column].isna().any():
        raise ValueError(
            f"Target column '{target_column}' contains missing values."
        )

    if allowed_target_values is not None:
        actual_values = set(df[target_column].unique())
        allowed_values = set(allowed_target_values)

        invalid_values = actual_values - allowed_values

        if invalid_values:
            raise ValueError(
                f"Target column '{target_column}' contains invalid values: "
                f"{invalid_values}. Expected values: {allowed_values}."
            )


def validate_classification_target(
    df: pd.DataFrame,
    target_column: str,
) -> None:
    """Ensure a classification target contains at least two distinct classes."""

    validate_dataset(df, target_column)

    if df[target_column].nunique() < 2:
        raise ValueError(
            f"Target column '{target_column}' must contain at least "
            "two distinct classes."
        )


def class_distribution(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """Return the count and percentage of each target class."""

    validate_classification_target(df, target_column)

    counts = df[target_column].value_counts().sort_index()

    return pd.DataFrame(
        {
            "class": counts.index,
            "count": counts.values,
            "percentage": (counts.values / len(df) * 100).round(2),
        }
    )

def split_dataset(
    df: pd.DataFrame,
    target_column: str,
    *,
    exclude_columns: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a classification dataset into stratified training and test sets."""

    validate_classification_target(df, target_column)

    if not 0 < test_size < 1:
        raise ValueError("test_size must be greater than 0 and less than 1.")

    exclude_columns = exclude_columns or []

    if target_column in exclude_columns:
        raise ValueError("The target column cannot be an excluded feature.")

    missing_columns = [
        column for column in exclude_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Excluded columns not found in dataset: {missing_columns}"
        )

    X = df.drop(columns=[target_column, *exclude_columns])
    y = df[target_column]

    if X.empty or X.shape[1] == 0:
        raise ValueError("Dataset contains no predictive feature columns.")

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a reproducible, compact data-quality summary."""
    if df.empty:
        raise ValueError("Cannot inspect an empty dataset.")

    if df.columns.has_duplicates:
        raise ValueError("Dataset contains duplicate column names.")

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "all_missing_columns": [
            str(column)
            for column in df.columns
            if df[column].isna().all()
        ],
        "columns_info": {
            str(column): {
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_percent": round(
                    float(df[column].isna().mean() * 100), 2
                ),
                "unique_values": int(df[column].nunique(dropna=True)),
            }
            for column in df.columns
        },
    }
