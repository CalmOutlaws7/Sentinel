
"""Utilities for loading and inspecting tabular CSV datasets."""

from pathlib import Path

import pandas as pd


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
