
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/default of credit card clients.xls")
TARGET = "default payment next month"


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download it from the official UCI page first."
        )

    # The original workbook has a descriptive row above its column headers.
    df = pd.read_excel(DATA_PATH, header=1)

    # Normalize column names for consistent inspection.
    df.columns = [str(column).strip() for column in df.columns]

    print("\n=== SENTINEL: DATASET PROFILE ===")
    print(f"File: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print(f"Total missing cells: {df.isna().sum().sum():,}")

    print("\n=== COLUMN NAMES AND TYPES ===")
    print(df.dtypes.to_string())

    print("\n=== MISSING VALUES BY COLUMN ===")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "None")

    if TARGET not in df.columns:
        print("\nTarget column not found. Actual columns:")
        print(df.columns.tolist())
        raise ValueError(f"Expected target column {TARGET!r} is missing.")

    print("\n=== TARGET DISTRIBUTION ===")
    counts = df[TARGET].value_counts(dropna=False).sort_index()
    print(counts.to_string())
    print("\nTarget percentages:")
    print((df[TARGET].value_counts(normalize=True, dropna=False) * 100)
          .sort_index().round(2).to_string())

    print("\n=== FIRST FIVE ROWS ===")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
