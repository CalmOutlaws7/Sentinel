from pathlib import Path

import pandas as pd

from sentinel.data import (
    class_distribution,
    inspect_dataset,
    validate_dataset,
)

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

    # Validate the target and the expected binary classification labels.
    validate_dataset(
        df,
        TARGET,
        allowed_target_values={0, 1},
    )

    # Generate reusable data-quality and target-distribution summaries.
    profile = inspect_dataset(df)
    distribution = class_distribution(df, TARGET)

    print("\n=== SENTINEL: DATASET PROFILE ===")
    print(f"File: {DATA_PATH}")
    print(f"Rows: {profile['rows']:,}")
    print(f"Columns: {profile['columns']}")
    print(f"Duplicate rows: {profile['duplicate_rows']:,}")

    total_missing = sum(
        column["missing_count"]
        for column in profile["columns_info"].values()
    )
    print(f"Total missing cells: {total_missing:,}")

    print("\n=== COLUMN NAMES AND TYPES ===")
    for column, info in profile["columns_info"].items():
        print(f"{column:<30} {info['dtype']}")

    print("\n=== MISSING VALUES BY COLUMN ===")
    missing = {
        column: info["missing_count"]
        for column, info in profile["columns_info"].items()
        if info["missing_count"] > 0
    }
    print(missing if missing else "None")

    print("\n=== ALL-MISSING COLUMNS ===")
    print(profile["all_missing_columns"] or "None")

    print("\n=== TARGET DISTRIBUTION ===")
    print(distribution.to_string(index=False))

    print("\n=== FIRST FIVE ROWS ===")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
