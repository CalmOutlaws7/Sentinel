
# Dataset Documentation

## Dataset
Default of Credit Card Clients

## Source
UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

## Target
`default payment next month`
- 0: no default
- 1: default

## Initial profile
- Rows: 30,000
- Columns: 25, including the target
- Missing cells: 0
- Exact duplicate rows: 0
- Target 0: 23,364 (77.88%)
- Target 1: 6,636 (22.12%)

## Limitations
This is historical data from credit-card clients in Taiwan.
Its results may not generalize to other populations, periods,
or financial products.

Sentinel is an educational project, not a validated lending
or credit-approval system.

## Local data handling
The original workbook belongs in `data/raw/`.
Do not commit raw dataset files to Git.
