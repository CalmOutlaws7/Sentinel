# Sentinel — Explainable Credit Risk Prediction Engine

Sentinel is an educational machine learning project for building a reproducible, explainable credit-risk prediction pipeline with data validation, model evaluation, and a tested API.

The project aims to demonstrate practical ML engineering: reliable data handling, leakage-aware experimentation, meaningful evaluation under class imbalance, and maintainable software design.

> **Project status:** Early development. The initial project structure, dataset documentation, data inspection utilities, and foundational tests are in place. Model training, explainability, and API functionality are planned.

## Objectives

- Build a reusable pipeline for validating and preparing tabular credit-risk datasets.
- Establish reproducible training and evaluation workflows.
- Compare a logistic regression baseline against a tree-based model.
- Evaluate performance using metrics appropriate for imbalanced classification.
- Explain model predictions and analyze decision-threshold trade-offs.
- Expose predictions through a validated API.
- Test important data, modeling, and API behavior.

## Current Implementation

- Python package structure using `src/` layout.
- Project configuration and development dependencies managed through `pyproject.toml`.
- CSV loading utilities with checks for missing files, empty datasets, and invalid column headers.
- Dataset inspection for missing values, duplicate rows, data types, and unique-value counts.
- Initial automated tests for data-loading and inspection behavior.
- Dataset source, target definition, and limitations documented.

See the repository's commit history for implementation progress.

## Planned Technical Stack

| Area | Technology |
|---|---|
| Language | Python |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn |
| API | FastAPI, Pydantic |
| Testing | pytest |
| Code quality | Ruff |
| Dataset | UCI Default of Credit Card Clients |

## Dataset

Sentinel uses the **Default of Credit Card Clients** dataset from the UCI Machine Learning Repository.

- **Source:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Prediction target:** `default payment next month`
- **Target encoding:** `0` represents no default; `1` represents default.
- **Dataset context:** Historical credit-card client data from Taiwan.

The initial dataset profile contains 30,000 records and 25 columns, including the target. The observed target distribution is approximately 77.88% non-default and 22.12% default.

Because the classes are imbalanced, accuracy alone will not be sufficient for evaluating future models. Sentinel will also assess precision, recall, F1-score, ROC-AUC, PR-AUC, confusion matrices, and threshold-dependent trade-offs.

Refer to [`data/README.md`](data/README.md) for dataset documentation.

### Download and place the dataset

1. Download the workbook from the official UCI source above.
2. Place the original Excel file at:

   `data/raw/default of credit card clients.xls`

3. Keep the raw dataset out of Git. It is excluded by the project's `.gitignore`.

The dataset is used for educational experimentation, not for making real financial decisions.

## Getting Started

### Prerequisites

- Python 3.12 or newer.
- Git.
- The dataset workbook, if you want to run the dataset profiling script.

### 1. Clone the repository

```bash
git clone https://github.com/CalmOutlaws7/Sentinel.git
cd Sentinel
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the following command for the current process only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the project

Upgrade pip and install the project with its development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

### 4. Run the tests

```bash
python -m pytest -q
```

### 5. Check code quality

```bash
python -m ruff check src tests scripts
```

### 6. Profile the dataset

After downloading the workbook and placing it in the expected location, run:

```bash
python scripts/profile_dataset.py
```

The script reports dataset dimensions, column information, missing values, duplicate rows, target distribution, and sample records.

## Evaluation Strategy

The planned evaluation workflow will prioritize reproducibility and appropriate handling of class imbalance.

1. Validate the dataset and target before training.
2. Split data into training and test sets, using stratification where appropriate.
3. Fit preprocessing steps on training data only to prevent leakage.
4. Establish logistic regression as a baseline.
5. Compare against a suitable tree-based model.
6. Evaluate precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrices.
7. Analyze false positives, false negatives, and threshold-dependent outcomes.
8. Record the configuration and results of each experiment.

Evaluation results will be reported only after the corresponding experiments have been implemented and run.

## Responsible Use and Limitations

Sentinel is an educational engineering project, not a validated lending or credit-approval system.

- The dataset represents a particular population, geography, and historical period; performance may not generalize to other contexts.
- Predictive performance does not establish fairness, suitability, or real-world financial validity.
- A model's output should not be interpreted as a definitive judgment about an individual.
- Future explanations will describe model behavior and limitations rather than establish causation.
- Dataset provenance, preprocessing choices, evaluation methods, and model limitations should be documented as the project develops.

## Repository Structure

```text
Sentinel/
├── data/
│   ├── raw/              # Local datasets; excluded from Git
│   ├── processed/        # Generated data; excluded from Git
│   └── README.md         # Dataset documentation
├── artifacts/            # Generated model artifacts; excluded from Git
├── reports/
│   └── figures/          # Generated figures; excluded from Git
├── scripts/
│   └── profile_dataset.py
├── src/
│   └── sentinel/
│       ├── __init__.py
│       └── data.py
├── tests/
│   └── test_data.py
├── pyproject.toml
└── .gitignore
```

## Development Roadmap

- [x] Establish Python project structure and configuration.
- [x] Document the dataset source, target, and limitations.
- [x] Implement initial CSV loading and data-inspection utilities.
- [x] Add foundational automated tests.
- [ ] Strengthen dataset schema and target validation.
- [ ] Build a reusable preprocessing and splitting pipeline.
- [ ] Train and compare baseline and tree-based models.
- [ ] Add comprehensive evaluation and threshold analysis.
- [ ] Implement explainability and model reporting.
- [ ] Develop and test a prediction API.
- [ ] Add reproducible experiment artifacts and continuous integration.

## License

No project license has been added yet. Dataset licensing and reuse conditions should be reviewed separately from the license chosen for the source code.
