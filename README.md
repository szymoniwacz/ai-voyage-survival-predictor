# AI Voyage Survival Predictor

A modular machine learning project for predicting passenger survival on the Titanic (Kaggle competition).

---

## Project Goal
Predict Titanic passenger survival using robust ML pipelines, real-world data preprocessing, feature engineering, and model comparison. Focus on clarity, reproducibility, and extensibility.

---

## Dataset
- Titanic dataset from [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- Files: `train.csv`, `test.csv`, `gender_submission.csv`
- All data files are included in `data/raw/` — no need to download separately.

---

## Tech Stack
- Python
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- pytest

---

## Quick Start
1. Create a virtual environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2. Data files are already present in `data/raw/`.
3. Run EDA, model comparison, training, and prediction from the CLI.

---

## CLI Usage

All main operations are available via the CLI:

```bash
python src/cli.py eda        --train data/raw/train.csv
python src/cli.py compare    --train data/raw/train.csv
python src/cli.py train_best --train data/raw/train.csv
python src/cli.py train_all  --train data/raw/train.csv
python src/cli.py predict    --train data/raw/train.csv --test data/raw/test.csv --model gradient_boosting
```

- `eda`: Generate a compact EDA summary report and save it to `artifacts/eda_summary.txt`.
- `compare`: Compare all models and print cross-validation results.
- `train_best`: Evaluate all models, train the best (by ROC-AUC), and save it to `artifacts/`.
- `train_all`: Train and save all supported models to `artifacts/`.
- `train`: Alias for `train_best` (backward compatibility).
- `predict`: Load a saved model and predict on test data, saving predictions to `artifacts/predictions.csv`.

Notes:
- `eda --output <path>` lets you change where the summary report is saved.
- `predict --model <name>` must match an existing artifact file: `artifacts/<name>.pkl`.
- The best model can change between runs; check training output before selecting `--model`.

---

## Project Structure

- `src/preprocessor.py`   — Feature engineering and preprocessing
- `src/model_factory.py`  — Model definitions and pipelines
- `src/trainer.py`        — Training, evaluation, model selection, artifact saving
- `src/cli.py`            — CLI entry point (no business logic)
- `src/formatters/`       — Output formatting helpers
- `data/raw/`             — Original dataset files
- `data/processed/`       — (optional) Processed datasets
- `artifacts/`            — Saved models, predictions
- `tests/`                — Unit tests (synthetic, fast, no I/O)

---

## Running Tests

All tests are synthetic and fast:

```bash
PYTHONPATH=src pytest tests/ -v
```

---

## Notes
- All code is modular, DRY, and KISS.
- No business logic in CLI.
- Feature engineering and model selection are fully reproducible.
- See `agents.md` for architecture and style rules.
