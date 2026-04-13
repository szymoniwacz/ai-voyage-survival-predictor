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

Recommended order:
```bash
python src/cli.py eda        --train data/raw/train.csv
python src/cli.py compare    --train data/raw/train.csv
python src/cli.py train_best --train data/raw/train.csv
python src/cli.py predict    --train data/raw/train.csv --test data/raw/test.csv --model <best_model_name>
```

---

## Results (Sample Run)

Cross-validation comparison from a recent run:

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| logistic_regression | 0.8193 +- 0.0125 | 0.7572 +- 0.0239 | 0.8637 +- 0.0179 |
| random_forest | 0.8417 +- 0.0084 | 0.7849 +- 0.0166 | 0.8796 +- 0.0214 |
| gradient_boosting | 0.8440 +- 0.0126 | 0.7882 +- 0.0199 | 0.8873 +- 0.0285 |

Best model in that run: `gradient_boosting`.

Interpretation:
- Gradient boosting gave the strongest ranking quality (ROC-AUC) and the best F1.
- Random forest was close, which is useful as a baseline sanity check.
- Logistic regression remained a solid lightweight reference model.

---

## Project Status

Current status: **v1 portfolio-ready**.

Completed end-to-end flow:
- EDA summary generation (`eda`) with console output and file artifact.
- Cross-validated model comparison (`compare`).
- Best-model training (`train_best`) and all-model training (`train_all`).
- Prediction workflow (`predict`) with saved model artifacts.
- Fast synthetic test suite covering core behavior.

---

## Trade-offs and Next Steps

Current trade-offs (intentional):
- Focused on readable, modular baseline workflow over advanced optimization tooling.
- Text-first EDA report instead of richer plots or dashboards.
- No hyperparameter search yet (kept model configs simple for learning clarity).

Natural next improvements:
- Add lightweight EDA plots saved to `artifacts/` (e.g., target distribution, age/fare histograms).
- Add a small `Results` artifact (JSON/CSV) saved from `compare` for easier run tracking.
- Add a compact model card section documenting assumptions and limitations.

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
