import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from model_factory import get_model, available_models

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"

CV_FOLDS = 5
RANDOM_STATE = 42

# Use string-based scorers so sklearn resolves the right API automatically.
SCORERS = ["accuracy", "f1", "roc_auc"]


def _split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a preprocessed DataFrame into features X and target y."""
    if "Survived" not in df.columns:
        raise ValueError("DataFrame must contain a 'Survived' column.")
    y = df["Survived"]
    X = df.drop(columns=["Survived"])
    return X, y


def evaluate_model(name: str, X: pd.DataFrame, y: pd.Series) -> dict:
    """Run cross-validated evaluation for a single model.

    Returns a dict with mean and std for each metric.
    """
    pipeline = get_model(name)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(pipeline, X, y, cv=cv, scoring=SCORERS)
    return {
        "accuracy_mean": float(np.mean(results["test_accuracy"])),
        "accuracy_std": float(np.std(results["test_accuracy"])),
        "f1_mean": float(np.mean(results["test_f1"])),
        "f1_std": float(np.std(results["test_f1"])),
        "roc_auc_mean": float(np.mean(results["test_roc_auc"])),
        "roc_auc_std": float(np.std(results["test_roc_auc"])),
    }


def compare_models(df: pd.DataFrame) -> dict[str, dict]:
    """Evaluate all available models and return results keyed by model name."""
    X, y = _split_features_target(df)
    return {name: evaluate_model(name, X, y) for name in available_models()}


def _select_best_model_name(df: pd.DataFrame) -> str:
    """Select best model by ROC-AUC mean from cross-validation comparison."""
    comparison = compare_models(df)
    return max(comparison, key=lambda n: comparison[n]["roc_auc_mean"])


def _artifact_model_name(name: str, model_name_prefix: str) -> str:
    return f"{model_name_prefix}{name}" if model_name_prefix else name


def train_best_model(
    df: pd.DataFrame, model_name_prefix: str = ""
) -> tuple[str, object]:
    """Train the best model (by ROC-AUC) on the full dataset and save it.

    Returns the saved model artifact name and the fitted pipeline.
    """
    best_name = _select_best_model_name(df)
    X, y = _split_features_target(df)
    best_pipeline = get_model(best_name)
    best_pipeline.fit(X, y)
    saved_name = _artifact_model_name(best_name, model_name_prefix)
    _save_model(saved_name, best_pipeline)
    return saved_name, best_pipeline


def train_all_models(
    df: pd.DataFrame, model_name_prefix: str = ""
) -> dict[str, object]:
    """Train all available models on the full dataset and save each artifact.

    Returns a mapping: saved_model_name -> fitted pipeline.
    """
    X, y = _split_features_target(df)
    trained_models: dict[str, object] = {}

    for model_name in available_models():
        pipeline = get_model(model_name)
        pipeline.fit(X, y)
        saved_name = _artifact_model_name(model_name, model_name_prefix)
        _save_model(saved_name, pipeline)
        trained_models[saved_name] = pipeline

    return trained_models


def _save_model(name: str, pipeline: object) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    return path


def load_model(name: str) -> object:
    path = ARTIFACTS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(pipeline: object, df: pd.DataFrame) -> np.ndarray:
    """Generate binary predictions from an already-preprocessed feature DataFrame."""
    return pipeline.predict(df)
