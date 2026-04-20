from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Each entry defines a named model configuration.
# The pipeline wraps each model with a StandardScaler so that
# gradient-sensitive models (e.g. logistic regression) are not
# penalised by feature scale differences.
MODEL_CONFIGS: dict[str, Pipeline] = {
    "logistic_regression": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    # liblinear avoids scipy/lbfgs crashes on some Python builds
                    # while remaining suitable for binary classification here.
                    solver="liblinear",
                    C=1.0,
                ),
            ),
        ]
    ),
    "random_forest": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    ),
    "gradient_boosting": Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=42,
                ),
            ),
        ]
    ),
}


def get_model(name: str) -> Pipeline:
    """Return a fresh (unfitted) pipeline for the given model name."""
    if name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_CONFIGS.keys())}"
        )
    return clone(MODEL_CONFIGS[name])


def available_models() -> list[str]:
    return list(MODEL_CONFIGS.keys())
