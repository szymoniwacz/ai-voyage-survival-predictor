import pytest
import numpy as np
import pandas as pd

from model_factory import get_model, available_models
from trainer import (
    _split_features_target,
    evaluate_model,
    compare_models,
    train_best_model,
    train_all_models,
    predict,
)


# ---------------------------------------------------------------------------
# Shared fixture — small synthetic preprocessed dataset
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_df():
    """A minimal preprocessed DataFrame that mirrors preprocess() output."""
    rng = np.random.default_rng(0)
    n = 60
    return pd.DataFrame(
        {
            "Survived": rng.integers(0, 2, n),
            "Pclass": rng.choice([1, 2, 3], n),
            "Sex": rng.integers(0, 2, n),
            "Age": rng.uniform(1, 80, n),
            "SibSp": rng.integers(0, 3, n),
            "Parch": rng.integers(0, 3, n),
            "Fare": rng.uniform(5, 200, n),
            "Embarked": rng.choice([0, 1, 2], n),
            "Title": rng.choice([0, 1, 2, 3, 4], n),
            "HasCabin": rng.integers(0, 2, n),
            "CabinDeck": rng.choice(range(9), n),
            "FamilySize": rng.integers(1, 8, n),
            "IsAlone": rng.integers(0, 2, n),
            "TicketGroupSize": rng.integers(1, 5, n),
            "FarePerPerson": rng.uniform(2, 100, n),
        }
    )


# ---------------------------------------------------------------------------
# model_factory
# ---------------------------------------------------------------------------


def test_available_models_returns_list():
    models = available_models()
    assert isinstance(models, list)
    assert len(models) >= 1


def test_available_models_contains_expected():
    models = available_models()
    assert "logistic_regression" in models
    assert "random_forest" in models
    assert "gradient_boosting" in models


def test_get_model_returns_pipeline():
    from sklearn.pipeline import Pipeline

    pipeline = get_model("logistic_regression")
    assert isinstance(pipeline, Pipeline)


def test_get_model_raises_for_unknown():
    with pytest.raises(ValueError, match="Unknown model"):
        get_model("not_a_model")


def test_each_model_has_scaler_and_model_steps():
    for name in available_models():
        pipeline = get_model(name)
        step_names = [s for s, _ in pipeline.steps]
        assert "scaler" in step_names
        assert "model" in step_names


# ---------------------------------------------------------------------------
# trainer – _split_features_target
# ---------------------------------------------------------------------------


def test_split_separates_survived(synthetic_df):
    X, y = _split_features_target(synthetic_df)
    assert "Survived" not in X.columns
    assert y.name == "Survived"
    assert len(X) == len(y)


def test_split_raises_without_survived():
    df = pd.DataFrame({"Age": [20, 30]})
    with pytest.raises(ValueError, match="Survived"):
        _split_features_target(df)


# ---------------------------------------------------------------------------
# trainer – evaluate_model
# ---------------------------------------------------------------------------


def test_evaluate_model_returns_expected_keys(synthetic_df):
    X, y = _split_features_target(synthetic_df)
    result = evaluate_model("logistic_regression", X, y)
    expected_keys = {
        "accuracy_mean",
        "accuracy_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
    }
    assert expected_keys == set(result.keys())


def test_evaluate_model_metrics_are_floats(synthetic_df):
    X, y = _split_features_target(synthetic_df)
    result = evaluate_model("logistic_regression", X, y)
    for key, value in result.items():
        assert isinstance(value, float), f"{key} is not a float"


def test_evaluate_model_metrics_in_range(synthetic_df):
    X, y = _split_features_target(synthetic_df)
    result = evaluate_model("random_forest", X, y)
    assert 0.0 <= result["accuracy_mean"] <= 1.0
    assert 0.0 <= result["f1_mean"] <= 1.0
    assert 0.0 <= result["roc_auc_mean"] <= 1.0


# ---------------------------------------------------------------------------
# trainer – compare_models
# ---------------------------------------------------------------------------


def test_compare_models_covers_all_models(synthetic_df):
    results = compare_models(synthetic_df)
    for name in available_models():
        assert name in results


def test_compare_models_each_result_has_metrics(synthetic_df):
    results = compare_models(synthetic_df)
    for name, metrics in results.items():
        assert "roc_auc_mean" in metrics, f"{name} missing roc_auc_mean"


# ---------------------------------------------------------------------------
# trainer – train_best_model
# ---------------------------------------------------------------------------


def test_train_best_model_returns_name_and_pipeline(
    synthetic_df, tmp_path, monkeypatch
):
    import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "ARTIFACTS_DIR", tmp_path)

    best_name, pipeline = train_best_model(synthetic_df)
    assert best_name in available_models()
    assert pipeline is not None


def test_train_best_model_saves_pkl(synthetic_df, tmp_path, monkeypatch):
    import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "ARTIFACTS_DIR", tmp_path)

    best_name, _ = train_best_model(synthetic_df)
    assert (tmp_path / f"{best_name}.pkl").exists()


def test_train_all_models_saves_all_pkls(synthetic_df, tmp_path, monkeypatch):
    import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "ARTIFACTS_DIR", tmp_path)

    trained = train_all_models(synthetic_df)
    assert set(trained.keys()) == set(available_models())

    for model_name in available_models():
        assert (tmp_path / f"{model_name}.pkl").exists()


def test_train_best_model_with_prefix_saves_prefixed_artifact(
    synthetic_df, tmp_path, monkeypatch
):
    import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "ARTIFACTS_DIR", tmp_path)

    saved_name, _ = train_best_model(synthetic_df, model_name_prefix="baseline_")
    assert saved_name.startswith("baseline_")
    assert (tmp_path / f"{saved_name}.pkl").exists()


def test_train_all_models_with_prefix_saves_prefixed_artifacts(
    synthetic_df, tmp_path, monkeypatch
):
    import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "ARTIFACTS_DIR", tmp_path)

    trained = train_all_models(synthetic_df, model_name_prefix="baseline_")
    assert len(trained) == len(available_models())
    for saved_name in trained:
        assert saved_name.startswith("baseline_")
        assert (tmp_path / f"{saved_name}.pkl").exists()


# ---------------------------------------------------------------------------
# trainer – predict
# ---------------------------------------------------------------------------


def test_predict_returns_binary_array(synthetic_df, tmp_path, monkeypatch):
    import trainer as trainer_module

    monkeypatch.setattr(trainer_module, "ARTIFACTS_DIR", tmp_path)

    _, pipeline = train_best_model(synthetic_df)

    X, _ = _split_features_target(synthetic_df)
    preds = predict(pipeline, X)
    assert set(preds).issubset({0, 1})
    assert len(preds) == len(X)


# ---------------------------------------------------------------------------
# formatters
# ---------------------------------------------------------------------------


def test_format_comparison_table_contains_model_names():
    from formatters.results import format_comparison_table

    results = {
        "logistic_regression": {
            "accuracy_mean": 0.8,
            "accuracy_std": 0.02,
            "f1_mean": 0.75,
            "f1_std": 0.03,
            "roc_auc_mean": 0.82,
            "roc_auc_std": 0.02,
        }
    }
    table = format_comparison_table(results)
    assert "logistic_regression" in table
    assert "0.8000" in table


def test_format_best_model_includes_name():
    from formatters.results import format_best_model

    metrics = {
        "accuracy_mean": 0.85,
        "accuracy_std": 0.01,
        "f1_mean": 0.80,
        "f1_std": 0.01,
        "roc_auc_mean": 0.88,
        "roc_auc_std": 0.01,
    }
    text = format_best_model("random_forest", metrics)
    assert "random_forest" in text
    assert "0.8800" in text


def test_format_feature_set_delta_table_contains_deltas():
    from formatters.results import format_feature_set_delta_table

    baseline = {
        "random_forest": {
            "accuracy_mean": 0.83,
            "f1_mean": 0.76,
            "roc_auc_mean": 0.87,
        }
    }
    engineered = {
        "random_forest": {
            "accuracy_mean": 0.84,
            "f1_mean": 0.78,
            "roc_auc_mean": 0.89,
        }
    }

    table = format_feature_set_delta_table(baseline, engineered)
    assert "random_forest" in table
    assert "+0.0100" in table
    assert "+0.0200" in table
    assert "Winner" not in table


def test_format_fold_stability_table_shows_degraded_folds():
    from formatters.results import format_fold_stability_table

    # 3 folds: engineered is worse in 2 of them
    baseline = {"random_forest": [0.88, 0.90, 0.85]}
    engineered = {"random_forest": [0.86, 0.92, 0.84]}

    table = format_fold_stability_table(baseline, engineered)
    assert "random_forest" in table
    assert "1 / 3" in table  # improved
    assert "2 / 3" in table  # degraded
    assert "-0.02" in table  # worst fold delta shows negative
