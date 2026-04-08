import pytest
from model_factory import get_model, available_models, MODEL_CONFIGS
from sklearn.pipeline import Pipeline


def test_available_models_returns_list():
    models = available_models()
    assert isinstance(models, list)
    assert len(models) >= 1


def test_available_models_matches_model_configs():
    models = set(available_models())
    configs = set(MODEL_CONFIGS.keys())
    assert models == configs


def test_get_model_returns_pipeline():
    for name in available_models():
        pipeline = get_model(name)
        assert isinstance(pipeline, Pipeline)


def test_get_model_raises_for_unknown():
    with pytest.raises(ValueError):
        get_model("not_a_model")


def test_pipeline_has_scaler_and_model():
    for name in available_models():
        pipeline = get_model(name)
        step_names = [step for step, _ in pipeline.steps]
        assert "scaler" in step_names
        assert "model" in step_names
