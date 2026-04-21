"""Tests for src/cli.py.

The CLI is a thin delegation layer, so tests focus on:
- argument routing (right command handler is called)
- that handlers call the correct modules
- that output files are written in the expected place

Heavy ML work is mocked out so tests run instantly.
"""

import argparse
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

import cli
from cli import _split, cmd_compare, cmd_train, cmd_train_all, cmd_predict, cmd_eda


# ---------------------------------------------------------------------------
# _split helper
# ---------------------------------------------------------------------------


def test_split_returns_X_and_y():
    df = pd.DataFrame({"Survived": [0, 1], "Age": [22.0, 38.0], "Fare": [7.25, 71.28]})
    X, y = _split(df)
    assert "Survived" not in X.columns
    assert list(y) == [0, 1]
    assert list(X.columns) == ["Age", "Fare"]


# ---------------------------------------------------------------------------
# cmd_compare
# ---------------------------------------------------------------------------


@patch("cli.format_comparison_table", return_value="TABLE")
@patch("cli.compare_models_folds", return_value={"logistic_regression": [0.8, 0.85]})
@patch(
    "cli.compare_models", return_value={"logistic_regression": {"roc_auc_mean": 0.8}}
)
@patch("cli.preprocess", return_value=pd.DataFrame({"Survived": [0, 1]}))
@patch("cli.preprocess_baseline", return_value=pd.DataFrame({"Survived": [0, 1]}))
@patch("cli.load_data", return_value=pd.DataFrame({"a": [1]}))
def test_cmd_compare_calls_compare_models(
    mock_load,
    mock_preprocess_baseline,
    mock_preprocess,
    mock_compare,
    mock_folds,
    mock_fmt,
    capsys,
):
    args = argparse.Namespace(train="data/raw/train.csv", feature_set="engineered")
    cmd_compare(args)
    mock_compare.assert_called_once()
    mock_preprocess.assert_called_once()
    out = capsys.readouterr().out
    assert "TABLE" in out


@patch("cli.format_comparison_table", return_value="TABLE")
@patch("cli.compare_models_folds", return_value={"logistic_regression": [0.8, 0.85]})
@patch(
    "cli.compare_models", return_value={"logistic_regression": {"roc_auc_mean": 0.8}}
)
@patch("cli.preprocess", return_value=pd.DataFrame({"Survived": [0, 1]}))
@patch("cli.preprocess_baseline", return_value=pd.DataFrame({"Survived": [0, 1]}))
@patch("cli.load_data", return_value=pd.DataFrame({"a": [1]}))
def test_cmd_compare_baseline_uses_baseline_preprocessor(
    mock_load,
    mock_preprocess_baseline,
    mock_preprocess,
    mock_compare,
    mock_folds,
    mock_fmt,
    capsys,
):
    args = argparse.Namespace(train="data/raw/train.csv", feature_set="baseline")
    cmd_compare(args)
    mock_compare.assert_called_once()
    mock_preprocess_baseline.assert_called_once()
    mock_preprocess.assert_not_called()
    out = capsys.readouterr().out
    assert "TABLE" in out


@patch("cli.format_comparison_table", return_value="TABLE")
@patch("cli.format_feature_set_delta_table", return_value="DELTA")
@patch("cli.format_fold_stability_table", return_value="STABILITY")
@patch("cli.compare_models_folds", return_value={"logistic_regression": [0.85, 0.82]})
@patch(
    "cli.compare_models", return_value={"logistic_regression": {"roc_auc_mean": 0.8}}
)
@patch("cli.preprocess", return_value=pd.DataFrame({"Survived": [0, 1]}))
@patch("cli.preprocess_baseline", return_value=pd.DataFrame({"Survived": [0, 1]}))
@patch("cli.load_data", return_value=pd.DataFrame({"a": [1]}))
def test_cmd_compare_both_runs_two_comparisons(
    mock_load,
    mock_preprocess_baseline,
    mock_preprocess,
    mock_compare,
    mock_folds,
    mock_stability,
    mock_delta,
    mock_fmt,
    capsys,
):
    args = argparse.Namespace(train="data/raw/train.csv", feature_set="both")
    cmd_compare(args)
    assert mock_compare.call_count == 2
    assert mock_folds.call_count == 2
    mock_preprocess_baseline.assert_called_once()
    mock_preprocess.assert_called_once()
    mock_delta.assert_called_once()
    mock_stability.assert_called_once()
    out = capsys.readouterr().out
    assert "Baseline features" in out
    assert "Engineered features" in out
    assert "Engineered vs Baseline (delta)" in out
    assert "DELTA" in out
    assert "STABILITY" in out


@patch("cli.plot_comparison_charts", return_value="artifacts/comparison_metrics.png")
@patch("cli.format_comparison_table", return_value="TABLE")
@patch("cli.format_feature_set_delta_table", return_value="DELTA")
@patch("cli.format_fold_stability_table", return_value="STABILITY")
@patch("cli.compare_models_folds", return_value={"logistic_regression": [0.85, 0.82]})
@patch(
    "cli.compare_models", return_value={"logistic_regression": {"roc_auc_mean": 0.8}}
)
@patch("cli.preprocess", return_value=pd.DataFrame({"Survived": [0, 1]}))
@patch("cli.preprocess_baseline", return_value=pd.DataFrame({"Survived": [0, 1]}))
@patch("cli.load_data", return_value=pd.DataFrame({"a": [1]}))
def test_cmd_compare_both_with_charts_saves_charts(
    mock_load,
    mock_preprocess_baseline,
    mock_preprocess,
    mock_compare,
    mock_folds,
    mock_stability,
    mock_delta,
    mock_fmt,
    mock_plot_comparison,
    capsys,
):
    args = argparse.Namespace(
        train="data/raw/train.csv", feature_set="both", charts=True
    )
    cmd_compare(args)
    mock_plot_comparison.assert_called_once()
    out = capsys.readouterr().out
    assert "Chart saved to" in out
    assert "comparison_metrics.png" in out


# ---------------------------------------------------------------------------
# cmd_train
# ---------------------------------------------------------------------------


@patch("cli.train_best_model", return_value=("random_forest", MagicMock()))
@patch("cli.format_best_model", return_value="BEST")
@patch("cli.format_comparison_table", return_value="TABLE")
@patch(
    "cli.evaluate_model",
    return_value={
        "accuracy_mean": 0.8,
        "accuracy_std": 0.01,
        "f1_mean": 0.75,
        "f1_std": 0.02,
        "roc_auc_mean": 0.82,
        "roc_auc_std": 0.01,
    },
)
@patch("cli.available_models", return_value=["random_forest"])
@patch("cli.preprocess")
@patch("cli.load_data", return_value=pd.DataFrame({"a": [1]}))
def test_cmd_train_calls_train_best_model(
    mock_load,
    mock_preprocess,
    mock_available,
    mock_eval,
    mock_fmt,
    mock_best_fmt,
    mock_train,
    capsys,
):
    processed = pd.DataFrame(
        {
            "Survived": [0, 1],
            "Age": [22.0, 38.0],
        }
    )
    mock_preprocess.return_value = processed
    args = argparse.Namespace(train="data/raw/train.csv", feature_set="engineered")
    cmd_train(args)
    mock_train.assert_called_once()
    out = capsys.readouterr().out
    assert "TABLE" in out


@patch("cli.train_best_model", return_value=("baseline_random_forest", MagicMock()))
@patch("cli.format_best_model", return_value="BEST")
@patch("cli.format_comparison_table", return_value="TABLE")
@patch(
    "cli.evaluate_model",
    return_value={
        "accuracy_mean": 0.8,
        "accuracy_std": 0.01,
        "f1_mean": 0.75,
        "f1_std": 0.02,
        "roc_auc_mean": 0.82,
        "roc_auc_std": 0.01,
    },
)
@patch("cli.available_models", return_value=["random_forest"])
@patch("cli.preprocess")
@patch("cli.preprocess_baseline")
@patch("cli.load_data", return_value=pd.DataFrame({"a": [1]}))
def test_cmd_train_baseline_uses_baseline_preprocessor_and_prefix(
    mock_load,
    mock_preprocess_baseline,
    mock_preprocess,
    mock_available,
    mock_eval,
    mock_fmt,
    mock_best_fmt,
    mock_train,
    capsys,
):
    processed = pd.DataFrame({"Survived": [0, 1], "Age": [22.0, 38.0]})
    mock_preprocess_baseline.return_value = processed
    args = argparse.Namespace(train="data/raw/train.csv", feature_set="baseline")

    cmd_train(args)

    mock_preprocess_baseline.assert_called_once()
    mock_preprocess.assert_not_called()
    mock_train.assert_called_once_with(processed, model_name_prefix="baseline_")
    out = capsys.readouterr().out
    assert "Model saved to artifacts/baseline_random_forest.pkl" in out


@patch(
    "cli.train_best_model",
    side_effect=[
        ("baseline_random_forest", MagicMock()),
        ("random_forest", MagicMock()),
    ],
)
@patch("cli.format_best_model", return_value="BEST")
@patch("cli.format_comparison_table", return_value="TABLE")
@patch(
    "cli.evaluate_model",
    return_value={
        "accuracy_mean": 0.8,
        "accuracy_std": 0.01,
        "f1_mean": 0.75,
        "f1_std": 0.02,
        "roc_auc_mean": 0.82,
        "roc_auc_std": 0.01,
    },
)
@patch("cli.available_models", return_value=["random_forest"])
@patch("cli.preprocess")
@patch("cli.preprocess_baseline")
@patch("cli.load_data", return_value=pd.DataFrame({"a": [1]}))
def test_cmd_train_both_runs_baseline_and_engineered(
    mock_load,
    mock_preprocess_baseline,
    mock_preprocess,
    mock_available,
    mock_eval,
    mock_fmt,
    mock_best_fmt,
    mock_train,
    capsys,
):
    baseline_processed = pd.DataFrame({"Survived": [0, 1], "Age": [22.0, 38.0]})
    engineered_processed = pd.DataFrame(
        {"Survived": [0, 1], "Age": [22.0, 38.0], "Title": [0, 1]}
    )
    mock_preprocess_baseline.return_value = baseline_processed
    mock_preprocess.return_value = engineered_processed
    args = argparse.Namespace(train="data/raw/train.csv", feature_set="both")

    cmd_train(args)

    assert mock_train.call_count == 2
    out = capsys.readouterr().out
    assert "Evaluating models on baseline features" in out
    assert "Evaluating models on engineered features" in out


@patch("cli.train_all_models", return_value={"random_forest": MagicMock()})
@patch("cli.format_comparison_table", return_value="TABLE")
@patch(
    "cli.evaluate_model",
    return_value={
        "accuracy_mean": 0.8,
        "accuracy_std": 0.01,
        "f1_mean": 0.75,
        "f1_std": 0.02,
        "roc_auc_mean": 0.82,
        "roc_auc_std": 0.01,
    },
)
@patch("cli.available_models", return_value=["random_forest"])
@patch("cli.preprocess")
@patch("cli.load_data", return_value=pd.DataFrame({"a": [1]}))
def test_cmd_train_all_calls_train_all_models(
    mock_load,
    mock_preprocess,
    mock_available,
    mock_eval,
    mock_fmt,
    mock_train_all,
    capsys,
):
    processed = pd.DataFrame(
        {
            "Survived": [0, 1],
            "Age": [22.0, 38.0],
        }
    )
    mock_preprocess.return_value = processed
    args = argparse.Namespace(train="data/raw/train.csv", feature_set="engineered")
    cmd_train_all(args)
    mock_train_all.assert_called_once()
    out = capsys.readouterr().out
    assert "TABLE" in out


# ---------------------------------------------------------------------------
# cmd_predict
# ---------------------------------------------------------------------------


@patch("cli.predict", return_value=np.array([0, 1]))
@patch("cli.load_model", return_value=MagicMock())
@patch("cli.preprocess")
@patch("cli.load_data")
def test_cmd_predict_saves_predictions_csv(
    mock_load,
    mock_preprocess,
    mock_load_model,
    mock_predict,
    tmp_path,
    monkeypatch,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "artifacts").mkdir()

    train_df = pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Survived": [0, 1],
            "Age": [22.0, 38.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "PassengerId": [3, 4],
            "Age": [26.0, 35.0],
        }
    )

    mock_load.side_effect = [train_df, test_df]

    combined_processed = pd.DataFrame(
        {
            "Age": [22.0, 38.0, 26.0, 35.0],
        }
    )
    mock_preprocess.return_value = combined_processed

    args = argparse.Namespace(
        train="data/raw/train.csv",
        test="data/raw/test.csv",
        model="random_forest",
    )
    cmd_predict(args)
    out = capsys.readouterr().out

    output_csv = tmp_path / "artifacts" / "predictions.csv"
    assert output_csv.exists()
    result = pd.read_csv(output_csv)
    assert list(result.columns) == ["PassengerId", "Survived"]
    assert len(result) == 2
    assert "Prediction summary: 1 survived, 1 did not survive (total: 2)" in out


@patch("cli.predict", return_value=np.array([0, 1]))
@patch(
    "cli.load_model",
    side_effect=FileNotFoundError(
        "No saved model found at /fake/path/random_forest.pkl"
    ),
)
@patch("cli.preprocess", return_value=pd.DataFrame({"Age": [22.0, 38.0, 26.0, 35.0]}))
@patch("cli.load_data")
def test_cmd_predict_missing_model_prints_error_and_exits(
    mock_load, mock_preprocess, mock_load_model, mock_predict, capsys
):
    train_df = pd.DataFrame(
        {"PassengerId": [1, 2], "Survived": [0, 1], "Age": [22.0, 38.0]}
    )
    test_df = pd.DataFrame({"PassengerId": [3, 4], "Age": [26.0, 35.0]})
    mock_load.side_effect = [train_df, test_df]
    args = argparse.Namespace(
        train="data/raw/train.csv",
        test="data/raw/test.csv",
        model="random_forest",
    )
    with pytest.raises(SystemExit) as excinfo:
        cmd_predict(args)
    out = capsys.readouterr().out
    assert "ERROR: No saved model found at" in out
    assert "Did you run 'python src/cli.py train'?" in out
    assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# cmd_eda
# ---------------------------------------------------------------------------


@patch("cli.save_eda_summary")
@patch("cli.format_eda_summary", return_value="EDA Summary\nRows: 2\n")
@patch("cli.build_eda_summary", return_value={"row_count": 2, "column_count": 2})
@patch("cli.load_data", return_value=pd.DataFrame({"Survived": [0, 1]}))
def test_cmd_eda_builds_and_saves_summary(
    mock_load, mock_build, mock_format, mock_save, capsys
):
    mock_save.return_value = "artifacts/eda_summary.txt"
    args = argparse.Namespace(
        train="data/raw/train.csv", output="artifacts/eda_summary.txt"
    )

    cmd_eda(args)

    mock_load.assert_called_once_with("data/raw/train.csv", data_dir=".")
    mock_build.assert_called_once()
    mock_format.assert_called_once_with({"row_count": 2, "column_count": 2})
    mock_save.assert_called_once_with(
        {"row_count": 2, "column_count": 2}, "artifacts/eda_summary.txt"
    )
    out = capsys.readouterr().out
    assert "EDA Summary" in out
    assert "Rows: 2" in out
    assert "EDA summary saved to artifacts/eda_summary.txt" in out


# ---------------------------------------------------------------------------
# main() – argument parsing
# ---------------------------------------------------------------------------


def test_main_unknown_command_exits():
    with pytest.raises(SystemExit):
        with patch("sys.argv", ["cli.py"]):
            cli.main()


def test_main_routes_compare(monkeypatch):
    called = {}

    def fake_compare(args):
        called["cmd"] = "compare"

    monkeypatch.setattr(cli, "cmd_compare", fake_compare)
    with patch(
        "sys.argv",
        [
            "cli.py",
            "compare",
            "--train",
            "data/raw/train.csv",
            "--feature-set",
            "baseline",
        ],
    ):
        cli.main()

    assert called.get("cmd") == "compare"


def test_main_routes_eda(monkeypatch):
    called = {}

    def fake_eda(args):
        called["cmd"] = "eda"

    monkeypatch.setattr(cli, "cmd_eda", fake_eda)
    with patch("sys.argv", ["cli.py", "eda", "--train", "data/raw/train.csv"]):
        cli.main()

    assert called.get("cmd") == "eda"


def test_main_routes_train(monkeypatch):
    called = {}

    def fake_train(args):
        called["cmd"] = "train"

    monkeypatch.setattr(cli, "cmd_train", fake_train)
    with patch("sys.argv", ["cli.py", "train", "--train", "data/raw/train.csv"]):
        cli.main()

    assert called.get("cmd") == "train"


def test_main_routes_train_best(monkeypatch):
    called = {}

    def fake_train(args):
        called["cmd"] = "train"

    monkeypatch.setattr(cli, "cmd_train", fake_train)
    with patch("sys.argv", ["cli.py", "train_best", "--train", "data/raw/train.csv"]):
        cli.main()

    assert called.get("cmd") == "train"


def test_main_routes_train_all(monkeypatch):
    called = {}

    def fake_train_all(args):
        called["cmd"] = "train_all"

    monkeypatch.setattr(cli, "cmd_train_all", fake_train_all)
    with patch("sys.argv", ["cli.py", "train_all", "--train", "data/raw/train.csv"]):
        cli.main()

    assert called.get("cmd") == "train_all"


def test_main_routes_predict(monkeypatch):
    called = {}

    def fake_predict(args):
        called["cmd"] = "predict"

    monkeypatch.setattr(cli, "cmd_predict", fake_predict)
    with patch("sys.argv", ["cli.py", "predict", "--model", "random_forest"]):
        cli.main()

    assert called.get("cmd") == "predict"
