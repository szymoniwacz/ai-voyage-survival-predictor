"""CLI entry point.

Commands:
    train_best — compare all models, pick the best one, save it to artifacts/
    train_all  — train all models and save each to artifacts/
    train      — alias for train_best (backward compatibility)
    compare    — compare all models and print a results table (no saving)
    eda        — generate a compact EDA summary and save it to artifacts/
    predict    — load a saved model and predict on test.csv

Usage:
    python -m src.cli train_best --train data/raw/train.csv
    python -m src.cli train_all  --train data/raw/train.csv
    python -m src.cli train      --train data/raw/train.csv
    python -m src.cli compare    --train data/raw/train.csv
    python -m src.cli eda        --train data/raw/train.csv
    python -m src.cli predict    --train data/raw/train.csv --test data/raw/test.csv --model random_forest
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is on the path when invoked from the project root
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data
from preprocessor import preprocess, preprocess_baseline
from eda import build_eda_summary, format_eda_summary, save_eda_summary
from trainer import (
    compare_models,
    compare_models_folds,
    train_best_model,
    train_all_models,
    load_model,
    predict,
    evaluate_model,
)
from model_factory import available_models
from formatters.results import (
    format_comparison_table,
    format_best_model,
    format_feature_set_delta_table,
    format_fold_stability_table,
)


def _prepare_feature_set(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    return preprocess_baseline(df) if feature_set == "baseline" else preprocess(df)


def cmd_compare(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.train} …")
    df = load_data(args.train, data_dir=".")

    def run_compare(feature_set: str) -> tuple[dict, dict]:
        if feature_set == "baseline":
            prepared = preprocess_baseline(df)
        else:
            prepared = preprocess(df)
        return compare_models(prepared), compare_models_folds(prepared)

    if args.feature_set == "both":
        print("Running cross-validation for baseline features …\n")
        baseline_results, baseline_folds = run_compare("baseline")
        print("Baseline features")
        print(format_comparison_table(baseline_results))

        print("\nRunning cross-validation for engineered features …\n")
        engineered_results, engineered_folds = run_compare("engineered")
        print("Engineered features")
        print(format_comparison_table(engineered_results))

        print("\nEngineered vs Baseline (delta)")
        print(format_feature_set_delta_table(baseline_results, engineered_results))

        print("\n" + format_fold_stability_table(baseline_folds, engineered_folds))
        return

    print(f"Running cross-validation for {args.feature_set} features …\n")
    results, _ = run_compare(args.feature_set)
    print(format_comparison_table(results))


def cmd_train(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.train} …")
    df = load_data(args.train, data_dir=".")
    feature_set = getattr(args, "feature_set", "engineered")

    def run_train_best(selected_set: str) -> None:
        processed = _prepare_feature_set(df, selected_set)

        print(f"Evaluating models on {selected_set} features …")
        X, y = _split(processed)
        results = {name: evaluate_model(name, X, y) for name in available_models()}
        print(format_comparison_table(results))

        best_name = max(results, key=lambda n: results[n]["roc_auc_mean"])
        model_prefix = "baseline_" if selected_set == "baseline" else ""
        print(f"\nTraining best model ({best_name}) on full dataset …")
        saved_name, _ = train_best_model(processed, model_name_prefix=model_prefix)
        print(f"Model saved to artifacts/{saved_name}.pkl")
        print(format_best_model(best_name, results[best_name]))

    if feature_set == "both":
        run_train_best("baseline")
        print("\n" + "-" * 58 + "\n")
        run_train_best("engineered")
        return

    run_train_best(feature_set)


def cmd_train_all(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.train} …")
    df = load_data(args.train, data_dir=".")
    feature_set = getattr(args, "feature_set", "engineered")

    def run_train_all(selected_set: str) -> None:
        processed = _prepare_feature_set(df, selected_set)

        print(f"Evaluating models on {selected_set} features …")
        X, y = _split(processed)
        results = {name: evaluate_model(name, X, y) for name in available_models()}
        print(format_comparison_table(results))

        model_prefix = "baseline_" if selected_set == "baseline" else ""
        print("\nTraining all models on full dataset …")
        trained = train_all_models(processed, model_name_prefix=model_prefix)
        for model_name in trained:
            print(f"Model saved to artifacts/{model_name}.pkl")

    if feature_set == "both":
        run_train_all("baseline")
        print("\n" + "-" * 58 + "\n")
        run_train_all("engineered")
        return

    run_train_all(feature_set)


def cmd_predict(args: argparse.Namespace) -> None:
    print(f"Loading training data from {args.train} to build preprocessor context …")
    train_df = load_data(args.train, data_dir=".")

    print(f"Loading test data from {args.test} …")
    test_df = load_data(args.test, data_dir=".")

    # Combine for consistent ticket group sizes / fare per person
    combined = preprocess(pd.concat([train_df, test_df], ignore_index=True))
    test_rows = combined.iloc[len(train_df) :]
    X_test = test_rows.drop(columns=["Survived"], errors="ignore")

    try:
        pipeline = load_model(args.model)
    except FileNotFoundError as e:
        print(
            f"ERROR: {e}\nDid you run 'python src/cli.py train'? Model must be trained and saved before prediction."
        )
        sys.exit(1)

    predictions = predict(pipeline, X_test)

    output = pd.DataFrame(
        {"PassengerId": test_df["PassengerId"], "Survived": predictions}
    )
    survivors_count = int(output["Survived"].sum())
    total_count = len(output)
    non_survivors_count = total_count - survivors_count

    print(
        "Prediction summary: "
        f"{survivors_count} survived, {non_survivors_count} did not survive "
        f"(total: {total_count})"
    )

    output_path = Path("artifacts") / "predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def cmd_eda(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.train} ...")
    df = load_data(args.train, data_dir=".")

    summary = build_eda_summary(df)
    print("\n" + format_eda_summary(summary))
    output_path = save_eda_summary(summary, args.output)
    print(f"EDA summary saved to {output_path}")


def _split(processed):
    y = processed["Survived"]
    X = processed.drop(columns=["Survived"])
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Titanic survival predictor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare all models")
    p_compare.add_argument(
        "--train", default="data/raw/train.csv", help="Path to training CSV"
    )
    p_compare.add_argument(
        "--feature-set",
        choices=["engineered", "baseline", "both"],
        default="engineered",
        help="Which features to compare models on",
    )

    # eda
    p_eda = subparsers.add_parser("eda", help="Generate EDA summary report")
    p_eda.add_argument(
        "--train", default="data/raw/train.csv", help="Path to training CSV"
    )
    p_eda.add_argument(
        "--output",
        default="artifacts/eda_summary.txt",
        help="Path to output EDA summary report",
    )

    # train_best
    p_train_best = subparsers.add_parser(
        "train_best", help="Train and save the best model"
    )
    p_train_best.add_argument(
        "--train", default="data/raw/train.csv", help="Path to training CSV"
    )
    p_train_best.add_argument(
        "--feature-set",
        choices=["engineered", "baseline", "both"],
        default="engineered",
        help="Which features to use for training",
    )

    # train_all
    p_train_all = subparsers.add_parser("train_all", help="Train and save all models")
    p_train_all.add_argument(
        "--train", default="data/raw/train.csv", help="Path to training CSV"
    )
    p_train_all.add_argument(
        "--feature-set",
        choices=["engineered", "baseline", "both"],
        default="engineered",
        help="Which features to use for training",
    )

    # train (alias for train_best)
    p_train_alias = subparsers.add_parser(
        "train", help="Alias for train_best (train and save best model)"
    )
    p_train_alias.add_argument(
        "--train", default="data/raw/train.csv", help="Path to training CSV"
    )
    p_train_alias.add_argument(
        "--feature-set",
        choices=["engineered", "baseline", "both"],
        default="engineered",
        help="Which features to use for training",
    )

    # predict
    p_predict = subparsers.add_parser("predict", help="Predict on test data")
    p_predict.add_argument(
        "--train", default="data/raw/train.csv", help="Path to training CSV"
    )
    p_predict.add_argument(
        "--test", default="data/raw/test.csv", help="Path to test CSV"
    )
    p_predict.add_argument(
        "--model", required=True, help="Saved model name (without .pkl)"
    )

    args = parser.parse_args()

    if args.command == "compare":
        cmd_compare(args)
    elif args.command == "eda":
        cmd_eda(args)
    elif args.command in {"train", "train_best"}:
        cmd_train(args)
    elif args.command == "train_all":
        cmd_train_all(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
