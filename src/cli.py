"""CLI entry point.

Commands:
    train_best — compare all models, pick the best one, save it to artifacts/
    train_all  — train all models and save each to artifacts/
    train      — alias for train_best (backward compatibility)
    compare.   — compare all models and print a results table (no saving)
    eda        — generate a compact EDA summary and save it to artifacts/
    predict    — load a saved model and predict on test.csv

Usage:
    python src/cli.py train_best --train data/raw/train.csv
    python src/cli.py train_all  --train data/raw/train.csv
    python src/cli.py train      --train data/raw/train.csv
    python src/cli.py compare.   --train data/raw/train.csv
    python src/cli.py eda        --train data/raw/train.csv
  python src/cli.py predict --train data/raw/train.csv --test data/raw/test.csv --model random_forest
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is on the path when invoked from the project root
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data
from preprocessor import preprocess
from eda import build_eda_summary, format_eda_summary, save_eda_summary
from trainer import (
    compare_models,
    train_best_model,
    train_all_models,
    load_model,
    predict,
    evaluate_model,
)
from model_factory import available_models
from formatters.results import format_comparison_table, format_best_model


def cmd_compare(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.train} …")
    df = load_data(args.train, data_dir=".")
    processed = preprocess(df)

    print("Running cross-validation for all models …\n")
    results = compare_models(processed)
    print(format_comparison_table(results))


def cmd_train(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.train} …")
    df = load_data(args.train, data_dir=".")
    processed = preprocess(df)

    print("Evaluating models …")
    X, y = _split(processed)
    results = {name: evaluate_model(name, X, y) for name in available_models()}
    print(format_comparison_table(results))

    best_name = max(results, key=lambda n: results[n]["roc_auc_mean"])
    print(f"\nTraining best model ({best_name}) on full dataset …")
    best_name, _ = train_best_model(processed)
    print(f"Model saved to artifacts/{best_name}.pkl")
    print(format_best_model(best_name, results[best_name]))


def cmd_train_all(args: argparse.Namespace) -> None:
    print(f"Loading data from {args.train} …")
    df = load_data(args.train, data_dir=".")
    processed = preprocess(df)

    print("Evaluating models …")
    X, y = _split(processed)
    results = {name: evaluate_model(name, X, y) for name in available_models()}
    print(format_comparison_table(results))

    print("\nTraining all models on full dataset …")
    trained = train_all_models(processed)
    for model_name in trained:
        print(f"Model saved to artifacts/{model_name}.pkl")


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

    # train_all
    p_train_all = subparsers.add_parser("train_all", help="Train and save all models")
    p_train_all.add_argument(
        "--train", default="data/raw/train.csv", help="Path to training CSV"
    )

    # train (alias for train_best)
    p_train_alias = subparsers.add_parser(
        "train", help="Alias for train_best (train and save best model)"
    )
    p_train_alias.add_argument(
        "--train", default="data/raw/train.csv", help="Path to training CSV"
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
