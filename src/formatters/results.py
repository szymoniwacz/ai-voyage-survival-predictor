"""Formatting helpers for trainer results."""

from __future__ import annotations


def format_comparison_table(results: dict[str, dict]) -> str:
    """Return a plain-text table summarising model comparison results."""
    header = f"{'Model':<25} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}"
    separator = "-" * len(header)
    rows = [header, separator]
    for model_name, metrics in results.items():
        row = (
            f"{model_name:<25} "
            f"{metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.4f}  "
            f"{metrics['f1_mean']:.4f}±{metrics['f1_std']:.4f}  "
            f"{metrics['roc_auc_mean']:.4f}±{metrics['roc_auc_std']:.4f}"
        )
        rows.append(row)
    return "\n".join(rows)


def format_best_model(name: str, metrics: dict) -> str:
    return (
        f"Best model: {name}\n"
        f"  ROC-AUC : {metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}\n"
        f"  Accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}\n"
        f"  F1      : {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}"
    )


def format_feature_set_delta_table(
    baseline_results: dict[str, dict], engineered_results: dict[str, dict]
) -> str:
    """Return a table with metric deltas (engineered - baseline) per model."""
    header = f"{'Model':<25} {'Δ Accuracy':>10} {'Δ F1':>10} {'Δ ROC-AUC':>12}"
    separator = "-" * len(header)
    rows = [
        header,
        separator,
    ]

    model_names = sorted(set(baseline_results) & set(engineered_results))
    for model_name in model_names:
        base = baseline_results[model_name]
        eng = engineered_results[model_name]

        delta_accuracy = eng["accuracy_mean"] - base["accuracy_mean"]
        delta_f1 = eng["f1_mean"] - base["f1_mean"]
        delta_roc_auc = eng["roc_auc_mean"] - base["roc_auc_mean"]

        rows.append(
            f"{model_name:<25} "
            f"{delta_accuracy:+.4f} "
            f"{delta_f1:+.4f} "
            f"{delta_roc_auc:+.4f}"
        )

    rows.append("Note: Positive deltas mean engineered features performed better.")
    return "\n".join(rows)


def format_fold_stability_table(
    baseline_folds: dict[str, list[float]],
    engineered_folds: dict[str, list[float]],
) -> str:
    """Return a per-fold ROC-AUC stability table (engineered vs baseline).

    Shows for each model how many CV folds engineered improved or degraded
    the score, and the worst single-fold delta.
    """
    n_folds = len(next(iter(baseline_folds.values())))
    header = f"{'Model':<25} {'Improved':>9} {'Degraded':>9} {'Worst fold Δ':>13}"
    separator = "-" * len(header)
    rows = [
        f"Fold-level ROC-AUC stability ({n_folds} folds, engineered vs baseline)",
        header,
        separator,
    ]

    model_names = sorted(set(baseline_folds) & set(engineered_folds))
    for model_name in model_names:
        base = baseline_folds[model_name]
        eng = engineered_folds[model_name]
        deltas = [e - b for e, b in zip(eng, base)]
        improved = sum(1 for d in deltas if d >= 0)
        degraded = n_folds - improved
        worst_delta = min(deltas)
        rows.append(
            f"{model_name:<25} "
            f"{improved:>4} / {n_folds:<3} "
            f"{degraded:>4} / {n_folds:<3} "
            f"{worst_delta:>+.4f}"
        )

    return "\n".join(rows)
