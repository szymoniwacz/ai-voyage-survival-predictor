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
