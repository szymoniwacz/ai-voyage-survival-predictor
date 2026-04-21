"""Formatting helpers for trainer results.

Domain logic only — visual layout is delegated to formatters.table.render_table.
"""

from __future__ import annotations

from formatters.table import render_table


def format_comparison_table(results: dict[str, dict]) -> str:
    """Return a plain-text table summarising model comparison results."""
    headers = ["Model", "Accuracy", "F1", "ROC-AUC"]
    rows = [
        [
            model_name,
            f"{m['accuracy_mean']:.4f}±{m['accuracy_std']:.4f}",
            f"{m['f1_mean']:.4f}±{m['f1_std']:.4f}",
            f"{m['roc_auc_mean']:.4f}±{m['roc_auc_std']:.4f}",
        ]
        for model_name, m in results.items()
    ]
    return render_table(headers, rows)


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
    headers = ["Model", "Δ Accuracy", "Δ F1", "Δ ROC-AUC"]
    rows = []
    for model_name in sorted(set(baseline_results) & set(engineered_results)):
        base = baseline_results[model_name]
        eng = engineered_results[model_name]
        rows.append(
            [
                model_name,
                f"{eng['accuracy_mean'] - base['accuracy_mean']:+.4f}",
                f"{eng['f1_mean'] - base['f1_mean']:+.4f}",
                f"{eng['roc_auc_mean'] - base['roc_auc_mean']:+.4f}",
            ]
        )
    return render_table(
        headers,
        rows,
        footer="Note: Positive deltas mean engineered features performed better.",
    )


def format_fold_stability_table(
    baseline_folds: dict[str, list[float]],
    engineered_folds: dict[str, list[float]],
) -> str:
    """Return a per-fold ROC-AUC stability table (engineered vs baseline).

    Shows for each model how many CV folds engineered improved or degraded
    the score, and the worst single-fold delta.
    """
    n_folds = len(next(iter(baseline_folds.values())))
    headers = ["Model", "Improved", "Degraded", "Worst fold Δ"]
    rows = []
    for model_name in sorted(set(baseline_folds) & set(engineered_folds)):
        base = baseline_folds[model_name]
        eng = engineered_folds[model_name]
        deltas = [e - b for e, b in zip(eng, base)]
        improved = sum(1 for d in deltas if d >= 0)
        degraded = n_folds - improved
        rows.append(
            [
                model_name,
                f"{improved}/{n_folds}",
                f"{degraded}/{n_folds}",
                f"{min(deltas):+.4f}",
            ]
        )
    return render_table(
        headers,
        rows,
        title=f"Fold-level ROC-AUC stability ({n_folds} folds, engineered vs baseline)",
    )
