"""Chart generators for model comparison results.

Each function saves one chart to output_dir and returns the saved path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_BASELINE_COLOR = "#6c757d"
_ENGINEERED_COLOR = "#2196F3"

_METRICS = [
    ("accuracy_mean", "Accuracy"),
    ("f1_mean", "F1"),
    ("roc_auc_mean", "ROC-AUC"),
]

_MODEL_DISPLAY = {
    "logistic_regression": "Logistic\nRegression",
    "random_forest": "Random\nForest",
    "gradient_boosting": "Gradient\nBoosting",
}


def plot_comparison_charts(
    baseline_results: dict[str, dict],
    engineered_results: dict[str, dict],
    output_dir: Path = Path("artifacts"),
) -> Path:
    """Save a grouped bar chart comparing baseline vs engineered per metric.

    Three subplots: Accuracy, F1, ROC-AUC.
    Returns path to the saved PNG file.
    """
    model_names = sorted(set(baseline_results) & set(engineered_results))
    labels = [_MODEL_DISPLAY.get(m, m) for m in model_names]
    x = np.arange(len(model_names))
    bar_width = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle(
        "Feature Engineering Impact — Baseline vs Engineered",
        fontsize=13,
        fontweight="bold",
    )

    for col, (mean_key, metric_label) in enumerate(_METRICS):
        ax = axes[col]
        base_vals = [baseline_results[m][mean_key] for m in model_names]
        eng_vals = [engineered_results[m][mean_key] for m in model_names]

        ax.bar(
            x - bar_width / 2,
            base_vals,
            bar_width,
            color=_BASELINE_COLOR,
            label="Baseline",
            alpha=0.85,
        )
        ax.bar(
            x + bar_width / 2,
            eng_vals,
            bar_width,
            color=_ENGINEERED_COLOR,
            label="Engineered",
            alpha=0.85,
        )

        lo = min(base_vals + eng_vals) - 0.04
        hi = max(base_vals + eng_vals) + 0.04
        ax.set_ylim(lo, hi)
        ax.set_title(metric_label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        if col == 0:
            ax.legend(fontsize=9)

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "comparison_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "comparison_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
