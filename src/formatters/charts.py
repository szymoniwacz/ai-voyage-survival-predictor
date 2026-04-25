"""Chart generators for model comparison results.

Each function saves one chart to output_dir and returns the saved path.
"""

from __future__ import annotations

import matplotlib.patches as mpatches
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_BASELINE_COLOR = "#6c757d"
_BETTER_COLOR = "#2196F3"  # engineered better than baseline
_WORSE_COLOR = "#F44336"  # engineered worse than baseline

_MODEL_DISPLAY = {
    "logistic_regression": "Logistic\nRegression",
    "random_forest": "Random\nForest",
    "gradient_boosting": "Gradient\nBoosting",
}


def plot_comparison_charts(
    baseline_folds: dict[str, list[float]],
    engineered_folds: dict[str, list[float]],
    output_dir: Path = Path("artifacts"),
) -> Path:
    """Save a self-explanatory fold-by-fold comparison: baseline vs engineered ROC-AUC.

    Each pair of bars shows how feature engineering changed the result for a given fold.
    Improvements and degradations are annotated directly on the chart.
    Returns path to the saved PNG file.
    """
    model_names = sorted(set(baseline_folds) & set(engineered_folds))
    n_folds = len(next(iter(baseline_folds.values())))
    fold_labels = [f"Fold {i + 1}" for i in range(n_folds)]
    x = np.arange(n_folds)
    bar_width = 0.35

    fig, axes = plt.subplots(
        1, len(model_names), figsize=(5 * len(model_names), 5), sharey=True
    )
    fig.suptitle(
        "Baseline vs Engineered Performance per Fold",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    if len(model_names) == 1:
        axes = [axes]

    # Shared zoomed Y range across all models and folds
    all_vals = [v for m in model_names for v in baseline_folds[m] + engineered_folds[m]]
    y_lo = max(0.0, min(all_vals) - 0.02)
    y_hi = max(all_vals) + 0.06  # headroom for value labels and annotations

    for ax, model_name in zip(axes, model_names):
        base_vals = baseline_folds[model_name]
        eng_vals = engineered_folds[model_name]

        eng_colors = [
            _BETTER_COLOR if eng >= base else _WORSE_COLOR
            for eng, base in zip(eng_vals, base_vals)
        ]

        ax.bar(
            x - bar_width / 2,
            base_vals,
            bar_width,
            color=_BASELINE_COLOR,
            alpha=0.75,
            label="Baseline",
        )
        for xi, ev, ec in zip(x, eng_vals, eng_colors):
            ax.bar(xi + bar_width / 2, ev, bar_width, color=ec, alpha=0.85)

        # Value labels above each bar
        for xi, bv, ev in zip(x, base_vals, eng_vals):
            ax.text(
                xi - bar_width / 2,
                bv + 0.001,
                f"{bv:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#444444",
            )
            ax.text(
                xi + bar_width / 2,
                ev + 0.001,
                f"{ev:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#222222",
                fontweight="bold",
            )

        # ↑ better / ↓ worse annotations near top of the shared y range
        annotation_y = y_hi - 0.04
        for xi, bv, ev in zip(x, base_vals, eng_vals):
            if ev > bv:
                ax.text(
                    xi,
                    annotation_y,
                    "↑ better",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=_BETTER_COLOR,
                    fontweight="bold",
                )
            else:
                ax.text(
                    xi,
                    annotation_y,
                    "↓ worse",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=_WORSE_COLOR,
                    fontweight="bold",
                )

        ax.set_ylim(y_lo, y_hi)
        ax.set_title(_MODEL_DISPLAY.get(model_name, model_name), fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.set_ylabel("ROC-AUC", fontsize=10)

    fig.text(
        0.5,
        -0.10,
        "Y-axis zoomed for readability",
        ha="center",
        fontsize=8,
        color="#888888",
        style="italic",
    )

    legend_handles = [
        mpatches.Patch(color=_BASELINE_COLOR, alpha=0.75, label="Baseline"),
        mpatches.Patch(color=_BETTER_COLOR, alpha=0.85, label="Engineered (better)"),
        mpatches.Patch(color=_WORSE_COLOR, alpha=0.85, label="Engineered (worse)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )

    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fold_comparison_clear.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
