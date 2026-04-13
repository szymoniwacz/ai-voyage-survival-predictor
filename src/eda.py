from pathlib import Path

import pandas as pd

# EDA (Exploratory Data Analysis) = initial data analysis
# before modeling: dataset size, missing values, and target distribution.


def build_eda_summary(df: pd.DataFrame) -> dict:
    """Build a compact EDA summary for quick inspection.

    Returns a dict with:
    - row_count
    - column_count
    - missing_values (column -> missing count)
    - target_distribution (when Survived exists)
    """
    summary = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": {
            column: int(count)
            for column, count in df.isnull().sum().to_dict().items()
            if int(count) > 0
        },
    }

    if "Survived" in df.columns:
        target_counts = df["Survived"].value_counts(dropna=False).to_dict()
        summary["target_distribution"] = {
            str(label): int(count) for label, count in target_counts.items()
        }

    return summary


def format_eda_summary(summary: dict) -> str:
    """Format EDA summary to a human-readable text report."""
    lines = [
        "EDA Summary",
        "===========",
        f"Rows: {summary['row_count']}",
        f"Columns: {summary['column_count']}",
        "",
        "Missing Values",
        "--------------",
    ]

    missing_values = summary.get("missing_values", {})
    if not missing_values:
        lines.append("No missing values.")
    else:
        for column, count in sorted(missing_values.items()):
            lines.append(f"- {column}: {count}")

    target_distribution = summary.get("target_distribution")
    if target_distribution is not None:
        lines.extend(
            [
                "",
                "Target Distribution (Survived)",
                "------------------------------",
            ]
        )
        for label, count in sorted(target_distribution.items()):
            lines.append(f"- {label}: {count}")

    return "\n".join(lines) + "\n"


def save_eda_summary(summary: dict, output_path: str | Path) -> Path:
    """Save formatted EDA summary to a text file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_eda_summary(summary), encoding="utf-8")
    return path
