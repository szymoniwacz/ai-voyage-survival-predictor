from pathlib import Path

import pandas as pd

from eda import build_eda_summary, save_eda_summary


def test_build_eda_summary_contains_core_fields():
    df = pd.DataFrame(
        {
            "Survived": [0, 1, 1],
            "Age": [22.0, None, 35.0],
            "Fare": [7.25, 71.83, 8.05],
        }
    )

    summary = build_eda_summary(df)

    assert summary["row_count"] == 3
    assert summary["column_count"] == 3
    assert summary["missing_values"]["Age"] == 1
    assert summary["target_distribution"] == {"0": 1, "1": 2}


def test_save_eda_summary_writes_report(tmp_path):
    summary = {
        "row_count": 2,
        "column_count": 2,
        "missing_values": {"Age": 1},
        "target_distribution": {"0": 1, "1": 1},
    }

    output_path = tmp_path / "artifacts" / "eda_summary.txt"
    saved_path = save_eda_summary(summary, output_path)

    assert saved_path == output_path
    assert saved_path.exists()
    content = saved_path.read_text(encoding="utf-8")
    assert "EDA Summary" in content
    assert "Rows: 2" in content
    assert "- Age: 1" in content
