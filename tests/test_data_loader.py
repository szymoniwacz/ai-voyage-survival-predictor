import pytest
from data_loader import load_data
import pandas as pd
from pathlib import Path


def test_load_train_csv():
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    df = load_data("train.csv", data_dir=str(data_dir))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Survived" in df.columns


def test_file_not_found():
    data_dir = Path(__file__).parent.parent / "data" / "raw"
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent.csv", data_dir=str(data_dir))
