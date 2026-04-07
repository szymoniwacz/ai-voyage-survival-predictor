import pandas as pd
from pathlib import Path


def load_data(filename: str, data_dir: str = "../data/raw") -> pd.DataFrame:
    path = Path(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    # Example usage: load and show info about train.csv
    df = load_data("train.csv")
    print(df.info())
    print(df.head())
