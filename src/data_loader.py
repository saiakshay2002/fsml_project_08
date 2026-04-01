from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = ROOT_DIR / "data" / "processed"


class DataFileNotFoundError(FileNotFoundError):
    """Raised when one or more expected processed CSV files are missing."""


def _validate_file(path: Path) -> None:
    if not path.exists():
        raise DataFileNotFoundError(f"Expected file not found: {path}")


def load_split(split_name: str, processed_dir: Path | str = DEFAULT_PROCESSED_DIR) -> pd.DataFrame:
    """Load one processed split by name.

    Parameters
    ----------
    split_name:
        One of: 'train', 'val', 'test'.
    processed_dir:
        Directory containing processed CSV files.
    """
    processed_dir = Path(processed_dir)
    path = processed_dir / f"{split_name}.csv"
    _validate_file(path)
    return pd.read_csv(path)


def load_processed_splits(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, and test splits from disk."""
    train_df = load_split("train", processed_dir)
    val_df = load_split("val", processed_dir)
    test_df = load_split("test", processed_dir)
    return train_df, val_df, test_df


def split_features_target(df: pd.DataFrame, target_col: str = "label") -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into features X and target y."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    return X, y
