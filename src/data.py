from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

from .config import (
    BASE_DIR,
    DATA_DIR,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    TARGET_COL,
    DROP_COLS,
    ID_COLS,
)


def ensure_dirs() -> None:
    for d in [DATA_DIR, PROCESSED_DIR, BASE_DIR / "models", BASE_DIR / "logs"]:
        d.mkdir(parents=True, exist_ok=True)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Replace '?' with NA uniformly
    df = df.replace("?", pd.NA)
    # Normalize all pandas NA to numpy nan so sklearn imputers work reliably
    df = df.where(pd.notna(df), np.nan)
    # Drop high-missing columns if present
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)
    return df


def map_target(df: pd.DataFrame) -> pd.Series:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in DataFrame")
    y = df[TARGET_COL].map(lambda v: 1 if v == "<30" else 0)
    return y.astype(int)


@dataclass
class Splits:
    train: Path
    val: Path
    test: Path


def create_splits(
    csv_path: Path = RAW_DATA_PATH,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Splits:
    ensure_dirs()
    df = pd.read_csv(csv_path)
    df = basic_clean(df)
    # Keep both raw string target and numeric mapping (for stratification)
    y_raw = df[TARGET_COL].copy()
    y = map_target(df)

    # Remove target and IDs from features
    feature_cols = [c for c in df.columns if c not in set([TARGET_COL] + ID_COLS)]
    X = df[feature_cols].copy()

    # Stratified splits
    from sklearn.model_selection import train_test_split

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=(1 - train_ratio), stratify=y, random_state=random_state
    )
    rel_test = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=rel_test, stratify=y_tmp, random_state=random_state
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    # Write original string targets aligned to each split
    y_train_raw = y_raw.loc[X_train.index]
    y_val_raw = y_raw.loc[X_val.index]
    y_test_raw = y_raw.loc[X_test.index]

    X_train.assign(**{TARGET_COL: y_train_raw}).to_csv(train_path, index=False)
    X_val.assign(**{TARGET_COL: y_val_raw}).to_csv(val_path, index=False)
    X_test.assign(**{TARGET_COL: y_test_raw}).to_csv(test_path, index=False)

    return Splits(train=train_path, val=val_path, test=test_path)


def load_split(path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df = basic_clean(df)
    y = map_target(df)
    X = df.drop(columns=[TARGET_COL])
    return X, y
