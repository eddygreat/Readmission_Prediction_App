from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from .config import LOGS_DIR, MODEL_PATH
from .data import create_splits, load_split


def log_metrics(payload: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, default=str)
    with open(LOGS_DIR / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    # Ensure splits exist
    splits = create_splits()

    X_test, y_test = load_split(splits.test)
    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Confusion Matrix:", cm)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stage": "evaluation",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }
    log_metrics(payload)


if __name__ == "__main__":
    main()
