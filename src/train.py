from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from scipy.stats import randint

from .config import MODEL_PATH, MODELS_DIR
from .data import create_splits, load_split
from .pipeline import build_preprocess


def main() -> None:
    splits = create_splits()

    # Load train/val
    X_train, y_train = load_split(splits.train)
    X_val, y_val = load_split(splits.val)

    # Build preprocess based on training columns
    pre = build_preprocess(X_train)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=None)
    pipe = Pipeline(steps=[("pre", pre), ("clf", rf)])

    # Randomized search distributions
    param_distributions = {
        "clf__n_estimators": randint(80, 201),  # 80..200
        "clf__max_depth": [None, 10, 20],       # small discrete set
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Safe precision scorer: return 0 when the positive class is absent
    def _safe_precision(y_true, y_pred):
        # Expect binary labels {0,1}
        if 1 not in np.unique(y_true):
            return 0.0
        return precision_score(y_true, y_pred, zero_division=0)

    scorer = make_scorer(_safe_precision)
    gs = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=8,
        scoring=scorer,
        n_jobs=-1,
        cv=cv,
        refit=True,
        verbose=1,
        error_score=0.0,
        random_state=42,
    )

    gs.fit(X_train, y_train)

    # Evaluate on validation set
    val_pred = gs.predict(X_val)
    val_precision = precision_score(y_val, val_pred, zero_division=0)
    print({"val_precision": float(val_precision), "best_params": gs.best_params_})

    # Persist model and feature columns used
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(gs.best_estimator_, MODEL_PATH)

    feature_cols = X_train.columns.tolist()
    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_cols}, f)


if __name__ == "__main__":
    main()
