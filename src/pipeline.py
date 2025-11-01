from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .config import ID_COLS, LENGTH_OF_STAY_COL, MEDICATION_COLS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # prior_visits
    for col in ["number_outpatient", "number_emergency", "number_inpatient"]:
        if col not in out.columns:
            out[col] = 0
    out["prior_visits"] = (
        out["number_outpatient"].fillna(0).astype(float)
        + out["number_emergency"].fillna(0).astype(float)
        + out["number_inpatient"].fillna(0).astype(float)
    )

    # length_of_stay (alias time_in_hospital)
    if LENGTH_OF_STAY_COL in out.columns:
        out["length_of_stay"] = out[LENGTH_OF_STAY_COL]
    else:
        out["length_of_stay"] = np.nan

    # medication_count: count of medication columns where value != 'No'
    meds = [c for c in MEDICATION_COLS if c in out.columns]
    if meds:
        out["medication_count"] = (
            out[meds]
            .apply(lambda r: (r.astype(str).str.lower() != "no").sum(), axis=1)
            .astype(float)
        )
    else:
        out["medication_count"] = 0.0

    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        # Ensure input is a DataFrame
        X_df = pd.DataFrame(X).copy()
        return engineer_features(X_df)


def split_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        c for c in X.columns if c not in numeric_cols and c not in ID_COLS
    ]
    return numeric_cols, categorical_cols


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    # Determine columns after feature engineering
    X_eng = engineer_features(X)
    numeric_cols, categorical_cols = split_feature_types(X_eng)

    num_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    ct = ColumnTransformer(
        transformers=[
            ("num", num_tf, numeric_cols),
            ("cat", cat_tf, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Full preprocessing pipeline: feature engineering first, then column-wise transforms
    pre_pipeline = Pipeline(
        steps=[
            ("feat", FeatureEngineer()),
            ("ct", ct),
        ]
    )
    return pre_pipeline
