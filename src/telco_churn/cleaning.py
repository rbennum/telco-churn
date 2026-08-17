"""Raw-data preparation, matching notebook 01 exactly.

The app must transform inputs the same way training data was transformed. Keeping
this in the package rather than in the app is what prevents the two from drifting:
a second implementation of cleaning logic is how an app silently starts scoring a
different feature space from the one the model was fitted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Churn"

SERVICE_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

FEATURE_COLS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORY_LEVELS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "No internet service", "Yes"],
    "OnlineBackup": ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport": ["No", "No internet service", "Yes"],
    "StreamingTV": ["No", "No internet service", "Yes"],
    "StreamingMovies": ["No", "No internet service", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ],
}


def _is_text(s: pd.Series) -> bool:
    """True for object- or string-dtype columns, across pandas 2 and 3."""
    return pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)


def prepare_raw(df: pd.DataFrame, drop_id: bool = True) -> pd.DataFrame:
    """Apply the notebook-01 cleaning steps to a raw Telco frame.

    Handles: blank TotalCharges, SeniorCitizen 0/1 to Yes/No, whitespace, the
    structural-zero domain rule, and target mapping when Churn is present.
    """
    out = df.copy()

    if drop_id and "customerID" in out.columns:
        out = out.drop(columns=["customerID"])

    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(
            out["TotalCharges"].astype(str).str.strip(), errors="coerce"
        )

    if "SeniorCitizen" in out.columns and out["SeniorCitizen"].dtype != object:
        out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # pandas 3 gives text columns dtype "str" rather than "object", so selecting on
    # object alone silently skips them. Test both.
    for c in out.columns:
        if _is_text(out[c]):
            out[c] = out[c].astype(str).str.strip()

    # Domain rule: never billed means zero billed, not missing.
    if {"TotalCharges", "tenure"}.issubset(out.columns):
        out.loc[out["TotalCharges"].isna() & (out["tenure"] == 0), "TotalCharges"] = 0.0

    if TARGET in out.columns and _is_text(out[TARGET]):
        out[TARGET] = out[TARGET].map({"Yes": 1, "No": 0}).astype("int8")

    return out


def validate_for_scoring(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check a frame can be scored. Returns (ok, list of problems)."""
    problems = []

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        problems.append(f"Missing required columns: {', '.join(missing)}")

    for col, levels in CATEGORY_LEVELS.items():
        if col not in df.columns:
            continue
        unknown = sorted(set(df[col].dropna().unique()) - set(levels))
        if unknown:
            problems.append(f"{col} has unrecognised values: {unknown}")

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            problems.append(f"{col} is not numeric after cleaning")

    if "MonthlyCharges" in df.columns and (df["MonthlyCharges"] <= 0).any():
        problems.append("MonthlyCharges contains values at or below zero")

    nulls = [c for c in FEATURE_COLS if c in df.columns and df[c].isna().any()]
    if nulls:
        problems.append(f"Null values remain in: {', '.join(nulls)}")

    return len(problems) == 0, problems
