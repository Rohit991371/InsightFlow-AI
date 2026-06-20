"""
cleaner.py
----------
Agent 1: Data Cleaning Agent

Responsibility: inspect dataset quality and surface issues
(missing values, duplicates, invalid types, empty columns)
without silently mutating the user's data. Cleaning suggestions
are returned alongside the report so downstream agents and the
UI can decide what (if anything) to apply.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def run_data_cleaning_agent(df: pd.DataFrame) -> dict:
    """
    Inspect a DataFrame and produce a data quality report.

    Returns a dict matching the shape described in the project vision doc:
    {
        "missing_values": int,
        "duplicates": int,
        "numeric_columns": int,
        "categorical_columns": int,
        ...plus extra detail used by later agents/report writer
    }
    """
    missing_per_column = df.isnull().sum()
    total_missing = int(missing_per_column.sum())
    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "str", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    empty_columns = [col for col in df.columns if df[col].isnull().all()]

    # Columns that are >90% missing are flagged as "near empty" — common
    # real-world junk columns worth calling out separately from empty_columns.
    near_empty_columns = [
        col for col in df.columns
        if 0.9 <= df[col].isnull().mean() < 1.0
    ]

    # Detect columns that look numeric but were loaded as text
    # (e.g. "$1,200" or "1200 units") — common in messy exports.
    suspected_numeric_as_text = []
    for col in categorical_cols:
        sample = df[col].dropna().astype(str).head(50)
        if len(sample) == 0:
            continue
        numeric_like = sample.str.replace(r"[,$%\s]", "", regex=True).str.match(r"^-?\d+\.?\d*$")
        if numeric_like.mean() > 0.8:
            suspected_numeric_as_text.append(col)

    cleaning_suggestions = []
    if total_missing > 0:
        cleaning_suggestions.append(
            f"Fill or drop {total_missing} missing values across "
            f"{(missing_per_column > 0).sum()} column(s)."
        )
    if duplicate_rows > 0:
        cleaning_suggestions.append(f"Remove {duplicate_rows} duplicate row(s).")
    if empty_columns:
        cleaning_suggestions.append(f"Drop fully empty column(s): {', '.join(empty_columns)}.")
    if near_empty_columns:
        cleaning_suggestions.append(
            f"Review near-empty column(s) (>90% missing): {', '.join(near_empty_columns)}."
        )
    if suspected_numeric_as_text:
        cleaning_suggestions.append(
            f"Convert text-formatted numeric column(s) to numeric type: "
            f"{', '.join(suspected_numeric_as_text)}."
        )
    if not cleaning_suggestions:
        cleaning_suggestions.append("No major data quality issues detected.")

    report = {
        "missing_values": total_missing,
        "duplicates": duplicate_rows,
        "numeric_columns": len(numeric_cols),
        "categorical_columns": len(categorical_cols),
        "datetime_columns": len(datetime_cols),
        "numeric_column_names": numeric_cols,
        "categorical_column_names": categorical_cols,
        "datetime_column_names": datetime_cols,
        "empty_columns": empty_columns,
        "near_empty_columns": near_empty_columns,
        "suspected_numeric_as_text": suspected_numeric_as_text,
        "missing_by_column": {
            col: int(count) for col, count in missing_per_column.items() if count > 0
        },
        "cleaning_suggestions": cleaning_suggestions,
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
    }
    return report


def apply_basic_cleaning(df: pd.DataFrame, drop_duplicates: bool = True,
                          drop_empty_columns: bool = True) -> pd.DataFrame:
    """
    Optional helper to actually apply low-risk cleaning steps.
    Not called automatically — the orchestrator decides whether to
    use the raw or cleaned frame for downstream agents.
    """
    cleaned = df.copy()

    if drop_empty_columns:
        cleaned = cleaned.dropna(axis=1, how="all")

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates()

    return cleaned
