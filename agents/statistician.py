"""
statistician.py
----------------
Agent 2: Statistician Agent

Responsibility: generate exploratory data analysis (EDA) insights —
descriptive statistics, correlations, top categories, distributions,
and outliers.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


def run_statistician_agent(df: pd.DataFrame) -> dict:
    """
    Compute EDA statistics for a cleaned DataFrame.

    Returns a dict with descriptive stats, correlation matrix,
    top categories per categorical column, and detected outliers.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df.loc[:, numeric_df.notna().any(axis=0)]
    categorical_cols = df.select_dtypes(include=["object", "str", "category", "bool"]).columns.tolist()

    descriptive_stats = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        descriptive_stats[col] = {
            "mean": round(float(series.mean()), 4),
            "median": round(float(series.median()), 4),
            "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
        }

    # Correlation matrix (only meaningful with 2+ numeric columns)
    correlation_matrix = {}
    strong_correlations = []
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True).round(3)
        correlation_matrix = corr.to_dict()

        seen_pairs = set()
        for col_a in corr.columns:
            for col_b in corr.columns:
                if col_a == col_b:
                    continue
                pair = tuple(sorted((col_a, col_b)))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                value = corr.loc[col_a, col_b]
                if pd.notna(value) and abs(value) >= 0.6:
                    direction = "positively" if value > 0 else "negatively"
                    strong_correlations.append({
                        "column_a": col_a,
                        "column_b": col_b,
                        "correlation": float(value),
                        "description": f"{col_a} is {direction} correlated with {col_b} ({value:.2f})",
                    })
        strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

    # Top categories per categorical column (top 5 each)
    top_categories = {}
    for col in categorical_cols:
        counts = df[col].value_counts(dropna=True).head(5)
        if not counts.empty:
            top_categories[col] = {str(k): int(v) for k, v in counts.items()}

    # Outlier detection via IQR method on numeric columns
    outliers = {}
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        if outlier_count > 0:
            outliers[col] = {
                "count": outlier_count,
                "percentage": round(outlier_count / len(series) * 100, 2),
            }

    return {
        "descriptive_stats": descriptive_stats,
        "correlation_matrix": correlation_matrix,
        "strong_correlations": strong_correlations,
        "top_categories": top_categories,
        "outliers": outliers,
        "numeric_columns_analyzed": list(numeric_df.columns),
        "categorical_columns_analyzed": categorical_cols,
    }
