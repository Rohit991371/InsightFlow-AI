"""
visualizer.py
-------------
Agent 3: Visualization Agent

Responsibility: automatically generate charts (histogram, bar chart,
pie chart, trend line, correlation heatmap) from the dataset and
save them to the charts/ directory for embedding into the PDF report
and Streamlit UI.
"""

from __future__ import annotations
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering — no GUI backend required


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def run_visualization_agent(df: pd.DataFrame, output_dir: str = "charts") -> dict:
    """
    Generate a set of standard charts from the dataset.

    Returns a dict mapping chart type -> list of saved file paths (unchanged
    shape, for backward compatibility with report_writer.py/app.py), PLUS a
    "chart_metadata" list of structured records — one per chart generated —
    used by the Chart Explanation Agent to know what each chart plots:
        {"chart_type": str, "x_axis": str|None, "y_axis": str|None,
         "column": str|None, "path": str}
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(
        include=["object", "str", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # Drop fully-empty numeric/categorical columns — they add noise to
    # histograms/heatmaps without contributing any signal.
    numeric_cols = [c for c in numeric_cols if df[c].notna().any()]
    categorical_cols = [c for c in categorical_cols if df[c].notna().any()]

    generated = {
        "histograms": [],
        "bar_charts": [],
        "pie_charts": [],
        "trend_lines": [],
        "heatmap": None,
        "chart_metadata": [],
    }

    sns.set_theme(style="whitegrid")

    # --- Histograms for up to 4 numeric columns ---
    for col in numeric_cols[:4]:
        series = df[col].dropna()
        if series.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series, bins=min(30, max(5, series.nunique())),
                color="#4C72B0", edgecolor="white")
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        path = out_dir / f"histogram_{_safe_filename(col)}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        generated["histograms"].append(str(path))
        generated["chart_metadata"].append({
            "chart_type": "Histogram", "x_axis": col, "y_axis": "Frequency",
            "column": col, "path": str(path),
        })

    # --- Bar charts: top categories for up to 3 categorical columns ---
    for col in categorical_cols[:3]:
        counts = df[col].value_counts(dropna=True).head(10)
        if counts.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=counts.values, y=counts.index.astype(
            str), ax=ax, color="#55A868")
        ax.set_title(f"Top categories: {col}")
        ax.set_xlabel("Count")
        fig.tight_layout()
        path = out_dir / f"bar_{_safe_filename(col)}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        generated["bar_charts"].append(str(path))
        generated["chart_metadata"].append({
            "chart_type": "Bar Chart", "x_axis": "Count", "y_axis": col,
            "column": col, "path": str(path),
        })

    # --- Pie chart for the first low-cardinality categorical column ---
    for col in categorical_cols:
        nunique = df[col].nunique(dropna=True)
        if 2 <= nunique <= 8:
            counts = df[col].value_counts(dropna=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.pie(counts.values, labels=counts.index.astype(
                str), autopct="%1.1f%%", startangle=90)
            ax.set_title(f"{col} distribution")
            fig.tight_layout()
            path = out_dir / f"pie_{_safe_filename(col)}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            generated["pie_charts"].append(str(path))
            generated["chart_metadata"].append({
                "chart_type": "Pie Chart", "x_axis": None, "y_axis": None,
                "column": col, "path": str(path),
            })
            break  # one pie chart is enough for the MVP

    # --- Trend line if a datetime column + numeric column exist ---
    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        value_col = numeric_cols[0]
        trend_df = df[[date_col, value_col]].dropna().sort_values(date_col)
        if not trend_df.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(trend_df[date_col], trend_df[value_col],
                    color="#C44E52", linewidth=1.5)
            ax.set_title(f"{value_col} over time")
            ax.set_xlabel(date_col)
            ax.set_ylabel(value_col)
            fig.autofmt_xdate()
            fig.tight_layout()
            path = out_dir / f"trend_{_safe_filename(value_col)}.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            generated["trend_lines"].append(str(path))
            generated["chart_metadata"].append({
                "chart_type": "Trend Line", "x_axis": date_col, "y_axis": value_col,
                "column": None, "path": str(path),
            })

    # --- Correlation heatmap ---
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(corr, annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        fig.tight_layout()
        path = out_dir / "heatmap.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        generated["heatmap"] = str(path)
        generated["chart_metadata"].append({
            "chart_type": "Correlation Heatmap", "x_axis": None, "y_axis": None,
            "column": None, "path": str(path),
        })

    return generated
