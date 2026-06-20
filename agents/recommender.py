"""
agents/recommender.py
----------------------
Recommendation Agent

Responsibility: proactively suggest useful analyses the user hasn't
already seen, based on the dataset's columns, types, correlations, and
trends. Each recommendation includes what to analyze, why it matters,
and its potential business value — and a ready-to-run chart_request
string the UI can hand straight to the Interactive Analytics Agent.

Runs after Visualization + Statistician so it can see what's already
been auto-generated and avoid recommending duplicates.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np

from utils.llm_client import call_llm_json, has_llm, LLMUnavailableError

MAX_RECOMMENDATIONS = 5

SYSTEM_PROMPT = """You are a senior data analyst proposing a short list of \
high-value follow-up analyses for a business user, based on a dataset's \
schema and statistical findings.

Produce a JSON object with EXACTLY this schema:
{
  "recommendations": [
    {
      "title": "short name for the analysis, e.g. 'Revenue vs Marketing Spend'",
      "purpose": "1 sentence: what question this analysis answers",
      "business_value": "1 sentence: why this matters for the business",
      "chart_request": "a natural-language chart request phrased the way a user \
would type it, e.g. 'scatter plot between revenue and marketing_spend' — must \
use EXACT column names from the list provided"
    }
  ]
}

Rules:
- Recommend at most 5 analyses, ranked by likely business value.
- Do not repeat any analysis already listed under "already generated".
- Every chart_request must reference only columns from the provided column list, \
using their exact names.
- Prefer analyses grounded in real signals given to you (strong correlations, \
high-cardinality categories, time trends) over generic suggestions.
- Output ONLY valid JSON, no markdown fences, no preamble.
"""


def _describe_already_generated(chart_metadata: list[dict]) -> list[str]:
    """Turn existing chart_metadata into short human-readable descriptions,
    used both to avoid duplicate LLM recommendations and in the rule-based
    fallback's own dedup check."""
    described = []
    for meta in chart_metadata or []:
        cols = [c for c in (meta.get("column"), meta.get("x_axis"), meta.get("y_axis")) if c]
        described.append(f"{meta.get('chart_type', 'chart')}: {', '.join(cols) if cols else 'overview'}")
    return described


def _rule_based_recommendations(
    df: pd.DataFrame,
    stats_report: dict,
    chart_metadata: list[dict],
) -> list[dict]:
    """
    Deterministic fallback: build recommendations directly from
    strong_correlations, top_categories, and datetime+numeric pairs —
    no LLM required.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "str", "category", "bool"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    already_cols = set()
    for meta in chart_metadata or []:
        for c in (meta.get("column"), meta.get("x_axis"), meta.get("y_axis")):
            if c:
                already_cols.add(c)

    recs = []

    # 1. Strong correlations -> scatter plots
    for corr in stats_report.get("strong_correlations", [])[:2]:
        a, b = corr["column_a"], corr["column_b"]
        recs.append({
            "title": f"{a} vs {b}",
            "purpose": f"Examine the relationship between {a} and {b} in detail.",
            "business_value": (
                f"These fields move together ({corr['correlation']:.2f} correlation) — "
                f"understanding this relationship can inform forecasting and resource allocation."
            ),
            "chart_request": f"scatter plot between {a} and {b}",
        })

    # 2. Datetime + numeric not already trended -> line chart
    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        for value_col in numeric_cols:
            if value_col in already_cols:
                continue
            recs.append({
                "title": f"{value_col} Trend Over Time",
                "purpose": f"Track how {value_col} changes over time.",
                "business_value": f"Identifying trends or seasonality in {value_col} supports planning and forecasting.",
                "chart_request": f"line chart between {date_col} and {value_col}",
            })
            break

    # 3. Categorical columns not yet bar-charted, with a numeric pairing -> bar
    for cat_col in categorical_cols:
        if cat_col in already_cols:
            continue
        if numeric_cols:
            value_col = numeric_cols[0]
            recs.append({
                "title": f"{value_col} by {cat_col}",
                "purpose": f"Compare average {value_col} across {cat_col} groups.",
                "business_value": f"Reveals which {cat_col} segments contribute most, supporting targeted decisions.",
                "chart_request": f"bar chart comparing {cat_col} and {value_col}",
            })
        break

    # 4. Outlier-heavy numeric column -> box plot
    for col, info in stats_report.get("outliers", {}).items():
        if info.get("percentage", 0) >= 5:
            recs.append({
                "title": f"{col} Outlier Review",
                "purpose": f"Inspect the spread and outliers in {col}.",
                "business_value": f"{info['percentage']}% of {col} values are outliers — worth investigating for data errors or genuine extreme cases.",
                "chart_request": f"boxplot for {col}",
            })
            break

    # 5. Correlation heatmap if 3+ numeric columns and not already shown standalone
    if len(numeric_cols) >= 3:
        recs.append({
            "title": "Full Correlation Overview",
            "purpose": "See how all numeric fields relate to each other at a glance.",
            "business_value": "Surfaces relationships that might not be obvious from individual charts.",
            "chart_request": "correlation heatmap for all numeric columns",
        })

    return recs[:MAX_RECOMMENDATIONS]


def run_recommendation_agent(
    df: pd.DataFrame,
    stats_report: dict,
    chart_metadata: Optional[list[dict]] = None,
) -> dict:
    """
    Generate up to MAX_RECOMMENDATIONS proactive analysis suggestions.

    Returns:
        {"recommendations": [{"title", "purpose", "business_value", "chart_request"}, ...],
         "source": "llm" | "rule_based"}
    Never raises.
    """
    chart_metadata = chart_metadata or []
    fallback = {
        "recommendations": _rule_based_recommendations(df, stats_report, chart_metadata),
        "source": "rule_based",
    }

    if not has_llm():
        return fallback

    columns = list(df.columns)
    already_generated = _describe_already_generated(chart_metadata)

    user_prompt = (
        f"Available columns: {columns}\n\n"
        f"Already generated charts (do not repeat): {already_generated}\n\n"
        f"Strong correlations: {stats_report.get('strong_correlations', [])}\n\n"
        f"Top categories: {stats_report.get('top_categories', {})}\n\n"
        f"Outliers: {stats_report.get('outliers', {})}\n\n"
        f"Numeric columns: {df.select_dtypes(include=[np.number]).columns.tolist()}\n"
        f"Categorical columns: {df.select_dtypes(include=['object', 'str', 'category', 'bool']).columns.tolist()}\n"
        f"Datetime columns: {df.select_dtypes(include=['datetime64']).columns.tolist()}"
    )

    try:
        parsed = call_llm_json(SYSTEM_PROMPT, user_prompt, max_tokens=1000)
        recs = parsed.get("recommendations", [])
        if not recs:
            return fallback
        valid_cols = set(columns)
        # Defensive: keep a recommendation only if at least one real column
        # name appears verbatim in its chart_request text — guards against
        # the LLM hallucinating a column that doesn't exist in this dataset.
        cleaned = [
            r for r in recs
            if isinstance(r.get("chart_request"), str)
            and any(col in r["chart_request"] for col in valid_cols)
        ]
        if not cleaned:
            return fallback
        return {"recommendations": cleaned[:MAX_RECOMMENDATIONS], "source": "llm"}
    except LLMUnavailableError:
        return fallback
    except Exception:  # noqa: BLE001 — any LLM/JSON failure falls back cleanly
        return fallback
