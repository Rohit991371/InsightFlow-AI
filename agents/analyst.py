"""
analyst.py
----------
Agent 4: Business Analyst Agent

Responsibility: convert technical findings (cleaning report + stats)
into plain-language business insights and recommendations using an LLM.

Uses Groq's free-tier API with Llama-3.3-70B by default. Falls back to
a deterministic, template-based summary if no API key is configured or
the API call fails — so the pipeline never hard-fails on this step.
"""

from __future__ import annotations
import os
import json
from groq import Groq


DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a senior business analyst. You are given technical \
data-quality findings and statistical analysis results from a dataset. \
Translate them into clear, non-technical business insights and 3-5 \
actionable recommendations.

Rules:
- No statistical jargon (no "p-value", "IQR", "standard deviation" etc).
- Write like you're briefing a business executive.
- Be specific: reference actual numbers, column names, and trends given to you.
- Output ONLY valid JSON, no markdown fences, no preamble. Schema:
{
  "executive_summary": "2-4 sentence overview",
  "key_insights": ["insight 1", "insight 2", ...],
  "recommendations": ["recommendation 1", "recommendation 2", ...]
}
"""


def _build_user_prompt(cleaning_report: dict, stats_report: dict) -> str:
    payload = {
        "data_quality": {
            "total_rows": cleaning_report.get("total_rows"),
            "total_columns": cleaning_report.get("total_columns"),
            "missing_values": cleaning_report.get("missing_values"),
            "duplicates": cleaning_report.get("duplicates"),
            "issues": cleaning_report.get("cleaning_suggestions"),
        },
        "statistics": {
            "descriptive_stats": stats_report.get("descriptive_stats"),
            "strong_correlations": stats_report.get("strong_correlations"),
            "top_categories": stats_report.get("top_categories"),
            "outliers": stats_report.get("outliers"),
        },
    }
    return (
        "Here are the technical findings from an automated data analysis "
        "pipeline. Translate them into business insights:\n\n"
        f"{json.dumps(payload, indent=2, default=str)}"
    )


def _fallback_summary(cleaning_report: dict, stats_report: dict) -> dict:
    """
    Deterministic, no-LLM fallback so the pipeline still produces
    something useful if GROQ_API_KEY is missing or the call fails.
    """
    insights = []
    correlations = stats_report.get("strong_correlations", [])
    for corr in correlations[:3]:
        insights.append(corr["description"].capitalize() + ".")

    top_categories = stats_report.get("top_categories", {})
    for col, counts in list(top_categories.items())[:2]:
        if counts:
            top_item = max(counts, key=counts.get)
            insights.append(f"'{top_item}' is the most frequent value in '{col}' ({counts[top_item]} occurrences).")

    if not insights:
        insights.append("No strong patterns were detected in the available numeric or categorical data.")

    recommendations = list(cleaning_report.get("cleaning_suggestions", []))[:3]
    if not recommendations:
        recommendations.append("Data quality looks solid; focus next on deeper trend analysis.")

    summary = (
        f"The dataset contains {cleaning_report.get('total_rows', 0)} records across "
        f"{cleaning_report.get('total_columns', 0)} columns. "
        f"{len(correlations)} notable relationship(s) were found between numeric fields, "
        "summarized below."
    )

    return {
        "executive_summary": summary,
        "key_insights": insights,
        "recommendations": recommendations,
        "source": "fallback_template",
    }


def run_business_analyst_agent(cleaning_report: dict, stats_report: dict,
                                model: str = DEFAULT_MODEL) -> dict:
    """
    Generate business-language insights from technical findings.

    Requires GROQ_API_KEY in the environment. If absent or the call
    fails, falls back to a deterministic template-based summary so
    downstream agents always receive a usable result.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return _fallback_summary(cleaning_report, stats_report)

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(cleaning_report, stats_report)},
            ],
            temperature=0.4,
            max_tokens=900,
        )
        raw = response.choices[0].message.content.strip()
        # Defensive cleanup in case the model wraps output in markdown fences
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        parsed = json.loads(raw)
        parsed["source"] = "llm"
        return parsed
    except Exception as exc:  # noqa: BLE001 — intentional broad catch for pipeline resilience
        fallback = _fallback_summary(cleaning_report, stats_report)
        fallback["llm_error"] = str(exc)
        return fallback
