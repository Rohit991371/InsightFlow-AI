"""
report_writer.py
-----------------
Agent 5: Report Writer Agent

Responsibility: combine data quality, statistics, charts + explanations,
recommendations, business insights, and any custom/interactive session
content (custom charts, chat Q&A) into a single executive PDF report.
"""

from __future__ import annotations
from pathlib import Path

from utils.pdf_generator import generate_pdf_report


def run_report_writer_agent(
    dataset_name: str,
    dataset_info: dict,
    cleaning_report: dict,
    stats_report: dict,
    charts: dict,
    business_insights: dict,
    chart_explanations: list | None = None,
    recommendations: dict | None = None,
    custom_charts: list | None = None,
    custom_explanations: list | None = None,
    chat_history: list | None = None,
    output_dir: str = "reports",
) -> str:
    """
    Generate the final PDF report and return its file path.

    The new params (chart_explanations, recommendations, custom_charts,
    custom_explanations, chat_history) are all optional so this agent
    still works unchanged for callers that only run the core pipeline
    without ever touching the interactive layer.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in Path(dataset_name).stem)
    output_path = str(out_dir / f"{safe_name}_executive_report.pdf")

    return generate_pdf_report(
        output_path=output_path,
        dataset_name=dataset_name,
        dataset_info=dataset_info,
        cleaning_report=cleaning_report,
        stats_report=stats_report,
        charts=charts,
        business_insights=business_insights,
        chart_explanations=chart_explanations or [],
        recommendations=recommendations or {},
        custom_charts=custom_charts or [],
        custom_explanations=custom_explanations or [],
        chat_history=chat_history or [],
    )
