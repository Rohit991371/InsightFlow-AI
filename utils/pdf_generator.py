"""
pdf_generator.py
----------------
Builds the final Executive Report PDF from the outputs of every agent.

Sections (Enhancement Phase spec):
  Executive Summary
  Dataset Overview
  Data Quality Assessment
  Statistical Findings
  Automated Visualizations + Automated Chart Explanations (paired, chart-by-chart)
  Business Insights
  Recommended Analyses
  Custom User Visualizations + Custom Chart Explanations (paired, if any)
  User Questions & Answers (if any)
  Strategic Recommendations
  Conclusion

Custom charts, chat history, etc. are optional — sections are skipped
cleanly when empty so the report still reads well for a dataset where
the user never opened the chat or requested a custom chart.
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
)


def _styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="ReportTitle", parent=base["Title"], fontSize=24, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="SectionHeading", parent=base["Heading1"], fontSize=15,
        spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#1F3864"),
    ))
    base.add(ParagraphStyle(
        name="SubHeading", parent=base["Heading3"], fontSize=11.5,
        spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#2E4D7B"),
    ))
    base.add(ParagraphStyle(
        name="Body", parent=base["BodyText"], fontSize=10.5, leading=15,
    ))
    base.add(ParagraphStyle(
        name="Caption", parent=base["BodyText"], fontSize=9, leading=12,
        textColor=colors.HexColor("#555555"), spaceAfter=10,
    ))
    return base


def _kpi_table(rows: list[tuple[str, str]]) -> Table:
    table = Table(rows, colWidths=[7 * cm, 7 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _data_table(table_payload: dict, max_col_width_cm: float = 16.0) -> Table:
    """
    Render a {"columns": [...], "rows": [[...], ...]} payload (e.g. from
    the Dataset Chat Agent) as a ReportLab Table with an arbitrary number
    of columns, splitting the available width evenly.
    """
    columns = table_payload["columns"]
    rows = table_payload["rows"]
    data = [columns] + [[str(v) if v is not None else "" for v in row] for row in rows]

    col_width = min(max_col_width_cm / max(len(columns), 1), 6.0) * cm
    table = Table(data, colWidths=[col_width] * len(columns))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def _render_chart_with_explanation(story, styles, chart_meta: dict, explanation: dict | None):
    """
    Render one chart image followed immediately by its explanation
    (overview, variables, comparison, observations, interpretation,
    actions) — keeps each chart and its narrative together rather than
    grouping all images first and all text after.
    """
    img_path = chart_meta.get("path")
    chart_type = chart_meta.get("chart_type", "Chart")
    cols = [c for c in (chart_meta.get("column"), chart_meta.get("x_axis"), chart_meta.get("y_axis")) if c]
    title = f"{chart_type}" + (f" — {', '.join(cols)}" if cols else "")

    story.append(Paragraph(title, styles["SubHeading"]))
    if img_path and Path(img_path).exists():
        story.append(Image(img_path, width=13 * cm, height=8 * cm, kind="proportional"))
        story.append(Spacer(1, 0.15 * cm))

    if not explanation:
        story.append(Spacer(1, 0.3 * cm))
        return

    if explanation.get("chart_overview"):
        story.append(Paragraph(explanation["chart_overview"], styles["Body"]))

    var_defs = explanation.get("variable_definitions") or {}
    if var_defs:
        for col, desc in var_defs.items():
            story.append(Paragraph(f"<b>{col}:</b> {desc}", styles["Caption"]))

    if explanation.get("comparison_summary"):
        story.append(Paragraph(explanation["comparison_summary"], styles["Body"]))

    observations = explanation.get("key_observations") or []
    if observations:
        story.append(Paragraph("Key Observations:", styles["Caption"]))
        for obs in observations:
            story.append(Paragraph(f"• {obs}", styles["Body"]))

    if explanation.get("business_interpretation"):
        story.append(Paragraph(f"<b>Business Interpretation:</b> {explanation['business_interpretation']}", styles["Body"]))

    actions = explanation.get("recommended_actions") or []
    if actions:
        for action in actions:
            story.append(Paragraph(f"➤ {action}", styles["Body"]))

    story.append(Spacer(1, 0.4 * cm))


def generate_pdf_report(
    output_path: str,
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
) -> str:
    """
    Assemble the executive PDF report.

    New optional params (Enhancement Phase):
        chart_explanations: list of explanation dicts, same order/length as
            charts["chart_metadata"] — output of the Chart Explanation Agent.
        recommendations: output of the Recommendation Agent, i.e.
            {"recommendations": [...], "source": str}.
        custom_charts: list of chart_meta dicts from user-requested charts
            (Interactive Analytics Agent), in the order they were created.
        custom_explanations: matching explanations for custom_charts, same
            order/length.
        chat_history: list of {"question": str, ...answer fields...} dicts
            from the Dataset Chat Agent, in chronological order.

    Returns the output_path on success.
    """
    chart_explanations = chart_explanations or []
    recommendations = recommendations or {}
    custom_charts = custom_charts or []
    custom_explanations = custom_explanations or []
    chat_history = chat_history or []

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    story = []

    # --- Title page ---
    story.append(Paragraph("InsightFlow AI", styles["ReportTitle"]))
    story.append(Paragraph("Executive Data Analysis Report", styles["Heading2"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Dataset: {dataset_name}", styles["Body"]))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Body"]))
    story.append(Spacer(1, 0.6 * cm))

    # --- Executive Summary ---
    story.append(Paragraph("Executive Summary", styles["SectionHeading"]))
    story.append(Paragraph(business_insights.get("executive_summary", "N/A"), styles["Body"]))

    # --- Dataset Overview ---
    story.append(Paragraph("Dataset Overview", styles["SectionHeading"]))
    overview_rows = [
        ("Metric", "Value"),
        ("Rows", str(dataset_info.get("rows", "N/A"))),
        ("Columns", str(dataset_info.get("columns", "N/A"))),
        ("Numeric Columns", str(cleaning_report.get("numeric_columns", "N/A"))),
        ("Categorical Columns", str(cleaning_report.get("categorical_columns", "N/A"))),
    ]
    story.append(_kpi_table(overview_rows))

    # --- Data Quality Assessment ---
    story.append(Paragraph("Data Quality Assessment", styles["SectionHeading"]))
    quality_rows = [
        ("Check", "Result"),
        ("Missing values", str(cleaning_report.get("missing_values", 0))),
        ("Duplicate rows", str(cleaning_report.get("duplicates", 0))),
        ("Empty columns", str(len(cleaning_report.get("empty_columns", [])))),
    ]
    story.append(_kpi_table(quality_rows))
    story.append(Spacer(1, 0.3 * cm))
    for suggestion in cleaning_report.get("cleaning_suggestions", []):
        story.append(Paragraph(f"• {suggestion}", styles["Body"]))

    # --- Statistical Findings ---
    story.append(Paragraph("Statistical Findings", styles["SectionHeading"]))
    correlations = stats_report.get("strong_correlations", [])
    if correlations:
        for corr in correlations[:5]:
            story.append(Paragraph(f"• {corr['description'].capitalize()}.", styles["Body"]))
    else:
        story.append(Paragraph("No strong correlations were detected between numeric columns.", styles["Body"]))

    outliers = stats_report.get("outliers", {})
    if outliers:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Paragraph("Outliers detected:", styles["Body"]))
        for col, info in outliers.items():
            story.append(Paragraph(
                f"• {col}: {info['count']} outlier value(s) ({info['percentage']}% of records).",
                styles["Body"]
            ))

    # --- Automated Visualizations + Automated Chart Explanations (paired) ---
    story.append(PageBreak())
    story.append(Paragraph("Automated Visualizations & Explanations", styles["SectionHeading"]))

    chart_metadata = charts.get("chart_metadata", [])
    if chart_metadata:
        for i, meta in enumerate(chart_metadata):
            explanation = chart_explanations[i] if i < len(chart_explanations) else None
            _render_chart_with_explanation(story, styles, meta, explanation)
    else:
        story.append(Paragraph("No charts were generated for this dataset.", styles["Body"]))

    # --- Business Insights ---
    story.append(PageBreak())
    story.append(Paragraph("Business Insights", styles["SectionHeading"]))
    for insight in business_insights.get("key_insights", []):
        story.append(Paragraph(f"• {insight}", styles["Body"]))

    # --- Recommended Analyses ---
    rec_list = recommendations.get("recommendations", [])
    if rec_list:
        story.append(Paragraph("Recommended Analyses", styles["SectionHeading"]))
        for rec in rec_list:
            story.append(Paragraph(rec.get("title", "Untitled analysis"), styles["SubHeading"]))
            if rec.get("purpose"):
                story.append(Paragraph(f"<b>Purpose:</b> {rec['purpose']}", styles["Body"]))
            if rec.get("business_value"):
                story.append(Paragraph(f"<b>Business Value:</b> {rec['business_value']}", styles["Body"]))
            story.append(Spacer(1, 0.2 * cm))

    # --- Custom User Visualizations + Custom Chart Explanations (paired) ---
    if custom_charts:
        story.append(PageBreak())
        story.append(Paragraph("Custom User Visualizations", styles["SectionHeading"]))
        for i, meta in enumerate(custom_charts):
            explanation = custom_explanations[i] if i < len(custom_explanations) else None
            _render_chart_with_explanation(story, styles, meta, explanation)

    # --- User Questions & Answers ---
    if chat_history:
        story.append(PageBreak())
        story.append(Paragraph("User Questions & Answers", styles["SectionHeading"]))
        for turn in chat_history:
            question = turn.get("question", "")
            if not question:
                continue
            story.append(Paragraph(f"Q: {question}", styles["SubHeading"]))
            if turn.get("success") is False:
                story.append(Paragraph(f"Could not be answered: {turn.get('error', 'unknown error')}", styles["Body"]))
                story.append(Spacer(1, 0.2 * cm))
                continue
            if turn.get("answer") is not None:
                story.append(Paragraph(f"<b>Answer:</b> {turn['answer']}", styles["Body"]))
            table_payload = turn.get("table")
            if table_payload and table_payload.get("rows"):
                story.append(Spacer(1, 0.15 * cm))
                story.append(_data_table(table_payload))
                story.append(Spacer(1, 0.15 * cm))
            if turn.get("reasoning"):
                story.append(Paragraph(f"<b>Reasoning:</b> {turn['reasoning']}", styles["Body"]))
            if turn.get("supporting_statistics"):
                story.append(Paragraph(f"<b>Supporting Statistics:</b> {turn['supporting_statistics']}", styles["Body"]))
            if turn.get("business_interpretation"):
                story.append(Paragraph(f"<b>Business Interpretation:</b> {turn['business_interpretation']}", styles["Body"]))
            story.append(Spacer(1, 0.3 * cm))

    # --- Strategic Recommendations ---
    story.append(PageBreak())
    story.append(Paragraph("Strategic Recommendations", styles["SectionHeading"]))
    for rec in business_insights.get("recommendations", []):
        story.append(Paragraph(f"• {rec}", styles["Body"]))

    # --- Conclusion ---
    story.append(Paragraph("Conclusion", styles["SectionHeading"]))
    story.append(Paragraph(
        "This report was generated by InsightFlow AI's multi-agent analytics "
        "pipeline — Data Cleaning, Statistician, Visualization, Chart "
        "Explanation, Recommendation, Business Analyst, and Report Writer "
        "agents, plus any interactive chart requests or questions explored "
        "during the session. Review the findings above and validate key "
        "results against domain knowledge before acting on them.",
        styles["Body"]
    ))

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
    )
    doc.build(story)
    return output_path
