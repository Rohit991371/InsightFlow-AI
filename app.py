"""
app.py
------
InsightFlow AI — Streamlit front-end.

Upload a CSV/Excel file, run it through the automated multi-agent
pipeline, then interact with the dataset live:
  - Request custom charts in natural language (Interactive Analytics Agent)
  - Ask questions about the data (Dataset Chat Agent)

Everything generated in the interactive layer (custom charts + their
explanations, chat Q&A) is kept in session state and folded into the
final PDF report when re-generated.
"""

from __future__ import annotations
import os
from pathlib import Path

import streamlit as st
import pandas as pd

from utils.data_loader import load_dataset, basic_dataset_info
from workflow.graph import run_pipeline
from agents.interactive_analytics import run_interactive_analytics_agent
from agents.chart_explainer import run_chart_explanation_agent
from agents.dataset_chat import run_dataset_chat_agent
from agents.report_writer import run_report_writer_agent


st.set_page_config(page_title="InsightFlow AI", page_icon="📊", layout="wide")

UPLOAD_DIR = "uploads"
CHARTS_DIR = "charts"
REPORTS_DIR = "reports"


def _init_session_state():
    defaults = {
        "result": None,
        "df": None,
        "dataset_name": None,
        "custom_charts": [],       # list of chart_meta dicts
        "custom_explanations": [], # matching explanations, same order/length
        "chat_history": [],        # list of dataset_chat_agent result dicts (+ "question")
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _regenerate_report():
    """Re-run the Report Writer with current pipeline result + everything
    generated interactively since, so the downloaded PDF always reflects
    the full session."""
    result = st.session_state["result"]
    report_path = run_report_writer_agent(
        dataset_name=st.session_state["dataset_name"],
        dataset_info=result["dataset_info"],
        cleaning_report=result["cleaning_report"],
        stats_report=result["stats_report"],
        charts=result["charts"],
        business_insights=result["business_insights"],
        chart_explanations=result.get("chart_explanations", []),
        recommendations=result.get("recommendations", {}),
        custom_charts=st.session_state["custom_charts"],
        custom_explanations=st.session_state["custom_explanations"],
        chat_history=st.session_state["chat_history"],
        output_dir=REPORTS_DIR,
    )
    st.session_state["result"]["report_path"] = report_path


def main():
    _init_session_state()

    st.title("📊 InsightFlow AI")
    st.caption("Autonomous Multi-Agent Data Analyst — upload a dataset, get an executive report, then dig deeper.")

    with st.sidebar:
        st.subheader("Settings")
        groq_key_input = st.text_input(
            "Groq API Key (optional)",
            type="password",
            help="Powers chart explanations, recommendations, custom chart "
                 "requests, and the dataset chat. Without it, everything "
                 "still works via rule-based fallbacks — just less flexible.",
        )
        if groq_key_input:
            os.environ["GROQ_API_KEY"] = groq_key_input
        st.caption("Get a free key at console.groq.com")

    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.info("Upload a CSV or Excel file to get started.")
        return

    # Reset interactive session state whenever a new file is uploaded.
    if st.session_state["dataset_name"] != uploaded_file.name:
        st.session_state["result"] = None
        st.session_state["custom_charts"] = []
        st.session_state["custom_explanations"] = []
        st.session_state["chat_history"] = []
        st.session_state["dataset_name"] = uploaded_file.name

    Path(UPLOAD_DIR).mkdir(exist_ok=True)
    save_path = Path(UPLOAD_DIR) / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        df = load_dataset(str(save_path))
    except Exception as exc:
        st.error(f"Failed to load file: {exc}")
        return
    st.session_state["df"] = df

    info = basic_dataset_info(df)
    col1, col2 = st.columns(2)
    col1.metric("Rows", f"{info['rows']:,}")
    col2.metric("Columns", info["columns"])

    with st.expander("Preview data"):
        st.dataframe(df.head(50), use_container_width=True)

    if st.button("🚀 Run Analysis", type="primary"):
        with st.spinner("Running multi-agent pipeline... this may take a minute."):
            try:
                result = run_pipeline(
                    df,
                    dataset_name=uploaded_file.name,
                    charts_dir=CHARTS_DIR,
                    reports_dir=REPORTS_DIR,
                )
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                return

        st.success("Analysis complete!")
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        return

    df = st.session_state["df"]

    # ------------------------------------------------------------------
    # Generated Insights
    # ------------------------------------------------------------------
    st.divider()
    st.header("Generated Insights")

    insights = result.get("business_insights", {})
    st.subheader("Executive Summary")
    st.write(insights.get("executive_summary", "N/A"))

    ic1, ic2 = st.columns(2)
    with ic1:
        st.subheader("Key Insights")
        for item in insights.get("key_insights", []):
            st.markdown(f"- {item}")
    with ic2:
        st.subheader("Recommendations")
        for item in insights.get("recommendations", []):
            st.markdown(f"- {item}")

    if insights.get("source") == "fallback_template":
        st.caption("⚠️ Generated via fallback template (no Groq API key provided or call failed).")

    # ------------------------------------------------------------------
    # Data Quality Report
    # ------------------------------------------------------------------
    st.divider()
    st.header("Data Quality Report")
    cleaning = result.get("cleaning_report", {})
    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Missing values", cleaning.get("missing_values", 0))
    qc2.metric("Duplicate rows", cleaning.get("duplicates", 0))
    qc3.metric("Empty columns", len(cleaning.get("empty_columns", [])))
    for suggestion in cleaning.get("cleaning_suggestions", []):
        st.markdown(f"- {suggestion}")

    # ------------------------------------------------------------------
    # Charts + Explanations (paired)
    # ------------------------------------------------------------------
    st.divider()
    st.header("Charts & Explanations")
    chart_metadata = result.get("charts", {}).get("chart_metadata", [])
    chart_explanations = result.get("chart_explanations", [])

    if chart_metadata:
        for i, meta in enumerate(chart_metadata):
            explanation = chart_explanations[i] if i < len(chart_explanations) else None
            cols_involved = [c for c in (meta.get("column"), meta.get("x_axis"), meta.get("y_axis")) if c]
            label = meta.get("chart_type", "Chart") + (f" — {', '.join(cols_involved)}" if cols_involved else "")
            with st.expander(label, expanded=False):
                img_path = meta.get("path")
                if img_path and Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                if explanation:
                    st.markdown(f"**Overview:** {explanation.get('chart_overview', '')}")
                    var_defs = explanation.get("variable_definitions") or {}
                    if var_defs:
                        st.markdown("**Variables:**")
                        for col, desc in var_defs.items():
                            st.markdown(f"- **{col}**: {desc}")
                    if explanation.get("comparison_summary"):
                        st.markdown(f"**What's being compared:** {explanation['comparison_summary']}")
                    observations = explanation.get("key_observations") or []
                    if observations:
                        st.markdown("**Key Observations:**")
                        for obs in observations:
                            st.markdown(f"- {obs}")
                    if explanation.get("business_interpretation"):
                        st.markdown(f"**Business Interpretation:** {explanation['business_interpretation']}")
                    actions = explanation.get("recommended_actions") or []
                    if actions:
                        st.markdown("**Recommended Actions:**")
                        for action in actions:
                            st.markdown(f"- {action}")
                    if explanation.get("source") == "fallback_template":
                        st.caption("⚠️ Explanation generated via fallback template.")
    else:
        st.write("No charts generated.")

    # ------------------------------------------------------------------
    # Recommended Analyses
    # ------------------------------------------------------------------
    st.divider()
    st.header("Recommended Analyses")
    rec_data = result.get("recommendations", {})
    rec_list = rec_data.get("recommendations", [])
    if rec_list:
        for rec in rec_list:
            with st.container(border=True):
                rc1, rc2 = st.columns([3, 1])
                with rc1:
                    st.markdown(f"**{rec.get('title', 'Untitled analysis')}**")
                    if rec.get("purpose"):
                        st.caption(f"Purpose: {rec['purpose']}")
                    if rec.get("business_value"):
                        st.caption(f"Business value: {rec['business_value']}")
                with rc2:
                    if st.button("Generate", key=f"rec_gen_{rec.get('title')}"):
                        _generate_custom_chart(rec.get("chart_request", ""), df, result)
                        st.rerun()
        if rec_data.get("source") == "rule_based":
            st.caption("⚠️ Recommendations generated via rule-based fallback (no Groq API key configured).")
    else:
        st.write("No additional analyses recommended.")

    # ------------------------------------------------------------------
    # Interactive Analytics — custom chart requests
    # ------------------------------------------------------------------
    st.divider()
    st.header("🎨 Request a Custom Chart")
    st.caption(
        "Try things like: \"scatter plot between age and salary\", "
        "\"show distribution of revenue\", \"boxplot for marketing_spend\"."
    )
    chart_request_text = st.text_input("Describe the chart you want", key="chart_request_input")
    if st.button("Generate Chart") and chart_request_text.strip():
        _generate_custom_chart(chart_request_text.strip(), df, result)
        st.rerun()

    if st.session_state["custom_charts"]:
        st.subheader("Your Custom Charts")
        for i, meta in enumerate(st.session_state["custom_charts"]):
            explanation = (
                st.session_state["custom_explanations"][i]
                if i < len(st.session_state["custom_explanations"]) else None
            )
            cols_involved = [c for c in (meta.get("column"), meta.get("x_axis"), meta.get("y_axis")) if c]
            label = meta.get("chart_type", "Chart") + (f" — {', '.join(cols_involved)}" if cols_involved else "")
            with st.expander(label, expanded=(i == len(st.session_state["custom_charts"]) - 1)):
                img_path = meta.get("path")
                if img_path and Path(img_path).exists():
                    st.image(img_path, use_container_width=True)
                if explanation:
                    st.markdown(f"**Overview:** {explanation.get('chart_overview', '')}")
                    observations = explanation.get("key_observations") or []
                    if observations:
                        st.markdown("**Key Observations:**")
                        for obs in observations:
                            st.markdown(f"- {obs}")
                    if explanation.get("business_interpretation"):
                        st.markdown(f"**Business Interpretation:** {explanation['business_interpretation']}")

    # ------------------------------------------------------------------
    # Dataset Chat
    # ------------------------------------------------------------------
    st.divider()
    st.header("💬 Ask Your Data")
    st.caption(
        "Try things like: \"What is the average revenue?\", "
        "\"Which product generated the highest revenue?\", "
        "\"How many records are there?\""
    )

    for turn in st.session_state["chat_history"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            if turn.get("success"):
                st.write(turn.get("answer"))
                table = turn.get("table")
                if table:
                    table_df = pd.DataFrame(table["rows"], columns=table["columns"])
                    st.dataframe(table_df, use_container_width=True, hide_index=True)
                if turn.get("business_interpretation"):
                    st.caption(turn["business_interpretation"])
                with st.expander("Details"):
                    if turn.get("reasoning"):
                        st.markdown(f"**Reasoning:** {turn['reasoning']}")
                    if turn.get("supporting_statistics"):
                        st.markdown(f"**Supporting statistics:** {turn['supporting_statistics']}")
                    if turn.get("expression"):
                        st.code(turn["expression"], language="python")
            else:
                st.error(turn.get("error", "Could not answer that question."))

    question = st.chat_input("Ask a question about your data")
    if question:
        with st.spinner("Analyzing..."):
            chat_result = run_dataset_chat_agent(question, df)
        chat_result["question"] = question
        st.session_state["chat_history"].append(chat_result)
        st.rerun()

    # ------------------------------------------------------------------
    # Report download (regenerated to include interactive session content)
    # ------------------------------------------------------------------
    st.divider()
    if st.session_state["custom_charts"] or st.session_state["chat_history"]:
        if st.button("🔄 Refresh Report with Custom Charts & Q&A"):
            with st.spinner("Rebuilding report..."):
                _regenerate_report()
            st.success("Report updated.")

    report_path = result.get("report_path")
    if report_path and Path(report_path).exists():
        with open(report_path, "rb") as f:
            st.download_button(
                "⬇️ Download PDF Report",
                data=f.read(),
                file_name=Path(report_path).name,
                mime="application/pdf",
                type="primary",
            )


def _generate_custom_chart(request_text: str, df, result: dict):
    """Run the Interactive Analytics Agent for a chart request, explain it,
    and append both to session state. Surfaces errors via st.error/st.warning
    rather than raising, since this is called from button click handlers."""
    if not request_text:
        return

    chart_result = run_interactive_analytics_agent(request_text, df, output_dir=CHARTS_DIR)
    if not chart_result["success"]:
        if chart_result.get("suggestion"):
            st.warning(f"{chart_result['error']}")
        else:
            st.error(chart_result["error"])
        return

    chart_meta = chart_result["chart_meta"]
    explanation = run_chart_explanation_agent(
        chart_meta, df=df, stats_report=result.get("stats_report")
    )
    st.session_state["custom_charts"].append(chart_meta)
    st.session_state["custom_explanations"].append(explanation)


if __name__ == "__main__":
    main()
