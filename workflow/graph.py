"""
graph.py
--------
Orchestrator Agent (LangGraph)

Wires the pipeline agents into a linear flow per the Enhancement Phase spec:

  START -> Data Cleaning -> Statistician -> Visualization
        -> Chart Explanation -> Recommendation
        -> Business Analyst -> Report Writer -> END

A linear graph remains sufficient here — each step's output is independent
of its sibling steps, and a linear graph is simpler to debug than true
parallel branching for this scope.

The Interactive Analytics Agent (custom user charts) and Dataset Chat Agent
are NOT part of this graph — they're invoked directly from the Streamlit UI
on-demand, outside the automated pipeline, since they're driven by live user
input rather than running once per uploaded dataset. See agents/
interactive_analytics.py and agents/dataset_chat.py.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Any

import pandas as pd
from langgraph.graph import StateGraph, END

from agents.cleaner import run_data_cleaning_agent
from agents.statistician import run_statistician_agent
from agents.visualizer import run_visualization_agent
from agents.chart_explainer import run_chart_explanation_agent
from agents.recommender import run_recommendation_agent
from agents.analyst import run_business_analyst_agent
from agents.report_writer import run_report_writer_agent
from utils.data_loader import basic_dataset_info


class PipelineState(TypedDict, total=False):
    # Inputs
    dataframe: pd.DataFrame
    dataset_name: str
    charts_dir: str
    reports_dir: str

    # Intermediate / outputs
    dataset_info: dict
    cleaning_report: dict
    stats_report: dict
    charts: dict
    chart_explanations: list  # one explanation dict per auto-generated chart, same order as charts["chart_metadata"]
    recommendations: dict
    business_insights: dict
    report_path: str
    error: Optional[str]


def _node_data_cleaning(state: PipelineState) -> PipelineState:
    df = state["dataframe"]
    state["dataset_info"] = basic_dataset_info(df)
    state["cleaning_report"] = run_data_cleaning_agent(df)
    return state


def _node_statistician(state: PipelineState) -> PipelineState:
    df = state["dataframe"]
    state["stats_report"] = run_statistician_agent(df)
    return state


def _node_visualization(state: PipelineState) -> PipelineState:
    df = state["dataframe"]
    charts_dir = state.get("charts_dir", "charts")
    state["charts"] = run_visualization_agent(df, output_dir=charts_dir)
    return state


def _node_chart_explanation(state: PipelineState) -> PipelineState:
    df = state["dataframe"]
    stats_report = state["stats_report"]
    chart_metadata = state["charts"].get("chart_metadata", [])
    state["chart_explanations"] = [
        run_chart_explanation_agent(meta, df=df, stats_report=stats_report)
        for meta in chart_metadata
    ]
    return state


def _node_recommendation(state: PipelineState) -> PipelineState:
    df = state["dataframe"]
    chart_metadata = state["charts"].get("chart_metadata", [])
    state["recommendations"] = run_recommendation_agent(
        df, state["stats_report"], chart_metadata=chart_metadata
    )
    return state


def _node_business_analyst(state: PipelineState) -> PipelineState:
    state["business_insights"] = run_business_analyst_agent(
        cleaning_report=state["cleaning_report"],
        stats_report=state["stats_report"],
    )
    return state


def _node_report_writer(state: PipelineState) -> PipelineState:
    reports_dir = state.get("reports_dir", "reports")
    state["report_path"] = run_report_writer_agent(
        dataset_name=state.get("dataset_name", "dataset"),
        dataset_info=state["dataset_info"],
        cleaning_report=state["cleaning_report"],
        stats_report=state["stats_report"],
        charts=state["charts"],
        chart_explanations=state.get("chart_explanations", []),
        recommendations=state.get("recommendations", {}),
        business_insights=state["business_insights"],
        output_dir=reports_dir,
    )
    return state


def build_graph():
    """
    Construct and compile the LangGraph state machine.
    """
    graph = StateGraph(PipelineState)

    graph.add_node("data_cleaning", _node_data_cleaning)
    graph.add_node("statistician", _node_statistician)
    graph.add_node("visualization", _node_visualization)
    graph.add_node("chart_explanation", _node_chart_explanation)
    graph.add_node("recommendation", _node_recommendation)
    graph.add_node("business_analyst", _node_business_analyst)
    graph.add_node("report_writer", _node_report_writer)

    graph.set_entry_point("data_cleaning")
    graph.add_edge("data_cleaning", "statistician")
    graph.add_edge("statistician", "visualization")
    graph.add_edge("visualization", "chart_explanation")
    graph.add_edge("chart_explanation", "recommendation")
    graph.add_edge("recommendation", "business_analyst")
    graph.add_edge("business_analyst", "report_writer")
    graph.add_edge("report_writer", END)

    return graph.compile()


def run_pipeline(df: pd.DataFrame, dataset_name: str = "dataset",
                  charts_dir: str = "charts", reports_dir: str = "reports") -> dict:
    """
    Convenience wrapper: run the full pipeline on a DataFrame and
    return the final state dict with every agent's output.
    """
    app = build_graph()
    initial_state: PipelineState = {
        "dataframe": df,
        "dataset_name": dataset_name,
        "charts_dir": charts_dir,
        "reports_dir": reports_dir,
    }
    final_state = app.invoke(initial_state)
    return final_state
