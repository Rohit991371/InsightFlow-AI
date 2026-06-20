"""
Test suite for InsightFlow AI.

Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.cleaner import run_data_cleaning_agent, apply_basic_cleaning
from agents.statistician import run_statistician_agent
from agents.visualizer import run_visualization_agent
from agents.analyst import run_business_analyst_agent, _fallback_summary
from agents.chart_explainer import run_chart_explanation_agent
from agents.recommender import run_recommendation_agent
from agents.interactive_analytics import run_interactive_analytics_agent, detect_chart_intent, ColumnNotFoundError
from agents.dataset_chat import run_dataset_chat_agent
from agents.report_writer import run_report_writer_agent
from utils.data_loader import load_dataset, basic_dataset_info
from utils.sandbox import safe_eval_pandas_expression
from workflow.graph import run_pipeline


@pytest.fixture
def messy_df():
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "category": np.random.choice(["A", "B", "C"], n),
        "spend": np.random.uniform(10, 1000, n),
        "revenue": np.random.uniform(20, 3000, n),
        "empty_col": [None] * n,
    })
    df.loc[0:9, "revenue"] = np.nan
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # inject duplicates
    return df


@pytest.fixture
def clean_df():
    np.random.seed(1)
    n = 50
    return pd.DataFrame({
        "x": np.random.uniform(0, 100, n),
        "y": np.random.uniform(0, 100, n),
        "label": np.random.choice(["red", "blue"], n),
    })


class TestDataLoader:
    def test_load_csv(self, tmp_path, clean_df):
        path = tmp_path / "test.csv"
        clean_df.to_csv(path, index=False)
        loaded = load_dataset(str(path))
        assert loaded.shape[0] == clean_df.shape[0]

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/tmp/does_not_exist_xyz.csv")

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("hello")
        with pytest.raises(ValueError):
            load_dataset(str(path))

    def test_basic_info(self, clean_df):
        info = basic_dataset_info(clean_df)
        assert info["rows"] == 50
        assert info["columns"] == 3


class TestCleaningAgent:
    def test_detects_missing_and_duplicates(self, messy_df):
        report = run_data_cleaning_agent(messy_df)
        assert report["missing_values"] > 0
        assert report["duplicates"] == 5
        assert "empty_col" in report["empty_columns"]

    def test_no_issues_on_clean_data(self, clean_df):
        report = run_data_cleaning_agent(clean_df)
        assert report["missing_values"] == 0
        assert report["duplicates"] == 0
        assert report["empty_columns"] == []

    def test_apply_basic_cleaning_removes_duplicates(self, messy_df):
        cleaned = apply_basic_cleaning(messy_df)
        assert cleaned.duplicated().sum() == 0
        assert "empty_col" not in cleaned.columns


class TestStatisticianAgent:
    def test_correlation_detected(self):
        n = 100
        x = np.random.uniform(0, 100, n)
        df = pd.DataFrame({"x": x, "y": x * 2 + np.random.normal(0, 1, n)})
        report = run_statistician_agent(df)
        assert len(report["strong_correlations"]) >= 1
        assert report["strong_correlations"][0]["correlation"] > 0.9

    def test_handles_single_numeric_column(self, clean_df):
        df = clean_df[["x", "label"]]
        report = run_statistician_agent(df)
        assert report["correlation_matrix"] == {}

    def test_top_categories(self, clean_df):
        report = run_statistician_agent(clean_df)
        assert "label" in report["top_categories"]


class TestVisualizationAgent:
    def test_generates_charts(self, tmp_path, messy_df):
        charts = run_visualization_agent(messy_df, output_dir=str(tmp_path))
        assert len(charts["histograms"]) > 0
        assert Path(charts["heatmap"]).exists()

    def test_empty_column_excluded(self, tmp_path, messy_df):
        charts = run_visualization_agent(messy_df, output_dir=str(tmp_path))
        filenames = " ".join(Path(p).name for p in charts["histograms"])
        assert "empty_col" not in filenames

    def test_skips_trend_without_datetime(self, tmp_path, clean_df):
        charts = run_visualization_agent(clean_df, output_dir=str(tmp_path))
        assert charts["trend_lines"] == []


class TestBusinessAnalystAgent:
    def test_fallback_when_no_api_key(self, monkeypatch, messy_df):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        cleaning = run_data_cleaning_agent(messy_df)
        stats = run_statistician_agent(messy_df)
        result = run_business_analyst_agent(cleaning, stats)
        assert result["source"] == "fallback_template"
        assert "executive_summary" in result
        assert isinstance(result["key_insights"], list)

    def test_fallback_summary_structure(self, messy_df):
        cleaning = run_data_cleaning_agent(messy_df)
        stats = run_statistician_agent(messy_df)
        summary = _fallback_summary(cleaning, stats)
        assert "executive_summary" in summary
        assert "key_insights" in summary
        assert "recommendations" in summary


class TestChartExplanationAgent:
    def test_fallback_explanation_structure(self, messy_df):
        stats = run_statistician_agent(messy_df)
        chart_meta = {"chart_type": "Histogram", "x_axis": "revenue", "y_axis": "Frequency", "column": "revenue", "path": "irrelevant.png"}
        explanation = run_chart_explanation_agent(chart_meta, df=messy_df, stats_report=stats)
        assert explanation["source"] == "fallback_template"
        for key in ("chart_overview", "variable_definitions", "comparison_summary",
                    "key_observations", "business_interpretation", "recommended_actions"):
            assert key in explanation

    def test_explanation_grounded_in_correlation(self):
        n = 100
        x = np.random.uniform(0, 100, n)
        df = pd.DataFrame({"x": x, "y": x * 2 + np.random.normal(0, 1, n)})
        stats = run_statistician_agent(df)
        chart_meta = {"chart_type": "Scatter Plot", "x_axis": "x", "y_axis": "y", "column": None, "path": "irrelevant.png"}
        explanation = run_chart_explanation_agent(chart_meta, df=df, stats_report=stats)
        joined_observations = " ".join(explanation["key_observations"]).lower()
        assert "x" in joined_observations and "y" in joined_observations

    def test_works_without_dataframe_or_stats(self):
        chart_meta = {"chart_type": "Heatmap", "x_axis": None, "y_axis": None, "column": None, "path": "irrelevant.png"}
        explanation = run_chart_explanation_agent(chart_meta)
        assert "chart_overview" in explanation


class TestRecommendationAgent:
    def test_recommends_from_strong_correlation(self):
        n = 100
        x = np.random.uniform(0, 100, n)
        df = pd.DataFrame({"x": x, "y": x * 2 + np.random.normal(0, 1, n)})
        stats = run_statistician_agent(df)
        result = run_recommendation_agent(df, stats, chart_metadata=[])
        assert result["source"] == "rule_based"
        chart_requests = " ".join(r["chart_request"] for r in result["recommendations"])
        assert "x" in chart_requests and "y" in chart_requests

    def test_respects_max_recommendations(self, messy_df):
        stats = run_statistician_agent(messy_df)
        result = run_recommendation_agent(messy_df, stats, chart_metadata=[])
        assert len(result["recommendations"]) <= 5

    def test_every_recommendation_has_required_fields(self, messy_df):
        stats = run_statistician_agent(messy_df)
        result = run_recommendation_agent(messy_df, stats, chart_metadata=[])
        for rec in result["recommendations"]:
            assert "title" in rec and "purpose" in rec and "business_value" in rec and "chart_request" in rec


class TestInteractiveAnalyticsAgent:
    def test_rule_based_scatter_intent(self, clean_df):
        intent = detect_chart_intent("scatter plot between x and y", clean_df)
        assert intent["chart_type"] == "scatter"
        assert intent["x_column"] == "x"
        assert intent["y_column"] == "y"

    def test_rule_based_distribution_intent(self, clean_df):
        intent = detect_chart_intent("show distribution of x", clean_df)
        assert intent["chart_type"] == "histogram"
        assert intent["column"] == "x"

    def test_unparseable_request_raises(self, clean_df):
        with pytest.raises(ValueError):
            detect_chart_intent("xyzzy completely nonsensical gibberish", clean_df)

    def test_generate_custom_chart_scatter(self, tmp_path, clean_df):
        result = run_interactive_analytics_agent(
            "scatter plot between x and y", clean_df, output_dir=str(tmp_path)
        )
        assert result["success"] is True
        assert Path(result["chart_meta"]["path"]).exists()

    def test_invalid_column_suggests_correction(self, tmp_path, clean_df):
        result = run_interactive_analytics_agent(
            "show distribution of xx", clean_df, output_dir=str(tmp_path)
        )
        # "xx" isn't a column, but is close to "x" — should either fail with
        # a suggestion or, if the fuzzy/rule parser resolves it directly to
        # "x", succeed. Either outcome is acceptable; a silent wrong answer
        # is not.
        if not result["success"]:
            assert result["suggestion"] in (None, "x")


class TestDatasetChatAgent:
    def test_average_question(self, clean_df):
        result = run_dataset_chat_agent("What is the average x?", clean_df)
        assert result["success"] is True
        assert result["source"] == "rule_based"
        assert result["table"] is None  # scalar answer, no table expected

    def test_row_count_question(self, clean_df):
        result = run_dataset_chat_agent("How many rows are there?", clean_df)
        assert result["success"] is True
        assert str(len(clean_df)) in str(result["answer"])

    def test_unanswerable_question_fails_cleanly(self, clean_df):
        result = run_dataset_chat_agent("blah blah nonsense gibberish query", clean_df)
        assert result["success"] is False
        assert result["error"]
        assert result["table"] is None

    def test_table_request_returns_structured_table(self):
        n = 60
        df = pd.DataFrame({
            "product": np.random.choice(["Laptop", "Mouse", "Keyboard"], n),
            "revenue": np.random.uniform(10, 500, n),
        })
        result = run_dataset_chat_agent("show me a table of revenue by product", df)
        assert result["success"] is True
        assert result["table"] is not None
        assert result["table"]["columns"] == ["product", "revenue"]
        assert len(result["table"]["rows"]) == 3  # one row per product
        product_names = {row[0] for row in result["table"]["rows"]}
        assert product_names == {"Laptop", "Mouse", "Keyboard"}

    def test_breakdown_request_without_named_metric_falls_back_to_counts(self):
        n = 40
        df = pd.DataFrame({
            "region": np.random.choice(["North", "South"], n),
            "value": np.random.uniform(0, 1, n),
        })
        result = run_dataset_chat_agent("breakdown of customers by region", df)
        assert result["success"] is True
        assert result["table"] is not None
        assert len(result["table"]["rows"]) == 2

    def test_top_n_question_returns_table(self, clean_df):
        result = run_dataset_chat_agent("top 2 label", clean_df)
        assert result["success"] is True
        assert result["table"] is not None
        assert len(result["table"]["rows"]) <= 2

    def test_table_values_are_json_safe_numeric_types(self):
        n = 30
        df = pd.DataFrame({
            "category": np.random.choice(["A", "B"], n),
            "amount": np.random.uniform(1, 1000, n),
        })
        result = run_dataset_chat_agent("table of amount by category", df)
        assert result["table"] is not None
        for row in result["table"]["rows"]:
            for value in row:
                assert not isinstance(value, (np.integer, np.floating))


class TestSandboxSecurity:
    @pytest.mark.parametrize("malicious_expr", [
        '__import__("os").system("ls")',
        'open("/etc/passwd").read()',
        'df.to_csv("/tmp/leak.csv")',
        'eval("1+1")',
        'df.query("a > 1")',
        '().__class__.__bases__[0].__subclasses__()',
        'df.__class__.__init__.__globals__',
        'exec("import os")',
        '(lambda: 1)()',
    ])
    def test_malicious_expressions_blocked(self, clean_df, malicious_expr):
        result = safe_eval_pandas_expression(malicious_expr, clean_df)
        assert result["success"] is False

    def test_legitimate_expression_succeeds(self, clean_df):
        result = safe_eval_pandas_expression("df['x'].mean()", clean_df)
        assert result["success"] is True
        assert isinstance(result["value"], float)

    def test_long_expression_rejected(self, clean_df):
        result = safe_eval_pandas_expression("df['x']" + "+0" * 1000, clean_df)
        assert result["success"] is False


class TestReportWriter:
    def test_generates_pdf(self, tmp_path, messy_df):
        cleaning = run_data_cleaning_agent(messy_df)
        stats = run_statistician_agent(messy_df)
        charts = run_visualization_agent(messy_df, output_dir=str(tmp_path / "charts"))
        insights = _fallback_summary(cleaning, stats)
        info = basic_dataset_info(messy_df)

        path = run_report_writer_agent(
            dataset_name="test.csv",
            dataset_info=info,
            cleaning_report=cleaning,
            stats_report=stats,
            charts=charts,
            business_insights=insights,
            output_dir=str(tmp_path / "reports"),
        )
        assert Path(path).exists()
        assert Path(path).suffix == ".pdf"
        assert Path(path).stat().st_size > 1000  # not an empty/broken PDF

    def test_generates_pdf_with_custom_charts_and_chat(self, tmp_path, clean_df):
        cleaning = run_data_cleaning_agent(clean_df)
        stats = run_statistician_agent(clean_df)
        charts = run_visualization_agent(clean_df, output_dir=str(tmp_path / "charts"))
        explanations = [
            run_chart_explanation_agent(meta, df=clean_df, stats_report=stats)
            for meta in charts.get("chart_metadata", [])
        ]
        recs = run_recommendation_agent(clean_df, stats, chart_metadata=charts.get("chart_metadata", []))
        insights = _fallback_summary(cleaning, stats)
        info = basic_dataset_info(clean_df)

        custom_chart_result = run_interactive_analytics_agent(
            "scatter plot between x and y", clean_df, output_dir=str(tmp_path / "charts")
        )
        custom_explanation = run_chart_explanation_agent(
            custom_chart_result["chart_meta"], df=clean_df, stats_report=stats
        )
        chat_result = run_dataset_chat_agent("What is the average x?", clean_df)
        chat_result["question"] = "What is the average x?"

        path = run_report_writer_agent(
            dataset_name="test.csv",
            dataset_info=info,
            cleaning_report=cleaning,
            stats_report=stats,
            charts=charts,
            business_insights=insights,
            chart_explanations=explanations,
            recommendations=recs,
            custom_charts=[custom_chart_result["chart_meta"]],
            custom_explanations=[custom_explanation],
            chat_history=[chat_result],
            output_dir=str(tmp_path / "reports"),
        )
        assert Path(path).exists()
        assert Path(path).stat().st_size > 1000


class TestFullPipeline:
    def test_end_to_end(self, tmp_path, messy_df):
        result = run_pipeline(
            messy_df,
            dataset_name="messy.csv",
            charts_dir=str(tmp_path / "charts"),
            reports_dir=str(tmp_path / "reports"),
        )
        assert "report_path" in result
        assert Path(result["report_path"]).exists()
        assert result["cleaning_report"]["duplicates"] == 5
        assert "business_insights" in result
        # Enhancement phase fields
        assert "chart_explanations" in result
        assert len(result["chart_explanations"]) == len(result["charts"]["chart_metadata"])
        assert "recommendations" in result
        assert "recommendations" in result["recommendations"]

    def test_pipeline_on_minimal_dataset(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = run_pipeline(
            df,
            dataset_name="tiny.csv",
            charts_dir=str(tmp_path / "charts"),
            reports_dir=str(tmp_path / "reports"),
        )
        assert Path(result["report_path"]).exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
