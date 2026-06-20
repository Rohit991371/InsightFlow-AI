"""
agents/interactive_analytics.py
---------------------------------
Interactive Analytics Agent

Responsibility: let users request custom charts via natural language
after the automated analysis is done. Handles:
  1. Intent detection — parse free text into a structured chart request
     (chart_type, x_column, y_column/column).
  2. Column validation — verify requested columns exist; suggest the
     closest real column name if not (fuzzy match) rather than just
     failing.
  3. Chart generation — render the requested chart and save it.
  4. Hands off to the Chart Explanation Agent automatically so every
     custom chart also gets an explanation (never returned bare).

Supported chart types: scatter, line, bar, histogram, pie, box, heatmap, area.
"""

from __future__ import annotations
import difflib
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils.llm_client import call_llm_json, has_llm, LLMUnavailableError

SUPPORTED_CHART_TYPES = {
    "scatter", "line", "bar", "histogram", "pie", "box", "heatmap", "area",
}

SYSTEM_PROMPT = """You convert a user's natural-language chart request into a \
structured JSON object describing the chart to generate.

Output ONLY valid JSON, no markdown fences, no preamble, matching EXACTLY \
this schema:
{
  "chart_type": "scatter" | "line" | "bar" | "histogram" | "pie" | "box" | "heatmap" | "area",
  "x_column": "<column name or null>",
  "y_column": "<column name or null>",
  "column": "<column name or null, used for single-variable charts like histogram/pie/box>"
}

Rules:
- "heatmap" with no specific columns mentioned -> use the correlation heatmap \
  convention: x_column=null, y_column=null, column=null (it covers all numeric columns).
- For single-variable requests (histogram, pie, box, distribution), set "column" \
  and leave x_column/y_column null.
- For two-variable requests (scatter, line, area, "compare X and Y"), set \
  x_column and y_column.
- For bar charts comparing a category to a value, x_column=category, y_column=value.
- Use the EXACT column names as given in the available columns list — do not \
  rename, abbreviate, or guess a column that isn't listed.
- If the user's intended column is ambiguous or doesn't closely match any \
  available column, still pick your best guess from the list — a downstream \
  validation step will catch mismatches.
"""


class ColumnNotFoundError(Exception):
    """Raised when a requested column doesn't exist and no close match is found."""
    def __init__(self, requested: str, suggestion: Optional[str] = None):
        self.requested = requested
        self.suggestion = suggestion
        msg = f"Column '{requested}' not found."
        if suggestion:
            msg += f" Did you mean '{suggestion}'?"
        super().__init__(msg)


def _rule_based_intent(text: str, columns: list[str]) -> Optional[dict]:
    """
    Lightweight keyword-based fallback intent parser, used when no LLM is
    configured. Handles the common phrasing patterns from the spec
    ("scatter plot between X and Y", "distribution of X", "boxplot for X").
    Returns None if it can't confidently parse the request.
    """
    lowered = text.lower()

    type_keywords = {
        "scatter": "scatter", "line": "line", "trend": "line",
        "bar": "bar", "histogram": "histogram", "distribution": "histogram",
        "pie": "pie", "box": "box", "boxplot": "box",
        "heatmap": "heatmap", "correlation": "heatmap", "area": "area",
    }
    chart_type = None
    for kw, ctype in type_keywords.items():
        if kw in lowered:
            chart_type = ctype
            break
    if chart_type is None:
        return None

    # Find column names mentioned in the text, in the order they appear
    # in the user's text (not sorted by length) — this matters because
    # "between X and Y" should map x_column=X, y_column=Y in that order.
    # Word-boundary matching avoids false positives like "salary" inside
    # "emp_salary"; a simple optional trailing "s" handles common plurals
    # ("ages" -> "age") without resorting to a full stemmer.
    import re
    found = []  # list of (start_index, column_name)
    for col in columns:
        pattern = r"\b" + re.escape(col.lower()) + r"s?\b"
        match = re.search(pattern, lowered)
        if match:
            found.append((match.start(), col))
    found.sort(key=lambda pair: pair[0])
    mentioned = [col for _, col in found]

    if chart_type == "heatmap" and not mentioned:
        return {"chart_type": "heatmap", "x_column": None, "y_column": None, "column": None}

    if chart_type in ("histogram", "pie", "box") :
        if mentioned:
            return {"chart_type": chart_type, "x_column": None, "y_column": None, "column": mentioned[0]}
        return None

    if chart_type in ("scatter", "line", "area", "bar"):
        if len(mentioned) >= 2:
            return {"chart_type": chart_type, "x_column": mentioned[0], "y_column": mentioned[1], "column": None}
        return None

    return None


def detect_chart_intent(user_text: str, df: pd.DataFrame) -> dict:
    """
    Parse a natural-language chart request into structured form.

    Returns:
        {"chart_type": str, "x_column": str|None, "y_column": str|None,
         "column": str|None, "source": "llm"|"rule_based"}

    Raises ValueError if intent could not be determined at all (neither
    LLM nor rule-based parsing produced a usable result).
    """
    columns = list(df.columns)

    if has_llm():
        try:
            user_prompt = (
                f"User request: \"{user_text}\"\n\n"
                f"Available columns: {columns}"
            )
            parsed = call_llm_json(SYSTEM_PROMPT, user_prompt, temperature=0.1)
            if parsed.get("chart_type") in SUPPORTED_CHART_TYPES:
                parsed["source"] = "llm"
                return parsed
        except LLMUnavailableError:
            pass
        except Exception:  # noqa: BLE001 — fall through to rule-based parsing
            pass

    rule_result = _rule_based_intent(user_text, columns)
    if rule_result:
        rule_result["source"] = "rule_based"
        return rule_result

    raise ValueError(
        "Could not determine what chart to generate from that request. "
        "Try phrasing like 'scatter plot between age and salary' or "
        "'show distribution of revenue'."
    )


def validate_columns(intent: dict, df: pd.DataFrame) -> dict:
    """
    Verify every column referenced in `intent` actually exists in `df`.
    Raises ColumnNotFoundError (with a fuzzy-match suggestion) on the
    first invalid column found.
    """
    columns = list(df.columns)
    for key in ("x_column", "y_column", "column"):
        requested = intent.get(key)
        if not requested:
            continue
        if requested in columns:
            continue
        close_matches = difflib.get_close_matches(requested, columns, n=1, cutoff=0.5)
        suggestion = close_matches[0] if close_matches else None
        raise ColumnNotFoundError(requested, suggestion)
    return intent


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


def generate_custom_chart(intent: dict, df: pd.DataFrame, output_dir: str = "charts") -> dict:
    """
    Render the chart described by `intent` and save it to disk.

    Returns chart_meta dict compatible with the Chart Explanation Agent:
        {"chart_type": str, "x_axis": str|None, "y_axis": str|None,
         "column": str|None, "path": str}

    Raises ValueError for unsupported chart types or unusable data
    (e.g. requesting a heatmap on a dataset with <2 numeric columns).
    """
    chart_type = intent["chart_type"]
    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ValueError(f"Unsupported chart type: '{chart_type}'.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    x_col, y_col, col = intent.get("x_column"), intent.get("y_column"), intent.get("column")

    if chart_type == "scatter":
        if not (x_col and y_col):
            raise ValueError("Scatter plot requires both x_column and y_column.")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(df[x_col], df[y_col], alpha=0.6, color="#4C72B0", edgecolor="white", s=40)
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} vs {x_col}")
        fname = f"custom_scatter_{_safe_filename(x_col)}_{_safe_filename(y_col)}.png"

    elif chart_type == "line":
        if not (x_col and y_col):
            raise ValueError("Line chart requires both x_column and y_column.")
        plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(plot_df[x_col], plot_df[y_col], color="#C44E52", linewidth=1.5)
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} over {x_col}")
        fig.autofmt_xdate()
        fname = f"custom_line_{_safe_filename(x_col)}_{_safe_filename(y_col)}.png"

    elif chart_type == "area":
        if not (x_col and y_col):
            raise ValueError("Area chart requires both x_column and y_column.")
        plot_df = df[[x_col, y_col]].dropna().sort_values(x_col)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(plot_df[x_col], plot_df[y_col], color="#4C72B0", alpha=0.4)
        ax.plot(plot_df[x_col], plot_df[y_col], color="#4C72B0", linewidth=1.2)
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} over {x_col}")
        fig.autofmt_xdate()
        fname = f"custom_area_{_safe_filename(x_col)}_{_safe_filename(y_col)}.png"

    elif chart_type == "bar":
        if not (x_col and y_col):
            # Allow bar(category) -> top value counts, same as the auto agent.
            target = x_col or y_col or col
            if not target:
                raise ValueError("Bar chart requires at least one column.")
            counts = df[target].value_counts(dropna=True).head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="#55A868")
            ax.set_title(f"Top categories: {target}")
            ax.set_xlabel("Count")
            fname = f"custom_bar_{_safe_filename(target)}.png"
            x_col, y_col, col = "Count", target, target
        else:
            agg = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x=agg.values, y=agg.index.astype(str), ax=ax, color="#55A868")
            ax.set_title(f"{y_col} by {x_col}")
            ax.set_xlabel(f"Average {y_col}")
            fname = f"custom_bar_{_safe_filename(x_col)}_{_safe_filename(y_col)}.png"

    elif chart_type == "histogram":
        target = col or x_col
        if not target:
            raise ValueError("Histogram requires a column.")
        series = df[target].dropna()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(series, bins=min(30, max(5, series.nunique())), color="#4C72B0", edgecolor="white")
        ax.set_title(f"Distribution of {target}")
        ax.set_xlabel(target); ax.set_ylabel("Frequency")
        fname = f"custom_histogram_{_safe_filename(target)}.png"
        col = target

    elif chart_type == "pie":
        target = col or x_col
        if not target:
            raise ValueError("Pie chart requires a column.")
        counts = df[target].value_counts(dropna=True).head(8)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{target} distribution")
        fname = f"custom_pie_{_safe_filename(target)}.png"
        col = target

    elif chart_type == "box":
        target = col or y_col or x_col
        if not target:
            raise ValueError("Box plot requires a column.")
        fig, ax = plt.subplots(figsize=(5, 4.5))
        if x_col and x_col != target:
            sns.boxplot(data=df, x=x_col, y=target, ax=ax, color="#4C72B0")
            ax.set_title(f"{target} by {x_col}")
        else:
            sns.boxplot(y=df[target].dropna(), ax=ax, color="#4C72B0")
            ax.set_title(f"Distribution of {target}")
        fname = f"custom_box_{_safe_filename(target)}.png"
        col = target

    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            raise ValueError("Heatmap requires at least 2 numeric columns in the dataset.")
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap")
        fname = "custom_heatmap.png"

    fig.tight_layout()
    path = out_dir / fname
    fig.savefig(path, dpi=120)
    plt.close(fig)

    return {
        "chart_type": chart_type.capitalize() if chart_type != "scatter" else "Scatter Plot",
        "x_axis": x_col, "y_axis": y_col, "column": col,
        "path": str(path),
    }


def run_interactive_analytics_agent(
    user_text: str,
    df: pd.DataFrame,
    output_dir: str = "charts",
) -> dict:
    """
    Full pipeline for a single custom chart request: intent detection ->
    column validation -> chart generation.

    Returns:
        {"success": True, "chart_meta": {...}, "intent": {...}}
        or
        {"success": False, "error": str, "suggestion": str|None}
    Never raises — all failures are captured for clean display in the UI.
    """
    try:
        intent = detect_chart_intent(user_text, df)
    except ValueError as exc:
        return {"success": False, "error": str(exc), "suggestion": None}

    try:
        validate_columns(intent, df)
    except ColumnNotFoundError as exc:
        return {"success": False, "error": str(exc), "suggestion": exc.suggestion}

    try:
        chart_meta = generate_custom_chart(intent, df, output_dir=output_dir)
    except ValueError as exc:
        return {"success": False, "error": str(exc), "suggestion": None}

    return {"success": True, "chart_meta": chart_meta, "intent": intent}
