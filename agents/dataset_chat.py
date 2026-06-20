"""
agents/dataset_chat.py
------------------------
Dataset Chat Agent

Responsibility: let users ask plain-English questions about their data
and get back an answer grounded in an actual pandas computation, not an
LLM guess.

Workflow:
  1. Data Understanding   — build a compact schema summary (columns, dtypes,
                             sample values) so the LLM knows what it's
                             querying without seeing the full dataset.
  2. Pandas Query Generation — ask the LLM for a SINGLE pandas expression
                             (not a script) that answers the question.
  3. Data Retrieval        — evaluate that expression through utils.sandbox,
                             never via raw eval()/exec() on untrusted code.
  4. Natural Language Explanation — turn the raw result into a structured,
                             business-readable answer.

If no LLM is configured, falls back to a small set of rule-based query
patterns (max/min/mean/count/top-N) so basic questions still work.

SECURITY NOTE: LLM-generated code is NEVER executed directly. It only ever
reaches pandas via utils.sandbox.safe_eval_pandas_expression, which parses
the expression as an AST, allowlists names/attributes, strips builtins,
and enforces a timeout — see utils/sandbox.py for the full threat model.
"""

from __future__ import annotations
import re
from typing import Optional
import pandas as pd
import numpy as np

from utils.llm_client import call_llm_json, call_llm_text, has_llm, LLMUnavailableError
from utils.sandbox import safe_eval_pandas_expression

MAX_RETRY = 1  # if the generated expression fails validation/execution, retry once with the error fed back

QUERY_SYSTEM_PROMPT = """You translate a business user's question about a \
dataset into a SINGLE pandas expression that computes the answer.

You will be given the dataframe's column names, dtypes, and a few sample \
values per column. The dataframe is available as `df`.

Produce a JSON object with EXACTLY this schema:
{
  "expression": "a single pandas expression, e.g. df.groupby('product')['revenue'].sum().idxmax()",
  "explanation_hint": "1 short sentence describing in plain terms what this expression computes"
}

STRICT RULES for "expression":
- Must be ONE expression, not multiple statements. No semicolons, no \
assignment (=), no import, no def/lambda, no exec/eval, no df.query(...).
- Must reference only `df`, `pd`, `np`, and exact column names from the list given.
- Never call methods that write/export data (to_csv, to_excel, to_sql, etc.) — \
read-only analysis only.
- If the question asks for a single number, name, or yes/no, return an expression \
that evaluates to a scalar.
- If the question asks for a table, breakdown, list, comparison across multiple \
items, or uses words like "table", "list", "show", "breakdown", "for each", \
"by <column>", or "top N <plural noun>" — return an expression that evaluates \
to a pandas DataFrame or Series with one row per item (e.g. \
df.groupby('product')['revenue'].sum().reset_index() or \
df.groupby(['product','region'])['revenue'].sum().reset_index()), not a single \
aggregated value. Use .reset_index() so grouped results keep their group column(s) \
as real columns rather than an index.
- Cap tabular results to a reasonable size with .head(20) unless the question \
asks for "all" or "every".
- Output ONLY valid JSON, no markdown fences, no preamble.
"""

ANSWER_SYSTEM_PROMPT = """You are a senior business analyst. You are given a \
user's question, the pandas expression used to answer it, and the raw \
result. Write a clean, business-friendly response.

Produce a JSON object with EXACTLY this schema:
{
  "answer": "direct, short answer to the question (a value, name, or number) — \
leave this as a brief 1-sentence summary if the result is a table; do NOT try \
to transcribe table rows into this field",
  "reasoning": "1-2 sentences on how this was determined",
  "supporting_statistics": "1 sentence with any relevant supporting numbers",
  "business_interpretation": "1-2 sentences on what this means for the business, no jargon"
}

Rules:
- Be specific and reference actual numbers/values from the result.
- No statistical jargon.
- Output ONLY valid JSON, no markdown fences, no preamble.
"""


def _build_schema_summary(df: pd.DataFrame, max_samples: int = 3) -> dict:
    """Compact schema description: dtype + a few sample values per column,
    kept small so it's cheap to include in every chat-turn prompt."""
    summary = {}
    for col in df.columns:
        series = df[col].dropna()
        samples = series.head(max_samples).tolist()
        summary[col] = {
            "dtype": str(df[col].dtype),
            "sample_values": [str(v) for v in samples],
        }
    return summary


def _stringify_result(value) -> str:
    """Render a sandbox result compactly for the answer-generation prompt."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_string(max_rows=20)
    return str(value)


def _to_table_payload(value) -> Optional[dict]:
    """
    If the sandbox result is genuinely tabular (a DataFrame, or a Series
    with more than one entry), convert it into a small JSON-safe structure
    the UI/PDF can render as a real table instead of flattened text.

    A single-value Series (e.g. df['x'].mean() never produces one, but
    df.groupby(...).sum() on a 1-row group could) is still treated as a
    scalar, since a 1x1 "table" isn't worth a table widget.

    Returns:
        {"columns": [str, ...], "rows": [[...], ...]} or None if the
        result isn't tabular (i.e. it's a plain scalar).
    """
    if isinstance(value, pd.DataFrame):
        df_display = value.reset_index(drop=isinstance(value.index, pd.RangeIndex))
        if df_display.shape[0] == 0:
            return None
        return {
            "columns": [str(c) for c in df_display.columns],
            "rows": [[_json_safe(v) for v in row] for row in df_display.itertuples(index=False)],
        }

    if isinstance(value, pd.Series):
        if len(value) <= 1:
            return None
        name = value.name or "value"
        index_name = value.index.name or "index"
        return {
            "columns": [str(index_name), str(name)],
            "rows": [[_json_safe(idx), _json_safe(v)] for idx, v in value.items()],
        }

    return None


def _json_safe(value):
    """Coerce a single pandas/numpy scalar into a plain Python type so the
    table payload can be safely passed through JSON-ish dict handling and
    rendered by Streamlit/ReportLab without numpy-type surprises.

    Note: DataFrame.itertuples() yields native Python floats/ints (not
    numpy scalars) even when the source column is float64/int64, so both
    native and numpy numeric types must be handled here."""
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 4)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# ---------------------------------------------------------------------------
# Rule-based fallback (no LLM configured)
# ---------------------------------------------------------------------------

def _find_column(token: str, columns: list[str]) -> Optional[str]:
    token_lower = token.strip().lower().replace(" ", "_")
    for col in columns:
        if col.lower() == token_lower:
            return col
    for col in columns:
        if token_lower in col.lower() or col.lower() in token_lower:
            return col
    return None


def _rule_based_query(question: str, df: pd.DataFrame) -> Optional[dict]:
    """
    Handles a handful of common question shapes without an LLM:
      - "table/breakdown/list of <num_col> by <cat_col>" (and "for each <cat_col>")
      - "average/mean <col>"
      - "highest/maximum/max <col>"
      - "lowest/minimum/min <col>"
      - "how many <rows/records> ... > / < / >= / <= <number>" on a column
      - "top N <col>"
      - "which <category_col> has the highest/most <numeric_col>"
    Returns {"expression": str, "explanation_hint": str} or None if no
    pattern matches confidently.
    """
    q = question.lower().strip()
    columns = list(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "str", "category", "bool"]).columns.tolist()

    # "table/breakdown/list/show ... <num_col> by/for each <cat_col>"
    # Explicit table-style requests — checked first so they don't get
    # mis-captured by the narrower scalar patterns below (e.g. "highest").
    wants_table = bool(re.search(r"\b(table|breakdown|list|show)\b", q))
    by_match = re.search(r"(?:by|for each|per)\s+([\w\s]+?)(?:\?|$)", q)
    if wants_table and by_match:
        group_col = _find_column(by_match.group(1), categorical_cols) or _find_column(by_match.group(1), columns)
        value_col = next((c for c in numeric_cols if c.lower() in q), None)
        if group_col and value_col:
            return {
                "expression": f"df.groupby('{group_col}')['{value_col}'].sum().reset_index().head(20)",
                "explanation_hint": f"Total {value_col} broken down by {group_col}.",
            }
        if group_col and not value_col:
            # No specific metric named — fall back to row counts per group,
            # still genuinely tabular and useful (e.g. "table of customers by region").
            return {
                "expression": f"df['{group_col}'].value_counts().reset_index().head(20)",
                "explanation_hint": f"Row count broken down by {group_col}.",
            }

    # "which <cat_col> ... highest/most <num_col>" / "<cat_col> generated the highest revenue"
    # Only fires when the captured word actually resolves to a real
    # categorical column — otherwise "What is the highest X" would
    # wrongly match here (e.g. capturing "is") before the simpler
    # "highest <col>" pattern below gets a chance.
    m = re.search(r"(?:which|what)\s+(\w+)", q)
    if m and ("highest" in q or "most" in q or "top" in q):
        cat_guess = _find_column(m.group(1), categorical_cols)
        if cat_guess:
            num_guess = None
            for col in numeric_cols:
                if col.lower() in q:
                    num_guess = col
                    break
            num_guess = num_guess or (numeric_cols[0] if numeric_cols else None)
            if num_guess:
                return {
                    "expression": f"df.groupby('{cat_guess}')['{num_guess}'].sum().idxmax()",
                    "explanation_hint": f"Group by {cat_guess} and find the one with the highest total {num_guess}.",
                }

    # "average/mean <col>"
    m = re.search(r"(?:average|mean)\s+([\w\s]+)", q)
    if m:
        col = _find_column(m.group(1), numeric_cols)
        if col:
            return {"expression": f"df['{col}'].mean()", "explanation_hint": f"Average of {col}."}

    # "highest/maximum/max <col>"
    m = re.search(r"(?:highest|maximum|max)\s+([\w\s]+)", q)
    if m:
        col = _find_column(m.group(1), numeric_cols)
        if col:
            return {"expression": f"df['{col}'].max()", "explanation_hint": f"Maximum value of {col}."}

    # "lowest/minimum/min <col>"
    m = re.search(r"(?:lowest|minimum|min)\s+([\w\s]+)", q)
    if m:
        col = _find_column(m.group(1), numeric_cols)
        if col:
            return {"expression": f"df['{col}'].min()", "explanation_hint": f"Minimum value of {col}."}

    # "how many records/rows/entries are there" (no specific threshold/column)
    if re.search(r"how many\s+(?:records|rows|entries|customers|items)\b", q) and "than" not in q:
        return {"expression": "len(df)", "explanation_hint": "Total number of rows in the dataset."}

    # "how many ... > NUMBER" on a mentioned column
    m = re.search(r"more than\s*\$?([\d,]+\.?\d*)", q)
    if m:
        threshold = m.group(1).replace(",", "")
        col = None
        for c in numeric_cols:
            if c.lower() in q:
                col = c
                break
        col = col or (numeric_cols[0] if numeric_cols else None)
        if col:
            return {
                "expression": f"int((df['{col}'] > {threshold}).sum())",
                "explanation_hint": f"Count of rows where {col} is greater than {threshold}.",
            }

    # "top N <col>"
    m = re.search(r"top\s+(\d+)\s+([\w\s]+)", q)
    if m:
        n = int(m.group(1))
        col = _find_column(m.group(2), categorical_cols) or _find_column(m.group(2), columns)
        if col:
            return {
                "expression": f"df['{col}'].value_counts().head({n})",
                "explanation_hint": f"Top {n} most frequent values in {col}.",
            }

    # "which month had the highest <col>" / generic time-based highest, when a
    # datetime column exists
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if datetime_cols and ("month" in q or "date" in q) and ("highest" in q or "most" in q):
        date_col = datetime_cols[0]
        num_guess = next((c for c in numeric_cols if c.lower() in q), numeric_cols[0] if numeric_cols else None)
        if num_guess:
            return {
                "expression": f"df.groupby(df['{date_col}'].dt.to_period('M'))['{num_guess}'].sum().idxmax()",
                "explanation_hint": f"Month with the highest total {num_guess}.",
            }

    return None


def _fallback_answer(question: str, expr_meta: dict, sandbox_result: dict) -> dict:
    """Deterministic answer formatting when no LLM is available."""
    value = sandbox_result.get("value")
    table = _to_table_payload(value)
    return {
        "answer": _stringify_result(value) if table is None else f"Showing {len(table['rows'])} row(s) — see table below.",
        "table": table,
        "reasoning": expr_meta.get("explanation_hint", "Computed directly from the dataset."),
        "supporting_statistics": f"Computed via: {expr_meta.get('expression', 'n/a')}",
        "business_interpretation": (
            "Review this figure alongside other metrics to confirm it aligns with expectations."
        ),
        "source": "rule_based",
    }


def _generate_expression(question: str, df: pd.DataFrame, prior_error: Optional[str] = None) -> Optional[dict]:
    """Ask the LLM for a pandas expression; returns None if unavailable."""
    if not has_llm():
        return None
    schema = _build_schema_summary(df)
    user_prompt = (
        f"Question: \"{question}\"\n\n"
        f"Columns and schema: {schema}\n\n"
        f"Row count: {len(df)}"
    )
    if prior_error:
        user_prompt += (
            f"\n\nA previous attempt failed with this error — fix it and try a "
            f"different valid expression:\n{prior_error}"
        )
    try:
        return call_llm_json(QUERY_SYSTEM_PROMPT, user_prompt, temperature=0.1)
    except LLMUnavailableError:
        return None
    except Exception:  # noqa: BLE001 — malformed JSON etc, treat as unavailable for this turn
        return None


def run_dataset_chat_agent(question: str, df: pd.DataFrame) -> dict:
    """
    Answer a natural-language question about `df`.

    Returns:
        {
          "success": bool,
          "answer": str | None,
          "table": {"columns": [str,...], "rows": [[...],...]} | None,
          "reasoning": str | None,
          "supporting_statistics": str | None,
          "business_interpretation": str | None,
          "expression": str | None,   # the pandas expression actually run, for transparency
          "source": "llm" | "rule_based",
          "error": str | None,
        }
    Never raises. "table" is populated whenever the underlying pandas
    result has more than one row worth showing (a DataFrame, or a Series
    with 2+ entries) — the UI should render it as an actual table rather
    than relying on "answer" alone, which is kept to a short summary.
    """
    expr_meta = _generate_expression(question, df)
    source = "llm"

    if expr_meta is None or "expression" not in expr_meta:
        expr_meta = _rule_based_query(question, df)
        source = "rule_based"

    if expr_meta is None:
        return {
            "success": False, "answer": None, "table": None, "reasoning": None,
            "supporting_statistics": None, "business_interpretation": None,
            "expression": None, "source": "none",
            "error": (
                "Could not understand that question well enough to query the data. "
                "Try asking about a specific column, e.g. 'What is the average revenue?' "
                "or 'Which product has the highest sales?'"
            ),
        }

    result = safe_eval_pandas_expression(expr_meta["expression"], df)

    # One retry with the error fed back, only if we used the LLM path —
    # the rule-based path doesn't have an LLM to correct itself with.
    if not result["success"] and source == "llm":
        for _ in range(MAX_RETRY):
            retried = _generate_expression(question, df, prior_error=result["error"])
            if not retried or "expression" not in retried:
                break
            expr_meta = retried
            result = safe_eval_pandas_expression(expr_meta["expression"], df)
            if result["success"]:
                break

    if not result["success"]:
        return {
            "success": False, "answer": None, "table": None, "reasoning": None,
            "supporting_statistics": None, "business_interpretation": None,
            "expression": expr_meta.get("expression"), "source": source,
            "error": f"Could not compute an answer: {result['error']}",
        }

    if has_llm():
        try:
            user_prompt = (
                f"Question: \"{question}\"\n"
                f"Pandas expression used: {expr_meta['expression']}\n"
                f"Result: {_stringify_result(result['value'])}"
            )
            parsed = call_llm_json(ANSWER_SYSTEM_PROMPT, user_prompt, temperature=0.2)
            parsed["expression"] = expr_meta["expression"]
            parsed["source"] = source
            parsed["success"] = True
            parsed["error"] = None
            parsed["table"] = _to_table_payload(result["value"])
            return parsed
        except Exception:  # noqa: BLE001 — fall through to deterministic formatting
            pass

    fallback = _fallback_answer(question, expr_meta, result)
    fallback["expression"] = expr_meta.get("expression")
    fallback["success"] = True
    fallback["error"] = None
    fallback["source"] = source
    return fallback
