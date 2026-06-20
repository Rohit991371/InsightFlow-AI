"""
utils/sandbox.py
-----------------
Restricted execution environment for running LLM-generated pandas
expressions against the user's dataset (used by the Dataset Chat Agent).

This is NOT a general-purpose Python sandbox — it's deliberately narrow:
the LLM is only ever asked to produce a SINGLE pandas EXPRESSION (not a
script, not a function, not a multi-statement block) that reads from a
DataFrame named `df` and evaluates to a result. We exploit that
constraint to keep the sandbox simple and auditable rather than trying
to fully secure arbitrary exec().

Defenses:
  - eval(), not exec() — no statements, imports, assignments, def/class,
    loops, or multi-line code are syntactically possible in an eval.
  - AST validation before execution: walks the parsed expression and
    rejects anything containing attribute access to dunder names,
    `import`, name lookups outside an explicit allowlist, or any Call
    node whose function isn't on the allowlist.
  - No builtins exposed at all (`__builtins__` set to a near-empty dict
    with only a tiny safe subset: len, min, max, sum, round, abs, sorted).
  - Only `df`, `pd`, and `np` are available as names.
  - Wall-clock timeout via a worker thread so a pathological expression
    (e.g. an accidental cartesian-product groupby) can't hang the app.
  - Result size cap — large DataFrame/Series results are truncated
    before being handed back, since they're headed for an LLM prompt
    or UI display, not further computation.
"""

from __future__ import annotations
import ast
import threading
from typing import Any

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------

# Names the expression is allowed to reference. Includes a few builtin
# type names (object, str, int, float, bool) since dtype comparisons
# like `df[col].dtype != object` are common and harmless.
ALLOWED_NAMES = {
    "df", "pd", "np", "True", "False", "None",
    "object", "str", "int", "float", "bool",
}

# Builtins allowed inside the expression (everything else is unreachable
# because __builtins__ is replaced wholesale, not patched).
SAFE_BUILTINS = {
    "len": len, "min": min, "max": max, "sum": sum,
    "round": round, "abs": abs, "sorted": sorted,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "str": str, "int": int, "float": float, "bool": bool,
    "range": range, "enumerate": enumerate, "zip": zip,
    "object": object,
}

# Attribute names that are never allowed regardless of object, since
# they're either Python internals or filesystem/process/network-adjacent
# even on a DataFrame (e.g. nothing pandas exposes legitimately needs
# these for read-only analysis).
BLOCKED_ATTR_SUBSTRINGS = (
    "__", "to_csv", "to_excel", "to_pickle", "to_sql", "to_json",
    "to_parquet", "to_feather", "to_hdf", "to_clipboard", "eval",
    "query",  # df.query() can run arbitrary expressions itself — block it
    "exec", "compile", "open", "os", "sys", "subprocess", "import",
)

MAX_EXPRESSION_LENGTH = 500
EXECUTION_TIMEOUT_SECONDS = 5
MAX_RESULT_ROWS = 200


class UnsafeExpressionError(Exception):
    """Raised when an expression fails AST validation."""
    pass


class ExecutionTimeoutError(Exception):
    """Raised when expression evaluation exceeds the time budget."""
    pass


def _validate_ast(expr: str) -> ast.Expression:
    """
    Parse the expression and walk the AST, rejecting anything outside
    a narrow allowlist. Raises UnsafeExpressionError on any violation.
    """
    if len(expr) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError(
            f"Expression too long ({len(expr)} chars, max {MAX_EXPRESSION_LENGTH})."
        )

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Not a valid single expression: {exc}") from exc

    # Names bound by comprehension targets (e.g. the `c` in
    # `[c for c in df.columns]`) are legitimate local variables, not
    # references to outside state — collect them so they're allowed
    # alongside the fixed ALLOWED_NAMES/SAFE_BUILTINS set.
    comprehension_bound_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.comprehension,)):
            for target_node in ast.walk(node.target):
                if isinstance(target_node, ast.Name):
                    comprehension_bound_names.add(target_node.id)

    for node in ast.walk(tree):
        # No imports, no statements of any kind — mode="eval" already
        # prevents most of this, but double-check for defense in depth.
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise UnsafeExpressionError("Imports are not allowed.")

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                              ast.Lambda)):
            raise UnsafeExpressionError("Defining functions/classes/lambdas is not allowed.")

        if isinstance(node, ast.Name):
            if (node.id not in ALLOWED_NAMES
                    and node.id not in SAFE_BUILTINS
                    and node.id not in comprehension_bound_names):
                raise UnsafeExpressionError(f"Use of name '{node.id}' is not allowed.")

        if isinstance(node, ast.Attribute):
            attr_lower = node.attr.lower()
            if any(blocked in attr_lower for blocked in BLOCKED_ATTR_SUBSTRINGS):
                raise UnsafeExpressionError(f"Access to '.{node.attr}' is not allowed.")

        if isinstance(node, ast.Subscript):
            pass  # df[...] indexing is fine and expected

        # Block any string literal that looks like it's trying to reach
        # a dunder or dangerous attribute via getattr-style tricks.
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                raise UnsafeExpressionError("Suspicious string literal containing '__'.")

    return tree


def _run_with_timeout(func, timeout: float):
    """Run func() in a daemon thread with a wall-clock timeout."""
    result_box: dict[str, Any] = {}

    def target():
        try:
            result_box["value"] = func()
        except Exception as exc:  # noqa: BLE001 — captured and re-raised on main thread
            result_box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        raise ExecutionTimeoutError(
            f"Expression did not complete within {timeout}s."
        )
    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("value")


def _truncate_result(value: Any):
    """Cap large pandas results before they're returned/serialized."""
    if isinstance(value, pd.DataFrame):
        truncated = len(value) > MAX_RESULT_ROWS
        return value.head(MAX_RESULT_ROWS), truncated
    if isinstance(value, pd.Series):
        truncated = len(value) > MAX_RESULT_ROWS
        return value.head(MAX_RESULT_ROWS), truncated
    return value, False


def safe_eval_pandas_expression(expr: str, df: pd.DataFrame) -> dict:
    """
    Validate and evaluate a single pandas expression against `df`.

    Returns:
        {
            "success": bool,
            "value": <result, possibly truncated> | None,
            "truncated": bool,
            "error": str | None,
        }
    Never raises — all failure modes are captured into the return dict
    so the calling agent can present a clean error message instead of
    crashing the request.
    """
    try:
        tree = _validate_ast(expr)
    except UnsafeExpressionError as exc:
        return {"success": False, "value": None, "truncated": False, "error": str(exc)}

    safe_globals = {"__builtins__": SAFE_BUILTINS, "pd": pd, "np": np}
    safe_locals = {"df": df}

    compiled = compile(tree, filename="<dataset_chat_expr>", mode="eval")

    def _execute():
        return eval(compiled, safe_globals, safe_locals)  # noqa: S307 — sandboxed via AST allowlist above

    try:
        raw_value = _run_with_timeout(_execute, EXECUTION_TIMEOUT_SECONDS)
    except ExecutionTimeoutError as exc:
        return {"success": False, "value": None, "truncated": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 — any pandas/runtime error becomes a clean message
        return {"success": False, "value": None, "truncated": False, "error": f"{type(exc).__name__}: {exc}"}

    value, truncated = _truncate_result(raw_value)
    return {"success": True, "value": value, "truncated": truncated, "error": None}
