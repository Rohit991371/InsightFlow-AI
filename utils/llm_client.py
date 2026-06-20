"""
utils/llm_client.py
--------------------
Shared helper for calling Groq with a system/user prompt pair and parsing
a JSON response, with consistent error handling across all agents that
use an LLM (Business Analyst, Chart Explanation, Interactive Analytics,
Recommendation, Dataset Chat).

Centralizing this avoids each agent re-implementing markdown-fence
stripping, JSON parsing, and the no-API-key/call-failure fallback path.
"""

from __future__ import annotations
import os
import json
from typing import Optional

from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class LLMUnavailableError(Exception):
    """Raised when no API key is configured. Callers should catch this
    and use a deterministic fallback rather than letting it propagate."""
    pass


def has_llm() -> bool:
    """Whether a Groq API key is currently configured."""
    return bool(os.environ.get("GROQ_API_KEY"))


def _strip_markdown_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw


def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 900,
) -> dict:
    """
    Call Groq's chat completion API expecting a JSON object response.

    Raises:
        LLMUnavailableError: if GROQ_API_KEY is not set.
        Exception: if the API call fails or returns invalid JSON
                   (callers should catch broadly and fall back).
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMUnavailableError("GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content
    cleaned = _strip_markdown_fences(raw)
    return json.loads(cleaned)


def call_llm_text(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 600,
) -> str:
    """Same as call_llm_json but returns raw text (no JSON parsing)."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMUnavailableError("GROQ_API_KEY is not configured.")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
