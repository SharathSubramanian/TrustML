"""LLM client for UF NaviGator Toolkit (OpenAI-compatible API).

Used only for planning (which tool next), the executive-summary wording and the
data-collection suggestions. Never for numbers. Every output is validated.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_BASE_URL = "https://api.ai.it.ufl.edu/v1"
DEFAULT_MODEL = "gpt-oss-120b"  # UF-hosted; student keys may not include cloud models (gpt-5.x, Claude)


def get_client():
    from openai import OpenAI  # imported lazily so the rest works without the package

    key = os.environ.get("NAVIGATOR_TOOLKIT_API_KEY")
    if not key:
        raise RuntimeError("Set NAVIGATOR_TOOLKIT_API_KEY in your .env file (see .env.example).")
    return OpenAI(api_key=key, base_url=os.environ.get("NAVIGATOR_BASE_URL", DEFAULT_BASE_URL), timeout=60)


def try_client():
    """Return a NaviGator client if a key is configured, else None (no error)."""
    if not os.environ.get("NAVIGATOR_TOOLKIT_API_KEY"):
        return None
    try:
        return get_client()
    except Exception:  # noqa: BLE001
        return None


def planner_model() -> str:
    return os.environ.get("TRUSTML_PLANNER_MODEL", DEFAULT_MODEL)


def chat(client, model: str, **kwargs):
    """chat.completions.create with temperature 0 where the model allows it.

    Reasoning models (e.g. the GPT-5 family) reject a non-default temperature; in that case
    the call is retried once without it (the run is then less deterministic, which the
    validation layer tolerates).
    """
    try:
        return client.chat.completions.create(model=model, temperature=0, **kwargs)
    except Exception as e:  # noqa: BLE001
        if "temperature" in str(e).lower():
            return client.chat.completions.create(model=model, **kwargs)
        raise
