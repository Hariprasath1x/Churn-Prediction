"""
groq_client.py
Thin wrapper for the Groq API client.
Reads credentials from environment variables or Streamlit secrets.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Load .env file if present (development environment)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not required; env vars may be set another way


def get_api_key() -> Optional[str]:
    """Retrieve Groq API key from env or Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    # Try Streamlit secrets (deployment)
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None


def get_model_name() -> str:
    """Return the Groq model ID to use."""
    default = "llama-3.3-70b-versatile"
    return os.environ.get("GROQ_MODEL", default)


def call_groq(prompt: str, system_prompt: str) -> Optional[str]:
    """
    Call the Groq API and return the response text.

    Returns None on any failure; callers should handle gracefully.
    """
    api_key = get_api_key()
    if not api_key:
        logger.warning("GROQ_API_KEY not set. Falling back to rule-based recommendation.")
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        model = get_model_name()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
            timeout=30,
        )
        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        return None


def is_groq_configured() -> bool:
    """Return True if a Groq API key is available."""
    return bool(get_api_key())
