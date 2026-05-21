"""
llm_client.py
Thin wrapper around the HuggingFace Inference API for clinical text generation.
"""

import os
import time

from dotenv import load_dotenv

# ─── Token loading ────────────────────────────────────────────────────────────
load_dotenv()
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# Fallback: read from Streamlit secrets (Streamlit Cloud deployment)
if not HF_TOKEN:
    try:
        import streamlit as st
        HF_TOKEN = st.secrets.get("HF_TOKEN", "")
    except Exception:
        pass

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Add it to your .env file or Streamlit secrets.")

_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
_BACKOFF = [2, 4, 8]  # seconds between retry attempts


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _is_retriable(exc: Exception) -> bool:
    """Return True for 503 / 429 / rate-limit class errors that warrant retry."""
    try:
        from huggingface_hub.utils import HfHubHTTPError
        if isinstance(exc, HfHubHTTPError):
            resp = getattr(exc, "response", None)
            if resp is not None:
                return resp.status_code in (429, 503)
    except ImportError:
        pass
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in ("503", "429", "rate limit", "too many requests", "service unavailable")
    )


# ─── Public API ───────────────────────────────────────────────────────────────

def generate(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """Generate a completion for *prompt* using the Qwen 2.5-7B Instruct model.

    Retries up to 3 times with exponential back-off (2 → 4 → 8 seconds) on
    503 or rate-limit errors.  Non-retriable errors are re-raised immediately.
    Returns a user-facing fallback string if all attempts fail.

    Args:
        prompt:      The full prompt string to send to the model.
        max_tokens:  Maximum tokens in the generated response.
        temperature: Sampling temperature — kept low (0.1) for factual accuracy.

    Returns:
        Generated text as a stripped string, or a fallback message on failure.
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        model=_MODEL,
        token=HF_TOKEN,
        timeout=30,
    )
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(3):
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            if _is_retriable(exc) and attempt < 2:
                time.sleep(_BACKOFF[attempt])
                continue
            if not _is_retriable(exc):
                raise
            # Retriable error on the final attempt — fall through to return below

    return "Service temporarily unavailable. Please try again."
