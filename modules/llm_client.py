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

# Zephyr is non-gated and available on HF's free serverless inference tier.
# Uses text_generation (standard HF endpoint) instead of chat_completion
# (/v1/chat/completions) which is unreliable on the free tier.
_MODEL = "HuggingFaceH4/zephyr-7b-beta"
_BACKOFF = [2, 4, 8]  # seconds between retry attempts


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _format_zephyr_prompt(user_message: str) -> str:
    """Wrap the message in Zephyr's chat template so the model follows instructions."""
    return (
        "<|system|>\nYou are a precise clinical document analyst.</s>\n"
        f"<|user|>\n{user_message}</s>\n"
        "<|assistant|>\n"
    )


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
    max_tokens: int = 500,
    temperature: float = 0.1,
) -> str:
    """Generate a completion for *prompt* using the Zephyr-7B model.

    Uses text_generation (standard HF endpoint) which works on the free tier.
    Retries up to 3 times with exponential back-off on 503 / rate-limit errors.
    """
    from huggingface_hub import InferenceClient

    client = InferenceClient(token=HF_TOKEN, timeout=60)
    formatted = _format_zephyr_prompt(prompt)

    for attempt in range(3):
        try:
            response = client.text_generation(
                formatted,
                model=_MODEL,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
            )
            return response.strip()

        except Exception as exc:
            if _is_retriable(exc) and attempt < 2:
                time.sleep(_BACKOFF[attempt])
                continue
            raise

    return "Service temporarily unavailable. Please try again."
