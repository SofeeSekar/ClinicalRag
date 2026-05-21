"""
llm_client.py
Thin wrapper around the HuggingFace Inference API for clinical text generation.
"""

import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

MODEL = "Qwen/Qwen2.5-7B-Instruct"

_client = None


def get_client() -> InferenceClient:
    global _client
    if _client is None:
        token = None
        try:
            import streamlit as st
            token = st.secrets.get("HF_TOKEN")
        except Exception:
            pass
        if not token:
            token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN not found. Please add it in Streamlit secrets or .env file.")
        _client = InferenceClient(token=token)
    return _client


def generate(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.1,
) -> str:
    client = get_client()
    response = client.chat_completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a precise clinical document analyst."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
