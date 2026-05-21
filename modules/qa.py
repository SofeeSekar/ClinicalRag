"""
qa.py
Clinical document QA, compliance scanning, and summarisation.
"""

import json
import re

from modules.llm_client import generate
from modules.vector_store import VectorStore


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _format_chunks(chunks: list) -> str:
    """Render a list of chunk dicts as a labelled context block."""
    parts = [
        f"[Document: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in chunks
    ]
    return "\n\n".join(parts)


def _extract_json_tag(text: str, tag: str):
    """Extract and JSON-parse the content inside <tag>...</tag> in *text*.

    Strips Markdown code fences if the model wraps the JSON in them.
    Returns the parsed object, or raises json.JSONDecodeError / returns None
    if the tag is absent.
    """
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if not match:
        return None
    raw = match.group(1).strip()
    # Strip ```json … ``` fences that some models emit
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ─── Public API ───────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    vector_store: VectorStore,
    filter_source: str = None,
) -> dict:
    """Search the vector store and generate a cited answer to *question*.

    Args:
        question:      The clinical question to answer.
        vector_store:  Populated VectorStore instance.
        filter_source: If provided, restrict retrieval to this document name.

    Returns:
        {"answer": str, "sources": [{"source": str, "page": int}, ...]}
    """
    chunks = vector_store.search(question, n_results=5, filter_source=filter_source)

    if not chunks:
        return {
            "answer": (
                "No relevant information found in the loaded documents. "
                "Try uploading the relevant clinical document first."
            ),
            "sources": [],
        }

    context = _format_chunks(chunks)
    prompt = (
        "You are a precise clinical document analyst. "
        "Answer questions based ONLY on the provided document excerpts. "
        "Always cite the source document and page number. "
        "If the information is not in the provided context, say so clearly. "
        "Never speculate beyond what is documented.\n\n"
        "Document excerpts:\n"
        f"{context}\n\n"
        f"Clinical question: {question}\n"
        "Provide a precise answer with document citations:"
    )

    return {
        "answer": generate(prompt),
        "sources": [{"source": c["source"], "page": c["page"]} for c in chunks],
    }


# Queries used to sweep a document for compliance-relevant content
_COMPLIANCE_QUERIES = [
    "adverse events side effects safety signals",
    "contraindications warnings precautions",
    "dosage administration overdose",
    "patient population exclusion criteria",
    "storage handling disposal",
]

_COMPLIANCE_FALLBACK = [
    {
        "level": "REVIEW",
        "title": "Manual Review Required",
        "excerpt": "Automated parsing failed",
        "recommendation": "Review document manually",
    }
]


def compliance_scan(doc_name: str, vector_store: VectorStore) -> list:
    """Run targeted searches against *doc_name* and identify compliance findings.

    Returns a list of finding dicts:
    {"level": "CRITICAL|REVIEW|COMPLIANT", "title": str,
     "excerpt": str, "recommendation": str}
    """
    # Collect unique chunks across all targeted queries
    seen_texts: set = set()
    unique_chunks: list = []
    for query in _COMPLIANCE_QUERIES:
        for chunk in vector_store.search(query, n_results=3, filter_source=doc_name):
            if chunk["text"] not in seen_texts:
                seen_texts.add(chunk["text"])
                unique_chunks.append(chunk)

    if not unique_chunks:
        return _COMPLIANCE_FALLBACK

    context = _format_chunks(unique_chunks)
    prompt = (
        "You are a clinical compliance specialist. Review these document excerpts "
        "and identify compliance findings.\n"
        "For each finding output a JSON object with exactly these keys:\n"
        '{"level": "CRITICAL|REVIEW|COMPLIANT", '
        '"title": "short finding title", '
        '"excerpt": "the relevant text from the document", '
        '"recommendation": "specific action required or confirmation"}\n'
        "Wrap all findings in <findings>[...]</findings> tags.\n"
        "Be specific. Only flag genuine clinical safety concerns.\n\n"
        f"Document excerpts:\n{context}"
    )

    raw = generate(prompt, max_tokens=500)

    try:
        findings = _extract_json_tag(raw, "findings")
        if isinstance(findings, list) and findings:
            return findings
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    return _COMPLIANCE_FALLBACK


_SUMMARY_EMPTY: dict = {
    "document_type": "",
    "purpose": "",
    "key_entities": [],
    "critical_dates": [],
    "safety_signals": [],
    "regulatory_references": [],
}


def generate_summary(doc_name: str, vector_store: VectorStore) -> dict:
    """Generate a structured summary of *doc_name* from its indexed content.

    Returns a dict with keys: document_type, purpose, key_entities,
    critical_dates, safety_signals, regulatory_references.
    """
    chunks = vector_store.search(
        "purpose objective scope indication patient population compound drug organisation",
        n_results=6,
        filter_source=doc_name,
    )

    if not chunks:
        return _SUMMARY_EMPTY.copy()

    context = _format_chunks(chunks)
    prompt = (
        "Extract structured information from these clinical document excerpts. "
        "Return valid JSON with exactly these fields:\n"
        "  document_type  (string)\n"
        "  purpose        (string)\n"
        "  key_entities   (list of strings — drug/compound names, organisations)\n"
        "  critical_dates (list of strings)\n"
        "  safety_signals (list of strings)\n"
        "  regulatory_references (list of strings)\n"
        "Wrap the JSON in <summary>...</summary> tags.\n\n"
        f"Document excerpts:\n{context}"
    )

    raw = generate(prompt, max_tokens=500)

    try:
        summary = _extract_json_tag(raw, "summary")
        if isinstance(summary, dict):
            # Merge with defaults so all keys are always present
            return {**_SUMMARY_EMPTY, **summary}
    except (json.JSONDecodeError, TypeError, AttributeError):
        pass

    return _SUMMARY_EMPTY.copy()
