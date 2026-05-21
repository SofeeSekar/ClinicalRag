"""
document_loader.py
Utilities for loading and chunking clinical PDF and DOCX documents.
"""

import io
import uuid


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _find_sentence_boundary(text: str) -> int:
    """Return the index just after the last sentence-ending character
    (period, question mark, exclamation mark, or newline) in *text*.
    Returns -1 if no boundary is found."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ".?!\n":
            return i + 1
    return -1


def _load_pdf(file_obj) -> list:
    """Extract text page-by-page from a PDF using pdfplumber.
    Returns [(page_number, page_text), ...] (1-based page numbers).
    Pages with fewer than 20 characters are skipped."""
    try:
        import pdfplumber
    except ImportError:
        print("WARNING: pdfplumber is not installed. Cannot load PDF.")
        return []

    pages = []
    try:
        with pdfplumber.open(file_obj) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if len(text) < 20:
                    continue
                pages.append((i, text))
    except Exception as exc:
        print(f"WARNING: Failed to extract PDF — {exc}")
        return []
    return pages


def _load_docx(file_obj) -> list:
    """Extract paragraphs from a DOCX and group them into ~400-word blocks.
    Returns [(block_number, block_text), ...].
    Blocks with fewer than 20 characters are skipped."""
    try:
        from docx import Document
    except ImportError:
        print("WARNING: python-docx is not installed. Cannot load DOCX.")
        return []

    try:
        doc = Document(file_obj)
        raw_paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    except Exception as exc:
        print(f"WARNING: Failed to extract DOCX — {exc}")
        return []

    blocks = []
    block_num = 1
    current_paras: list[str] = []
    current_word_count = 0
    target = 400

    for para in raw_paragraphs:
        word_count = len(para.split())
        # Flush current block when adding this paragraph would exceed the target
        if current_paras and current_word_count + word_count > target:
            block_text = "\n".join(current_paras)
            if len(block_text) >= 20:
                blocks.append((block_num, block_text))
                block_num += 1
            current_paras = [para]
            current_word_count = word_count
        else:
            current_paras.append(para)
            current_word_count += word_count

    # Flush the final block
    if current_paras:
        block_text = "\n".join(current_paras)
        if len(block_text) >= 20:
            blocks.append((block_num, block_text))

    return blocks


# ─── Public API ───────────────────────────────────────────────────────────────

def load_document(uploaded_file) -> list:
    """Load a Streamlit UploadedFile (PDF or DOCX) and return a list of
    (page_number, text) tuples.  Returns [] and prints a warning on failure."""
    name = uploaded_file.name.lower()
    # Wrap bytes in a BytesIO so both pdfplumber and python-docx can seek
    file_bytes = io.BytesIO(uploaded_file.read())

    if name.endswith(".pdf"):
        return _load_pdf(file_bytes)
    elif name.endswith(".docx"):
        return _load_docx(file_bytes)
    else:
        print(f"WARNING: Unsupported file type for '{uploaded_file.name}'. "
              "Only PDF and DOCX are supported.")
        return []


def chunk_text(
    pages: list,
    chunk_size: int = 400,
    overlap: int = 80,
) -> list:
    """Split a (page_number, text) list into overlapping word-based chunks,
    cutting at the nearest sentence boundary.

    Args:
        pages:      List of (page_number, text) tuples from load_document.
        chunk_size: Target maximum number of words per chunk.
        overlap:    Number of words carried over into the next chunk.

    Returns:
        List of dicts: {"text": str, "page": int, "chunk_id": str}
    """
    chunks: list[dict] = []

    for page_num, text in pages:
        words = text.split()
        total = len(words)
        if total == 0:
            continue

        start = 0
        while start < total:
            end = min(start + chunk_size, total)
            candidate = " ".join(words[start:end])

            # Try to cut at a sentence boundary when not at the page end
            if end < total:
                boundary = _find_sentence_boundary(candidate)
                if boundary != -1:
                    candidate = candidate[:boundary].rstrip()

            candidate = candidate.strip()
            word_count = len(candidate.split())

            if word_count >= 40:
                chunks.append(
                    {
                        "text": candidate,
                        "page": page_num,
                        "chunk_id": str(uuid.uuid4()),
                    }
                )

            # Advance by (actual chunk words - overlap), at minimum 1 word
            advance = max(1, word_count - overlap)
            start += advance

    return chunks
