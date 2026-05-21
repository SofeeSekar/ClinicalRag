"""
vector_store.py
In-memory vector store for clinical document RAG using numpy cosine
similarity and sentence-transformers (no external vector DB required).
"""

print("Loading embedding model...")
import numpy as np
from sentence_transformers import SentenceTransformer as _SentenceTransformer

_model = _SentenceTransformer("all-MiniLM-L6-v2")


def _cosine_similarity(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between *mat* (N×D) and *vec* (D,)."""
    mat_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    vec_norm = np.linalg.norm(vec)
    denom = mat_norms.squeeze() * vec_norm
    denom = np.where(denom == 0, 1e-10, denom)
    return (mat @ vec) / denom


class VectorStore:
    """Pure-numpy in-memory vector store with document-level operations."""

    def __init__(self) -> None:
        # Each entry: {"text": str, "source": str, "page": int, "embedding": np.ndarray}
        self._chunks: list[dict] = []

    # ── Write operations ──────────────────────────────────────────────────────

    def add_document(self, chunks: list, doc_name: str) -> int:
        self.delete_document(doc_name)

        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = _model.encode(texts, show_progress_bar=False)  # (N, D)

        for chunk, emb in zip(chunks, embeddings):
            self._chunks.append(
                {
                    "text": chunk["text"],
                    "source": doc_name,
                    "page": chunk["page"],
                    "embedding": emb,
                }
            )

        n = len(chunks)
        print(f"Indexed {n} chunks from {doc_name}")
        return n

    def delete_document(self, doc_name: str) -> None:
        self._chunks = [c for c in self._chunks if c["source"] != doc_name]

    # ── Read operations ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_source: str = None,
    ) -> list:
        pool = (
            [c for c in self._chunks if c["source"] == filter_source]
            if filter_source
            else self._chunks
        )
        if not pool:
            return []

        query_emb = _model.encode([query], show_progress_bar=False)[0]  # (D,)
        mat = np.stack([c["embedding"] for c in pool])  # (N, D)
        scores = _cosine_similarity(mat, query_emb)

        top_idx = np.argsort(scores)[::-1][: n_results]
        results = []
        for i in top_idx:
            score = float(round(scores[i], 4))
            if score >= 0.22:
                results.append(
                    {
                        "text": pool[i]["text"],
                        "source": pool[i]["source"],
                        "page": pool[i]["page"],
                        "score": score,
                    }
                )
        return results

    def list_documents(self) -> list:
        return sorted({c["source"] for c in self._chunks})

    def get_chunk_count(self, doc_name: str) -> int:
        return sum(1 for c in self._chunks if c["source"] == doc_name)
