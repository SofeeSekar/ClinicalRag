"""
vector_store.py
In-memory vector store for clinical document RAG using ChromaDB
and sentence-transformers.
"""

print("Loading embedding model...")
from sentence_transformers import SentenceTransformer as _SentenceTransformer

_model = _SentenceTransformer("all-MiniLM-L6-v2")


class VectorStore:
    """Thin wrapper around a ChromaDB ephemeral collection that provides
    document-level add / search / delete operations."""

    def __init__(self) -> None:
        import chromadb

        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(
            name="clinical_docs",
            metadata={"hnsw:space": "cosine"},
        )

    # ── Write operations ──────────────────────────────────────────────────────

    def add_document(self, chunks: list, doc_name: str) -> int:
        """Index *chunks* (output of chunk_text) under *doc_name*.

        Any previously indexed data for this document is replaced first.
        Returns the number of chunks indexed.
        """
        self.delete_document(doc_name)

        if not chunks:
            return 0

        texts = [c["text"] for c in chunks]
        embeddings = _model.encode(texts, show_progress_bar=False).tolist()

        self._collection.add(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"source": doc_name, "page": c["page"]} for c in chunks],
        )

        n = len(chunks)
        print(f"Indexed {n} chunks from {doc_name}")
        return n

    def delete_document(self, doc_name: str) -> None:
        """Remove all chunks whose source metadata matches *doc_name*."""
        try:
            existing = self._collection.get(where={"source": doc_name})
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
        except Exception:
            pass  # Collection may be empty or doc not present

    # ── Read operations ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_source: str = None,
    ) -> list:
        """Embed *query* and return the closest matching chunks.

        Args:
            query:         Natural-language question.
            n_results:     Maximum number of results to consider.
            filter_source: If provided, restrict search to this document name.

        Returns:
            List of dicts sorted by score descending:
            {"text": str, "source": str, "page": int, "score": float}
            Only results with cosine similarity >= 0.22 are returned.
        """
        try:
            total = self._collection.count()
            if total == 0:
                return []

            actual_n = min(n_results, total)
            query_embedding = _model.encode([query], show_progress_bar=False).tolist()
            where = {"source": filter_source} if filter_source else None

            raw = self._collection.query(
                query_embeddings=query_embedding,
                n_results=actual_n,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            print(f"WARNING: Search failed — {exc}")
            return []

        results = []
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            # ChromaDB cosine distance: distance = 1 - similarity
            score = round(1.0 - dist, 4)
            if score >= 0.22:
                results.append(
                    {
                        "text": doc,
                        "source": meta["source"],
                        "page": meta["page"],
                        "score": score,
                    }
                )

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def list_documents(self) -> list:
        """Return a sorted list of unique document names in the collection."""
        try:
            result = self._collection.get(include=["metadatas"])
            return sorted({m["source"] for m in result["metadatas"]})
        except Exception:
            return []

    def get_chunk_count(self, doc_name: str) -> int:
        """Return the number of chunks stored for *doc_name*."""
        try:
            result = self._collection.get(where={"source": doc_name})
            return len(result["ids"])
        except Exception:
            return 0
