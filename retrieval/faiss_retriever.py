"""FAISS semantic retrieval over news chunks."""

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .embeddings import EmbeddingModel


Float32Array = NDArray[np.float32]


class Encoder(Protocol):
    """Interface required by :class:`FAISSRetriever`."""

    def encode_documents(self, texts: list[str]) -> Float32Array:
        ...

    def encode_query(self, query: str) -> Float32Array:
        ...


class FAISSRetriever:
    """Rank news chunks by cosine similarity using a FAISS inner-product index."""

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        encoder: Encoder | None = None,
    ) -> None:
        self.chunks = chunks
        self.encoder = encoder if encoder is not None else EmbeddingModel()
        self._index = None

        if not chunks:
            return

        import faiss

        document_texts = [
            f"{str(chunk.get('title') or '')} {str(chunk.get('text') or '')}"
            for chunk in chunks
        ]
        embeddings = self._as_document_matrix(
            self.encoder.encode_documents(document_texts)
        )
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)

    @property
    def index(self) -> Any:
        """Return the underlying FAISS index, or ``None`` for an empty corpus."""

        return self._index

    @staticmethod
    def _as_document_matrix(embeddings: Any) -> Float32Array:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Document embeddings must be a non-empty 2D matrix")
        return np.ascontiguousarray(matrix, dtype=np.float32)

    @staticmethod
    def _as_query_vector(embedding: Any) -> Float32Array:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            return vector
        return np.ascontiguousarray(vector, dtype=np.float32)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return up to ``top_k`` chunks ordered by descending similarity."""

        if not query or not query.strip() or top_k <= 0 or not self.chunks:
            return []
        if self._index is None:
            return []

        query_embedding = self._as_query_vector(self.encoder.encode_query(query))
        if query_embedding.size == 0:
            return []
        query_matrix = query_embedding.reshape(1, -1)
        if query_matrix.shape[1] != self._index.d:
            raise ValueError(
                "Query embedding dimension does not match document embedding dimension"
            )

        result_count = min(top_k, len(self.chunks))
        scores, indices = self._index.search(query_matrix, result_count)
        ranked = sorted(
            (
                float(scores[0][position]),
                int(indices[0][position]),
            )
            for position in range(result_count)
            if int(indices[0][position]) >= 0
        )
        ranked.reverse()

        results: list[dict[str, Any]] = []
        for rank, (score, index) in enumerate(ranked, start=1):
            chunk = self.chunks[index]
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "article_id": chunk.get("article_id", ""),
                    "title": chunk.get("title", ""),
                    "source": chunk.get("source", ""),
                    "url": chunk.get("url", ""),
                    "published_at": chunk.get("published_at", ""),
                    "text": chunk.get("text", ""),
                }
            )
        return results
