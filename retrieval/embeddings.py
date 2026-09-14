"""Sentence Transformer embeddings for news retrieval."""

import numpy as np
from numpy.typing import NDArray


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
Float32Array = NDArray[np.float32]


class EmbeddingModel:
    """Small wrapper around a CPU-compatible Sentence Transformer model."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        # Import lazily so tests can inject a fake encoder without downloading or
        # importing the optional Hugging Face stack.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode_documents(self, texts: list[str]) -> Float32Array:
        """Encode document texts as a normalized float32 matrix."""

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> Float32Array:
        """Encode one query as a normalized float32 vector."""

        if not query or not query.strip():
            return np.empty((0,), dtype=np.float32)

        embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embedding, dtype=np.float32).reshape(-1)
