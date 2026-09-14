"""Local inference adapters: embedding, reranking (M2), entailment (M5)."""

from scholiast.infrastructure.ml.embedder import BGE_QUERY_PREFIX, LocalEmbedder

__all__ = ["BGE_QUERY_PREFIX", "LocalEmbedder"]
