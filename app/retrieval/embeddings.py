"""
Local sentence-transformers embeddings -- not Cohere. This is a student
hackathon budget, the corpus is small, and a legal-domain-general embedding
model at this scale is not your bottleneck -- corpus correctness is.
"""

import threading

from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None
# Guards first-load only. encode() calls after that are thread-safe
# (SentenceTransformer.encode is stateless given a fixed model).
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    global _model
    # Double-checked locking: the outer check avoids acquiring the lock on
    # every call once the model is loaded (the common path). The inner check
    # is the safety net -- if two threads both pass the outer None check
    # before either enters the lock, only one of them will actually load the
    # model; the other will find it already set when it gets the lock.
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def preload_model() -> None:
    """Call once at app startup so the model download/load cost is paid
    before the first request, not during it."""
    get_model()


def embed(text: str) -> list[float]:
    model = get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Batch-encode -- one matmul instead of N Python-level calls. Not
    benchmarked in this sandbox (no network to fetch the model here) --
    time it yourself with `time python -m app.ingestion.ingest`."""
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vectors.tolist()
