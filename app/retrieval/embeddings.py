"""
Local sentence-transformers embeddings - not Cohere. Reasoning: this is a
student hackathon budget, the corpus is small (a few thousand chunks for
India-only), and a legal-domain-general embedding model at this scale is not
your bottleneck - corpus correctness is.
"""

from sentence_transformers import SentenceTransformer

from app.config import settings

_model = None


def get_model() -> SentenceTransformer:
    global _model
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
    """Batch-encode -- sentence-transformers processes a batch as one matmul
    instead of N separate Python-level calls, which matters once ingestion
    is doing this a few thousand times (the original ingest.py called
    embed() once per chunk in a loop). Not benchmarked in this environment
    (no network access to download the model here) -- verify the speed
    difference yourself with `time python -m app.ingestion.ingest`.
    """
    if not texts:
        return []
    model = get_model()
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
    return vectors.tolist()
