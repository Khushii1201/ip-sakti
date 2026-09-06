"""
Local sentence-transformers embeddings - not Cohere. Reasoning: this is a
student hackathon budget, the corpus is small (a few thousand chunks for
India-only), and a legal-domain-general embedding model at this scale is not
your bottleneck - corpus correctness is. Don't spend your one scarce resource
(time, not compute) wiring up a paid embedding API for a marginal quality gain
you won't be able to prove out in 7 days anyway.

If retrieval quality turns out to be the actual weak point later (you'll know
because the confidence scores in rag.py stay low even on well-covered
questions), swap EMBEDDING_MODEL for a larger local model before reaching for
a paid API.
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
