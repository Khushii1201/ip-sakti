import asyncio

import asyncpg

from app.config import settings
from app.llm.groq_client import generate_answer
from app.retrieval.db import search_chunks
from app.retrieval.embeddings import embed

ABSTAIN_MESSAGE = (
    "The corpus doesn't have a clear answer to this specific fact pattern. "
    "Please consult a registered patent agent or AYUSH-recognized IP cell."
)


async def answer_query(
    pool: asyncpg.Pool,
    query: str,
    jurisdiction: str,
    category: str | None = None,
) -> dict:
    if not pool:
        return {"answer": ABSTAIN_MESSAGE, "citations": [], "confidence": 0.0, "abstained": True}

    # embed() and generate_answer() are both synchronous, CPU/network-bound
    # calls being invoked directly inside an async handler -- that blocks the
    # whole event loop for their entire duration, so one slow request stalls
    # every other concurrent request on the same worker. Fine with one
    # person testing locally; a real risk with several judges hitting /query
    # around the same time on demo day. asyncio.to_thread hands each call to
    # a worker thread instead.
    query_embedding = await asyncio.to_thread(embed, query)
    rows = await search_chunks(pool, query_embedding, jurisdiction, category, settings.RETRIEVAL_TOP_K)

    if not rows:
        return {"answer": ABSTAIN_MESSAGE, "citations": [], "confidence": 0.0, "abstained": True}

    # Confidence = mean similarity of top-3 retrieved chunks. Crude, but honest:
    # it's tied to actual retrieval quality, not a made-up number the LLM reports
    # about itself. Tune CONFIDENCE_THRESHOLD against real queries before demo day.
    top_scores = [r["score"] for r in rows[:3]]
    confidence = sum(top_scores) / len(top_scores)

    if confidence < settings.CONFIDENCE_THRESHOLD:
        return {
            "answer": ABSTAIN_MESSAGE,
            "citations": [dict(r) for r in rows],
            "confidence": confidence,
            "abstained": True,
        }

    sources = [dict(r) for r in rows]
    answer_text = await asyncio.to_thread(generate_answer, query, sources)

    return {"answer": answer_text, "citations": sources, "confidence": confidence, "abstained": False}
