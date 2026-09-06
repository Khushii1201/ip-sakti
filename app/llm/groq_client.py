import logging

from groq import Groq

from app.config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are IP-SAKTI Sahayak, a legal-regulatory assistant for Ayurveda IP and \
regulatory questions in India. You must ONLY answer using the provided source excerpts. Every \
factual claim must cite the exact section/rule reference given in the sources. If the sources do \
not clearly answer the question, say so explicitly and recommend consulting a registered patent \
agent or AYUSH-recognized IP cell. Never invent a section number, and never state a section number \
that is not present verbatim in the provided sources."""

LLM_FAILURE_MESSAGE = (
    "Retrieval found relevant sources, but the answer-generation step failed "
    "(the model backend didn't respond). The citations below are still real "
    "and safe to read directly; please retry the query, or consult a "
    "registered patent agent or AYUSH-recognized IP cell in the meantime."
)


def get_client() -> Groq:
    return Groq(api_key=settings.GROQ_API_KEY)


def generate_answer(query: str, sources: list[dict]) -> str:
    context = "\n\n".join(
        f"[{s['act_name']} - {s['section_ref']}]\n{s['chunk_text']}" for s in sources
    )
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=0.1,
        )
    except Exception:
        logger.exception("Groq completion failed for model=%s", MODEL_NAME)
        return LLM_FAILURE_MESSAGE

    answer = resp.choices[0].message.content
    if not answer or not answer.strip():
        logger.warning("Groq returned an empty completion for model=%s", MODEL_NAME)
        return LLM_FAILURE_MESSAGE
    return answer
