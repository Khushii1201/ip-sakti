from groq import Groq

from app.config import settings

SYSTEM_PROMPT = """You are IP-SAKTI Sahayak, a legal-regulatory assistant for Ayurveda IP and \
regulatory questions in India. You must ONLY answer using the provided source excerpts. Every \
factual claim must cite the exact section/rule reference given in the sources. If the sources do \
not clearly answer the question, say so explicitly and recommend consulting a registered patent \
agent or AYUSH-recognized IP cell. Never invent a section number, and never state a section number \
that is not present verbatim in the provided sources."""


def get_client() -> Groq:
    return Groq(api_key=settings.GROQ_API_KEY)


def generate_answer(query: str, sources: list[dict]) -> str:
    context = "\n\n".join(
        f"[{s['act_name']} - {s['section_ref']}]\n{s['chunk_text']}" for s in sources
    )
    client = get_client()
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content
