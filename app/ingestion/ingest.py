"""
Run with: python -m app.ingestion.ingest

Drop plain-text statute files into app/ingestion/corpus/ and list them in
CORPUS_MANIFEST below. This is the actual bottleneck of the whole build -
see README. Do not let this script's simplicity fool you into thinking corpus
prep is quick; sourcing and cleaning the statute text is the real work.
"""

import asyncio
from pathlib import Path

from app.ingestion.chunker import chunk_by_section
from app.retrieval.db import get_pool
from app.retrieval.embeddings import embed

# (filename in corpus/, act_name, jurisdiction, category, source_url)
CORPUS_MANIFEST = [
    ("patents_act_1970.txt", "Patents Act, 1970", "india", "patent_law", None),
    ("drugs_and_cosmetics_act_1940.txt", "Drugs & Cosmetics Act, 1940", "india", "asu_regulatory", None),
    ("biological_diversity_act_2002.txt", "Biological Diversity Act, 2002", "india", "biodiversity", None),
]


async def ingest_file(pool, filepath: Path, act_name: str, jurisdiction: str, category: str, source_url):
    full_text = filepath.read_text(encoding="utf-8")
    chunks = chunk_by_section(full_text, act_name)  # raises before we touch the DB if malformed

    async with pool.acquire() as conn:
        statute_id = await conn.fetchval(
            """insert into statutes (act_name, jurisdiction, category, source_url, full_text)
               values ($1, $2, $3, $4, $5) returning id""",
            act_name,
            jurisdiction,
            category,
            source_url,
            full_text,
        )
        for c in chunks:
            vec = str(embed(c["chunk_text"]))
            await conn.execute(
                """insert into chunks (statute_id, jurisdiction, category, section_ref, chunk_text, embedding)
                   values ($1, $2, $3, $4, $5, $6::vector)""",
                statute_id,
                jurisdiction,
                category,
                c["section_ref"],
                c["chunk_text"],
                vec,
            )
    print(f"Ingested {len(chunks)} chunks from {act_name}")


async def main():
    pool = await get_pool()
    corpus_dir = Path(__file__).parent / "corpus"
    for filename, act_name, jurisdiction, category, url in CORPUS_MANIFEST:
        fp = corpus_dir / filename
        if not fp.exists():
            print(f"MISSING: {fp} - drop the statute text here before running ingest.")
            continue
        await ingest_file(pool, fp, act_name, jurisdiction, category, url)


if __name__ == "__main__":
    asyncio.run(main())
