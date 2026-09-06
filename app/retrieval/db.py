import asyncpg

from app.config import settings


async def get_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(settings.DATABASE_URL)


async def search_chunks(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    jurisdiction: str,
    category: str | None,
    top_k: int,
):
    """Jurisdiction (and optionally category) is a hard SQL WHERE filter, not a
    prompt instruction. This is the single decision the PS doc flags as most
    important - an India-mode query must be structurally unable to retrieve
    international-mode chunks, regardless of what the LLM does with them."""
    filters = ["jurisdiction = $2"]
    params: list = [str(query_embedding), jurisdiction]
    if category:
        filters.append("category = $3")
        params.append(category)

    where_clause = " AND ".join(filters)
    sql = f"""
        select c.id, c.chunk_text, c.section_ref, c.jurisdiction, c.category,
               s.act_name, s.source_url,
               1 - (c.embedding <=> $1::vector) as score
        from chunks c
        join statutes s on s.id = c.statute_id
        where {where_clause}
        order by c.embedding <=> $1::vector
        limit {top_k}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return rows
