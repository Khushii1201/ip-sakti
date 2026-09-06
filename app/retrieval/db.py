import logging

import asyncpg

from app.config import settings

logger = logging.getLogger(__name__)


async def get_pool() -> asyncpg.Pool:
    # Explicit pool sizing + command timeout instead of asyncpg's defaults --
    # cheap insurance against pool exhaustion or one runaway query hanging
    # the whole server during a demo. Tune via env vars if you need to.
    return await asyncpg.create_pool(
        settings.DATABASE_URL,
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        command_timeout=settings.DB_COMMAND_TIMEOUT_SECONDS,
    )


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
    next_param = 3
    if category:
        filters.append(f"category = ${next_param}")
        params.append(category)
        next_param += 1

    top_k_param = next_param
    params.append(top_k)

    where_clause = " AND ".join(filters)
    sql = f"""
        select c.id, c.chunk_text, c.section_ref, c.jurisdiction, c.category,
               s.act_name, s.source_url,
               1 - (c.embedding <=> $1::vector) as score
        from chunks c
        join statutes s on s.id = c.statute_id
        where {where_clause}
        order by c.embedding <=> $1::vector
        limit ${top_k_param}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return rows
