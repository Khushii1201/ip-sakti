import logging

import asyncpg

from app.config import settings

logger = logging.getLogger(__name__)


async def get_pool() -> asyncpg.Pool:
    async def _init_connection(conn):
        await conn.execute(f"SET ivfflat.probes = {settings.PGVECTOR_PROBES}")

    return await asyncpg.create_pool(
        settings.DATABASE_URL,
        min_size=settings.DB_POOL_MIN_SIZE,
        max_size=settings.DB_POOL_MAX_SIZE,
        command_timeout=settings.DB_COMMAND_TIMEOUT_SECONDS,
        init=_init_connection,
    )


async def search_chunks(
    pool: asyncpg.Pool,
    query_embedding: list[float],
    jurisdiction: str,
    category: str | None,
    top_k: int,
):
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


async def fetch_all_chunks_for_bm25(pool: asyncpg.Pool, jurisdiction: str, category: str | None):
    """Pulls every chunk in scope (same jurisdiction/category filter as
    search_chunks) so hybrid.py builds a BM25 index over exactly the set
    dense search is allowed to return from - never a broader corpus than
    the hard filter permits."""
    filters = ["jurisdiction = $1"]
    params: list = [jurisdiction]
    if category:
        filters.append("category = $2")
        params.append(category)

    where_clause = " AND ".join(filters)
    sql = f"""
        select c.id, c.chunk_text, c.section_ref, c.jurisdiction, c.category,
               s.act_name, s.source_url
        from chunks c
        join statutes s on s.id = c.statute_id
        where {where_clause}
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [dict(r) for r in rows]
