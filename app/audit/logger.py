import json

import asyncpg


async def log_query(
    pool: asyncpg.Pool,
    query: str,
    classification_branch: str | None,
    jurisdiction: str,
    sources_cited: list,
    confidence: float,
    abstained: bool,
):
    if not pool:
        return
    async with pool.acquire() as conn:
        await conn.execute(
            """insert into audit_log
               (query, classification_branch, jurisdiction, sources_cited, confidence_score, abstained)
               values ($1, $2, $3, $4, $5, $6)""",
            query,
            classification_branch,
            jurisdiction,
            json.dumps(sources_cited, default=str),
            confidence,
            abstained,
        )
