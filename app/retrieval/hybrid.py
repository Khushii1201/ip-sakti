"""
Wires the previously-stubbed BM25 module into actual retrieval, fused with
dense (pgvector) results via reciprocal rank fusion - plus a second,
deterministic layer on top: exact-citation pinning.

IMPORTANT FINDING FROM TESTING (don't skip this): plain RRF turned out to be
weaker than expected at the corpus size this project actually has. RRF is
purely rank-based - it discards *how much* better a match is, only *where*
it ranked. Tested this directly: a chunk that BM25 scored 1.75 (a clean,
unambiguous "Section 3(d)" exact match) against two chunks BM25 scored 0.0,
fused with dense results, only moved up ONE rank position (from 3rd to 2nd),
not to 1st - because a mediocre dense-rank-0 match's aggregate rank-sum
still beat it. This isn't a k-tuning problem: swept k from 60 down to 2 and
the ordering never changed, because RRF's math structurally can't let a
rank-2/rank-0 combination beat a rank-0/rank-1 combination regardless of k.
If you only take one thing from this file, take that - "just fuse it" was
not sufficient for the exact use case (typing a section number verbatim)
this feature was built for.

The actual fix: for queries that contain an explicit "Section N" / "Rule N"
reference, don't rely on statistical fusion at all for that part - check
directly whether a chunk with that exact section_ref exists in scope, and if
so, pin it to the front, deterministically. This is consistent with how the
rest of this codebase already treats legally load-bearing decisions
(classifier.py's whole docstring is "a judge should see exactly which rule
fired", not "the model decided") - a citation lookup should work the same
way: exact match wins outright, not "gets a fusion score boost you'd have to
explain the math behind."

RRF fusion is kept for everything else - it's still a real (if modest)
improvement for queries that don't name an exact section, blending lexical
and semantic signal. Just don't oversell what it does on its own.
"""

import re

from app.retrieval.bm25 import build_bm25_index, bm25_search, reciprocal_rank_fusion
from app.retrieval.db import fetch_all_chunks_for_bm25

# {(jurisdiction, category_or_none): (bm25_index, chunk_rows)}
_bm25_cache: dict = {}

_QUERY_CITATION_PATTERN = re.compile(
    r"(Section\s+\d+[A-Za-z()]*|Rule\s+\d+[A-Za-z()]*)", re.IGNORECASE
)


def _normalize_ref(ref: str) -> str:
    return re.sub(r"\s+", " ", ref.strip().lower())


def _extract_query_citation(query: str) -> str | None:
    m = _QUERY_CITATION_PATTERN.search(query)
    return _normalize_ref(m.group(1)) if m else None


async def get_bm25_index(pool, jurisdiction: str, category: str | None):
    key = (jurisdiction, category)
    if key in _bm25_cache:
        return _bm25_cache[key]

    chunk_rows = await fetch_all_chunks_for_bm25(pool, jurisdiction, category)
    if not chunk_rows:
        _bm25_cache[key] = (None, [])
        return _bm25_cache[key]

    index = build_bm25_index(chunk_rows)
    _bm25_cache[key] = (index, chunk_rows)
    return _bm25_cache[key]


def clear_bm25_cache() -> int:
    """Call after re-ingesting into a live server - the BM25 index is built
    once and cached for the process lifetime, so it goes stale otherwise.
    Exposed via POST /admin/reindex-bm25 in main.py."""
    n = len(_bm25_cache)
    _bm25_cache.clear()
    return n


async def hybrid_search(pool, query: str, query_embedding: list[float], jurisdiction: str,
                         category: str | None, top_k: int, dense_search_fn):
    dense_rows = await dense_search_fn(pool, query_embedding, jurisdiction, category, top_k)
    dense_ranked = [dict(r) for r in dense_rows]

    bm25_index, chunk_rows = await get_bm25_index(pool, jurisdiction, category)
    if bm25_index is None or not dense_ranked:
        return dense_ranked

    pinned = None
    cited_ref = _extract_query_citation(query)
    if cited_ref:
        for row in chunk_rows:
            if _normalize_ref(row["section_ref"]) == cited_ref:
                pinned = row
                break

    sparse_ranked = bm25_search(bm25_index, chunk_rows, query, top_k=top_k)
    fused_scores = reciprocal_rank_fusion(dense_ranked, sparse_ranked)
    by_id = {r["id"]: r for r in dense_ranked}
    for row, _score in sparse_ranked:
        by_id.setdefault(row["id"], row)
    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    fused_rows = [by_id[cid] for cid in ranked_ids if cid in by_id]

    if pinned is not None:
        fused_rows = [r for r in fused_rows if r["id"] != pinned["id"]]
        fused_rows = [pinned] + fused_rows

    return fused_rows[:top_k]
