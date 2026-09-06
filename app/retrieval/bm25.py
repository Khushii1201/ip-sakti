"""
NOT wired into rag.py yet - see README "What's real vs stubbed". Dense-only
retrieval ships first because it's the smaller, provably-correct slice; BM25+RRF
fusion is Day 3-4 work once the corpus is large enough that dense-only misses
exact-term legal queries (e.g. someone typing "Section 3(d)" verbatim, where
sparse search should dominate over semantic similarity).
"""

from rank_bm25 import BM25Okapi


def build_bm25_index(chunk_rows: list[dict]):
    corpus = [row["chunk_text"].split() for row in chunk_rows]
    return BM25Okapi(corpus)


def bm25_search(bm25: BM25Okapi, chunk_rows: list[dict], query: str, top_k: int):
    scores = bm25.get_scores(query.split())
    ranked = sorted(zip(chunk_rows, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def reciprocal_rank_fusion(dense_ranked: list[dict], sparse_ranked: list[tuple[dict, float]], k: int = 60):
    """dense_ranked: rows in descending relevance order (already sorted by db.py).
    sparse_ranked: (row, bm25_score) pairs, descending. Returns {chunk_id: fused_score}."""
    scores: dict = {}
    for rank, item in enumerate(dense_ranked):
        scores[item["id"]] = scores.get(item["id"], 0) + 1 / (k + rank + 1)
    for rank, (item, _score) in enumerate(sparse_ranked):
        scores[item["id"]] = scores.get(item["id"], 0) + 1 / (k + rank + 1)
    return scores
