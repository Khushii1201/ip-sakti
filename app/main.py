import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.audit.logger import get_recent_queries, log_query
from app.classifier.classifier import classify
from app.models.schemas import ClassificationResult, ClassifierAnswers, QueryRequest, QueryResponse
from app.rag import answer_query
from app.retrieval.db import get_pool
from app.retrieval.embeddings import preload_model

logger = logging.getLogger(__name__)

pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    try:
        pool = await get_pool()
    except Exception as e:
        print(f"Warning: Database connection pool initialization skipped or failed: {e}")
        pool = None
    preload_model()
    yield
    if pool:
        await pool.close()


app = FastAPI(title="IP-SAKTI Sahayak Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify", response_model=ClassificationResult)
def classify_formulation(answers: ClassifierAnswers):
    """Runs before retrieval. The frontend should call this first, then pass the
    result's `category` into /query so retrieval is filtered before the RAG core
    even fires - matches the doc's Layer 2 -> Layer 1 ordering."""
    return classify(answers)


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    category = None
    if req.classification and req.classification.category != "unclassified":
        category = req.classification.category

    result = await answer_query(pool, req.query, req.jurisdiction, category)

    # Bug fix: audit logging is a side effect, not part of the contract with
    # the user -- if the insert fails (transient DB hiccup, pool exhausted),
    # the original code let that exception propagate and turn a perfectly
    # good, already-generated answer into a 500. Log the failure for
    # yourself; don't let it take down the response.
    try:
        await log_query(
            pool,
            req.query,
            req.classification.category if req.classification else None,
            req.jurisdiction,
            result["citations"],
            result["confidence"],
            result["abstained"],
        )
    except Exception:
        logger.exception("Failed to write audit_log entry for query=%r", req.query)

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        confidence=result["confidence"],
        abstained=result["abstained"],
    )


@app.get("/audit/recent")
async def audit_recent(limit: int = 20):
    limit = min(limit, 100)
    return await get_recent_queries(pool, limit)


@app.get("/health")
def health():
    return {"status": "ok"}
