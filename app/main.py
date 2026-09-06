from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.audit.logger import log_query
from app.classifier.classifier import classify
from app.models.schemas import ClassificationResult, ClassifierAnswers, QueryRequest, QueryResponse
from app.rag import answer_query
from app.retrieval.db import get_pool

pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    try:
        pool = await get_pool()
    except Exception as e:
        print(f"Warning: Database connection pool initialization skipped or failed: {e}")
        pool = None
    yield
    if pool:
        await pool.close()


app = FastAPI(title="IP-SAKTI Sahayak Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
    category = req.classification.category if req.classification else None
    result = await answer_query(pool, req.query, req.jurisdiction, category)

    await log_query(
        pool,
        req.query,
        req.classification.category if req.classification else None,
        req.jurisdiction,
        result["citations"],
        result["confidence"],
        result["abstained"],
    )

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        confidence=result["confidence"],
        abstained=result["abstained"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}
