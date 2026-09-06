from typing import Literal, Optional

from pydantic import BaseModel

Jurisdiction = Literal["india", "international"]

FormulationCategory = Literal[
    "classical_asu",
    "patent_proprietary_asu",
    "phytopharmaceutical",
    "cosmetic",
    "nutraceutical",
    "unclassified",
]


class ClassifierAnswers(BaseModel):
    internal_or_external: Literal["internal", "external"]
    follows_classical_text_exactly: bool
    makes_disease_claim: bool
    uses_indian_biological_resource: bool
    free_text_description: Optional[str] = None


class ClassificationResult(BaseModel):
    category: FormulationCategory
    legal_pathway: str
    bda_flag: bool
    reasoning: str
    needs_review: bool = False


class Citation(BaseModel):
    act_name: str
    section_ref: str
    jurisdiction: Jurisdiction
    source_url: Optional[str] = None
    chunk_text: str
    # Optional, not required: hybrid retrieval can surface a chunk that only
    # matched via BM25 (exact-term match), which never computed a dense
    # cosine similarity at all. Making this required broke serialization the
    # moment such a row reached the API response - confirmed by testing
    # Citation(**row_without_score) directly, not just reasoned about.
    score: Optional[float] = None


class QueryRequest(BaseModel):
    query: str
    jurisdiction: Jurisdiction = "india"
    classification: Optional[ClassificationResult] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: float
    abstained: bool
    disclaimer: str = (
        "This is not legal advice. Consult a registered patent agent or "
        "AYUSH-recognized IP cell."
    )
