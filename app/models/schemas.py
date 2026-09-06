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
    """These map 1:1 to the four clarifying questions in the doc's Layer 2 table.
    Deliberately structured, not free text - a legally load-bearing branch should
    not depend on an LLM correctly parsing intent under time pressure."""

    internal_or_external: Literal["internal", "external"]
    follows_classical_text_exactly: bool
    makes_disease_claim: bool
    uses_indian_biological_resource: bool
    # Optional: used only to distinguish phytopharmaceutical from proprietary ASU
    # when a disease claim is present. Free text here, deliberately narrow-scoped.
    free_text_description: Optional[str] = None


class ClassificationResult(BaseModel):
    category: FormulationCategory
    legal_pathway: str
    bda_flag: bool  # true if BDA Section 6 / NBA clearance applies - cross-cutting, not a category
    reasoning: str
    # True whenever this result came out of the free-text-keyword heuristic
    # branch (phytopharmaceutical vs proprietary ASU) or the no-match/
    # insufficient-info branches. The frontend should render this as "flagged
    # for human/legal review", not silently treat it the same as a clean
    # deterministic branch (cosmetic, classical_asu, nutraceutical). Stress
    # testing showed the keyword heuristic can't reliably handle negation or
    # ambiguous phrasing, so its outputs need a visible confidence signal,
    # not just a black-box category string.
    needs_review: bool = False


class Citation(BaseModel):
    act_name: str
    section_ref: str
    jurisdiction: Jurisdiction
    source_url: Optional[str] = None
    chunk_text: str
    score: float


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
