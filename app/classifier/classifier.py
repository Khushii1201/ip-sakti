"""
Deliberately NOT an ML model. This is a transparent decision tree, and that's
a feature, not a shortcut: in a compliance context, a judge (or a real user)
being able to see exactly which rule fired is more trustworthy than a model
that "usually gets it right." If you're tempted to replace this with an LLM
call later, don't - keep the LLM confined to free-text intent parsing that
FEEDS these structured answers, never to making the branching decision itself.
"""

from app.models.schemas import ClassificationResult, ClassifierAnswers

PHYTOPHARM_KEYWORDS = ("purified", "standardized extract", "bioactive marker", "isolated compound")


def classify(answers: ClassifierAnswers) -> ClassificationResult:
    bda_flag = answers.uses_indian_biological_resource

    # 1. External use, no disease claim -> Cosmetic
    if answers.internal_or_external == "external" and not answers.makes_disease_claim:
        return ClassificationResult(
            category="cosmetic",
            legal_pathway="Cosmetic Rules - separate from D&C Act ASU licensing, no GMP-facility requirement.",
            bda_flag=bda_flag,
            reasoning="External use, no disease claim.",
        )

    # 2. Disease claim + not a classical formulation -> Proprietary ASU or Phytopharmaceutical
    if answers.makes_disease_claim and not answers.follows_classical_text_exactly:
        desc = (answers.free_text_description or "").lower()
        if any(kw in desc for kw in PHYTOPHARM_KEYWORDS):
            return ClassificationResult(
                category="phytopharmaceutical",
                legal_pathway=(
                    "CDSCO Phytopharmaceutical Drug guidelines - distinct new drug class, "
                    "requires >=4 qualitatively/quantitatively assessed bioactive compounds, "
                    "full CDSCO new-drug pathway."
                ),
                bda_flag=bda_flag,
                reasoning="Disease claim + purified/standardized-extract description.",
            )
        return ClassificationResult(
            category="patent_proprietary_asu",
            legal_pathway=(
                "Patent or Proprietary ASU medicine, Section 3(a)/(h) D&C Act - "
                "requires pilot-study proof of effectiveness under Rule 158B."
            ),
            bda_flag=bda_flag,
            reasoning="Classical base with a new combination or new indication.",
        )

    # 3. Follows a named classical text exactly -> Classical ASU drug
    if answers.follows_classical_text_exactly:
        return ClassificationResult(
            category="classical_asu",
            legal_pathway=(
                "Classical ASU drug, Section 3(a) D&C Act - manufactured/labeled per the "
                "classical text; generally not treated as a 'new drug', no DCGI pre-approval."
            ),
            bda_flag=bda_flag,
            reasoning="Made exactly per an authoritative classical text (e.g. Charaka/Sushruta Samhita).",
        )

    # 4. Internal use, wellness claim, no disease claim -> Nutraceutical
    if answers.internal_or_external == "internal" and not answers.makes_disease_claim:
        return ClassificationResult(
            category="nutraceutical",
            legal_pathway="FSSAI jurisdiction - food/wellness claim, not CDSCO/AYUSH.",
            bda_flag=bda_flag,
            reasoning="Internal use, wellness claim, no disease claim.",
        )

    # 5. Nothing matched cleanly - do not force a guess into a legal category.
    return ClassificationResult(
        category="unclassified",
        legal_pathway="Insufficient information to classify safely - ask follow-up questions before routing.",
        bda_flag=bda_flag,
        reasoning="Answers did not match a known branch. Abstain from classification rather than guess.",
    )
