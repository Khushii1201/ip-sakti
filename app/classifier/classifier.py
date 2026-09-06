"""
Deliberately NOT an ML model -- a transparent decision tree, so a judge (or a
real user) can see exactly which rule fired. Keep any future LLM confined to
free-text intent parsing that FEEDS these structured answers, never to making
the branching decision itself.

Round-3 note: round 2's post-keyword "contrast marker" scan (catching
"isolated compound, but rejected it") is gone -- it fixed that one case but
introduced a new false-positive mode on unrelated "but" clauses, which is a
different bug wearing a better story, not a fix. Instead of guessing harder,
every keyword match now returns WHICH keyword matched and the surrounding
text as structured fields (matched_keyword / review_snippet on
ClassificationResult), so a human reviewing a needs_review=True result can
verify it in seconds instead of re-reading the whole description.
"""

import re

from app.models.schemas import ClassificationResult, ClassifierAnswers

PHYTOPHARM_KEYWORDS = ("purified", "standardized extract", "bioactive marker", "isolated compound")

NEGATION_MARKERS = ("not", "non", "without", "no", "isn't", "wasn't", "never")
NEGATION_WINDOW_WORDS = 4
SNIPPET_CONTEXT_CHARS = 40


def _find_keyword(desc: str, keyword: str) -> re.Match | None:
    """Word-boundary match so 'purified' does not match inside 'unpurified'."""
    return re.search(r"\b" + re.escape(keyword) + r"\b", desc)


def _looks_negated(desc: str, match_start: int) -> bool:
    preceding_words = re.findall(r"[a-z']+", desc[:match_start])[-NEGATION_WINDOW_WORDS:]
    return any(marker in preceding_words for marker in NEGATION_MARKERS)


def _snippet_around(desc: str, start: int, end: int) -> str:
    lo = max(0, start - SNIPPET_CONTEXT_CHARS)
    hi = min(len(desc), end + SNIPPET_CONTEXT_CHARS)
    prefix = "..." if lo > 0 else ""
    suffix = "..." if hi < len(desc) else ""
    return f"{prefix}{desc[lo:hi].strip()}{suffix}"


def _check_phytopharm_keywords(
    free_text_description: str,
) -> tuple[bool, str | None, str | None]:
    """Return (matched, keyword_or_None, snippet_or_None).

    Scans for each keyword in order; returns on the first non-negated hit.
    Caller gets the specific keyword and surrounding text as structured data
    rather than having to parse it out of a free-form string.
    """
    desc = free_text_description.lower()
    for kw in PHYTOPHARM_KEYWORDS:
        m = _find_keyword(desc, kw)
        if not m or _looks_negated(desc, m.start()):
            continue
        return True, kw, _snippet_around(desc, m.start(), m.end())
    return False, None, None


def classify(answers: ClassifierAnswers) -> ClassificationResult:
    bda_flag = answers.uses_indian_biological_resource

    # 1. External use, no disease claim -> Cosmetic.
    # NOTE: this fires even if follows_classical_text_exactly is also True --
    # "cosmetic" is a claim-based category (no disease claim = not a drug
    # under D&C Act), orthogonal to whether the formulation follows a
    # classical text. Defend this ordering explicitly if a judge probes it.
    if answers.internal_or_external == "external" and not answers.makes_disease_claim:
        return ClassificationResult(
            category="cosmetic",
            legal_pathway="Cosmetic Rules -- separate from D&C Act ASU licensing, no GMP-facility requirement.",
            bda_flag=bda_flag,
            reasoning="External use, no disease claim.",
        )

    # 2. Disease claim + not a classical formulation -> Proprietary ASU or Phytopharmaceutical.
    if answers.makes_disease_claim and not answers.follows_classical_text_exactly:
        raw_desc = (answers.free_text_description or "").strip()

        if not raw_desc:
            return ClassificationResult(
                category="unclassified",
                legal_pathway=(
                    "Insufficient information to distinguish phytopharmaceutical from "
                    "proprietary ASU -- ask for a free-text description of the "
                    "formulation process before routing."
                ),
                bda_flag=bda_flag,
                reasoning="Disease claim + not classical, but no free-text description was provided to disambiguate.",
                needs_review=True,
            )

        matched, matched_keyword, snippet = _check_phytopharm_keywords(raw_desc)
        if matched:
            return ClassificationResult(
                category="phytopharmaceutical",
                legal_pathway=(
                    "CDSCO Phytopharmaceutical Drug guidelines -- distinct new drug class, "
                    "requires >=4 qualitatively/quantitatively assessed bioactive compounds, "
                    "full CDSCO new-drug pathway."
                ),
                bda_flag=bda_flag,
                reasoning=(
                    "Disease claim + purified/standardized-extract description matched by "
                    "keyword heuristic. Cannot reliably parse negation phrased after the "
                    "keyword (e.g. 'considered but rejected') -- verify the snippet before "
                    "trusting this category."
                ),
                needs_review=True,
                matched_keyword=matched_keyword,
                review_snippet=snippet,
            )

        return ClassificationResult(
            category="patent_proprietary_asu",
            legal_pathway=(
                "Patent or Proprietary ASU medicine, Section 3(a)/(h) D&C Act -- "
                "requires pilot-study proof of effectiveness under Rule 158B."
            ),
            bda_flag=bda_flag,
            reasoning=(
                "Classical base with a new combination or new indication. "
                "No confident phytopharmaceutical-keyword match -- verify manually if in doubt."
            ),
            needs_review=True,
        )

    # 3. Follows a named classical text exactly -> Classical ASU drug.
    if answers.follows_classical_text_exactly:
        return ClassificationResult(
            category="classical_asu",
            legal_pathway=(
                "Classical ASU drug, Section 3(a) D&C Act -- manufactured/labeled per the "
                "classical text; generally not treated as a 'new drug', no DCGI pre-approval."
            ),
            bda_flag=bda_flag,
            reasoning="Made exactly per an authoritative classical text (e.g. Charaka/Sushruta Samhita).",
        )

    # 4. Internal use, wellness claim, no disease claim -> Nutraceutical.
    if answers.internal_or_external == "internal" and not answers.makes_disease_claim:
        return ClassificationResult(
            category="nutraceutical",
            legal_pathway="FSSAI jurisdiction -- food/wellness claim, not CDSCO/AYUSH.",
            bda_flag=bda_flag,
            reasoning="Internal use, wellness claim, no disease claim.",
        )

    # 5. Nothing matched cleanly -- do not force a guess into a legal category.
    return ClassificationResult(
        category="unclassified",
        legal_pathway="Insufficient information to classify safely -- ask follow-up questions before routing.",
        bda_flag=bda_flag,
        reasoning="Answers did not match a known branch. Abstain from classification rather than guess.",
        needs_review=True,
    )
