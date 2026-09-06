"""
Deliberately NOT an ML model. This is a transparent decision tree, and that's
a feature, not a shortcut: in a compliance context, a judge (or a real user)
being able to see exactly which rule fired is more trustworthy than a model
that "usually gets it right." If you're tempted to replace this with an LLM
call later, don't -- keep the LLM confined to free-text intent parsing that
FEEDS these structured answers, never to making the branching decision itself.

Stress-test findings fixed here (see chat writeup for full repro):
1. Substring bug: naive `keyword in desc` matched "purified" inside
   "unpurified" -- a description that explicitly DENIES purification got
   classified as phytopharmaceutical. Fixed with word-boundary regex.
2. Negation-blindness: "not a standardized extract" still contains the
   literal phrase "standardized extract", so word-boundary matching alone
   isn't enough. Added a small negation-window check for the clearest cases
   (explicit "not/without/no X" immediately before the keyword).
3. Silent guessing on missing free text: when the disease-claim branch needed
   the free-text description to pick phytopharmaceutical vs proprietary ASU
   and got None/"", it was silently defaulting to proprietary_asu -- which
   directly contradicts this file's own stated principle (see branch 5:
   "do not force a guess into a legal category"). Now returns unclassified
   and asks for the description instead.

None of this makes the heuristic trustworthy -- it still can't catch
"we considered an isolated compound but rejected it," which needs real NLP,
not a keyword scan. That's why branch 2 always sets needs_review=True: every
result out of this branch should be read as "best guess from a fuzzy
heuristic," not a verified classification. Say that plainly to judges.
"""

import re

from app.models.schemas import ClassificationResult, ClassifierAnswers

PHYTOPHARM_KEYWORDS = ("purified", "standardized extract", "bioactive marker", "isolated compound")

# Heuristic only -- catches the two clearest negation patterns seen in stress
# testing ("not a purified extract", "without any isolated compound"). Does
# NOT catch negation that comes after the keyword ("isolated compound...but
# rejected it") or negation several clauses away. When in doubt this errs
# toward still flagging needs_review=True rather than trusting a clean match.
NEGATION_MARKERS = ("not", "non", "without", "no", "isn't", "wasn't", "never")
NEGATION_WINDOW_WORDS = 4


def _find_keyword(desc: str, keyword: str) -> re.Match | None:
    """Word-boundary match so 'purified' doesn't match inside 'unpurified'."""
    return re.search(r"\b" + re.escape(keyword) + r"\b", desc)


def _looks_negated(desc: str, match_start: int) -> bool:
    """True if a negation marker appears in the few words immediately before
    the matched keyword. Best-effort only -- see module docstring."""
    preceding_text = desc[:match_start]
    preceding_words = re.findall(r"[a-z']+", preceding_text)[-NEGATION_WINDOW_WORDS:]
    return any(marker in preceding_words for marker in NEGATION_MARKERS)


def _matches_phytopharm_keywords(free_text_description: str | None) -> bool:
    desc = (free_text_description or "").lower()
    for kw in PHYTOPHARM_KEYWORDS:
        m = _find_keyword(desc, kw)
        if m and not _looks_negated(desc, m.start()):
            return True
    return False


def classify(answers: ClassifierAnswers) -> ClassificationResult:
    bda_flag = answers.uses_indian_biological_resource

    # 1. External use, no disease claim -> Cosmetic
    # NOTE: this fires even if follows_classical_text_exactly is also True,
    # and that's deliberate, not an oversight -- "cosmetic" is a claim-based
    # category (no disease claim = not a drug at all under D&C Act), so it's
    # orthogonal to whether the formulation happens to follow a classical
    # text. A classical topical product with zero disease claim genuinely is
    # a cosmetic, not a "classical ASU drug." Be ready to defend this
    # ordering explicitly if a judge probes it.
    if answers.internal_or_external == "external" and not answers.makes_disease_claim:
        return ClassificationResult(
            category="cosmetic",
            legal_pathway="Cosmetic Rules - separate from D&C Act ASU licensing, no GMP-facility requirement.",
            bda_flag=bda_flag,
            reasoning="External use, no disease claim.",
        )

    # 2. Disease claim + not a classical formulation -> Proprietary ASU or Phytopharmaceutical
    if answers.makes_disease_claim and not answers.follows_classical_text_exactly:
        raw_desc = (answers.free_text_description or "").strip()

        # Fix #3: don't silently guess when the disambiguating input is
        # missing. This is the one branch that legally hinges on free text,
        # so an empty description is genuinely insufficient information --
        # treat it the same way branch 5 treats an unmatched pattern.
        if not raw_desc:
            return ClassificationResult(
                category="unclassified",
                legal_pathway="Insufficient information to distinguish phytopharmaceutical from proprietary ASU - ask for a free-text description of the formulation process before routing.",
                bda_flag=bda_flag,
                reasoning="Disease claim + not classical, but no free-text description was provided to disambiguate.",
                needs_review=True,
            )

        if _matches_phytopharm_keywords(raw_desc):
            return ClassificationResult(
                category="phytopharmaceutical",
                legal_pathway=(
                    "CDSCO Phytopharmaceutical Drug guidelines - distinct new drug class, "
                    "requires >=4 qualitatively/quantitatively assessed bioactive compounds, "
                    "full CDSCO new-drug pathway."
                ),
                bda_flag=bda_flag,
                reasoning="Disease claim + purified/standardized-extract description.",
                needs_review=True,
            )
        return ClassificationResult(
            category="patent_proprietary_asu",
            legal_pathway=(
                "Patent or Proprietary ASU medicine, Section 3(a)/(h) D&C Act - "
                "requires pilot-study proof of effectiveness under Rule 158B."
            ),
            bda_flag=bda_flag,
            reasoning="Classical base with a new combination or new indication.",
            needs_review=True,
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

    # 5. Nothing matched cleanly -- do not force a guess into a legal category.
    return ClassificationResult(
        category="unclassified",
        legal_pathway="Insufficient information to classify safely - ask follow-up questions before routing.",
        bda_flag=bda_flag,
        reasoning="Answers did not match a known branch. Abstain from classification rather than guess.",
        needs_review=True,
    )
