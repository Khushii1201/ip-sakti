# IP-SAKTI Sahayak - Backend (India-only slice, v0)

## Setup
1. Create a free Supabase project. In its SQL editor, run `schema.sql`.
2. Copy `.env.example` to `.env` and fill in `DATABASE_URL` (Supabase connection
   string, "Session mode" pooler recommended) and `GROQ_API_KEY` (free at
   console.groq.com).
3. `pip install -r requirements.txt`
4. Drop statute text into `app/ingestion/corpus/` (see "The real bottleneck" below),
   then `python -m app.ingestion.ingest`.
5. `uvicorn app.main:app --reload`
6. `GET /health`, `POST /classify`, `POST /query`.

## What's real vs stubbed (don't overstate this in the pitch)
- **Real:** classifier (fully deterministic, all 6 branches + BDA flag), dense
  retrieval with hard jurisdiction/category SQL filtering, confidence-gated
  abstention, audit logging, section-boundary chunker that fails loud on bad
  source text instead of silently degrading citations.
- **Stubbed:** BM25 + RRF fusion (`app/retrieval/bm25.py` exists but isn't wired
  into `rag.py` yet - dense-only for now). No reranker (Cohere rerank was cut
  deliberately - see decisions below). No multilingual layer yet. International
  jurisdiction mode has the schema/filter support but no corpus.
- **Not started:** free-text intent parsing to auto-fill `ClassifierAnswers`
  from a user's natural-language description (currently the frontend must ask
  the 4 structured questions directly).

Say this plainly to judges rather than rounding up - the doc you're working
from specifically flags inflated completion numbers as a credibility risk in
this PS category, more so than most.

## Key decisions and why (so you can defend them live)
- **Postgres/pgvector over Qdrant:** corpus is small enough that ANN speed
  isn't the bottleneck; SQL WHERE filters are more auditable under judge
  questioning than nested Qdrant filter dicts, and you already know Supabase.
- **Local embeddings (sentence-transformers) over Cohere:** no API cost, no
  external dependency to keep alive on demo day, and at this corpus size the
  quality gap won't be your problem - corpus correctness will be.
- **Classifier is a hard-coded decision tree, not an LLM call:** a legally
  load-bearing branch should be inspectable and reproducible. If a judge asks
  "why did it classify this as X," the honest answer should be a line of code,
  not "the model decided."
- **Chunker raises on malformed source text instead of falling back to token
  windows:** a half-cited section is worse than a missing one. This will
  surface corpus quality problems early, which is the point.

## The real bottleneck: corpus, not code
This backend is close to feature-complete for an India-only demo slice. What's
missing is the actual statute text in `app/ingestion/corpus/` - Patents Act
1970, D&C Act 1940 + Rules, Biological Diversity Act 2002. That's legal-research
work, not engineering work, and it's where a wrong citation costs you more than
a missing feature would. Assign this to whoever's doing domain research, and
have them fact-check every section reference against the actual gazette text,
not a secondary summary site.

## 7-day plan against this codebase
- **Day 1-2:** Corpus sourcing + cleanup (the bottleneck - start this in
  parallel with everything else, not after). Get `ingest.py` running end to end
  against at least one real statute.
- **Day 3:** Wire up `/classify` -> `/query` end to end from a minimal frontend
  or curl. Confirm the abstention path actually fires on a query the corpus
  doesn't cover - don't just assume it works.
- **Day 4:** Wire BM25 + RRF fusion into `rag.py` if dense-only retrieval is
  visibly missing exact-term queries (e.g. "Section 3(d)" typed verbatim).
  Skip this step if dense-only is already scoring well - don't add complexity
  you can't defend under questioning.
- **Day 5:** BDA/NBA flag surfacing in the API response (currently returned as
  a boolean on `ClassificationResult` - the frontend needs to actually render
  it as a hard stop, not a footnote).
- **Day 6:** Add the international-mode corpus (WIPO GRATK treaty, TRIPS
  27.3(b), Nagoya Protocol) if India-only is solid. Don't start this before
  India-only is bulletproof.
- **Day 7:** Freeze the backend. Stop touching code before evaluation - per the
  doc's own advice from past winners, a smaller working scope beats a bigger
  half-wired one.

## Where this will break if you don't fix it
- `CONFIDENCE_THRESHOLD=0.55` is a guess, not a tuned value - run real queries
  against your actual corpus and adjust before demo day.
- The classifier's free-text keyword match for phytopharmaceutical vs
  proprietary-ASU (`PHYTOPHARM_KEYWORDS` in `classifier.py`) is a crude
  heuristic. It will misclassify an edge case if a judge phrases their example
  cleverly - know this going in, don't get caught flat-footed.
