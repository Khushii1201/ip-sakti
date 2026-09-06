-- IP-SAKTI Sahayak - Postgres schema (Supabase, pgvector)
-- Run this once against your Supabase project's SQL editor.

create extension if not exists vector;
create extension if not exists "pgcrypto"; -- for gen_random_uuid()

create table if not exists statutes (
    id uuid primary key default gen_random_uuid(),
    act_name text not null,          -- e.g. 'Patents Act, 1970'
    jurisdiction text not null,      -- 'india' | 'international'
    category text not null,          -- 'patent_law' | 'asu_regulatory' | 'biodiversity' | 'gi_trademark' | 'nutraceutical'
    source_url text,
    full_text text not null,
    created_at timestamptz default now()
);

create table if not exists chunks (
    id uuid primary key default gen_random_uuid(),
    statute_id uuid references statutes(id) on delete cascade,
    jurisdiction text not null,
    category text not null,
    section_ref text not null,       -- e.g. 'Section 3(d)' or 'Rule 158B' - never blank, never a page range
    chunk_text text not null,
    embedding vector(384),           -- matches bge-small-en-v1.5 / e5-small dim - change if you swap models
    created_at timestamptz default now()
);

-- ANN index for dense search
create index if not exists chunks_embedding_idx on chunks
    using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- This is the index that actually matters for the pitch: jurisdiction/category
-- are WHERE-clause filters, not something the LLM is trusted to respect.
create index if not exists chunks_jurisdiction_category_idx on chunks (jurisdiction, category);

create table if not exists audit_log (
    id uuid primary key default gen_random_uuid(),
    query text not null,
    classification_branch text,
    jurisdiction text,
    sources_cited jsonb,
    confidence_score float,
    abstained boolean default false,
    created_at timestamptz default now()
);
