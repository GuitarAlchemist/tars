-- Migration 003: Evidence and Assertions Tables
-- Internet ingestion pipeline: candidates → assertions → beliefs
-- "Evidence must be verified before becoming knowledge"

CREATE TABLE IF NOT EXISTS evidence_candidates (
    id UUID PRIMARY KEY,
    source_uri TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    raw_content TEXT NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL, -- 'Pending', 'Verified', 'Rejected', 'Conflicting'
    segments JSONB NOT NULL DEFAULT '[]',
    metadata JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_evidence_status ON evidence_candidates(status);
CREATE INDEX IF NOT EXISTS idx_evidence_fetched_at ON evidence_candidates(fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_content_hash ON evidence_candidates(content_hash);

CREATE TABLE IF NOT EXISTS proposed_assertions (
    id UUID PRIMARY KEY,
    evidence_id UUID NULL REFERENCES evidence_candidates(id),
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    source_section TEXT NULL,
    extractor_agent TEXT NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_assertions_evidence_id ON proposed_assertions(evidence_id);
CREATE INDEX IF NOT EXISTS idx_assertions_confidence ON proposed_assertions(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_assertions_extracted_at ON proposed_assertions(extracted_at DESC);
