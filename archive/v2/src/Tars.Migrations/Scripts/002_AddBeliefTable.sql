-- Migration 002: Beliefs Snapshot Table
-- Materialized view of current belief state for fast queries
-- "Symbols are earned, not assumed"

CREATE TABLE IF NOT EXISTS beliefs (
    id UUID PRIMARY KEY,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    provenance_source TEXT NOT NULL,
    provenance_agent TEXT NOT NULL,
    provenance_run_id UUID NULL,
    provenance_confidence DOUBLE PRECISION NOT NULL,
    provenance_timestamp TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_beliefs_subject ON beliefs(subject);
CREATE INDEX IF NOT EXISTS idx_beliefs_predicate ON beliefs(predicate);
CREATE INDEX IF NOT EXISTS idx_beliefs_object ON beliefs(object);
CREATE INDEX IF NOT EXISTS idx_beliefs_created_by ON beliefs(created_by);
CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON beliefs(confidence DESC);
