-- Migration 001: Initial Knowledge Ledger Schema
-- Creates the event log for belief lifecycle
-- "Evolution is logged, not forgotten"

CREATE TABLE IF NOT EXISTS knowledge_ledger (
    id UUID PRIMARY KEY,
    belief_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL,
    agent_id TEXT NOT NULL,
    run_id UUID NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ledger_belief_id ON knowledge_ledger(belief_id);
CREATE INDEX IF NOT EXISTS idx_ledger_timestamp ON knowledge_ledger(timestamp);
CREATE INDEX IF NOT EXISTS idx_ledger_agent_id ON knowledge_ledger(agent_id);
