-- Migration 004: Plans and Plan Events Tables
-- Event-sourced symbolic goals and hypotheses
-- "Plans evolve, assumptions are tracked, versions are remembered"

CREATE TABLE IF NOT EXISTS plans (
    id UUID PRIMARY KEY,
    goal TEXT NOT NULL,
    assumptions JSONB NOT NULL DEFAULT '[]',
    steps JSONB NOT NULL DEFAULT '[]',
    success_metrics JSONB NOT NULL DEFAULT '[]',
    risk_factors JSONB NOT NULL DEFAULT '[]',
    version INT NOT NULL DEFAULT 1,
    parent_version UUID NULL,
    status TEXT NOT NULL DEFAULT 'Draft',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    created_by TEXT NOT NULL,
    tags JSONB NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status);
CREATE INDEX IF NOT EXISTS idx_plans_created_by ON plans(created_by);
CREATE INDEX IF NOT EXISTS idx_plans_created_at ON plans(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_plans_goal ON plans USING gin(to_tsvector('english', goal));

CREATE TABLE IF NOT EXISTS plan_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id UUID NOT NULL,
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_plan_events_plan_id ON plan_events(plan_id);
CREATE INDEX IF NOT EXISTS idx_plan_events_timestamp ON plan_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_plan_events_type ON plan_events(event_type);
