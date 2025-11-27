# TARS v2 - A/B Testing & Versioning Strategy

## Overview

This document outlines the strategy for implementing A/B testing and agent versioning in TARS v2. The goal is to allow multiple versions of an agent to coexist, enabling safe evolution and performance comparison.

## Core Concepts

### 1. Agent Versioning

Every agent in the system will have a `Version` field.

- **Structure:** `Major.Minor.Patch` (Semantic Versioning)
- **Storage:** The `KernelContext` will map `AgentId` to a *list* or *map* of versions, or we use a composite key `{Id, Version}`.

### 2. Traffic Routing (The "Router")

A new component, `AgentRouter`, will sit between the `EventBus` and the agents.

- **Responsibility:** Decide which version of an agent receives a message.
- **Strategies:**
  - **Pinned:** A conversation/session is pinned to a specific version (e.g., `v1.0.0`).
  - **Canary:** 90% of new sessions go to `v1.0.0`, 10% go to `v1.1.0`.
  - **A/B Test:** 50/50 split between `vA` and `vB` to compare metrics.

### 3. Metrics & Evaluation

To make A/B testing useful, we need to measure success.

- **Metrics:**
  - Success rate (did the task complete?)
  - Token usage (cost)
  - Latency
  - User feedback (thumbs up/down)
- **Reporting:** The `Governance` module will aggregate these metrics per version.

## Implementation Plan

### Phase 1: Data Structure Updates

1. **Update `Agent` Record:** Add `Version: string` (or `SemVer` type).
2. **Update `KernelContext`:** Change `Agents: Map<AgentId, Agent>` to support multiple versions.
    - Option A: `Map<AgentId, Map<string, Agent>>` (Id -> Version -> Agent)
    - Option B: `Map<AgentId, Agent>` where `AgentId` includes the version? (Cleaner to keep Id stable).
    - **Decision:** Let's keep `AgentId` stable. The Kernel should store `ActiveAgents: Map<AgentId, Agent>` (the "primary" or "latest") and `AllVersions: Map<AgentId, List<Agent>>`.
    - *Refinement:* Actually, Erlang style: `AgentId` addresses the *process*. If we want A/B testing, we might have `AgentId_v1` and `AgentId_v2`. The *Router* maps a logical name ("Coder") to a specific `AgentId`.

### Phase 2: The Router

1. Create `Tars.Kernel.Router`.
2. Implement `selectAgent(logicalName: string, strategy: RoutingStrategy) -> AgentId`.

### Phase 3: Integration

1. Update `EventBus` to use the Router.
2. Update `Evolution` engine to deploy new versions as "Canary" first.

## Example Workflow

1. **Evolution:** The system evolves "CoderAgent" from v1 to v2.
2. **Deployment:** v2 is registered in the Kernel.
3. **Routing:** The Router is updated: "CoderAgent" -> { v1: 90%, v2: 10% }.
4. **Traffic:** New user requests are routed accordingly.
5. **Analysis:** After 100 runs, if v2 has higher success rate, Router updates to { v1: 0%, v2: 100% }.
