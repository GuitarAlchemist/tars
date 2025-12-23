# TARS v2 - Phase 9: Symbolic Knowledge & Free Skills Roadmap

**Date:** December 22, 2025  
**Status:** Planning  
**Source:** [ChatGPT-Claude skills free use.md](../conversations/ChatGPT-Claude%20skills%20free%20use.md)

---

## 🧠 Philosophical Foundation

> **LLMs as stochastic generators + Symbolic systems as memory, law, and self-control.**

This phase implements the core thesis of TARS:

| LLM-Only Problem | Symbolic Solution |
|------------------|-------------------|
| Amnesic across runs | Event-sourced ledger |
| No provenance | Every belief has a source |
| Plans "vibe-based" | Plans link to belief IDs |
| Can't explain changes | Version history + diffs |
| Contradictions hidden | Explicit contradiction detection |

**Reference:** [Architectural Vision](../../1_Vision/architectural_vision.md)

---

## 🎯 Executive Summary

This phase transforms TARS from an execution system into a **knowledge-accumulating, evolving intelligence** by implementing:

1. **Symbolic Knowledge Ledger** - Event-sourced belief storage with provenance
2. **Trusted Internet Ingestion** - Curated evidence streams (Wikipedia, arXiv, GitHub)
3. **Evolving Plans/Roadmaps** - Hypothesis-driven, evidence-based planning
4. **Free Local Skills Stack** - MCP + local LLMs + tool registry

---


## 📊 Free Local Stack (Already Implemented)

Based on research, TARS can achieve "Claude-level" capabilities entirely locally:

| Component | Model/Tool | Purpose | Status |
|-----------|------------|---------|--------|
| **Reasoning** | `qwen3:14b` | General thinking | ✅ Configured |
| **Math/Logic** | `deepseek-r1:14b` | Step-by-step reasoning | ✅ Routing enabled |
| **Coding** | `qwen2.5-coder:32b` | Code generation | 🔜 Add to routing |
| **Fast Agents** | `phi3:medium` | Sub-agents, validators | 🔜 Add to routing |
| **Embeddings** | `nomic-embed-text` | Semantic memory | ✅ Configured |
| **Inference** | llama.cpp | 75+ tok/s local | ✅ Installed |
| **Graph Memory** | Neo4j/Graphiti | Symbolic relations | ✅ Integrated |
| **Vector Memory** | Chroma | Semantic recall | ✅ Integrated |
| **Persistence** | PostgreSQL | Ledger, audit | 🔜 Phase 9.1 |

**Performance Achieved:**
- Prompt processing: ~162 tok/s
- Generation: ~75 tok/s
- GPU: RTX 5080 (compute 12.0)

---

## 🏗️ Phase 9.1: Symbolic Knowledge Ledger

**Priority:** Critical  
**Effort:** 1-2 weeks  

### Core Principle
> "The universe is cruel to systems that start with embeddings and vibes. Start with a ledger of explicit claims."

### 9.1.1 Atomic Belief Format

Define the fundamental knowledge unit:

```fsharp
type Belief = {
    Id: BeliefId
    Subject: EntityId
    Predicate: RelationType
    Object: EntityId
    Provenance: Provenance
    Confidence: float
    ValidFrom: DateTime
    InvalidAt: DateTime option
    Version: int
}

type Provenance = {
    Source: SourceType  // Run | Agent | External | User
    SourceUri: Uri option
    ExtractedBy: AgentId option
    ExtractedAt: DateTime
    ContentHash: string option
}
```

### 9.1.2 Event-Sourced Operations

Never mutate beliefs - append events:

```fsharp
type BeliefEvent =
    | Assert of Belief
    | Retract of BeliefId * reason: string
    | Weaken of BeliefId * newConfidence: float
    | Strengthen of BeliefId * newConfidence: float
    | Link of BeliefId * BeliefId * RelationType
    | SchemaEvolve of OntologyChange
```

### 9.1.3 Postgres Ledger Table

```sql
CREATE TABLE knowledge_ledger (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    belief_id UUID,
    payload JSONB NOT NULL,
    provenance JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    agent_id UUID,
    run_id UUID
);

CREATE INDEX idx_ledger_belief ON knowledge_ledger(belief_id);
CREATE INDEX idx_ledger_time ON knowledge_ledger(created_at);
```

### Tasks

- [ ] **9.1.1** Create `Tars.Knowledge` project
- [ ] **9.1.2** Define `Belief`, `BeliefEvent`, `Provenance` types
- [ ] **9.1.3** Implement `KnowledgeLedger` (Postgres-backed)
- [ ] **9.1.4** Add CLI: `tars know ingest <path>`
- [ ] **9.1.5** Parse `.trsx` outputs into assertions
- [ ] **9.1.6** Emit `knowledge_snapshot.trsx` each run

---

## 🌐 Phase 9.2: Internet Ingestion Pipeline

**Priority:** High  
**Effort:** 2-3 weeks

### Core Principle
> "The Internet never writes beliefs directly. It only produces evidence candidates."

### 9.2.1 Trusted Sources (Tier 1)

| Source | Entity Types | Relation Types |
|--------|--------------|----------------|
| **Wikipedia** | Concepts, Definitions | is_a, part_of, related_to |
| **arXiv** | Papers, Claims, Methods | claims, cites, improves |
| **GitHub** | Repos, APIs, Commits | depends_on, implements, forks |
| **IETF/RFCs** | Protocols, Specs | specifies, supersedes |

### 9.2.2 Pipeline Stages

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FETCH     │ => │   SEGMENT   │ => │   PROPOSE   │ => │   VERIFY    │
│   (dumb)    │    │ (mechanical)│    │ (LLM-gated) │    │ (symbolic)  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                  │                  │                  │
   Raw content      Sections/claims    Candidate beliefs   Ledger entry
   + content hash   + structure        + confidence        + provenance
```

### 9.2.3 Evidence Store

```fsharp
type EvidenceCandidate = {
    Id: Guid
    SourceUrl: Uri
    ContentHash: string
    ExtractedAt: DateTime
    Segments: Segment list
    ProposedAssertions: ProposedAssertion list
    Status: EvidenceStatus  // Pending | Verified | Rejected
}

type ProposedAssertion = {
    Subject: string
    Predicate: string
    Object: string
    SourceSection: string
    Confidence: float
    ExtractorAgent: AgentId
}
```

### 9.2.4 Verification Gate

Only promote to `ASSERT` if:
- Matches ontology schema
- Not contradictory (or explicitly marked)
- Supported by ≥ N sources OR trusted source
- Approved by Verifier Agent

### Tasks

- [ ] **9.2.1** Create `evidence_store` table (raw content)
- [ ] **9.2.2** Create `assertion_proposals` table
- [ ] **9.2.3** Implement Wikipedia fetcher
- [ ] **9.2.4** Implement segment extractor
- [ ] **9.2.5** Implement LLM-based assertion proposer
- [ ] **9.2.6** Implement Verifier Agent
- [ ] **9.2.7** Implement contradiction detection
- [ ] **9.2.8** Add CLI: `tars know fetch <url>`

---

## 📋 Phase 9.3: Evolving Plans

**Priority:** High  
**Effort:** 1-2 weeks

### Core Principle
> "Plans are hypotheses about future actions, not beliefs. Treat them as a different symbolic class."

### 9.3.1 Plan Object

```fsharp
type Plan = {
    Id: PlanId
    Goal: string
    Assumptions: BeliefId list  // References to beliefs
    Steps: PlanStep list
    SuccessMetrics: Metric list
    RiskFactors: Risk list
    Version: int
    ParentVersion: int option
    Status: PlanStatus  // Active | Paused | Invalidated | Completed
    CreatedAt: DateTime
    UpdatedAt: DateTime
}

type PlanStep = {
    Order: int
    Description: string
    EstimatedEffort: TimeSpan option
    Dependencies: PlanStepId list
    Status: StepStatus
}
```

### 9.3.2 Plan Events

```fsharp
type PlanEvent =
    | StepSucceeded of PlanId * stepOrder: int * evidence: string
    | StepFailed of PlanId * stepOrder: int * reason: string
    | AssumptionInvalidated of PlanId * BeliefId * reason: string
    | PlanForked of originalId: PlanId * newId: PlanId
    | ConfidenceUpdated of PlanId * newConfidence: float
```

### 9.3.3 The Loop

```
Beliefs → Hypotheses → Plans → Actions → Observations → Beliefs
```

Plans touch knowledge only through explicit belief references.

### Tasks

- [ ] **9.3.1** Create `Plan`, `PlanEvent` types
- [ ] **9.3.2** Create `plans` and `plan_events` tables
- [ ] **9.3.3** Implement `PlanManager` service
- [ ] **9.3.4** Link plan assumptions to belief IDs
- [ ] **9.3.5** Invalidate plans when beliefs retract
- [ ] **9.3.6** Add CLI: `tars plan new "<goal>"`
- [ ] **9.3.7** Add CLI: `tars plan status`

---

## 🔧 Phase 9.4: Skills & Tool Framework

**Priority:** Medium  
**Effort:** 1 week

### Core Principle
> "Skills are capabilities, not prompts. You already have everything Claude 'skills' pretend to offer."

### 9.4.1 Tool Registry (Already Implemented)

Current tools in `Tars.Tools`:

| Category | Tools | Status |
|----------|-------|--------|
| Standard | run_command, read_file, list_dir, http_get | ✅ |
| Git | git_status, git_diff, git_log | ✅ |
| Reasoning | lookup_docs, chain_of_thought | ✅ |
| Agent | delegate_task, list_agents, agent_status | ✅ |
| MCP | list_mcp_servers, configure_mcp | ✅ |
| **Skills** | list_skills, load_skill, create_skill | ✅ (New!) |

### 9.4.2 Agent Permission Model

```fsharp
type AgentPermissions = {
    AllowedTools: ToolId Set
    MaxCostPerCall: decimal<usd>
    MaxTokensPerCall: int<token>
    AllowExternalNetwork: bool
    AllowFileSystem: FileSystemAccess
    AllowShell: bool
}

type FileSystemAccess =
    | None
    | ReadOnly of paths: string list
    | ReadWrite of paths: string list
```

### 9.4.3 Skill-to-Container Mapping

| Container | Skill Type | Example |
|-----------|------------|---------|
| Graphiti | Belief inspection, ontology updates | `query_beliefs "X supports Y"` |
| Neo4j | Reasoning path search | `find_path "A" "B" max_hops=3` |
| Chroma | Memory recall | `recall "similar to last week"` |
| Postgres | Audit, metrics, replay | `audit run_id=abc` |
| Ollama | Cognition engine | (implicit) |

### Tasks

- [ ] **9.4.1** Define `AgentPermissions` type
- [ ] **9.4.2** Implement permission enforcement in tool execution
- [ ] **9.4.3** Add Graphiti query skill
- [ ] **9.4.4** Add Neo4j path search skill
- [ ] **9.4.5** Add Postgres audit skill
- [ ] **9.4.6** Document skill creation process

---

## 🎯 Minimal Viable Knowledge (MVK)

**Timeline:** 3 weekends

### Weekend 1: Ledger Foundation
1. Add `knowledge_ledger` table in Postgres
2. Add CLI: `tars know ingest <path>`
3. Parse `.trsx` + `agentic_trace.txt` into assertions
4. Emit `knowledge_snapshot.trsx` each run

### Weekend 2: Internet Ingestion
1. Ingest one Wikipedia page
2. Ingest one arXiv paper
3. Manually verify proposals
4. Add verifier agent

### Weekend 3: Evolving Plans
1. Introduce `plan.trsx` format
2. Tie plan assumptions to belief IDs
3. Invalidate plans when beliefs retract

**Success Criteria:**
- TARS accumulates symbolic knowledge from its own runs
- TARS can ingest external evidence with provenance
- Plans evolve based on evidence, not vibes

---

## 📊 Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                          TARS v2 Knowledge Stack                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       INTERFACE LAYER                            │    │
│  │  CLI: tars know | tars plan | tars run | tars evolve            │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                    │                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       COGNITIVE LAYER                            │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │ Planner      │  │ Verifier     │  │ Extractor    │           │    │
│  │  │ Agent        │  │ Agent        │  │ Agent        │           │    │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │    │
│  └─────────┼─────────────────┼─────────────────┼───────────────────┘    │
│            │                 │                 │                         │
│  ┌─────────▼─────────────────▼─────────────────▼───────────────────┐    │
│  │                       KNOWLEDGE LAYER                            │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │ Knowledge    │  │ Evidence     │  │ Plan         │           │    │
│  │  │ Ledger       │◀─│ Store        │  │ Manager      │           │    │
│  │  │ (Postgres)   │  │ (Postgres)   │  │ (Postgres)   │           │    │
│  │  └──────┬───────┘  └──────────────┘  └──────┬───────┘           │    │
│  └─────────┼────────────────────────────────────┼───────────────────┘    │
│            │                                    │                         │
│  ┌─────────▼────────────────────────────────────▼───────────────────┐    │
│  │                       MEMORY LAYER                               │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │ Neo4j        │  │ Chroma       │  │ Graphiti     │           │    │
│  │  │ (Graph)      │  │ (Vectors)    │  │ (Temporal)   │           │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       INFERENCE LAYER                            │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │    │
│  │  │ llama.cpp    │  │ Ollama       │  │ Cloud APIs   │           │    │
│  │  │ (Fast)       │  │ (Flexible)   │  │ (Fallback)   │           │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 References

- [ChatGPT-Claude skills free use.md](../conversations/ChatGPT-Claude%20skills%20free%20use.md) - Source conversation
- [Model Context Protocol](https://github.com/modelcontextprotocol) - Tool protocol
- [Graphiti](https://github.com/getzep/graphiti) - Temporal knowledge graphs
- [Ollama](https://ollama.ai/) - Local model serving
- [llama.cpp](https://github.com/ggml-org/llama.cpp) - Fast inference

---

## ✅ Success Criteria

At the end of Phase 9, TARS should:

1. **Accumulate knowledge** from its own runs (symbolic, versioned, auditable)
2. **Ingest external evidence** with full provenance (Wikipedia, arXiv, GitHub)
3. **Maintain evolving plans** that update based on evidence
4. **Answer provenance questions**: "Who asserted this? When? From what evidence?"
5. **Self-correct** when beliefs are retracted

This transforms TARS from a "project" into a "system".
