# 🤖 Agentic AI Interfaces: From Rigid Contracts to Soft Semantics

**Date:** 2025-11-27  
**Status:** Planning  
**Goal:** Define and implement a soft semantic interface layer for TARS v2 agents.  
**Research Basis:** `docs/__research/ChatGPT-Agentic AI Interfaces.md`

---

## Overview

This document outlines how TARS v2 will implement **Agentic AI Interfaces** - moving from rigid, deterministic contracts to soft contracts based on capabilities, protocols, and reputation.

## Core Principles

1. **Probabilistic:** Allow for confidence scores and error distributions
2. **Capability-driven:** Encode what the agent *can* do, not just what methods it has
3. **Failure-first:** Make partial success, degraded modes, and fallbacks part of the contract
4. **Evolvable:** Discoverable and adaptable at runtime

---

## Architecture: The Triad

### 1. Hard Shell (Compiler Enforced)

**Implemented in:** `Tars.Core/Domain.fs`, `Tars.Core/AgentWorkflow.fs`

```fsharp
type PartialFailure =
    | SubAgentTimeout of agentId: AgentId * taskId: Guid
    | ToolError of tool: string * error: string
    | LowConfidence of score: float * details: string
    | ProtocolViolation of message: string
    | ConstraintViolation of constraint: string

type ExecutionOutcome<'T> =
    | Success of payload: 'T
    | PartialSuccess of payload: 'T * warnings: PartialFailure list
    | Failure of errors: PartialFailure list
```

### 2. Soft Semantics (Schema Enforced)

**Implemented in:** `Tars.Core/Capabilities.fs`, `Tars.Cortex/CapabilityStore.fs`

- Capability taxonomies
- Natural-language descriptions
- I/O schemas (JSON Schema or FLUX DSL)
- Confidence scores

### 3. Empirical Layer (Observed Behavior)

**Implemented in:** `Tars.Cortex/ReputationStore.fs`, `Tars.Kernel/Router.fs`

- Historical success/failure rates
- Learned reputation models
- Embedding-based routing

---

## Implementation Plan

### Phase 1: Core Types (Hard Shell)

**Priority:** High  
**Effort:** 2-3 hours

- [ ] Define `PartialFailure` DU in `Tars.Core/Domain.fs`
- [ ] Define `ExecutionOutcome<'T>` DU
- [ ] Define `Capability` record type
- [ ] Define `CapabilityKind` DU
- [ ] Update `Agent` to include `Capabilities: Capability list`

### Phase 2: Agent Workflow CE

**Priority:** High  
**Effort:** 4-6 hours

- [ ] Create `Tars.Core/AgentWorkflow.fs`
- [ ] Implement `AgentWorkflow<'T>` type
- [ ] Implement `AgentBuilder` with `Bind`, `Return`, `ReturnFrom`
- [ ] Create `agent { }` computation expression
- [ ] Add helper functions: `succeed`, `fail`, `warnWith`

### Phase 3: Combinators

**Priority:** Medium  
**Effort:** 3-4 hours

- [ ] Implement `callAgentByCapability`
- [ ] Implement `withFallback`
- [ ] Implement `escalateIfLowConfidence`
- [ ] Implement `retryWithBackoff`
- [ ] Implement `aggregateResults`

### Phase 4: Capability System

**Priority:** Medium  
**Effort:** 4-5 hours

- [ ] Create `Tars.Core/Capabilities.fs`
- [ ] Define capability taxonomy
- [ ] Implement capability matching/scoring
- [ ] Add embedding-based similarity search
- [ ] Create `CapabilityStore` in `Tars.Cortex`

### Phase 5: Reputation & Routing

**Priority:** Low  
**Effort:** 5-6 hours

- [ ] Create `ReputationStore` in `Tars.Cortex`
- [ ] Track agent success/failure rates
- [ ] Implement confidence-weighted routing
- [ ] Add telemetry for agent performance

### Phase 6: Integration

**Priority:** High  
**Effort:** 3-4 hours

- [ ] Update `Graph.fs` to use `ExecutionOutcome`
- [ ] Update `Evolution/Engine.fs` to use agent workflows
- [ ] Update existing agents to declare capabilities
- [ ] Create migration guide for existing code

---

## Example Usage

### Simple Agent Workflow

```fsharp
let summarizeDocument (doc: string) : AgentWorkflow<string> =
    agent {
        // Check document length
        if doc.Length > 10000 then
            do! warnWith (LowConfidence(0.7, "Document is very long"))
        
        // Call summarization capability
        let! summary = callAgentByCapability CapabilityKind.Summarization doc
        
        // Validate result
        if summary.Length < 50 then
            return! fail (ToolError("summarizer", "Summary too short"))
        
        return summary
    }
```

### Delegation with Fallback

```fsharp
let enrichWithWebSearch (query: string) : AgentWorkflow<string> =
    agent {
        // Try primary web search
        let! webResults = 
            callAgentByCapability CapabilityKind.WebSearch query
            |> withFallback (agent { return "No web results available" })
        
        // Check confidence
        let confidence = 0.85
        let! validated = escalateIfLowConfidence 0.9 webResults confidence
        
        return validated
    }
```

---

## Benefits

1. **Type Safety:** Partial failures are values, not exceptions
2. **Composability:** Workflows compose naturally via CE
3. **Observability:** All failures and warnings are captured
4. **Testability:** Each combinator is independently testable
5. **Flexibility:** Runtime capability discovery and routing

---

## Next Steps

1. Review and approve this design
2. Implement Phase 1 (Core Types)
3. Create unit tests for `ExecutionOutcome` patterns
4. Implement Phase 2 (Agent Workflow CE)
5. Update roadmap with detailed timeline
