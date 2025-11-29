# Phase 6.5: Agentic Interfaces - Implementation Summary

**Date:** 2025-11-27  
**Status:** Planning Phase  
**Parent Phase:** Phase 6: Cognitive Architecture  
**Priority:** Critical (Immediate)

## Overview

Phase 6.5 introduces **Soft Semantic Contracts** for agentic AI - moving beyond rigid C#/F# interfaces to probabilistic, capability-driven contracts that embrace partial failure as a first-class concept.

## Research Foundation

- **Primary Source:** `docs/__research/ChatGPT-Agentic AI Interfaces.md`
- **Architecture Doc:** `docs/2_Analysis/Architecture/agentic_interfaces.md`
- **Architecture Overview Reference:** Section 1.2 in `docs/2_Analysis/Architecture/00_Overview/tars_v2_architecture.md`

## The Triad Architecture

### 1. Hard Shell (Compiler Enforced)

- `ExecutionOutcome<'T>` discriminated union
- `PartialFailure` type system
- Agent workflow computation expression
- F# type guarantees

### 2. Soft Semantics (Schema Enforced)

- Capability taxonomies
- Natural-language descriptions
- Confidence scores & SLOs
- JSON Schema / FLUX contracts

### 3. Empirical Layer (Observed Behavior)

- Historical success/failure rates
- Reputation models
- Embedding-based routing
- Runtime metrics

## Implementation Phases

### Phase 6.5.1: Core Types (Hard Shell)

**Effort:** 2-3 hours  
**Priority:** Critical

**Tasks:**

- [x] Define `PartialFailure` DU in `Tars.Core/Domain.fs`
  - `SubAgentTimeout of agentId: AgentId * taskId: Guid`
  - `ToolError of tool: string * error: string`
  - `LowConfidence of score: float * details: string`
  - `ProtocolViolation of message: string`
  - `ConstraintViolation of constraint: string`

- [x] Define `ExecutionOutcome<'T>` DU
  - `Success of payload: 'T`
  - `PartialSuccess of payload: 'T * warnings: PartialFailure list`
  - `Failure of errors: PartialFailure list`

- [x] Define `Capability` record type

  ```fsharp
  type Capability = {
      Kind: CapabilityKind
      Description: string
      InputSchema: string option
      OutputSchema: string option
  }
  ```

- [x] Define `CapabilityKind` DU
  - `Summarization | WebSearch | CodeGeneration | DataAnalysis | Planning | TaskExecution | Custom of string`

- [x] Update `Agent` type
  - Add `Capabilities: Capability list` field

**Acceptance Criteria:**

- All types compile without errors
- Types are exported from `Tars.Core`
- Existing code still compiles (backward compatible where possible)

### Phase 6.5.2: Agent Workflow CE

**Effort:** 4-6 hours  
**Priority:** Critical

**Tasks:**

- [x] Create `Tars.Core/AgentWorkflow.fs`
- [x] Define `AgentContext` type

  ```fsharp
  type AgentContext = {
      Self: Agent
      Registry: IAgentRegistry
      Logger: string -> unit
      CancellationToken: CancellationToken
  }
  ```

- [x] Define `AgentWorkflow<'T>` type alias

  ```fsharp
  type AgentWorkflow<'T> = AgentContext -> Async<ExecutionOutcome<'T>>
  ```

- [x] Implement `AgentBuilder` class
  - `Return` member
  - `ReturnFrom` member
  - **`Bind` member** (Railway-oriented):
    - Accumulates warnings from PartialSuccess
    - Short-circuits on Failure
    - Preserves full error context

- [x] Create `agent { }` computation expression binding

- [x] Add helper functions:
  - `succeed : 'T -> AgentWorkflow<'T>`
  - `fail : PartialFailure list -> AgentWorkflow<'T>`
  - `warnWith : PartialFailure -> AgentWorkflow<unit>`

**Acceptance Criteria:**

- CE compiles and works in unit tests
- `Bind` correctly accumulates warnings
- Failures propagate correctly
- Workflow composition is linear and readable

### Phase 6.5.3: Combinators

**Effort:** 3-4 hours  
**Priority:** High

**Tasks:**

- [x] Implement `callAgentByCapability`

  ```fsharp
  val callAgentByCapability : 
      CapabilityKind -> TaskSpec -> AgentWorkflow<string>
  ```

- [x] Implement `withFallback`

  ```fsharp
  val withFallback : 
      primary: AgentWorkflow<'T> -> 
      backup: AgentWorkflow<'T> -> 
      AgentWorkflow<'T>
  ```

- [x] Implement `escalateIfLowConfidence`

  ```fsharp
  val escalateIfLowConfidence : 
      minConfidence: float -> 
      result: string -> 
      confidence: float -> 
      AgentWorkflow<string>
  ```

- [x] Implement `retryWithBackoff`

  ```fsharp
  val retryWithBackoff : 
      AgentWorkflow<'T> -> 
      maxRetries: int -> 
      AgentWorkflow<'T>
  ```

- [x] Implement `aggregateResults`

  ```fsharp
  val aggregateResults : 
      AgentWorkflow<'T> list -> 
      AgentWorkflow<'T list>
  ```

**Acceptance Criteria:**

- Each combinator has unit tests
- Fallback logic is correct
- Escalation works with partial success
- Retry respects budget limits

### Phase 6.5.3.5: Agentic Patterns (New)

**Effort:** 2-3 hours
**Priority:** High

**Tasks:**

- [x] Create `Tars.Core/Patterns.fs`
- [x] Implement **Chain of Thought** pattern
  - `chainOfThought : (string -> AgentWorkflow<string>) list -> string -> AgentWorkflow<string>`
- [x] Implement **ReAct** pattern
  - `reAct : maxSteps:int -> goal:string -> AgentWorkflow<string>`
- [x] Implement **Plan & Execute** pattern
  - `planAndExecute : AgentWorkflow<string list> -> (string -> AgentWorkflow<string>) -> AgentWorkflow<string list>`

### Phase 6.5.4: Capability System

**Effort:** 4-5 hours  
**Priority:** Medium

**Tasks:**

- [ ] Create `Tars.Core/Capabilities.fs`
- [ ] Define capability taxonomy (extend `CapabilityKind`)
- [ ] Implement capability matching/scoring algorithm
- [ ] Add embedding-based similarity search (using existing VectorStore)
- [ ] Create `CapabilityStore` in `Tars.Cortex`
  - Store agent capabilities
  - Query by similarity
  - Track capability usage metrics

### Phase 6.5.4: Capability System

**Effort:** 4-5 hours  
**Priority:** Medium

**Tasks:**

- [x] Create `Tars.Core/Capabilities.fs` (Merged into `Domain.fs`)
- [x] Define capability taxonomy (extend `CapabilityKind`)
- [x] Implement capability matching/scoring algorithm (Basic filtering in `KernelRegistry`)
- [ ] Add embedding-based similarity search (using existing VectorStore)
- [ ] Create `CapabilityStore` in `Tars.Cortex`
  - Store agent capabilities
  - Query by similarity
  - Track capability usage metrics

**Acceptance Criteria:**

- Agents can register capabilities
- Router can find agents by capability match
- Similarity search returns ranked results
- Metrics track capability usage

### Phase 6.5.5: Circuit Combinators (New)

**Effort:** 3-4 hours
**Priority:** High

**Research:** `docs/__research/ChatGPT-Circuit-based AI architecture.md`

**Tasks:**

- [x] Implement **Transformer** (Impedance Matching) combinators
- [x] Implement **Inductor** (Context Inertia) combinator
- [x] Implement **Diode** (Directed Flow) combinator
- [x] Implement **Grounding** (Reference Potential) combinator

**Acceptance Criteria:**

- Combinators correctly modify workflow behavior
- `stepDown`/`stepUp` handle type conversions gracefully
- `stabilize` dampens high-frequency context changes
- `grounded` integrates with Epistemic Governor (Phase 6.8)

### Phase 6.5.6: Integration

**Effort:** 3-4 hours  
**Priority:** High

**Tasks:**

- [x] Update `Graph.fs` to use `ExecutionOutcome`
  - Modify `step` function signature
  - Preserve backward compatibility where possible

- [x] Update `Evolution/Engine.fs` to use agent workflows
  - Wrap existing logic in `agent { }` (Implemented `GraphExecutor` wrapper)
  - Handle partial success in task execution

- [ ] Update existing agents to declare capabilities
  - Add capabilities to Curriculum Agent
  - Add capabilities to Executor Agent

- [ ] Create migration guide document
  - How to convert existing agents
  - Examples of workflow patterns
  - Common pitfalls

**Acceptance Criteria:**

- All existing tests pass
- Evolution loop works with new types
- Graph execution preserves warnings
- Migration guide is clear and complete

## Overall Acceptance Criteria

1. **Type Safety:**
   - All agent execution returns `ExecutionOutcome<'T>`
   - No unhandled exceptions in agent workflows

2. **Partial Success Handling:**
   - Warnings are accumulated, not lost
   - Full trace of what succeeded/failed

3. **Capability Routing:**
   - Agents can be selected by capability matching
   - Routing decisions are logged and traceable

4. **Workflow Composition:**
   - CE allows linear, readable workflows
   - Complex delegation logic is simple to express

5. **Testing:**
   - PartialSuccess correctly accumulates warnings
   - Fallback logic is correct
   - Escalation triggers appropriately

## Dependencies

- **Requires:** Phase 6.0 (Type Safety, Versioning)
- **Blocks:** None (can proceed in parallel with other Phase 6 work)
- **Integrates With:**
  - Phase 6.2 (Semantic Bus)
  - Phase 6.3 (Fan-out Limiter)
  - Phase 6.4 (Adaptive Reflection)

## Timeline Estimate

- **Total Effort:** 19-26 hours
- **Recommended Sprint:** 1 week (part-time) or 3-4 days (full-time)
- **Critical Path:** 6.5.1 → 6.5.2 → 6.5.6
- **Parallel Work:** 6.5.3, 6.5.4, and 6.5.5 can run in parallel with 6.5.6

## Success Metrics

- **Code Quality:** 80%+ test coverage on workflow logic
- **Performance:** No measurable overhead vs. existing implementation
- **Usability:** 50% reduction in boilerplate for complex agent delegation
- **Observability:** 100% traceability of partial failures

## References

1. Research: `docs/__research/ChatGPT-Agentic AI Interfaces.md`
2. Architecture: `docs/2_Analysis/Architecture/agentic_interfaces.md`
3. Roadmap: `docs/3_Roadmap/implementation_plan.md` (Phase 6.5)
