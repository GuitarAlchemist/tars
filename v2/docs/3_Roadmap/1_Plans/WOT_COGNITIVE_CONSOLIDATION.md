# 🧠 WoT Cognitive Consolidation

**Status**: ✅ **VALIDATED & REFINED**  
**Date**: 2024-12-31  
**Purpose**: Unify TARS's fragmented cognitive patterns under a single Workflow-of-Thought execution spine  

---

## 🎯 Problem Statement

### Current Fragmentation

TARS currently has **5 parallel reasoning patterns** that operate independently:

| Pattern | File | Purpose | Integration |
|---------|------|---------|-------------|
| `chainOfThought` | Patterns.fs | Sequential reasoning | Standalone |
| `reAct` | Patterns.fs | Tool-augmented reasoning | Standalone |
| `planAndExecute` | Patterns.fs | Goal decomposition | Standalone |
| `graphOfThoughts` | Patterns.fs | Branching exploration | Standalone |
| `treeOfThoughts` | Patterns.fs | Beam search reasoning | Standalone |
| `workflowOfThought` | Patterns.fs | Traced execution | Standalone |

**Issues**:
1. **Duplication**: Each pattern re-implements logging, context building, and error handling
2. **No Memory**: Patterns don't learn from previous executions
3. **No Tracing**: Execution traces are ephemeral, not queryable
4. **No Cross-Pattern Knowledge**: GoT discoveries don't inform ReAct actions
5. **No Metrics Unification**: CognitiveBenchmarker only evaluates GoT

---

## 🏗️ Proposed Architecture

### WoT as the Universal Execution Spine

**Strategic Verdict**: This architecture transforms TARS from a "collection of smart patterns" into a **governed cognitive runtime**.

```
┌───────────────────────────────────────────────────────────────────┐
│                    TARS COGNITIVE STACK                           │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              USER / AGENT REQUEST                            │ │
│  └────────────────────────┬────────────────────────────────────┘ │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              PATTERN SELECTOR (Deterministic v1)            │ │
│  │   - Rules-based: Goal Size, Tools, History                  │ │
│  │   - NO "smart AI" selection yet (keep it auditable)         │ │
│  └────────────────────────┬────────────────────────────────────┘ │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              PATTERN COMPILER (The Law Layer)               │ │
│  │   compiles(CoT | ReAct) -> WoT (Canonized Graph)            │ │
│  └────────────────────────┬────────────────────────────────────┘ │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              WOT EXECUTION ENGINE (Universal)               │ │
│  │   - Executes minimal WoTNodes (Reason, Tool, Check)         │ │
│  │   - Produces Canonical Golden Traces                        │ │
│  │   - Guaranteed Determinism via Step IDs                     │ │
│  └────────────────────────┬────────────────────────────────────┘ │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              COGNITIVE STATE MANAGER                         │ │
│  │   - Variable d'état global (Mode, Entropy, Branching)       │ │
│  │   - Observable & Serializable (Don't overload it!)          │ │
│  └────────────────────────┬────────────────────────────────────┘ │
│                           ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              KNOWLEDGE GRAPH (Cold Start)                   │ │
│  │   - Phase 1: Append-only Triples (Facts)                    │ │
│  │   - Phase 2: Provenance (Why/When)                          │ │
│  │   - Phase 3: Inference                                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Core Design Principles

### 1. Pattern → WoT Compilation (The "Law Layer")

This is the most critical constraint: **No pattern executes directly.** All patterns MUST compile to a WoT graph first. This ensures:
- **Auditability**: We can inspect the plan before running it.
- **Diffability**: We can mathematically compare two reasoning traces.
- **Optimization**: We can rewrite the graph (e.g., prune dead branches) before execution.

```fsharp
// Before: Opaque execution
let reAct llm tools goal = ... // Runs internally

// After: Transparent Compilation
type WoTPlan = { Nodes: WoTNode list; Edges: WoTEdge list }

let reActCompiler = 
    PatternCompiler.compile "ReAct" (fun ctx -> 
        chain [
           Node.Reason("Plan", ctx.Goal)
           Node.Loop(
               Node.Tool("Action"),
               Node.Observe("Result")
           )
           Node.Reason("Final Answer")
        ])
```

### 2. Minimalist Node Types (Avoid the "God-Union")

We avoid a massive discriminated union that tries to do everything. Instead, we use a minimal set of `Kinds` and a flexible payload.

```fsharp
type WoTNodeKind =
    | Reason    // LLM Generation
    | Tool      // External Action
    | Validate  // Symbolic/Logic Check
    | Memory    // Read/Write State
    | Control   // Branch/Loop/Join

type WoTNode =
    { Id: Guid
      Kind: WoTNodeKind
      Payload: JsonElement // Versioned content (Prompt, ToolArgs, etc.)
      Metadata: NodeMetadata }
```

### 3. Cognitive State as Global Variable

The `CognitiveState` is the system's heartbeat. It must be **observable**, **serializable**, and **comparable**.

```fsharp
type CognitiveState =
    { Mode: CognitiveMode           // Exploratory | Convergent
      Entropy: float                 // 0.0 (Order) -> 1.0 (Chaos)
      BranchingFactor: float         // Metric of complexity
      ActivePattern: string          // "ReAct", "CoT"
      StepCount: int
      LastTransition: DateTime }
```

---

## 📋 Implementation Roadmap

### Phase A: WoT Core & Compilation (The "Spine")

**Goal**: Validate that WoT is a universal runtime by migrating ONE pattern.

**Tasks**:
1. [x] Define Minimal `WoTNode` types (Reason, Tool, Validate, Control).
2. [ ] **Action**: Create `PatternCompiler` module.
3. [ ] **Action**: Compile **ONE** existing pattern (e.g., `chainOfThought`) to `WoTPlan`.
4. [ ] **Action**: Execute this plan via `WoTExecutor`.
5. [ ] **Verify**: Ensure `Trace` -> `Golden` -> `Diff` works.

**Deliverable**:
`tars compile cot "goal"` -> outputs JSON Graph.
`tars run cot "goal"` -> executes via WoT.

### Phase B: Deterministic Pattern Selection

**Goal**: Smart routing *without* magic.

**Tasks**:
1. [ ] Implement `PatternSelector` (Rules Engine).
   - *Rule 1*: If Tools are present -> Force `ReAct` (or `WoT` with tools).
   - *Rule 2*: If Goal length > 100 chars -> `ChainOfThought`.
   - *Rule 3*: If `Entropy` > 0.8 (Confusion) -> Switch to `GraphOfThoughts` (Exploration).
2. [ ] Avoid using an LLM for selection in v1. Keep it fast and predictable.

### Phase C: Knowledge Graph (Cold Layout)

**Goal**: Durable memory without over-engineering.

**Tasks**:
1. [ ] **Stage 1 (Append-Only)**: Write execution traces as raw triples (`Run X -> Contains -> Step Y`).
2. [ ] **Stage 2 (Provenance)**: Add `(Fact Z) -> GeneratedBy -> (Step Y)`.
3. [ ] **Stage 3 (Query)**: Simple lookups ("Has this failed before?").
4. [ ] **Avoid**: Complex RDF ontologies or reasoning engines in v1.

### Phase D: Cognitive State Management

**Goal**: The feedback loop.

**Tasks**:
1. [ ] Monitor `Entropy` during execution.
2. [ ] Trigger `Mode` transitions (Exploratory -> Convergent).
3. [ ] Update UI to visualize this heartbeat.

---

## 🎯 Success Criteria

### The "Golden Run" Standard
- Every cognitive execution produces a **Canonical Trace**.
- We can **diff** two executions of the same goal.
- We can **replay** an execution deterministically (if LLM seed is fixed).

### Quantitative
| Metric | Current | Target |
|--------|---------|--------|
| Patterns running on WoT | 0 | 1 (Pilot), then All |
| Execution Determinism | Low | High (Auditable) |
| Trace Persistance | Ephemeral | Permanent (KG) |

---

## ⚡ Next Immediate Action (The "Physics")

**Do not try to build the whole system at once.**

1.  **Pick one pattern** (Recommendation: `ChainOfThought` or `ReAct`).
2.  **Write the Compiler**: `compile(Pattern) -> WoTPlan`.
3.  **Run it**.
4.  **Check the Golden Trace**.

Once this loop exists, the rest is just iteration.
