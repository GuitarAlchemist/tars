# TARS v2: Hybrid Brain Architecture

> **Core Thesis**: The LLM is a hypothesis generator and plan writer. The symbolic system (grammars + F# types + solvers) is the judge that validates, executes, and corrects. You don't "mix" everything - you orchestrate layers with very strict contracts.

## 1. The Mental Model: LLM = Proposes, Symbolic = Disposes

**Central Idea**: The LLM never directly "decides" a fact or action. It proposes an artifact that must pass through a series of deterministic safeguards.

### Typical Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│     LLM     │ ──► │   Parser/    │ ──► │  Typecheck  │ ──► │ Interpreter/ │
│  (Propose)  │     │   Grammar    │     │ Constraints │     │   Sandbox    │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
       │                   │                    │                    │
       │                   │                    │                    │
       ▼                   ▼                    ▼                    ▼
   Plan + DSL          Reject if           Reject if            Execute in
   JSON/F#/EBNF       non-conforming      violates invariants   controlled env
       │                                                             │
       │                                                             ▼
       │                                                    ┌──────────────┐
       │                                                    │  Proof/Tests │
       │                                                    │ "Does it hold?"│
       │                                                    └──────────────┘
       │                                                             │
       │          ┌─────────────────────────────────────────────────┘
       │          │ On Failure: Formal diagnostic (not "doesn't work lol")
       │          │             "Rule R12 violates invariant I3"
       │          ▼
       └─────► Retry with structured feedback
```

**It's like a compiler + CI, but for thoughts.**

---

## 2. Grammars: Your Cognitive Firewall

Two-level grammar architecture:

### Level A — Surface DSL (readable, versionable)
- `.trsx` / `.flux` : plans, tasks, agents, constraints, token budgets, authorized sources, etc.
- Human-readable, diff-friendly, versionable in git

### Level B — IR (Intermediate Representation) strict
- Typed F# AST, minimal, unambiguous
- Everything passes through here
- **The LLM never manipulates execution directly, only the IR via parsing**

### Why This is Magic Against Hallucination

> Hallucinations become compilation or validation errors — easy to detect and auto-correct.

```
EBNF / Parser Combinators → AST → Typed IR → Interpreter / Code Generator / Executor
```

### 2.1 The "Immutable Core" Contract

A critical distinction exists between **Core Code** and **Target Code**:

1.  **Immutable Core** (`src/Tars.Core`):
    *   TARS **NEVER** modifies its own core logic automatically.
    *   Changes to Core require a strict "Constitution Amendment" protocol (Phase 14).
    *   *Exception*: TARS can extend its capabilities via **Grammars** (Phase 17) and **Tools** (Phase 13), which are data, not code.

2.  **Mutable Target** (User Workspace):
    *   TARS **CAN** refactor, write, and delete code in the user's workspace (e.g., the Canonical Task).
    *   This is the "Work" in "Workflow of Thought".
    *   Subject to standard safety constraints (build pass, test pass, rollback).

---

## 3. F# as the "Language of Truth": Types + Computation Expressions

F# gives you three Jedi weapons:

### (a) Algebraic Types (Discriminated Unions)  
Perfect for modeling: authorized actions, sources, confidence levels, agent states, etc.

```fsharp
type Tool = 
    | WebSearch of query: string
    | FileRead of path: string
    | LlmCall of prompt: string
    | DatabaseQuery of sql: string

type ConfidenceLevel = 
    | High | Medium | Low | Unknown

type Source = 
    | FromKnowledgeGraph
    | FromWeb of url: string
    | FromLLM of model: string * confidence: float
    | FromUser
```

### (b) Phantom Types / State Types
Make it impossible to execute unless the plan is "Validated":

```fsharp
type Draft = interface end
type Parsed = interface end
type Validated = interface end
type Executable = interface end

type Plan<'State> = { Goal: string; Steps: Step list; Version: int }

// Functions enforce state transitions:
let parse: Plan<Draft> -> Result<Plan<Parsed>, ParseError list>
let validate: Plan<Parsed> -> Result<Plan<Validated>, ValidationError list>
let execute: Plan<Validated> -> Plan<Executable>  // Only validated plans can execute!
```

### (c) Computation Expressions = Executable DSL
Your internal orchestration language:

```fsharp
// Workflow orchestration
tars {
    let! plan = generatePlan goal
    do! verify plan
    let! result = execute plan
    return result
}

// Invariant verification
verify {
    require (plan.Budget <= maxBudget) "Budget exceeded"
    require (plan.Tools |> List.forall isAllowed) "Unauthorized tool"
    require (plan.Steps.Length > 0) "Empty plan"
}

// Evidence tracing
evidence {
    source "Wikipedia:France"
    confidence 0.85
    retrieved_at DateTime.Now
    claim "Paris is the capital of France"
}
```

The CE can carry:
- **Writer monad** (logs)
- **Result monad** (errors)
- **State monad** (memory)
- **Reader monad** (config/policy)

= A pure, testable pipeline.

---

## 4. The "LLM Contract": Structured Outputs, Not Blabber

**Draconian Rule**: The LLM must produce 2 channels:

| Channel | Purpose | Example |
|---------|---------|---------|
| **Narrative** (optional) | For humans | "I'll search for information about..." |
| **Formal** (mandatory) | DSL/JSON conforming to grammar | `{ "action": "search", "query": "..." }` |

**TARS only executes the formal channel.**

### Anti-Chaos Bonus: Require the formal channel to contain:

```json
{
  "action": "web_search",
  "query": "Paris population 2024",
  "assumptions": ["User wants current data", "English sources preferred"],
  "unknowns": ["Exact source reliability"],
  "requires_sources": ["Official statistics", "Wikipedia"],
  "confidence": 0.7  // Not used for decisions, just logging
}
```

---

## 5. "Useful" Symbolic: Where the LLM Struggles Most

You don't need to "solve world logic". You want symbolic where it's profitable:

### Plan Validation
- Preconditions / postconditions
- Security invariants (no deletion, no unauthorized network actions, etc.)
- Budgets (time, tokens, cost, memory)

### Rewriting / Normalization
- Transform "fuzzy plan" → "sequence of primitive actions"
- Normalize intentions into operators (mini STRIPS/HTN)

### Constraints & Solvers
- Scheduling (toposort)
- Tool choice (MCP/web/local) under constraints
- Conflict resolution between agents

### Local Truth Verification
- "Does this fact exist in symbolic memory / triple store?"
- If not: "fetch a source" or mark as "unknown"

---

## 6. Recommended Modular Architecture

A "compiler" structure:

```
┌─────────────────┐
│  Tars.Parsing   │  EBNF + parser → AST
├─────────────────┤
│    Tars.IR      │  Typed IR (the core)
├─────────────────┤
│ Tars.Validation │  Rules, invariants, policies (no-web, allow-web, etc.)
├─────────────────┤
│  Tars.Planning  │  Expansions / HTN / rewrite engine
├─────────────────┤
│ Tars.Execution  │  Sandbox + runners (process, file I/O, tool calls)
├─────────────────┤
│  Tars.Memory    │  .trsx + .smem + vector store + KG
├─────────────────┤
│   Tars.LLM      │  Adapters (Ollama/llama.cpp/OpenAI/etc.)
├─────────────────┤
│Tars.Diagnostics │  Structured logs + version diffs + meta-summary
└─────────────────┘
```

**Key Point**: Everything that is "LLM-output" must pass through:
```
Parsing → IR → Validation
```
Otherwise it's considered decorative text.

Parsing → IR → Validation
```
Otherwise it's considered decorative text.

### 6.1 Unified Execution Pipeline (Single IR)

To avoid fragmentation, **ALL** DSLs compile to the same Executable IR:

*   `.tars` (Metascripts) ↘
*   `.trsx` (Rich Scripts) → **Unified Parser** → **WorkflowGraph** → **Plan<Executable>**
*   `.flux` (Meta-logic)   ↗

There is only **ONE** runtime engine (`Tars.Core.HybridBrain`). Use the `CognitionCompiler` pipeline for everything.

---

## 7. The Game-Changer: Automatic "Formal Critique"

When it fails, TARS doesn't return "doesn't work" to the LLM. It returns a **patch request**:

```
┌──────────────────────────────────────────────────────────────┐
│ VALIDATION FAILURE REPORT                                    │
├──────────────────────────────────────────────────────────────┤
│ Parse Errors:                                                │
│   - Line 5, Column 12: Unexpected token '{'                  │
│                                                              │
│ Type Errors:                                                 │
│   - Step[3].Tool: Expected 'WebSearch', got 'string'         │
│                                                              │
│ Invariant Violations:                                        │
│   - INV-003: Budget constraint violated (150 > max 100)      │
│   - INV-007: Tool 'shell_exec' not in allowlist              │
│                                                              │
│ Minimal Counter-Example:                                     │
│   - Step sequence [1,3,5] violates precondition of Step 5    │
│                                                              │
│ Suggested Fix (rewrite rule):                                │
│   - Replace 'shell_exec' with 'sandbox_exec'                 │
│   - Reduce budget from 150 to 100                            │
└──────────────────────────────────────────────────────────────┘
```

**This transforms the LLM loop into compiler-assisted repair.**

---

## 8. Integration Example: Grammar ↔ CE ↔ Execution

The most powerful (and realistic) path:

```
1. LLM generates workflow in .trsx (or .flux)
         │
         ▼
2. Parser → AST
         │
         ▼
3. AST → Typed F# IR
         │
         ▼
4. IR executed via computation expression `tars { ... }`
         │
         ▼
5. Each step produces:
   - trace.json
   - agentic_trace.txt
   - meta-summary.trsx
         │
         ▼
6. Next version feeds from this diff (your trends system)
```

This aligns perfectly with the `output/versions/` + auto-improvement vision.

---

## 9. Implementation Order (ROI Priority)

| Priority | Component | Description |
|----------|-----------|-------------|
| **1** | Typed IR + States | `Draft/Validated/Executable` state machine |
| **2** | Minimal Grammar | Just enough to express plan + policy + steps |
| **3** | Validator | Security invariants + budget + tool allowlist |
| **4** | CE Runner | `Result + Writer + State` computation expression |
| **5** | Formal Critique | Validation diagnostics returned to LLM |
| **6** | Advanced | KG/triples, richer solvers, etc. |

---

## 10. Core IR Definition (Minimal)

```fsharp
/// The minimal IR for validated execution
type Goal = {
    Description: string
    SuccessMetrics: string list
    Budget: Budget
}

type Step = {
    Id: StepId
    Action: Action
    Preconditions: Condition list
    Postconditions: Condition list
    Evidence: EvidenceRequirement option
}

type ToolCall = {
    Tool: Tool
    Arguments: Map<string, obj>
    AllowedSources: Source list
    Timeout: TimeSpan
}

type Evidence = {
    Source: Source
    Claim: string
    Confidence: float
    RetrievedAt: DateTime
}

type Policy = {
    AllowedTools: Tool list
    ForbiddenActions: Action list
    MaxBudget: Budget
    RequiredEvidence: EvidenceLevel
}

type Budget = {
    MaxTokens: int
    MaxTime: TimeSpan
    MaxCost: decimal
    MaxMemory: int64
}
```

---

## 11. The `tars {}` Computation Expression

The CE that executes only `Step<Validated>`:

```fsharp
type TarsBuilder() =
    member _.Bind(step: Step<Validated>, f) = 
        // Execute validated step, propagate errors
        match executeStep step with
        | Ok result -> f result
        | Error e -> Error e
    
    member _.Return(x) = Ok x
    
    member _.ReturnFrom(x) = x
    
    member _.Zero() = Ok ()

let tars = TarsBuilder()

// Usage:
let workflow = tars {
    let! plan = validatePlan draftPlan  // Must be validated first!
    for step in plan.Steps do
        let! result = executeStep step
        do! logEvidence step result
    return! summarize plan
}
```

**This mechanically forces the entire architecture to stay honest.**

---

## Summary

TARS v2 already has a "cognition compiler" vibe. By pushing this direction, you get a system where:

> **Hallucination is no longer a mysterious bug, but a typed, recoverable error.**

And that's engineering happiness.

**Next Natural Step**: Define the IR (very small) for `Goal / Step / ToolCall / Evidence / Policy / Budget`, then write the `tars {}` CE that executes only `Step<Validated>`. This mechanically forces all architecture to remain honest.
