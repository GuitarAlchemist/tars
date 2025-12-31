# Phase 12: Workflow of Thought (WoT)

> **From "guessing" to "thinking" — reasoning as a governed graph.**

## Definition of Done ✅

A 5-node minimal workflow that proves WoT is "real", not just documentation:

### Required Components

| Component | Requirement | Status |
|-----------|-------------|--------|
| **5-node workflow** | Plan → Critique → Verify → Tool → Distill | ✅ Implemented in WotTypes.fs |
| **Typed I/O** | Each node has `NodeContent` input/output | ✅ Implemented |
| **Budget** | Each node has `NodeBudget` (tokens/time/tools) | ✅ Implemented |
| **Policy Gates** | Each node has `PolicyGate` (PII/PHI/risk) | ✅ Implemented |
| **Evidence** | Each node produces `NodeEvidence` (audit trail) | ✅ Implemented |
| **Happy Path** | 5 nodes execute in sequence | ✅ Test exists |
| **Failure Case** | PII detected → Redaction → Continue | ✅ Test exists |
| **LLM Integration** | Reason nodes use actual LLM | 🚧 Mock (needs wiring) |
| **CLI Command** | `tars wot run <workflow>` | 🚧 Pending |

### Key Insight: Reason Nodes vs Work Nodes

```fsharp
// REASON NODES: LLM thinks (no side effects)
type ReasonOperation =
    | Plan of goal: string
    | Critique of target: NodeId
    | Synthesize of sources: NodeId list
    | Explain of topic: string
    | Rewrite of target: NodeId * instruction: string

// WORK NODES: Tools act (has side effects)  
type WorkOperation =
    | ToolCall of tool: string * args: Map<string, obj>
    | Verify of check: string * expected: string
    | Redact of patterns: string list
    | Persist of location: string
    | Fetch of source: string
    | Transform of fn: string
```

This separation prevents the LLM from "pretending" to have done something.

### Implementation

- **Types**: `src/Tars.Core/WorkflowOfThought/WotTypes.fs`
- **Tests**: `DefinitionOfDone.runAll()`

---

## Core Conceptual Frameworks

AI reasoning is categorized by how it structures intermediate "thoughts" to reach a conclusion:

### Chain-of-Thought (CoT)
- **Structure**: Sequential, linear steps
- **Best for**: Arithmetic or simple logic
- **Weakness**: Fragile for complex, multi-stakeholder tasks

```
Step 1 → Step 2 → Step 3 → Answer
```

### Tree-of-Thought (ToT)
- **Structure**: Multiple divergent branches
- **Best for**: Strategic lookahead, problem-solving
- **Strength**: Backtracking if a path fails

```
       Step 1
      /      \
   Step 2a  Step 2b
    /   \      |
  2a-1  2a-2  2b-1
```

### Graph-of-Thought (GoT)
- **Structure**: Thoughts as nodes, relationships as edges
- **Behaviors**: Non-linear, merging partial answers from different branches
- **Edges**: `supports`, `contradicts`, `refines`

```
    [Thought A] ←── supports ──→ [Thought B]
         ↓                            ↓
    contradicts                   refines
         ↓                            ↓
    [Thought C] ←── merges ───→ [Conclusion]
```

### Workflow-of-Thought (WoT) — THE PRODUCTION PATTERN
- **Structure**: GoT + policy checks + tool use + verification
- **Unique**: Mixes reasoning nodes with "work nodes"
- **Work Nodes**: PII redaction, legal compliance, human-in-the-loop triggers

```
    [Plan Node] 
         ↓
    [Generate Node] ←──→ [Tool: Web Search]
         ↓
    [Policy Check: GDPR] ←── FAIL ──→ [Redact PII Node]
         ↓
    [Critique Node] ←── contradicts ──→ [Refine Node]
         ↓
    [Human Review Trigger] ←── high risk ──→ [HALT]
         ↓
    [Distill Node] → Final Output + Audit Trail
```

---

## Operational Workflow Steps

A typical advanced AI workflow (e.g., an Agentic Workflow) follows these stages:

### 1. Planning
A controller agent breaks a high-level goal into sub-goals and defines initial thought nodes.

```fsharp
type PlanNode = {
    Id: NodeId
    Goal: string
    SubGoals: NodeId list
    Constraints: Policy list
}
```

### 2. Generation
Specialized models generate candidate answers, perform research, or execute tool calls.

```fsharp
type GenerateNode = {
    Id: NodeId
    Generator: string  // Model or tool
    Input: ThoughtContent
    Output: ThoughtContent option
    ToolCalls: ToolCall list
}
```

### 3. Critique & Verification
A "critic" node evaluates output against hard constraints.

```fsharp
type CritiqueNode = {
    Id: NodeId
    Target: NodeId
    Constraints: Constraint list  // Schema, brand, legal (GDPR, HIPAA)
    Violations: Violation list
    Verdict: Pass | Fail | NeedsReview
}
```

### 4. Routing & Refinement
Based on critique, the system decides next action.

```fsharp
type RouteDecision =
    | Expand of NodeId         // Continue this branch
    | Merge of NodeId list     // Combine partial answers
    | Backtrack of NodeId      // Return to previous step
    | Escalate to Human        // Human-in-the-loop trigger
```

### 5. Distillation
The final reasoning graph is compressed into a polished response with an audit trail.

```fsharp
type DistillNode = {
    Id: NodeId
    ReasoningGraph: Node list
    CompressedOutput: string
    AuditTrail: Decision list  // Why specific decisions were made
}
```

---

## Key Interaction Patterns

### ReAct (Reason + Act)
The AI alternates between explicit internal reasoning and external actions, using observations from those actions to inform the next step.

```
Thought: I need to find the current population of France.
Action: web_search("France population 2025")
Observation: 68.4 million
Thought: Now I can answer the question with a verified source.
```

### Parallelization
The system runs multiple sub-tasks or reasoning paths simultaneously to increase speed and cross-verify results.

```fsharp
let! results = 
    [ searchWikipedia query
      searchWikidata query
      searchDBpedia query ]
    |> Async.Parallel
```

### Reflection
A dedicated cycle where the model reviews its own draft for inconsistencies before presenting it as a final output.

```fsharp
type ReflectionCycle = {
    Draft: ThoughtContent
    SelfCritique: Critique list
    Revisions: Revision list
    FinalOutput: ThoughtContent
}
```

---

## TARS Integration

### How WoT Fits TARS Architecture

| TARS Component | WoT Role |
|----------------|----------|
| `Tars.Core.HybridBrain` | Plan nodes with typed IR |
| `Tars.Knowledge.KnowledgeLedger` | Evidence storage for reasoning |
| `Tars.Core.PolicyEngine` | Constraint enforcement |
| `Tars.Symbolic.Invariants` | Critique validation |
| `Tars.Tools.*` | Tool calls (web, code, graph) |

### Example: Canonical Task as WoT

```
[Plan: Refactor file.fs]
        ↓
[Generate: Analyze code smells]  ←──→ [Tool: analyze_complexity]
        ↓
[Generate: Propose refactoring]  ←──→ [Tool: find_code_smells]
        ↓
[Critique: Validate plan]  ←── Tars.Core.HybridBrain.validate
        ↓ (FAIL)
[Backtrack: Revise plan]
        ↓ (PASS)
[Execute: Apply refactoring]  ←──→ [Tool: file_write]
        ↓
[Verify: Build & Test]  ←──→ [Tool: dotnet_build, dotnet_test]
        ↓ (PASS)
[Distill: Record evidence]  ←──→ [Ledger: Assert beliefs]
```

---

## Human-AI Collaboration Tips

### The 10-20-70 Rule
- **10%** effort on algorithms
- **20%** effort on technology
- **70%** effort on people and process integration

### Define Clear Handoffs
Establish exactly when a human must review an AI's intermediate "thought node":
- High-risk legal clauses
- Financial decisions above threshold
- Code modifications to production systems

### Use Reasoning Roadmaps
Instead of a rigid script, provide the AI with a flexible framework of steps to ensure logical consistency.

### Traceability
Save the reasoning graph, not just the output. This provides:
- Audit trail for compliance
- Debug capability for "hallucination cascades"
- Training data for improvement

---

## Implementation Priority

| Component | Priority | Status |
|-----------|----------|--------|
| GoT node types | P0 | 🔜 Planned |
| Policy check nodes | P0 | Exists in PolicyEngine |
| Tool integration | P0 | ✅ 124+ tools |
| Human-in-the-loop triggers | P1 | 🔜 Planned |
| Reasoning graph persistence | P1 | Exists in KnowledgeLedger |
| Audit trail generation | P2 | 🔜 Planned |

---

## Relationship to Other Phases

```
Phase 8 (GoT) ──────────────────────────────────────────────┐
                                                             │
Phase 12 (WoT) = GoT + Policy + Tools + Verification ◄──────┤
                                                             │
Phase 17 (Hybrid Brain) provides the typed IR for WoT ◄─────┘
```

WoT is the **operational application** of the theoretical foundations built in Phases 8 and 17.

---

*Last updated: 2025-12-26*
