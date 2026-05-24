# Phase 6 Integration Strategy: The Cognitive Nervous System

**Date:** November 28, 2025
**Status:** Draft
**Target:** TARS v2.1

## Overview

Phase 6 introduces several powerful but distinct components: **Budget Governor**, **Semantic Message Bus**, **Agentic Interfaces**, and **Circuit Flow Control**. This document defines the integration strategy to ensure these components work together as a cohesive "Cognitive Nervous System" rather than isolated features.

## The Integration Model

We view the system through three layers:

1. **The Governor (Resource Layer)**: `BudgetGovernor`
    * *Role*: The metabolic system. It limits energy (tokens/cost) consumption.
    * *Flow*: Passed down the call stack. Every action consumes from it.
2. **The Bus (Communication Layer)**: `SemanticMessage`
    * *Role*: The nervous system. It carries signals (Intent) with context.
    * *Flow*: Horizontal between agents.
3. **The Interfaces (Execution Layer)**: `ExecutionOutcome`
    * *Role*: The muscle system. It performs actions and reports status (Success/Partial/Failure).
    * *Flow*: Returns up the call stack.

## 1. Integrating Budget Governor

The `BudgetGovernor` must be threaded through the entire execution stack.

### Strategy: "Budget Context"

Instead of passing `BudgetGovernor` manually to every function, we will attach it to the `KernelContext` or `AgentContext`.

```fsharp
type AgentContext = {
    AgentId: AgentId
    Budget: BudgetGovernor  // <-- The Governor
    Memory: WorkingMemory
    Tools: ToolRegistry
}
```

### Integration Points (Budget)

1. **Metascript Engine**:
    * **Start**: Creates a root `BudgetGovernor` based on user config (e.g., `--max-cost 1.00`).
    * **Step**: Allocates a slice of the budget to each step (if parallel) or shares the governor (if sequential).
2. **Agents**:
    * **Thinking**: Checks `Budget.TryConsumeTokens()` before calling LLM.
    * **Acting**: Checks `Budget.TryConsumeCall()` before executing a tool.
3. **Tools**:
    * **Execution**: Reports actual consumption back to the governor (if measurable).

## 2. Integrating Semantic Message Bus

The `EventBus` moves from routing simple objects to routing `SemanticMessage<'T>`.

### Strategy: "Intent-Based Routing"

Agents will no longer just "receive messages". They will "listen for intents".

```fsharp
// Old
bus.Subscribe(agentId, handler)

// New
bus.SubscribeSemantic(agentId, {
    Performative = Some Request
    Ontology = Some "coding"
    Filter = fun msg -> msg.Constraints.MaxComplexity = Some "O(n)"
})
```

### Integration Points (Semantic Bus)

1. **Curriculum Agent**:
    * Emits `Request` messages with `TaskDefinition` as content.
    * Sets `Constraints` (e.g., `MaxTokens=1000`) based on the task difficulty.
2. **Executor Agent**:
    * Subscribes to `Request` messages where it matches the `Ontology`.
    * Replies with `Inform` (Success) or `Failure` (Error).
3. **Kernel**:
    * Enforces that `Sender` and `Receiver` fields match the actual channel.

## 3. Integrating Agentic Interfaces

The `ExecutionOutcome` type replaces raw `Result` types to handle partial failures.

### Strategy: "Outcome Propagation"

The `execution` computation expression will automatically handle budget checks and partial failures.

```fsharp
let solveTask ctx task = execution {
    // 1. Check Budget
    do! ctx.Budget.Check() 
    
    // 2. Plan
    let! plan = Planner.createPlan task
    
    // 3. Execute (with partial failure handling)
    let! result = Executor.run plan
    
    return result
}
```

### Integration Points (Agentic Interfaces)

1. **Evolution Loop**:
    * `runAgentLoop` now returns `ExecutionOutcome`.
    * If `PartialSuccess` (e.g., "Tests passed but lint failed"), the loop continues but logs a warning.
    * If `Failure` (e.g., "Budget Exceeded"), the loop terminates or backtracks.

## 4. The Unified Flow

Here is how a single task flows through the integrated system:

1. **User** runs `tars evolve --budget 1000`.
2. **Kernel** creates `BudgetGovernor(1000)`.
3. **Curriculum Agent** (using Governor) generates a task.
    * *Cost*: Consumes 50 tokens.
4. **Curriculum** sends `SemanticMessage(Request, Task)` to Bus.
5. **Bus** routes to **Executor Agent**.
6. **Executor** (using Governor) accepts task.
    * *Check*: Is budget > estimated cost?
7. **Executor** calls **LLM**.
    * *Cost*: Consumes 200 tokens.
8. **Executor** runs **Tool**.
    * *Result*: `PartialSuccess` (Tool ran but timed out).
9. **Executor** sends `SemanticMessage(Inform, Result)` back.
10. **Kernel** aggregates costs and logs the trace.

## 5. Implementation Order

1. **Budget Wiring**: Update `AgentContext` to include `BudgetGovernor`.
2. **Semantic Envelope**: Update `EventBus` to use `SemanticMessage`.
3. **Agent Update**: Refactor `Curriculum` and `Executor` to use the new types.
4. **Metascript Integration**: Update `Engine.fs` to initialize the budget.

## 6. Conclusion

This strategy ensures that "Cognitive Hardening" is not just a collection of features, but a fundamental shift in how TARS operates: **Resource-aware, Intent-driven, and Resilient.**
