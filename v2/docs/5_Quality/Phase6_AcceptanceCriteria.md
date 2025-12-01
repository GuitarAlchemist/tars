# Acceptance Criteria: Phase 6 - Cognitive Architecture

**Date:** November 28, 2025
**Status:** Draft
**Target Version:** v2.1

## Overview

This document defines the acceptance criteria and test scenarios for the Cognitive Architecture components of TARS v2. These criteria must be met before the features are considered "Done".

## Phase 6.1: Budget Governor

**Objective:** Prevent resource exhaustion (tokens, money, time) by enforcing strict budgets on all workflows.

### Scenarios (Budget Governor)

1. **Token Budget Enforcement**
    * **Given** a workflow with a budget of 1000 tokens.
    * **When** the agent attempts to consume 1001 tokens.
    * **Then** the operation should fail with `BudgetExceededException`.
    * **And** the remaining budget should be 0 (or unchanged if atomic).

2. **Budget Inheritance**
    * **Given** a parent task with 5000 tokens.
    * **When** it spawns 2 child tasks.
    * **Then** the child tasks should share the parent's budget (or have it allocated).
    * **And** the total consumption of parent + children must not exceed 5000.

3. **Graceful Degradation**
    * **Given** a budget is near exhaustion (e.g., < 10%).
    * **When** an agent requests a complex operation.
    * **Then** the system should suggest a cheaper alternative (e.g., "Summarize" instead of "Analyze").

## Phase 6.2: Semantic Speech Acts

**Objective:** Ensure clear intent in inter-agent communication using FIPA-ACL inspired performatives.

### Scenarios (Speech Acts)

1. **Explicit Intent**
    * **Given** an agent wants to ask for information.
    * **When** it sends a message.
    * **Then** the message `Performative` must be `Query`.
    * **And** the `Content` must be the question.

2. **Protocol Enforcement**
    * **Given** an agent receives a `Propose` message.
    * **When** it replies.
    * **Then** the reply must be `Accept`, `Reject`, or `CounterPropose`.
    * **And** a `Inform` reply should be flagged as a protocol violation (warning).

## Phase 6.3: Semantic Fan-out Limiter

**Objective:** Prevent combinatorial explosion of subtasks.

### Scenarios (Fan-out Limiter)

1. **Top-K Selection**
    * **Given** the Curriculum Agent generates 20 subtasks.
    * **And** the `MaxSubtasks` limit is set to 5.
    * **When** the tasks are processed.
    * **Then** only the top 5 tasks (by Score) should be scheduled.
    * **And** the other 15 should be logged as "Pruned".

2. **Scoring Validation**
    * **Given** a list of generated tasks.
    * **When** the limiter runs.
    * **Then** tasks must be sorted by `Importance` * `Urgency`.

## Phase 6.4: Adaptive Reflection

**Objective:** Optimize quality vs. cost by stopping reflection when returns diminish.

### Scenarios (Adaptive Reflection)

1. **Convergence Stop**
    * **Given** a reflection loop.
    * **When** the improvement between Iteration N and N+1 is < 5%.
    * **Then** the loop should terminate.
    * **And** the result from N+1 should be returned.

2. **Hard Limit Stop**
    * **Given** a reflection loop.
    * **When** the `MaxReflections` count (e.g., 3) is reached.
    * **Then** the loop should terminate regardless of improvement.

## Phase 6.5: Agentic Interfaces

**Objective:** Handle partial failures and soft contracts gracefully.

### Scenarios (Agentic Interfaces)

1. **Partial Success**
    * **Given** an agent executes a multi-step tool (e.g., "Search and Summarize").
    * **When** the search succeeds but summarization is truncated.
    * **Then** the result should be `PartialSuccess`.
    * **And** it should contain the search results and a warning about truncation.

2. **Computation Expression Flow**
    * **Given** a workflow using the `execution` computation expression.
    * **When** a step returns `PartialSuccess`.
    * **Then** the workflow should continue to the next step.
    * **And** the warnings should be accumulated in the final result.

## Phase 6.6: Semantic Message Bus

**Objective:** Route messages based on meaning and context.

### Scenarios (Semantic Message Bus)

1. **Ontology Routing**
    * **Given** a message with `Ontology = "finance"`.
    * **When** it is published.
    * **Then** it should be delivered to agents subscribed to "finance".
    * **And** agents subscribed only to "coding" should NOT receive it.

2. **JSON-LD Serialization**
    * **Given** a `SemanticMessage`.
    * **When** it is serialized.
    * **Then** the output must be valid JSON-LD.
    * **And** it must contain the `@context` field.

## Phase 6.7: Circuit Flow Control

**Objective:** Manage system stability under load.

### Scenarios (Circuit Flow Control)

1. **Backpressure**
    * **Given** the Executor Agent's inbox is full (Capacity=10).
    * **When** the Curriculum Agent tries to send the 11th task.
    * **Then** the send operation should wait (async) or return `BackpressureSignal`.
    * **And** the Curriculum Agent should pause generation.

2. **Circuit Breaker**
    * **Given** a tool fails 5 times in a row.
    * **When** the 6th attempt is made.
    * **Then** the call should fail immediately with `CircuitOpenException`.
    * **And** no actual tool execution should occur.

## Phase 6.8: Epistemic Governor

**Objective:** Ensure TARS learns generalizable principles, not just memorized solutions.

### Scenarios (Epistemic Governor)

1. **Variant Generation**
    * **Given** a coding task "Sort a list of integers".
    * **When** `GenerateVariants` is called with N=3.
    * **Then** it should return 3 distinct variations (e.g., "Sort floats", "Sort strings", "Sort empty list").
    * **And** the variations should preserve the core logic requirement.

2. **Principle Extraction**
    * **Given** a successful solution to a task.
    * **When** `ExtractPrinciple` is called.
    * **Then** it should return a `Belief` object.
    * **And** the `Statement` should be a high-level abstraction (e.g., "Use Quicksort for average O(n log n)").

3. **Evolution Integration**
    * **Given** the Evolution Engine completes a task successfully.
    * **When** the step finishes.
    * **Then** a new `Belief` should be stored in the `VectorStore`.
    * **And** the belief should be linked to the Task ID.

## Evidence Findings (K-Theory Validation)

* **$K_0$ (Resource Conservation)**:
  * Verified that `Budget` forms a commutative monoid under addition.
  * Verified conservation laws: `Remaining + Consumed = Total` holds for all bounded resources (Tokens, RAM, Nodes, etc.).
  * Verified allocation conservation: `ParentRemaining + ChildTotal = ParentTotal`.
  * **Test Suite**: `Tars.Tests.KTheoryTests` (All K0 tests passed).

* **$K_1$ (Cycle Detection)**:
  * Verified `GraphAnalyzer` correctly computes the kernel of the adjacency matrix ($I - A^T$).
  * Detected self-loops, 2-cycles, 3-cycles, and disjoint cycles.
  * Confirmed no false positives for DAGs (Directed Acyclic Graphs) and linear chains.
  * **Test Suite**: `Tars.Tests.KTheoryTests` (All K1 tests passed).
