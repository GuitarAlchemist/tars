# Epistemic Governor: The Scientist

## Overview

The **Epistemic Governor** is the "Scientist" component of TARS's cognitive architecture. Its primary role is to ensure that the system learns *generalizable principles* rather than just memorizing solutions to specific tasks. It acts as a meta-learning layer that observes the execution process and extracts abstract knowledge.

## Core Responsibilities

1. **Generalization Testing (Falsification)**:
    * Before accepting a solution as "valid knowledge," the Governor generates *variants* of the original task (changing inputs, constraints, or context).
    * It tests the proposed solution against these variants.
    * If the solution fails a variant, it is rejected or marked as "brittle."

2. **Principle Extraction (Abstraction)**:
    * Once a solution is verified, the Governor analyzes *why* it worked.
    * It distills the solution into a `Belief` (a high-level principle or pattern).
    * Example: "Use a Hash Map for O(1) lookups" instead of "Use `Dictionary<string, int>`".

3. **Belief Management (Knowledge Graph)**:
    * Extracted beliefs are stored in the `BeliefGraph` (a specialized Vector Store collection).
    * Beliefs have confidence scores and lineage (which tasks derived them).

## Architecture

### Components

* **`IEpistemicGovernor` Interface**:
  * `GenerateVariants(task, count)`: Returns a list of modified task descriptions.
  * `VerifyGeneralization(task, solution, variants)`: Returns a verification result (Pass/Fail).
  * `ExtractPrinciple(task, solution)`: Returns a `Belief` object.

* **`EpistemicGovernor` Implementation**:
  * Uses the LLM (Reasoning Model) to perform the above tasks.
  * Prompts are designed to encourage critical thinking and abstraction.

* **Integration with Evolution Engine**:
  * The Governor hooks into the `Engine.step` loop.
  * **Post-Execution**: After the Executor solves a task, the Governor intervenes to extract and store the principle.

### Data Structures

```fsharp
type EpistemicStatus =
    | Hypothesis          // Proposed solution, untested generalization
    | VerifiedFact        // Passed generalization tests
    | UniversalPrinciple  // Abstracted and reused successfully > N times
    | Heuristic           // Useful but known to be brittle (flagged)
    | Fallacy             // Proven false

type Belief = {
    Id: Guid
    Statement: string     // The abstract principle
    Context: string       // When does this apply?
    Status: EpistemicStatus
    Confidence: float
    DerivedFrom: Guid list // TaskIds that led to this belief
    CreatedAt: DateTime
    LastVerified: DateTime
}
```

## Workflow

1. **Task Solved**: The Executor Agent solves a task (e.g., "Implement a binary search").
2. **Extraction**: The Evolution Engine calls `Governor.ExtractPrinciple`.
3. **LLM Analysis**: The Governor asks the LLM: "What is the underlying principle here?"
4. **Belief Creation**: The LLM responds: "Sorted arrays allow O(log n) search via divide-and-conquer."
5. **Storage**: This belief is embedded and stored in the `tars-beliefs` vector collection.
6. **Future Use**: (Phase 6.8.4) The Curriculum Agent queries this store to find gaps in knowledge.

## Thermodynamic Model (Physics of Truth)

To rigorously manage the quality of information, the Epistemic Governor adopts a **Thermodynamic Model** of cognition.

### 1. Maxwell's Demon (Entropy Reduction)

* **Concept**: The context window naturally accumulates **Entropy ($S$)** (noise, hallucination, drift) over time.
* **Role**: The Governor acts as **Maxwell's Demon**, actively expending energy (compute budget) to sort "hot" (true, high-value) tokens from "cold" (noise) tokens.
* **Mechanism**:
  * **Entropy Measurement**: Periodically measure the semantic divergence of the current context from the core `BeliefGraph`.
  * **Active Cooling**: If entropy exceeds a threshold, trigger a "Cooling Cycle" (summarization, pruning, or re-grounding) to restore order.

### 2. Simulated Annealing (Search Strategy)

* **Concept**: Balancing exploration (high volatility) and exploitation (freezing).
* **Role**: The Governor modulates the system's "Temperature" ($T$).
* **Mechanism**:
  * **High Temp**: When exploring new domains or forming Hypotheses, allow high volatility (high LLM temperature, diverse agents).
  * **Quenching**: When verifying a `UniversalPrinciple`, rapidly cool the system (low temperature, strict validation) to "freeze" the truth into the Knowledge Graph.

## Future Roadmap

* **Active Falsification**: Actually *run* the code against generated test cases (currently static analysis).
* **Curriculum Feedback**: Use the density of beliefs in the embedding space to guide the Curriculum Agent (e.g., "Explore areas where we have few beliefs").
* **Belief Refinement**: Update confidence scores based on reuse success/failure.
