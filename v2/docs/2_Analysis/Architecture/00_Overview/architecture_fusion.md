# TARS v2: The Fusion Architecture

**Date:** November 26, 2025
**Status:** Draft
**Objective:** Synthesize the best ideas from TARS v1, TARS v2 (Functional), and Agent0 (Co-Evolution) into a unified F# architecture.

---

## 1. The Core Concept: "Evolutionary Graph Execution"

We define TARS v2 not just as an agent framework, but as a **Self-Evolving Runtime**.

* **From TARS v1:** We keep the **Metascript** concept, but redefine it as a serialized **Graph Definition**. We also keep the **Zero Tolerance** policy for quality.
* **From TARS v2:** We use the **Functional Micro-Kernel** and **Immutable State** to ensure that self-modification is safe and verifiable.
* **From Agent0:** We adopt the **Curriculum-Executor** loop as the primary driver of autonomy.

---

## 2. The Architecture Layers

### Layer 1: The Substrate (Tars.Core & Tars.Graph)

* **Immutable Kernel:** The event loop that passes messages.
* **Graph Engine:** A finite state machine that executes agent logic.
* **Sandbox:** A Docker-based isolation layer for executing generated code.

### Layer 2: The Cortex (Tars.Cortex)

* **LLM Abstraction:** Pluggable backends (Ollama, OpenAI).
* **Memory Grid:** Vector stores for long-term retention.
* **Grammar Engine:** Enforcing structured output (JSON/F#) from LLMs.

### Layer 3: The Soul (The Agent0 Loop)

This is the new "Main Loop" of the system.

#### A. The Curriculum Agent (The Teacher)

* **Role:** Analyzes the system's current capabilities and generates a **Training Task**.
* **Output:** A `TaskDefinition` (YAML/JSON) containing:
  * `Goal`: What to achieve.
  * `Constraints`: What tools to use/avoid.
  * `Validation`: A test case to prove success.

#### B. The Executor Agent (The Student)

* **Role:** Takes the `TaskDefinition` and attempts to solve it.
* **Process:**
    1. **Plan:** Converts the task into a `Graph` workflow.
    2. **Code:** Generates F# code or uses existing tools.
    3. **Execute:** Runs the code in the `Sandbox`.
    4. **Refine:** Loops until success or timeout.

#### C. The Evaluator (The Critic)

* **Role:** Verifies the result against the `Validation` criteria.
* **Feedback:** Updates the `Memory Grid` with the result (Success/Failure trace).

---

## 3. The "Metascript" Evolution

In v1, Metascripts were text files. In v2, they are **Graph Templates**.

```fsharp
// A Metascript is now a Graph Factory
type Metascript = 
    | Linear of steps: string list
    | Branching of condition: string * truePath: Metascript * falsePath: Metascript
    | Loop of condition: string * body: Metascript
```

The Curriculum Agent generates these structures, and the Graph Engine executes them.

---

## 4. Implementation Strategy

1. **Refine Tars.Graph:** Ensure it can support "Graph-in-Graph" (Subgraphs) to allow the Executor to spawn sub-agents.
2. **Build Tars.Evolution:** A new project containing the Co-Evolution loop logic.
3. **Define the Protocol:** The structured format for `TaskDefinition` and `ValidationResult`.

---
