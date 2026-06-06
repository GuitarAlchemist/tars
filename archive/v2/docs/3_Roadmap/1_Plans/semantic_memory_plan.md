# Semantic Memory (ViLoMem) Implementation Plan

## Overview

This document outlines the plan to integrate a **Semantic Memory** system into the TARS v2 Kernel, inspired by the "Agentic Learner with Grow-and-Refine Multimodal Semantic Memory" (ViLoMem) paper. This system will enable TARS to learn from its past episodes (both successes and failures) by storing, retrieving, and refining structured memory schemas.

## Goals

1. **Kernel-Level Service**: Establish `ISemanticMemory` as a core Kernel service, not just a Cortex tool.
2. **Dual-Stream Schemas**: Implement memory schemas with distinct **Logical** (reasoning, errors) and **Perceptual** (code structure, env state) streams.
3. **Memory Cycle**: Transform the agent execution loop into a `Retrieve -> Plan -> Execute -> Verify -> Grow` cycle.
4. **Grow & Refine**: Implement mechanisms to append new memories (`Grow`) and offline consolidation (`Refine`).

## Architecture

### 1. Core Types (`Tars.Core`)

We will define the fundamental types in `Tars.Core` to ensure they are accessible throughout the system.

* **`ErrorKind`**: Categorization of errors (e.g., `Perceptual`, `Logical`, `Mixed`).
* **`ErrorTag`**: Specific error labels (e.g., `Hallucination`, `SchemaMismatch`).
* **`LogicalMemory`**:
  * `ProblemSummary`: LLM-generated summary of the task.
  * `StrategySummary`: Summary of the approach taken.
  * `ErrorTags`: List of errors encountered.
  * `Outcome`: Success/Failure/Partial.
  * `Embedding`: Vector representation of the logical context.
* **`PerceptualMemory`**:
  * `TouchedResources`: Files, APIs, Tools accessed.
  * `EnvFingerprint`: Hash of the environment state.
  * `Embedding`: Vector representation of the structural context (leveraging AST data).
* **`MemorySchema`**: The top-level container binding Logical and Perceptual memories with metadata (ID, timestamps, usage stats).

### 2. Kernel Interface (`Tars.Kernel`)

The Kernel will expose the Semantic Memory service via the `ISemanticMemory` interface.

```fsharp
type MemoryQuery = {
    TaskId      : string
    TaskKind    : string
    TextContext : string
    Tags        : string list
}

type ISemanticMemory =
    abstract member Retrieve : MemoryQuery -> Async<MemorySchema list>
    abstract member Grow     : EpisodeTrace * VerificationReport -> Async<string> // Returns Schema ID
    abstract member Refine   : unit -> Async<unit>
```

### 3. Implementation (`Tars.Memory` - New Project?)

*Decision*: For Phase 1, we will implement this directly in `Tars.Core` or a new `Tars.Memory` project if dependencies warrant it. Given the need for Vector Store access, a separate project or `Tars.Kernel` implementation is likely best.

* **Storage**: Simple JSON-based storage for schemas (`data/semantic_memory/records/`) and a lightweight index (`data/semantic_memory/index.json`).
* **Vector Store Integration**: Use the existing `IVectorStore` to generate and query embeddings.

### 4. Data Ingestion & Chunking Strategy

Effective chunking is critical for RAG performance. Based on modern best practices (e.g., [DataCamp Chunking Strategies](https://www.datacamp.com/blog/chunking-strategies)), we will implement a multi-strategy approach:

* **Code**: Use **Recursive** or **AST-based** chunking to respect function/class boundaries.
* **Text/Documentation**: Use **Semantic Chunking** (embedding similarity) or **Sliding Window** with overlap to maintain context.
* **Agentic Chunking**: For complex episodes, use an LLM to determine logical breakpoints (e.g., "Problem Statement", "Attempt 1", "Error Analysis").

The `Tars.Cortex.Chunking` module will be upgraded to support these advanced strategies, moving beyond simple fixed-size splitting.

### 5. Integration Points

#### A. Kernel Bootstrap

* Initialize `SemanticMemory` service in `KernelFactory`.
* Inject `IVectorStore` and configuration paths.

#### B. Execution Loop (`Tars.Interface.Cli` / `Tars.Core`)

* **Pre-Execution**: Call `Retrieve` using the task description. Pass retrieved schemas to the Planner/Agent as context (e.g., "Lessons Learned").
* **Post-Execution**: Call `Grow` with the execution trace and verification results (Diagnostics).

#### C. CLI Commands

* `tars smem grow`: Manually trigger memory growth from past traces.
* `tars smem refine`: Trigger the refinement (deduplication/clustering) process.
* `tars smem query`: Test retrieval.

## Implementation Steps

### Phase 1: Core Types & Interface (Completed)

1. Create `src/Tars.Core/SemanticMemoryTypes.fs`.
2. Define `ISemanticMemory` interface in `src/Tars.Kernel/Interfaces.fs` (or similar).

### Phase 2: Basic Implementation (Completed)

1. Implement `SemanticMemoryService` (in `Tars.Kernel` or `Tars.Memory`).
2. Implement JSON persistence and simple cosine similarity retrieval using `IVectorStore`.
3. Register service in `KernelFactory`.

### Phase 3: The Memory Cycle (Completed)

1. [x] Update `EvolutionEngine` to retrieve memories before execution.
2. [x] Update `EvolutionEngine` to call `Grow` after execution/verification.
3. [x] Integrate `SemanticMemory` into `EvolutionContext`.

### Phase 4: CLI & Refinement (Completed)

1. [x] Add `SemanticMemory` command group to `Tars.Interface.Cli`.
2. [x] Implement a basic `Refine` method (deduplication based on embedding similarity).
3. [x] Implement `demo-perceptual` to validate ingestion and growth.
4. [x] Implement `query` to validate retrieval.

### Phase 5: Advanced Ingestion (Completed)

1. [x] Upgrade `Tars.Cortex.Chunking` with **Semantic Chunking** (using `Tars.Llm` embeddings).
2. [x] Implement **Agentic Chunking** for episode summarization.
3. [x] Integrate **AST-based Chunking** for code files in `PerceptualMemory`.

## Verification

* **Unit Tests**: Test storage, retrieval, and embedding logic.
* **Integration Test**: Run a "Golden Run", verify a memory schema is created, then run a similar task and verify the schema is retrieved.
