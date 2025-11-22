# TARS v2 Architecture Design

## Executive Summary

TARS v2 represents a shift from a monolithic engine to a **Modular, Event-Driven Agentic Platform**. The goal is to decouple the "Brain" (Inference/Reasoning) from the "Body" (Tools/Integrations) and the "Soul" (Agents/Persona).

## 1. High-Level Architecture: The "Micro-Kernel" Approach

The new architecture will follow a **Micro-Kernel** pattern with a **Hexagonal (Ports & Adapters)** architecture for the agents.

### Core Layers

1. **The Nucleus (Kernel)**:
   
   * Responsible for **Agent Lifecycle Management**, **Message Routing**, and **Context Orchestration**.
   * Lightweight, fast, and stable.
   * *Technology*: F# Core, Channels (System.Threading.Channels) for internal async messaging.

2. **The Cortex (Cognitive Layer)**:
   
   * Abstracts the AI/LLM providers.
   * Handles **Inference**, **Grammar Constraining**, and **Reasoning Loops**.
   * *Components*: `TarsInferenceEngine`, `GrammarDistillation`.

3. **The Memory Grid (State Layer)**:
   
   * Unified storage for short-term (Context), long-term (Vector), and structured (Graph) memory.
   * *Components*: `VectorStore`, `BeliefGraph`.

4. **The Exoskeleton (Interface Layer)**:
   
   * API Gateway (gRPC/REST/GraphQL).
   * CLI / TUI.
   * *Components*: `TarsApiServer`, `TarsInferenceCLI`.

## 2. Component Reuse & Migration Strategy

This section details exactly what we keep, what we refactor, and what we build from scratch.

### ♻️ REUSE (Lift & Shift / Minor Refactor)

*These components are high-value and can be ported with minimal changes.*

| Component               | Source Location                             | V2 Destination              | Rationale                                                                                                                               |
|:----------------------- |:------------------------------------------- |:--------------------------- |:--------------------------------------------------------------------------------------------------------------------------------------- |
| **BeliefGraph**         | `src/Tars.Core/BeliefGraph.fs`              | `Tars.Kernel.Memory.Graph`  | **Critical Asset.** The 100KB+ belief graph logic is the core of TARS's structured reasoning. It is pure F# and easily portable.        |
| **CudaKernels**         | `src/TarsEngine.FSharp.Core/CudaKernels.cu` | `Tars.Compute.Cuda`         | **High Performance.** Custom CUDA kernels for tensor operations are valuable and hard to rewrite. Keep as a native library.             |
| **GrammarDistillation** | `src/TarsEngine.GrammarDistillation`        | `Tars.Cortex.Grammar`       | **Unique Capability.** The logic for constraining LLM outputs is robust. We will extract this into a standalone library.                |
| **VectorStore**         | `src/TarsEngine.FSharp.Core/VectorStore`    | `Tars.Kernel.Memory.Vector` | **Standard Utility.** The vector storage logic is reusable, though we may want to abstract the backend (e.g., support Qdrant/Pinecone). |

### 🛠️ REFACTOR (Significant Changes Required)

*These components contain valuable logic but need structural changes to fit the new architecture.*

| Component               | Source Location                                    | V2 Destination            | Changes Needed                                                                                                                                                                                                    |
|:----------------------- |:-------------------------------------------------- |:------------------------- |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **AgentSystem**         | `src/TarsEngine.FSharp.Core/Agents/AgentSystem.fs` | `Tars.Agents.Definitions` | **Decouple.** The *logic* for `Cosmologist`, `DataScientist`, etc., is good, but they are currently hardcoded functions. We need to convert them into data-driven **Agent Definitions** that the Kernel can load. |
| **TarsInferenceEngine** | `src/TARS.AI.Inference/TarsInferenceEngine.fs`     | `Tars.Cortex.Inference`   | **Abstract.** Currently tightly coupled to specific execution paths. Needs to implement a generic `ICognitiveProvider` interface so we can swap models (Llama, GPT-4, Claude) easily.                             |
| **TarsApiServer**       | `src/TarsEngine.FSharp.Core/TarsApiServer.fs`      | `Tars.Interface.Api`      | **Modernize.** Likely needs to be updated to a newer ASP.NET Core version or Giraffe, with cleaner route separation.                                                                                              |

### 🆕 NEW (To Be Coded)

*These are the missing pieces required for the V2 architecture.*

| Component              | Purpose             | Description                                                                                                                               |
|:---------------------- |:------------------- |:----------------------------------------------------------------------------------------------------------------------------------------- |
| **Tars.Kernel**        | Core Orchestration  | The lightweight runtime that hosts agents. Needs a **Plugin System** for loading skills and an **Event Bus** for agent communication.     |
| **SkillRegistry**      | Tool Management     | A system to define, version, and load "Skills" (tools). Agents should request skills from the registry rather than having them hardcoded. |
| **ObservabilityTower** | Tracing & Debugging | A dedicated module for OpenTelemetry tracing. We need to see "inside" the agent's thought process in real-time (Agentic Trace Capture).   |
| **ConfigurationHub**   | Settings Management | Centralized, dynamic configuration (likely using a hot-reloadable `appsettings.json` or env vars) to manage model endpoints and API keys. |

## 3. Detailed Migration Plan

### Phase 1: The Foundation (The Kernel)

1. Create `Tars.Kernel` (F# Class Library).
2. Implement `IAgent` and `ISkill` interfaces.
3. Build the `EventBus` (using `System.Threading.Channels`).

### Phase 2: The Brain (The Cortex)

1. Create `Tars.Cortex` (F# Class Library).
2. Port `GrammarDistillation` logic here.
3. Port `TarsInferenceEngine` logic, wrapping it in `ICognitiveProvider`.

### Phase 3: The Soul (The Agents)

1. Create `Tars.Agents` (F# Class Library).
2. Extract `Cosmologist`, `DataScientist`, `TheoreticalPhysicist` from `AgentSystem.fs`.
3. Rewrite them to implement `IAgent` and use the `EventBus`.

### Phase 4: The Memory (The Grid)

```
# TARS v2 Architecture Design

## Executive Summary

TARS v2 represents a shift from a monolithic engine to a **Modular, Event-Driven Agentic Platform**. The goal is to decouple the "Brain" (Inference/Reasoning) from the "Body" (Tools/Integrations) and the "Soul" (Agents/Persona).

## 1. High-Level Architecture: The "Micro-Kernel" Approach

The new architecture will follow a **Micro-Kernel** pattern with a **Hexagonal (Ports & Adapters)** architecture for the agents.

### Core Layers

1.  **The Nucleus (Kernel)**:
    *   Responsible for **Agent Lifecycle Management**, **Message Routing**, and **Context Orchestration**.
    *   Lightweight, fast, and stable.
    *   *Technology*: F# Core, Channels (System.Threading.Channels) for internal async messaging.

2.  **The Cortex (Cognitive Layer)**:
    *   Abstracts the AI/LLM providers.
    *   Handles **Inference**, **Grammar Constraining**, and **Reasoning Loops**.
    *   *Components*: `TarsInferenceEngine`, `GrammarDistillation`.

3.  **The Memory Grid (State Layer)**:
    *   Unified storage for short-term (Context), long-term (Vector), and structured (Graph) memory.
    *   *Components*: `VectorStore`, `BeliefGraph`.

4.  **The Exoskeleton (Interface Layer)**:
    *   API Gateway (gRPC/REST/GraphQL).
    *   CLI / TUI.
    *   *Components*: `TarsApiServer`, `TarsInferenceCLI`.

## 2. Component Reuse & Migration Strategy

This section details exactly what we keep, what we refactor, and what we build from scratch.

### ♻️ REUSE (Lift & Shift / Minor Refactor)

*These components are high-value and can be ported with minimal changes.*

| Component | Source Location | V2 Destination | Rationale |
| :--- | :--- | :--- | :--- |
| **BeliefGraph** | `src/Tars.Core/BeliefGraph.fs` | `Tars.Kernel.Memory.Graph` | **Critical Asset.** The 100KB+ belief graph logic is the core of TARS's structured reasoning. It is pure F# and easily portable. |
| **CudaKernels** | `src/TarsEngine.FSharp.Core/CudaKernels.cu` | `Tars.Compute.Cuda` | **High Performance.** Custom CUDA kernels for tensor operations are valuable and hard to rewrite. Keep as a native library. |
| **GrammarDistillation** | `src/TarsEngine.GrammarDistillation` | `Tars.Cortex.Grammar` | **Unique Capability.** The logic for constraining LLM outputs is robust. We will extract this into a standalone library. |
| **VectorStore** | `src/TarsEngine.FSharp.Core/VectorStore` | `Tars.Kernel.Memory.Vector` | **Standard Utility.** The vector storage logic is reusable, though we may want to abstract the backend (e.g., support Qdrant/Pinecone). |

### 🛠️ REFACTOR (Significant Changes Required)

*These components contain valuable logic but need structural changes to fit the new architecture.*

| Component | Source Location | V2 Destination | Changes Needed |
| :--- | :--- | :--- | :--- |
| **AgentSystem** | `src/TarsEngine.FSharp.Core/Agents/AgentSystem.fs` | `Tars.Agents.Definitions` | **Decouple.** The *logic* for `Cosmologist`, `DataScientist`, etc., is good, but they are currently hardcoded functions. We need to convert them into data-driven **Agent Definitions** that the Kernel can load. |
| **TarsInferenceEngine** | `src/TARS.AI.Inference/TarsInferenceEngine.fs` | `Tars.Cortex.Inference` | **Abstract.** Currently tightly coupled to specific execution paths. Needs to implement a generic `ICognitiveProvider` interface so we can swap models (Llama, GPT-4, Claude) easily. |
| **TarsApiServer** | `src/TarsEngine.FSharp.Core/TarsApiServer.fs` | `Tars.Interface.Api` | **Modernize.** Likely needs to be updated to a newer ASP.NET Core version or Giraffe, with cleaner route separation. |

### 🆕 NEW (To Be Coded)

*These are the missing pieces required for the V2 architecture.*

| Component | Purpose | Description |
| :--- | :--- | :--- |
| **Tars.Kernel** | Core Orchestration | The lightweight runtime that hosts agents. Needs a **Plugin System** for loading skills and an **Event Bus** for agent communication. |
| **SkillRegistry** | Tool Management | A system to define, version, and load "Skills" (tools). Agents should request skills from the registry rather than having them hardcoded. |
| **ObservabilityTower** | Tracing & Debugging | A dedicated module for OpenTelemetry tracing. We need to see "inside" the agent's thought process in real-time (Agentic Trace Capture). |
| **ConfigurationHub** | Settings Management | Centralized, dynamic configuration (likely using a hot-reloadable `appsettings.json` or env vars) to manage model endpoints and API keys. |

## 3. Detailed Migration Plan

### Phase 1: The Foundation (The Kernel)

1.  Create `Tars.Kernel` (F# Class Library).
2.  Implement `IAgent` and `ISkill` interfaces.
3.  Build the `EventBus` (using `System.Threading.Channels`).

### Phase 2: The Brain (The Cortex)

1.  Create `Tars.Cortex` (F# Class Library).
2.  Port `GrammarDistillation` logic here.
3.  Port `TarsInferenceEngine` logic, wrapping it in `ICognitiveProvider`.

### Phase 3: The Soul (The Agents)

1.  Create `Tars.Agents` (F# Class Library).
2.  Extract `Cosmologist`, `DataScientist`, `TheoreticalPhysicist` from `AgentSystem.fs`.
3.  Rewrite them to implement `IAgent` and use the `EventBus`.

### Phase 4: The Memory (The Grid)

1.  Create `Tars.Memory` (F# Class Library).
2.  **Copy `BeliefGraph.fs`** (Lift & Shift).
3.  Port `VectorStore` logic.

### Phase 5: The Body (The Interface)

1.  Create `Tars.Server` (Web API).
2.  Wire up the Kernel to the API endpoints.

## 5. Advanced Capabilities & Esoteric Components

This section details how the advanced, non-standard components fit into the V2 architecture.

### 🌀 FLUX (The "Language of Thought")
*   **Current State**: `FluxIntegrationEngine.fs` (Tier 5). A multi-modal metascript engine supporting Wolfram, Julia, and F# Type Providers.
*   **V2 Placement**: **`Tars.Cortex.Flux`**.
*   **Role**: The primary "Reasoning Runtime" for agents. Instead of just "text-in/text-out", agents will generate **FLUX Metascripts**.
    *   *Example*: An agent needing to solve a differential equation will generate a `Wolfram` FLUX script, which the Cortex executes via the Wolfram engine, returning the result to the agent's context.

### 📐 Hyper-Complex Geometric DSL
*   **Current State**: `HyperComplexGeometricDSL.fs` (Tier 4). F# Computational Expressions for Sedenions (16D numbers) and Non-Euclidean geometry.
*   **V2 Placement**: **`Tars.Core.Mathematics`**.
*   **Role**: The fundamental mathematical primitives for the Memory Grid.
    *   **Non-Euclidean Vector Store**: Vector embeddings will not just be flat arrays. They will be stored in **Hyperbolic Space** (Poincaré disk) using this DSL to optimize for hierarchical data (like taxonomies or code trees).

### 🏭 AI-Enhanced Closure Factories
*   **Current State**: `AIEnhancedClosureFactory.fs`. Uses AI to write optimized F# functions (closures) at runtime.
*   **V2 Placement**: **`Tars.Kernel.Evolution`**.
*   **Role**: The "Self-Modification" engine.
    *   *Usage*: When an agent identifies a repetitive task, it requests the Kernel to "compile" a skill. The Closure Factory generates a highly optimized F# function (e.g., a specific data pipeline), compiles it, and hot-loads it into the `SkillRegistry`.

### 🕸️ Graphiti (The Knowledge Graph)
*   **Current State**: `BeliefGraph.fs` (Concept).
*   **V2 Placement**: **`Tars.Memory.Graphiti`**.
*   **Role**: The structured "Long-Term Memory".
    *   **Evolution**: Will move beyond simple nodes/edges to support **Hypergraph** structures (edges connecting edges) and **Temporal Beliefs** (facts that are only true for a specific duration).
    *   **Integration**: Will be tightly coupled with the **Non-Euclidean Vector Store**—nodes in the graph will have embeddings in Hyperbolic space to encode their semantic hierarchy.

### 🧬 Computational Expression Factories
*   **Current State**: Implicit in `DSL` and `Flux`.
*   **V2 Placement**: **`Tars.Core.Computation`**.
*   **Role**: Syntactic sugar for complex operations. We will expose these as a library so users can write "Agentic Workflows" using F# CE syntax (e.g., `agent { let! thought = ... }`).

## 6. Observability & Evolution

### 🕵️ Agentic Traces
*   **Current State**: `AgenticTraceCapture.fs`. A comprehensive system capturing every thought, decision, and state change.
*   **V2 Placement**: **`Tars.Kernel.Tracing`**.
*   **Role**: The "Black Box Recorder" for agents.
    *   **Granularity**: Captures not just logs, but **Architecture Snapshots**, **Grammar Evolution Events**, and **Inter-Agent Communication** payloads.
    *   **Visualization**: Traces will be exportable to OpenTelemetry and a custom React-based **"Thought Player"** that allows replaying an agent's reasoning process step-by-step.

### 🔄 TARS Auto-Evolution (Self-Modification)
*   **Current State**: `SelfModificationEngine.fs`.
*   **V2 Placement**: **`Tars.Kernel.Evolution`**.
*   **Role**: The engine that allows TARS to rewrite its own code.
    *   **Mechanism**: Agents can identify performance bottlenecks or missing capabilities. They submit a "Modification Request" to the Kernel.
    *   **Process**: The `SelfModificationEngine` uses LLMs to generate optimized F# code (e.g., a faster sorting algorithm or a new API client), compiles it into a dynamic assembly, and hot-swaps it into the running process.
    *   **Safety**: All self-modifications run through a "Sandbox Validator" before being applied.

### 🔗 Triple Stores: Ingestion & Production
*   **Current State**: `RdfTripleStore.fs` (Supports Virtuoso, File, Remote SPARQL).
*   **V2 Placement**: **`Tars.Integration.Rdf`**.
*   **Role**: The bridge between the Semantic Web and TARS.
    *   **Ingestion**: Agents can "read" the web as a graph. TARS can ingest RDF/Turtle files or query remote SPARQL endpoints (like Wikidata or DBpedia) to populate its `BeliefGraph`.
    *   **Production**: TARS doesn't just consume knowledge; it produces it. Agent reasoning chains can be serialized as RDF triples (Subject-Predicate-Object) and published back to a Triple Store, effectively allowing TARS to contribute to the global Semantic Web.

## 7. Agentic Engineering Patterns (The "AI Operating System")

This layer introduces bounded rationality into the system, treating the LLM not as a wizard, but as a stochastic CPU unit that requires deterministic guards.

### Layer 1: The Resource Controller (The "Wallet")
*Controls the raw fuel (tokens, money, time).*

*   **Token Budget Governor**:
    *   **Concept**: Prevents financial infinite loops (`while(goal_not_met)`).
    *   **Implementation**: A `Budget` object passed through the context.
    *   **Refinement**: **"Low Battery Mode"**. When `Budget.Remaining < 10%`, the agent switches to a cleanup routine (log state, exit) rather than starting new reasoning chains.
*   **Cognitive Bulkheads**:
    *   **Concept**: Isolates resource pools to prevent priority inversion.
    *   **Implementation**: Separate API keys/quotas for "Primary" (User-facing), "Critical" (Safety), and "Experimental" (Background) tasks.

### Layer 2: The Control Flow (The "Manager")
*Decides how to think.*

*   **Semantic Fan-out Limiter**:
    *   **Concept**: Prevents "Bureaucracy" where scoring takes more tokens than doing.
    *   **Implementation**: **Tiered Model Approach**. Use cheap models (GPT-4o-mini) to score/filter subtasks, passing only top-K to the expensive model (Claude 3.5 Sonnet/GPT-4o).
*   **Uncertainty-Gated Planner**:
    *   **Concept**: Solves the "Sledgehammer" problem and overconfidence.
    *   **Implementation**: Instead of asking "Confidence Score?", ask **"List 3 reasons why this might fail."** If the model can easily list 3, force the uncertainty score up and switch to a more robust planning mode.
*   **Semantic Watchdog**:
    *   **Concept**: Detects infinite loops that *look* like progress.
    *   **Implementation**: **State Awareness**. The watchdog monitors the agent's internal memory changes. If the reasoning looks new but the state ("Processed files: [A, B]") hasn't changed, it trips the circuit breaker.

### Layer 3: The Data Plane (The "Memory")
*Manages context window pressure.*

*   **Context Compaction Pipeline**:
    *   **Concept**: Prevents "Lossy Compression" of critical data.
    *   **Implementation**: **Anchor Points**. Keep the original `Goal` and the most recent `Error Message` raw and uncompressed. Only compress intermediate reasoning steps.
*   **Triage Summarizer**:
    *   **Concept**: RAG 2.0.
    *   **Implementation**: Retrieve Top-100 chunks -> Compress into a "Briefing" -> Let LLM decide which 5 to read in full.
```
