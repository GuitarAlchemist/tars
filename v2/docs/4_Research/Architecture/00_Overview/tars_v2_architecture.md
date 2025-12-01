# TARS v2 Architecture Design

## Executive Summary

TARS v2 represents a shift from a monolithic engine to a **Modular, Event-Driven Agentic Platform**. The goal is to decouple the "Brain" (Inference/Reasoning) from the "Body" (Tools/Integrations) and the "Soul" (Agents/Persona).

## 1. High-Level Architecture: The "Micro-Kernel" Approach

The new architecture will follow a **Micro-Kernel** pattern with a **Hexagonal (Ports & Adapters)** architecture for the agents.

### Core Layers

1. **The Nucleus (Kernel)**:

   * Responsible for **Agent Lifecycle Management**, **Semantic Bus Orchestration**, and **Protocol Enforcement**.
   * Acts as the "Physics Engine" of the agent world, enforcing constraints (tokens, time, complexity) carried in messages.
   * *Technology*: F# Core, `System.Threading.Channels`, **Semantic Message Protocol**.

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

### 1.1 The Semantic Bus Protocol

 Instead of simple DTOs, agents communicate via a **Semantic Envelope** (Hybrid Approach). This ensures that *intent*, *constraints*, and *context* are first-class citizens.

#### The Envelope Structure

 ```fsharp
 // The "Verb" of the message (FIPA-ACL inspired)
 type Performative = 
     | Request | Inform | Query | Propose | Refuse | Failure | NotUnderstood
 
 // The "Guardrails"
 type SemanticConstraints = {
     MaxTokens: int option
     MaxComplexity: string option
     Timeout: TimeSpan option
     KnowledgeBoundary: string list
 }
 
 // The Envelope
 type SemanticMessage<'TContent> = {
     Id: Guid
     CorrelationId: Guid
     Sender: AgentId
     Receiver: AgentId option
     Performative: Performative
     Constraints: SemanticConstraints
     Ontology: string option    // NEW: domain context (e.g., "coding", "finance")
     Language: string           // NEW: content type (e.g., "json", "fsharp")
     Content: 'TContent
     Metadata: Map<string,obj>
 }
 ```

* **JSON-LD Serialization**: Messages are serialized as JSON-LD to ensure semantic interoperability and potential future integration with external knowledge graphs.
* **Constraint Enforcement Point**: The Kernel runs a `SemanticEnvelopeGuard` on ingress (EventBus middleware) and again at execution start. Budgets/timeouts/knowledge-boundaries are reduced as they flow; violations are logged and downgraded to `Failure` envelopes (never silent drop).
* **Size Discipline**: JSON-LD is opt-in per envelope; envelopes carry a byte-budget header and are gzip’d when crossing process boundaries.

### 1.3 Agentic RAG Path (Memory Grid Realization)

1. **Ingestion**: Documents are chunked (size + semantic splits), stamped with schema (`source`, `kind`, `createdAt`, `version`, `scope`, `provenance`), deduped, and throttled via the Memory Ingress Governor (shared with the BudgetGovernor).
2. **Indexing**: Chunks go to `VectorStore`; entities/relations go to `BeliefGraph` (GraphAnalyzer builds candidates). Both use the same IDs and versions.
3. **Retrieval**: Given a query, we run hybrid retrieval: vector search + graph neighborhood pull; merge on ID; apply cheap reranker (LLM-lite or cosine-on-summary) and K-top fan-out.
4. **Context Assembly**: Apply the envelope’s `SemanticConstraints` (token/time) to budget the assembled context; compress long tails; emit a compact, provenance-tagged context block with scores.
5. **Provenance & Audit**: Every returned chunk includes `source`, `version`, `score`, and `why` (match rationale). Missing/empty retrieval degrades gracefully and logs to the Governance channel.

### 1.2 Agentic Interfaces: Soft Semantic Contracts

 **Reference:** `docs/2_Analysis/Architecture/agentic_interfaces.md`

 TARS v2 implements interfaces as a **Triad**:

 1. **Hard Shell**: `ExecutionOutcome<'T>` (Success/Partial/Failure) enforced by F# compiler.
 2. **Soft Semantics**: Capability-based routing and natural language contracts.
 3. **Empirical Layer**: Routing based on historical success rates.

## 2. Component Reuse & Migration Strategy

 (See original doc for details - no changes here)

 ...

## 7. Agentic Engineering Patterns (The "Cognitive Circuit Board")

 This layer introduces bounded rationality into the system, modeled after **Electrical Circuits**.

### Layer 1: Resistors (Throttling & Backpressure)

* **Token Budget Governor**:
  * **Role**: Prevents resource exhaustion.
  * **Implementation**: A `BudgetGovernor` tracks tokens, money, and time. It is passed down the call stack.
  * **Behavior**: If budget is low, agents switch to "Low Battery Mode" (summarize/exit).
* **Bounded Channels**:
  * **Role**: Prevents queue explosion.
  * **Implementation**: `System.Threading.Channels` with fixed capacity. Producers await if consumers are slow.

### Layer 2: Capacitors (Buffering & Memory)

* **Context Compaction**:
  * **Role**: Manages context window pressure.
  * **Implementation**: Intermediate reasoning steps are compressed; only Anchors (Goal, Error) are kept raw.
* **Buffer Agents**:
  * **Role**: Smooths bursty traffic.
  * **Implementation**: Agents that accumulate messages over time (or count) before flushing to a downstream processor.

### Layer 3: Transistors (Gating & Logic)

* **Semantic Fan-out Limiter**:
  * **Role**: Prevents "Bureaucracy".
  * **Implementation**: Use cheap models to score subtasks, passing only Top-K to expensive models.
* **Uncertainty Gates**:
  * **Role**: Stops overconfident hallucinations.
  * **Implementation**: If "Reasons to fail" are high, the gate closes or redirects to a robust planner.
* **Semantic Watchdog**:
  * **Role**: Detects infinite loops.
  * **Implementation**: Monitors state changes. If reasoning continues but state is static, it trips the circuit breaker.
