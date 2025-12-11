# TARS v2 Implementation Plan

**Date:** November 26, 2025
**Status:** Approved
**Goal:** Build TARS v2.0 - A Modular, Secure, and Agentic AI System.

---

## 🎯 v2.0 Alpha Scope (The "Cut Line")

**Required for v2.0-alpha:**

* ✅ **Phase 1 (Foundation):** Kernel, EventBus, Docker Sandbox, Security Core.
* ✅ **Phase 2 (Brain - Minimal):**
  * One LLM provider (OpenAI or Ollama).
  * One Vector Store collection (ChromaDB).
  * Minimal "Garden Shed" Grammar (parse simple goals).
* ✅ **Phase 3 (Body - Partial):** Terminal UI, Basic Tool Registry.
* ✅ **CLI:** `tars run script.trsx` (basic execution).

**Deferred to v2.x (Nice-to-have):**

* ❌ Full MCP Client (start with hardcoded tools).
* ❌ AutoGen Bridge (start with native F# agents).
* ❌ Complex Cost Budgeting.
* ❌ Web UI / Fancy TUI Dashboards.

---

## � The TARS Constitution (New in v2.1)

**Core Insight:** To prevent entropy in a self-evolving system, TARS needs a "Constitution" — a set of inviolable laws that precede modules.

### Phase 1: The Foundation (Kernel & Security)

**Goal:** Establish the secure runtime environment and core message passing.
**Acceptance Criteria:**

* [x] `dotnet run --project src/Tars.Interface.Cli -- demo-ping` works.
* [x] Kernel spins up `EventBus`.
* [x] Demo agent subscribes, receives message, and logs it.
* [x] **Golden Run:** 1 test that runs CLI, captures trace, and replays it.

* [x] **1.1 Project Setup**: Initialize `Tars.sln` with F# structure (Kernel, Core, Interface).
* [x] **1.2 Tars.Kernel**: Implement `EventBus` (System.Threading.Channels) and `IAgent`.
* [x] **1.3 Docker Sandbox**: Create `tars-sandbox` image (read-only fs, no network).
* [x] **1.4 Security Core**: Implement `CredentialVault` and `SandboxedProcess`.

### Phase 2: The Brain (Inference & Memory)

* [x] **2.4.3 Persistence**: Add JSON serialization for the graph.
* [x] **2.4.4 Querying**: Implement basic traversal (e.g., "Find all files created by Agent X").
  * *Status:* Working loop in `Evolve.fs`. Results are saved to `InMemoryVectorStore` (ephemeral).

### Phase 5: The Mind (Metascript Engine)

**Goal:** Enable complex, multi-step workflows defined in a DSL.

* [x] **5.1 Metascript DSL**: Define the schema for `.tars` files (JSON).
* [x] **5.2 Workflow Engine**: Implement the engine to parse and execute these scripts.
* [x] **5.3 CLI Integration**: `tars run <script>` to execute workflows.
  * *Status:* Working! `tars run sample.tars` executes multi-step workflows.

### Phase 6: Cognitive Architecture (Backpressure & Resource Control)

**Goal:** Implement AI-native backpressure patterns to prevent cognitive overload and budget explosions.

**Research Basis:**

* `docs/__research/ChatGPT-Backpressure in AI systems.md` (12 AI Design Patterns)
* `docs/__research/ChatGPT-Multi-agent system protocols.md` (Semantic Speech Acts)

**Design Principles:**

1. **"Semantic Load is Real"**: Tokens, LLM calls, and thinking depth are finite resources
2. **"Budget Everything"**: Every workflow must have resource limits
3. **"Fail Gracefully"**: When budget exhausted, return "best effort + limitations"
4. **"Compress Don't Drop"**: Summarize instead of losing signal
5. **"Immutability by Default"**: Version everything; never mutate in place
6. **"Time is First-Class"**: All state decays; prune aggressively

#### Phase 6.0: Architecture Hardening (The Constitution)

**Priority**: Critical (Immediate)

**Rationale**: Enforce immutability and safety before adding more complexity.

**Reference**: `docs/3_Roadmap/phase6_integration_strategy.md`

**Tasks:**

* [~] **Repo Structure**: Align folders (deferred - functional as-is)
* [x] **Type Safety**: Define strict F# DUs for `AgentState`, `Message`, `SkillResult`.
* [x] **Versioning**: Add `Version` and `ParentVersion` to `Agent`, `Skill`, `TaskDefinition`.
* [x] **Safety Gates**: Implement `SafetyGate` module in Kernel (Static Check -> Test -> Sandbox).

  ```fsharp
  type Performative = Request | Inform | Query | Propose | Refuse | Failure

  type SemanticConstraints = {
      MaxTokens: int option
      MaxComplexity: string option // "O(n)"
      Timeout: TimeSpan option
  }

  type SemanticMessage<'T> = {
      Id: Guid
      Sender: AgentId
      Receiver: AgentId
      Performative: Performative
      Constraints: SemanticConstraints
      Content: 'T
  }
  ```

* [x] Refactor `EventBus` to be a **Semantic Bus** that routes `SemanticMessage<'T>`.
* [x] Implement **Constraint Enforcement** middleware in the Kernel (reject messages that violate constraints).
* [x] Update Evolution Engine to use speech acts (Curriculum → Request, Executor → Inform).
* [x] Add telemetry/logging for all agent interactions by intent type

**Acceptance Criteria:**

* All agent messages have explicit `Intent`
* Logs show clear interaction patterns: "Curriculum ASKed → Executor TOLD"
* Tests validate routing by intent

#### Phase 6.1: Budget Governor (Resource Control)

* **Status**: Complete
* **K-Theory Integration**:
  * ✅ **K0 (Conservation)**: Validated via `KTheoryTests.fs`.
  * ✅ **K1 (Topology)**: Validated via `GraphAnalyzer.detectCycles`.

* **Resistors** = throttling (bounded queues, rate limits)

* [x] **Bounded Message Channels**: Implemented in `Tars.Core/BoundedChannel.fs`
* [x] Update `EventBus` to use bounded channels with configurable capacity
* [x] Add backpressure signals when channels are full
* [x] Implement adaptive throttling: slow down producers when consumers lag

##### Phase 6.7.2: Capacitors (Buffering & Batching)

* [x] **Buffer Agent**: Implemented in `Tars.Kernel/Capacitor.fs`
* [x] Use MailboxProcessor for automatic batching
* [x] `ContextCompressor` compresses message histories

* [ ] Implement working memory as a capacitor:
  
  ```fsharp
  type WorkingMemory<'T> = {
      Items: 'T Queue
      MaxSize: int
      DecayFn: 'T -> TimeSpan -> float  // importance decay
  }
  
  member memory.Add(item: 'T) : unit
  member memory.Prune() : unit  // Remove low-importance items
  ```

##### Phase 6.7.3: Transistors (Gating & Conditional Flow)

* [x] **Task Dependency Gates**: Implemented in `Tars.Kernel/Transistor.fs` and `Tars.Core/Gates.fs`
* [x] `JoinGate` for multi-input gates (wait for multiple signals)
* [x] **MutexGate**: Implemented with `TryAcquire`, `Release`, `WithLockAsync`

**Rationale**: Regulate cognitive load and ensure reasoning quality.

**Tasks**:

* [x] **6.7.1 Cognitive Analysis**:
  * `CognitiveAnalyzer`: Measure Eigenvalue (stability) and Entropy (disorder) from real agent states
  * `ContextCompression`: LLM-powered summarization with auto-compression on entropy spikes
* [x] **6.7.2 Epistemic Governance**:
  * `EpistemicGovernor`: Verify beliefs, generate variants, extract principles
  * `IAgentRegistry.GetAllAgents()`: Support global agent state analysis
* [x] **6.7.3 Integration**:
  * Evolution Engine curriculum generation influenced by epistemic guidance
  * Principle extraction from successful task completions stored in vector database
  * Metrics infrastructure for observability (agent workflow + budget tracking)

**Acceptance Criteria**:

* ✅ Context compression achieves >50% token reduction
* ✅ Cognitive state reflects actual agent activity (real-time calculation)
* ✅ Epistemic Governor influences curriculum generation
* ✅ Principles extracted and stored from successful tasks
* ✅ All tests passing (172/172 active tests)

#### Phase 6.8: The Epistemic Governor (Anti-Hack / Pro-Learning)

**Priority**: High (v2.2)

**Rationale**: Prevent overfitting and enforce deep learning.

**Reference**: `docs/2_Analysis/Architecture/epistemic_governor.md`

**Tasks:**

* [x] **6.8.1 Variant Generator**:
  * `EpistemicGovernor.GenerateVariants(task)`
  * Perturbs the task (change inputs, constraints) to test generalization.
* [x] **6.8.2 Principle Extractor**:
  * `EpistemicGovernor.ExtractPrinciple(solution)`
  * LLM prompt to distill "Why it works" into a `Belief`.
* [x] **6.8.3 Integration with Evolution Loop**:
  * Hook into `Evolution.Engine.step`.
  * Store extracted beliefs in `VectorStore` (temporary Belief Store).
* [x] **6.8.4 Curriculum Feedback**:
  * Use `BeliefGraph` density to guide `CurriculumAgent`.
  * "We have too many beliefs about 'Sorting', but few about 'Networking'. Generate networking tasks."

**Acceptance Criteria:**

* Solutions must pass N=3 variations of the task.
* Every solved task must produce at least one new `Belief` node.

### Phase 1: The Foundation (✅ Complete)

**Evidence**: `dotnet run --project src/Tars.Interface.Cli -- demo-ping` works.

* [x] Project Setup
* [x] EventBus with Channels
* [x] Docker Sandbox
* [x] Security Core (CredentialVault, FilesystemPolicy)
* [x] Golden Run test

### Phase 2: The Brain (Partial)

**Completed:**

* [x] **2.1 LLM Integration**: Ollama + vLLM clients, routing, embeddings
* [x] **2.3 Grammar Engine**: Basic EBNF parser for goals/tasks

**Deferred:**

* [x] **2.2 Memory Grid**: Simple Persistence (SQLite/File)
  * *Decision*: Start simple (SQLite/File) before adding complex Vector DBs.
  * *Future*: Evaluate ChromaDB/GAM only when scale demands it.
* [x] **2.4 Internal Knowledge Graph**: Graphiti-style internal graph
  * *Decision*: Drop external Triple Stores. Use internal graph for beliefs/lineage.
  * *Status*: Core temporal graph implemented in `Tars.Core/KnowledgeGraph.fs`. Integration pending.

#### Phase 2.5: Epistemic RAG (Graphiti)

**Goal**: Enable semantic understanding of the codebase structure and "Belief" management.

* [x] **2.5.1 Code Graph Ingestion**:
  * Use F# Compiler Services / Roslyn to parse code.
  * Extract nodes: `Module`, `Type`, `Function`, `Value`.
  * Extract edges: `Calls`, `Inherits`, `DependsOn`.
* [x] **2.5.2 Belief Store**:
  * `BeliefGraph.fs` - Full graph with edges, status tracking, relations
  * Evolution Engine stores principles via `EpistemicGovernor.ExtractPrinciple`
  * Beliefs stored in VectorStore collection `tars-beliefs`
* [x] **2.5.3 Hybrid Retrieval**:
  * Query: "Find all functions that call `LlmService.generate` and are not async."
  * Mechanism: Graph traversal + Vector similarity.
* [x] **2.5.4 Advanced RAG Capabilities**:
  * **Hybrid Search**: BM25 + Cosine Similarity.
  * **Query Routing**: Classify queries (Factual, Analytical, Conversational).
  * **Time Decay**: Prioritize fresher documents.
  * **Multi-Hop**: Traverse Knowledge Graph for deep answers.
  * **Metadata Filtering**: Precise context narrowing.

### Phase 3: The Body (Partial)

**Completed:**

* [x] **3.1 Terminal UI**: Spectre.Console with splash screen, spinner, interactive chat
* [x] **3.2 MCP Client**: Model Context Protocol for external tools
  * *Status*: Implemented basic client, stdio transport, and CLI command.

**Deferred:**

* [x] **3.3 Tool Registry**: Implemented in `Tars.Tools/Registry.fs`
  * Circuit breakers per-tool (line 15-40)
  * Resilient execution via `CircuitBreaker.ExecuteAsync`
* [x] **3.4 Cost Budget**: Implemented via `BudgetGovernor` (Phase 6.1)

### Phase 4: The Soul (✅ Complete)

**Evidence**: `tars evolve` runs the full co-evolution loop.

* [x] **4.1 Tars.Evolution Project**: Project structure created
* [x] **4.2 The Protocol**: `TaskDefinition`, `TaskResult` DUs
* [x] **4.3 Curriculum Agent**: Generates tasks using `qwen2.5-coder:1.5b`
* [x] **4.4 Executor Agent**: Solves tasks using `Graph.step` and `LlmService`
* [x] **4.5 The Loop**: Curriculum → Task → Executor → Result → Memory (InMemoryVectorStore)

### Phase 5: The Mind (✅ Complete)

**Evidence**: `tars run sample.tars` executes multi-step workflows.

* [x] **5.1 Metascript DSL**: JSON schema with agent + tool steps
* [x] **5.2 Workflow Engine**: Variable resolution, step execution, context passing
* [x] **5.3 CLI Integration**: `tars run <script>` command

### Phase 6: Cognitive Architecture (✅ Complete)

**Status**: Implemented in v2.1.

**Design Patterns** (from research):

1. Token Budget Governor ← **Complete** (BudgetGovernor)
2. Semantic Fan-out Limiter ← **Complete** (EventBus bounded channels)
3. Adaptive Reflection Loop ← **Complete** (ReflectionAgent)
4. Context Compaction Pipeline ← **Complete** (ContextCompressor.AutoCompress)
5. Uncertainty-Gated Planner ← **Complete** (UncertaintyGatedPlanner.fs)
6. Consensus Circuit Breaker ← **Complete** (ConsensusCircuitBreaker.fs)
7. Tool Circuit Breaker ← **Complete** (Registry.fs)
8. Semantic Watchdog ← **Complete** (SemanticWatchdog.fs)

**Implementation Priority:**

* **v2.1 (Alpha)**: 6.1, 6.2, 6.3, 6.4 (core backpressure)
* **v2.2 (Beta)**: Remaining patterns + MCP integration

---

## 📋 Architecture Decision Records (Key Decisions)

### ADR-001: F#-First Architecture

**Status**: Accepted  
**Rationale**: Functional programming ensures correctness and composability for AI orchestration.  
**Consequences**: Some integrations require F#/C# interop; Python tools need subprocess calls.

### ADR-002: In-Process First, Network Later

**Status**: Accepted  
**Rationale**: Faster iteration, easier debugging. Network transport (gRPC) added when multi-machine needed.  
**Implementation**: `EventBus` uses `Channel<T>`, future: gRPC backend via `IAgentBus` abstraction.

### ADR-003: InMemoryVectorStore for MVP

**Status**: Accepted (Temporary)  
**Rationale**: Simplifies Phase 4 implementation; sufficient for validation.  
**Future**: Switch to ChromaDB (v2.2) or GAM (v2.3) for production.

### ADR-004: Token Budget as First-Class Citizen

**Status**: Accepted  
**Rationale**: Research shows unbounded LLM calls lead to runaway costs and cognitive overload.  
**Implementation**: Phase 6.1 `BudgetGovernor` gates all workflows.

### ADR-005: Agent Speech Acts as Semantic Protocol

**Status**: Accepted  
**Rationale**: Aligns with emerging standards (LSP-style, JSON-RPC). Enables tracing, telemetry, and future remote agents.  
**Implementation**: Phase 6.2 `AgentIntent` DU in `Tars.Kernel`.

### ADR-006: Metascript as JSON (not YAML/Markdown)

**Status**: Accepted  
**Rationale**: Strong typing, tooling support, faster parsing.  
**Consequences**: Less human-friendly than YAML; acceptable for v2.0.

---

## 🎯 Success Criteria

### v2.0 Alpha (Current Target)

* [x] Phase 1: Foundation ✅
* [x] Phase 2: Brain (Partial) ✅
* [x] Phase 3: Body (Partial) ✅

* [x] Phase 4: Evolution Loop ✅
* [x] Phase 5: Metascript Engine ✅
* [x] Phase 6.1: Budget Governor (See `docs/QA/Phase6_AcceptanceCriteria.md`)
* [x] Phase 6.2: Speech Acts (See `docs/QA/Phase6_AcceptanceCriteria.md`)
* [x] Phase 6.3: Fan-out Limiter (See `docs/QA/Phase6_AcceptanceCriteria.md`)
* [x] Phase 6.4: Adaptive Reflection (See `docs/QA/Phase6_AcceptanceCriteria.md`)
* [x] Phase 6.5: Agentic Interfaces (Capability Store, Agent Capabilities, Workflow CE)

**Exit Criteria:**

1. `tars evolve` runs 10 generations without budget overrun
2. `tars run <script>` executes complex workflows with budgets
3. All tests pass (`dotnet test`)
4. Documentation complete for all implemented features

### v2.1 Beta (Next Milestone)

* [ ] Phase 2.2: Persistent Memory (ChromaDB)
* [ ] Phase 3.2: MCP Client
* [ ] Phase 6.5-6.8: Advanced backpressure patterns
* [x] Phase 6.6: Semantic Message Bus (JSON-LD + FIPA-ACL)
* [x] Phase 6.7: Circuit Flow Control (Resistors, Capacitors, Transistors)
* [ ] Benchmark: Evolution Loop on standard tasks (HotpotQA, LoCoMo)

### v2.2 Production

* [ ] GAM integration (if benchmarks validate)

* [ ] gRPC + NATS for multi-machine agents
* [ ] Full observability (traces, metrics, costs)
* [ ] Self-improvement: TARS improves its own code

---

## 🔬 Research Integration

### Research Documents Analyzed

1. ✅ `ChatGPT-Multi-agent system protocols.md` → Phase 6.2 (Speech Acts)
2. ✅ `ChatGPT-Backpressure in AI systems.md` → Phase 6.1-6.4 (Backpressure Patterns)
3. ✅ `ChatGPT-AI semantic bus research.md` → Phase 6.6 (Semantic Message Bus)
4. ✅ `ChatGPT-Circuit-based AI architecture.md` → Phase 6.7 (Circuit Flow Control)
5. ✅ `ChatGPT-Agentic AI Interfaces.md` → Phase 6.5 (Agentic Interfaces)
6. ⏳ `ChatGPT-K-theory and TARS query.md` → Future (theoretical foundations)
7. ⏳ `Grothendieck Groups and their Application to AI.md` → Future (category theory)

### Key Insights Applied

* **Backpressure Patterns**: 12 design patterns → 4 implemented in Phase 6
* **Semantic Bus**: Move from DTOs to rich semantic messages with context, constraints, and intent
* **Circuit Analogies**: Resistors (throttling), Capacitors (buffering), Transistors (gating) for flow control
* **Emerging Standards**: OpenAI-compatible REST + MCP + JSON-RPC → roadmap alignment
* **Semantic Load**: Tokens/calls/time as first-class resources → `BudgetGovernor`
* **Speech Acts**: ASK/TELL/PROPOSE → `AgentIntent` DU (extended with FIPA-ACL semantics)
* **Pre-LLM Pipeline**: Transformer stages before main LLM (safety, intent, summarization, rewriting)

### Research-Driven Decisions

1. **Budget Governor** (from Pattern #1: Token Budget Governor)
2. **Fan-out Limiter** (from Pattern #2: Semantic Fan-out Limiter)
3. **Adaptive Reflection** (from Pattern #3: Adaptive Reflection Loop)
4. **Agent Protocol** (from Multi-agent protocols analysis)
5. **Semantic Message Bus** (from AI semantic bus research - JSON-LD + FIPA ACL)
6. **Circuit-Inspired Flow Control** (from circuit-based architecture - resistors, capacitors, transistors)
7. **Pre-LLM Transformer Pipeline** (from circuit architecture - staged processing)

---

## 🚀 Next Actions

### Immediate (Next Session)

1. **Phase 3.2: MCP Client**: ✅ Completed.
2. **Phase 2.4: Internal Knowledge Graph**: Core temporal graph implemented. Integration pending.
3. **Benchmarks**: Run Evolution Loop on standard tasks.

### Short-Term (This Week)

1. Harden integration of all Phase 6 components.
2. Write v2.0 Alpha release notes.

### Medium-Term (This Month)

1. v2.1 Beta release.
2. Evaluate GAM integration.

### Completed (Phase 6.8)

* [x] **Epistemic Governor**: Implemented `GenerateVariants`, `ExtractPrinciple`, and Evolution Loop integration.
* [x] **Tests**: Verified with `GoldenRun` and `EvolutionTests`.

### Phase 1: The Foundation

#### 1.1 Project Setup & Kernel

* **Action**: Create `Tars.sln`, `Tars.Kernel`, `Tars.Core`, `Tars.Interface.Cli`.
* **Code**: Implement `IMessage`, `IAgent`, and `EventBus` (using `System.Threading.Channels`).
* **Demo**: Create a `DemoAgent` that logs messages to Serilog.

#### 1.2 CLI & Golden Run

* **Action**: Implement `tars demo-ping` command.
* **Flow**: CLI -> EventBus -> DemoAgent -> Log.
* **Test**: Create one Golden Run test that captures this flow and asserts success.

#### 1.3 Docker Sandbox

* **Action**: Create `docker/sandbox/Dockerfile`.
* **Specs**: Python 3.11, non-root user `tars`, read-only filesystem.
* **Verify**: Run a simple python script inside the container from F#.

#### 1.4 Security Core

* **Action**: Implement `Tars.Security`.
* **Features**: `CredentialVault` (Env vars) and `FilesystemPolicy` (path allowlisting).

---

### Phase 2: The Brain

#### 2.1 LLM Integration

* **Action**: Implement `Tars.Cortex.Grammar`.
* **Scope**: Parse a simple `.trsx` file with 2-3 block types (Goal, Task).
* **Output**: Generate strongly-typed F# AST messages.

#### 2.4 Temporal Knowledge Graph (Graphiti Integration)

* **Goal**: Implement a bi-temporal knowledge graph for belief tracking and pattern recognition.
* **Reference**: `docs/2_Analysis/Research/graphiti_integration_research.md`

* [x] **2.4.1 Core Temporal Graph**: Implemented in `TemporalKnowledgeGraph.fs`
  * `TemporalNode` with `Validity` (ValidFrom/InvalidAt)
  * `TemporalEdge` with temporal tracking
  * Episode ingestion in `IngestEpisode`
* [x] **2.4.2 Semantic Layer**: Implemented in `EntityExtractor.fs`
  * Entity resolution via `EntityResolver.resolveEntities`
  * Fact extraction via `FactExtractor` module
  * Contradiction detection via `ProcessInvalidation`
* [x] **2.4.3 Pattern Recognition**: Implemented in separate modules
  * `CommunityDetection.fs` - Label Propagation algorithm
  * `PatternRecognition.fs` - Structural/behavioral tagging
* [x] **2.4.4 Grammar Distillation**: Implemented in `GrammarDistillation.fs`
  * `GrammarDistiller` extracts rules from patterns
  * `HotReloadManager` for hot-reload support

---

## 🚀 First Coding Session Checklist

1. [x] Create `Tars.sln` and projects.
2. [x] Implement `EventBus` and `IAgent`.
3. [x] Create `DemoAgent` (logs to console).
4. [x] Implement `tars demo-ping` CLI command.
5. [x] Write 1 Golden Run test for `demo-ping`.

---

## ✅ Evidence of Completion

> **See full QA Report:** [docs/QA/Phase1_Sandbox.md](../QA/Phase1_Sandbox.md)

### Phase 1: Foundation

**1. CLI Demo Ping (`tars demo-ping`)**

```text
[15:09:28 INF] Starting TARS v2 Demo Ping...
[15:09:28 INF] DemoAgent received: PING
[15:09:31 INF] Ping sent.
DEBUG: Ping sent (Console).
[15:09:28 INF] Publishing message...
```

**2. Golden Run Test (`dotnet test`)**

```text
Test summary: total: 30, failed: 0, succeeded: 30, skipped: 0
```

**Test Coverage:**

* KernelTests (3 tests): Agent creation, state updates, message handling
* GrammarTests (4 tests): Goal/task parsing, whitespace handling
* SecurityTests (8 tests): CredentialVault, FilesystemPolicy
* GraphTests (8 tests): ResponseParser, PromptBuilder
* LlmServiceTests (4 tests): Routing, Ollama/vLLM clients
* OpenWebUiTests (1 test): Model listing with authentication
* GoldenRun (1 test): End-to-end CLI demo-ping

> Status: Phase 6.2, 6.5, and 6.7 complete. Integration hardening ongoing.
