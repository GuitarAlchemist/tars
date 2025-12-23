# TARS v2 Implementation Plan

**Date:** November 26, 2025  
**Updated:** December 22, 2025  
**Status:** Approved  
**Goal:** Build TARS v2.0 - A Modular, Secure, and Agentic AI System.

---

## 🧠 Core Thesis

> **LLMs as stochastic generators + Symbolic systems as memory, law, and self-control.**
>
> This is not a compromise — it's a division of labor.

| Component | Role |
|-----------|------|
| LLMs | Propose (cortex) |
| Symbolic Ledger | Decide + Remember (hippocampus + law) |
| Plans | Commit (prefrontal) |
| Contradictions | Learn (pain signals) |

**Reference:** [Architectural Vision](../../1_Vision/architectural_vision.md)

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

### Phase 5: The Mind (Metascript Engine & Grammar Bridge)

**Goal:** Enable complex, multi-step workflows defined in a rich DSL integrated with the Grammar system.

**Reference:** [Phase 5 Metascript Port](../2_Phases/phase_5_metascript_full_port.md)

* [x] **5.1 Metascript DSL**: Define the initial schema (JSON).
* [x] **5.2 Workflow Engine**: Implement the engine to parse and execute JSON scripts.
* [x] **5.3 CLI Integration**: `tars run <script>` for JSON workflows.
* [x] **5.4 V1 Metascript Full Port**: 
  * Port `.tars`/`.trsx` block-based parser.
  * Implement F# Interactive (FSI) handler for deterministic logic.
  * Integrate with `Tars.Cortex.Grammar` for grammar-constrained LLM blocks.
  * Support inline grammars and cross-language variable sharing.

### Phase 6: Cognitive Architecture (Backpressure & Resource Control)

**Goal:** Implement AI-native backpressure patterns to prevent cognitive overload and budget explosions.

**Research Basis:**

* [Backpressure in AI systems](../../4_Research/Conversations/ChatGPT-Backpressure in AI systems.md) (12 AI Design Patterns)
* [Multi-agent system protocols](../../4_Research/Conversations/ChatGPT-Multi-agent system protocols.md) (Semantic Speech Acts)

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

**Reference**: [Phase 6 Integration Strategy](./phase6_integration_strategy.md)

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

* [x] Phase 2.2: Persistent Memory (ChromaDB) - Integrated & Verified (v2 API)
* [x] Phase 3.2: MCP Client (Integrated with Augment Context Engine)
* [ ] Phase 6.5-6.8: Advanced backpressure patterns
* [x] Phase 6.6: Semantic Message Bus (JSON-LD + FIPA-ACL)
* [x] Phase 6.7: Circuit Flow Control (Resistors, Capacitors, Transistors)
* [x] Benchmark: Evolution Loop on standard tasks (5 tasks, 20K tokens, no overruns)

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

### Phase 7: Production Hardening

**Goal**: Make TARS production-ready with robust error handling, logging, and configuration.

**Status**: 🚧 In Progress (Started: 2025-12-21)

* [x] **7.1 Structured Logging**: JSON output, log levels, correlation tracking
* [x] **7.2 Health Checks**: Readiness/liveness probes, dependency checks
* [x] **7.3 Error Categories**: Typed error classification (12 categories)
* [ ] **7.4 Configuration Management**: Environment profiles, settings validation
* [ ] **7.5 Metrics Export**: Prometheus/OpenTelemetry format

---

### Phase 8: Advanced Prompting Techniques

**Goal**: Implement state-of-the-art prompting strategies for enhanced reasoning.

**Status**: 🔜 Planned

**Research Basis**: [Prompting Techniques Analysis](../../4_Research/Architecture/prompting_techniques_analysis.md)

**References**:
- [Awesome-Graph-Prompting](https://github.com/AndrewZhou924/Awesome-Graph-Prompting)
- [LearnPrompting.org](https://learnprompting.org/)
- [PromptingGuide.ai](https://www.promptingguide.ai/)

#### Current State (Already Implemented)

| Technique | Location | Status |
|-----------|----------|--------|
| ReAct (Reason-Act) | `Tars.Cortex/Patterns.fs` | ✅ |
| Chain of Thought | `Tars.Cortex/Patterns.fs` | ✅ |
| Few-Shot Prompting | `Tars.Core/Persona.fs` | ✅ |
| RAG | `Tars.Cortex/VectorStore.fs` | ✅ |
| Tool Use (MRKL) | `Tars.Tools/*` | ✅ |

#### 8.1 Tree of Thoughts (ToT)

**Priority**: High  
**Effort**: 4-6 hours

Explore multiple reasoning paths with BFS/DFS and backtracking.

```fsharp
let treeOfThoughts (llm) (branching) (maxDepth) (task) =
    // Decompose → Generate thoughts → Evaluate → Search → Return best
```

* [ ] Implement `ThoughtNode` type
* [ ] Implement BFS/DFS exploration
* [ ] Add self-evaluation scoring
* [ ] CLI command `tars agent tot <task>`

#### 8.2 Self-Consistency

**Priority**: High  
**Effort**: 2-3 hours

Generate multiple CoT paths and majority-vote the answer.

* [ ] Implement `selfConsistent` wrapper
* [ ] Add configurable sample count
* [ ] Implement majority voting logic

#### 8.3 Graph Prompting

**Priority**: High  
**Effort**: 6-8 hours

Leverage knowledge graph context in prompts.

**Key Papers**:
- StructGPT (reasoning over structured data)
- GraphPrompt (GNN prompting)
- PRODIGY (in-context learning over graphs)

* [ ] Create `GraphReasoning.fs` in `Tars.Cortex`
* [ ] Implement subgraph extraction from `TemporalKnowledgeGraph`
* [ ] Add graph context injection to prompts
* [ ] Enable "reason over knowledge" queries

#### 8.4 Prompt Chaining DSL

**Priority**: Medium  
**Effort**: 3-4 hours

Formalize complex task handoffs in Metascript.

* [ ] Extend DSL syntax for explicit chains
* [ ] Add intermediate validation points
* [ ] Support branching chains

#### 8.5 Zero-Shot CoT

**Priority**: Low  
**Effort**: 1 hour

Simple "Let's think step by step" enhancement.

* [ ] Add to prompt templates
* [ ] Auto-apply for reasoning tasks

---

### Phase 9: Symbolic Knowledge & Free Skills

**Goal**: Transform TARS into a knowledge-accumulating, evolving intelligence.

**Status**: 🔜 Planned (December 2025)

**Reference**: [Phase 9 Symbolic Knowledge Roadmap](./phase9_symbolic_knowledge.md)

**Research Basis**: [ChatGPT-Claude skills free use.md](../../conversations/ChatGPT-Claude%20skills%20free%20use.md)

#### Core Principles (from research)

1. **"Symbolic stays first-class"** - Embeddings are indexes, not truth
2. **"Event-source everything"** - Never mutate beliefs, append events
3. **"Internet never writes beliefs directly"** - Only produces evidence candidates
4. **"Plans are hypotheses"** - Not beliefs, but hypotheses about future actions
5. **"Provenance is non-negotiable"** - Every belief answers: Who? When? From what evidence?

#### 9.1 Symbolic Knowledge Ledger

* [ ] **9.1.1** Create `Tars.Knowledge` project  
* [ ] **9.1.2** Define `Belief`, `BeliefEvent`, `Provenance` types
* [ ] **9.1.3** Implement `KnowledgeLedger` (Postgres-backed)
* [ ] **9.1.4** Add CLI: `tars know ingest <path>`
* [ ] **9.1.5** Parse `.trsx` outputs into assertions
* [ ] **9.1.6** Emit `knowledge_snapshot.trsx` each run

#### 9.2 Internet Ingestion Pipeline

* [ ] **9.2.1** Create `evidence_store` table
* [ ] **9.2.2** Implement Wikipedia/arXiv/GitHub fetchers
* [ ] **9.2.3** Implement LLM-based assertion proposer  
* [ ] **9.2.4** Implement Verifier Agent
* [ ] **9.2.5** Implement contradiction detection

#### 9.3 Evolving Plans

* [ ] **9.3.1** Create `Plan`, `PlanEvent` types
* [ ] **9.3.2** Link plan assumptions to belief IDs
* [ ] **9.3.3** Invalidate plans when beliefs retract
* [ ] **9.3.4** Add CLI: `tars plan new "<goal>"`

#### 9.4 Free Local Stack (Already Implemented)

| Component | Tool | Status |
|-----------|------|--------|
| Reasoning | qwen3:14b | ✅ |
| Math/Logic | deepseek-r1:14b | ✅ |
| Fast Inference | llama.cpp | ✅ |
| Embeddings | nomic-embed-text | ✅ |
| Agent Skills | Tars.Tools | ✅ |

**Acceptance Criteria:**

1. TARS accumulates symbolic knowledge from runs
2. External evidence ingested with provenance
3. Plans evolve based on evidence
4. `tars know status` shows belief graph metrics

---

### Phase 10: 3D Knowledge Graph Visualization

**Goal**: Create an interactive 3D visualization of TARS belief graphs.

**Status**: 🔜 Planned (After Phase 9)

**Reference**: [Phase 10 3D Knowledge Graph Roadmap](./phase10_3d_knowledge_graph.md)

**Architecture**: `Graph Export → Layout → 3D Viewer` (Keep viewer dumb; keep semantics in TARS)

#### MVP View: Belief Graph

**Node Types**: Belief, Concept, Agent, Run, File  
**Edge Types**: supports, contradicts, derived_from, mentions, produced_by

#### 10.0 Data Contract

* [ ] **10.0.1** Define `GraphSliceDto` (nodes/edges JSON)
* [ ] **10.0.2** Define query params: rootId, depth, limit, filters

#### 10.1 Backend Graph Slice

* [ ] **10.1.1** Create `/api/graph/slice` endpoint
* [ ] **10.1.2** Implement Neo4j query templates (neighborhood, by-run, contradictions)
* [ ] **10.1.3** Add in-memory caching (5min TTL)

#### 10.2 Frontend 3D Viewer

* [ ] **10.2.1** Vite + React + Three.js + 3d-force-graph setup
* [ ] **10.2.2** Click node → side panel (full details, provenance)
* [ ] **10.2.3** Double-click → expand neighborhood (fetch API)
* [ ] **10.2.4** Filter chips (node types, edge types)
* [ ] **10.2.5** Confidence range slider
* [ ] **10.2.6** Search box with highlighting

#### 10.3 TARS Integration

* [ ] **10.3.1** "Explain cluster" button → LLM summary
* [ ] **10.3.2** "Create plan from contradictions" → plan.trsx draft
* [ ] **10.3.3** Time slider (version navigation)

**Visual Encoding**:
- Node color = type (Belief=Orange, Concept=Blue, Agent=Green)
- Node size = confidence
- Edge style: supports=thin solid, contradicts=thick dashed red

**Acceptance Criteria:**

1. `/api/graph/slice` returns proper JSON
2. 3D viewer renders with pan/zoom/orbit
3. Click/expand/filter interactions work
4. TARS can "explain" selected clusters

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

---

## 🔄 TARS V1 Reuse Strategy

To accelerate v2 development, high-value assets and logic from v1 are being ported and adapted.

**Reference:** [v1_reuse_strategy.md](../../4_Research/V1_Insights/v1_reuse_strategy.md)

### Phase 1: Immediate Porting (Epic 3 & 5)

* [ ] **AgenticTraceCapture**: Port `AgenticTraceCapture.fs` to `Tars.Observability.Tracing`.
* [ ] **Agent Registry**: Implement YAML loader in `Tars.Kernel.Registry` to bootstrap from `v1/.tars/agents/tars_agent_organization.yaml`.
* [ ] **Metascript V1**: Port `.tars` block-based parser and FSI execution handler to `Tars.Metascript`.

### Phase 2: Cortex Enhancement (Epic 2)

* [ ] **Grammar Migration**: Migrate `v1/.tars/grammars/*.tars` to `v2/resources/grammars/`.
* [ ] **Structured Output**: Integrate `GrammarConstraint` into `ICognitiveProvider` interface.

### Phase 3: Evolution Loop (Epic 6 & 8)

* [ ] **Self-Improvement**: Extract logic from `tars-self-improvement-cycle.trsx` into `Tars.Evolution.Engine`.
