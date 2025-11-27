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
* ✅ **CLI:** `tars run script.trsx` (basic execution).

**Deferred to v2.x (Nice-to-have):**

* ❌ Full MCP Client (start with hardcoded tools).
* ❌ AutoGen Bridge (start with native F# agents).
* ❌ Complex Cost Budgeting.
* ❌ Web UI / Fancy TUI Dashboards.

---

## � The TARS Constitution (New in v2.1)

**Core Insight:** To prevent entropy in a self-evolving system, TARS needs a "Constitution" — a set of inviolable laws that precede modules.

1. **Immutability by Default**: Configs, grammars, agent definitions, and skills are versioned and immutable once released. Mutation only happens via explicit evolution workflows.
2. **Universal Versioning**: Everything (Agents, Skills, Grammars, Beliefs) must have `version`, `parentVersion`, and `createdAt`. This enables rollback and evolution tracking.
3. **Time as First-Class Citizen**: All state must have `validFrom`, `lastUsed`, and `decayScore`. The system must actively prune and compact old state.
4. **Safety Gates (Jidoka)**: "Stop the Line" mentality. No code mutation without passing: Static Checks → Test Harness → Sandbox Execution. If a check fails, the process halts immediately.
5. **Omotenashi (DX First)**: Every error must be actionable. Every trace must be readable. The system serves the human, not the other way around.
6. **Kaizen (Evolution)**: The system favors small, frequent self-improvements (1% better every day) over large, risky leaps.
7. **Monozukuri (Craftsmanship)**: We value long-term maintainability. Code is written to last decades, not just to ship features.
8. **Hansei (Reflection)**: After every task, the system performs a self-reflection step to learn from mistakes.

---

## �📅 Phased Roadmap

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

### Phase 2.5: Cognitive Hardening (Safety & Types)

**Goal:** Enforce safety through F# type system.

* [x] **4.1 Tars.Evolution Project**: Create the project structure for the evolution engine.
* [x] **4.2 The Protocol**: Define `TaskDefinition` and `ValidationResult` DUs.
* [x] **4.3 Curriculum Agent**: Implement the "Teacher" that generates tasks.
  * *Status:* Implemented using `LlmService` and `qwen2.5-coder`.
* [x] **4.4 Executor Agent**: Implement the "Student" that solves tasks using the Graph.
  * *Status:* Implemented using `Graph.step` and `LlmService`.
* [x] **4.5 The Loop**: Wire them together: `Curriculum -> Task -> Executor -> Result -> Memory`.
  * *Status:* Working loop in `Evolve.fs`. Results are saved to `InMemoryVectorStore` (ephemeral).

### Phase 5: The Mind (Metascript Engine)

**Goal:** Enable complex, multi-step workflows defined in a DSL.

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

**Tasks:**

* [ ] **Repo Structure**: Align folders (Kernel, Cortex, Memory, Agents, Skills, Observability).
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

* [ ] Refactor `EventBus` to be a **Semantic Bus** that routes `SemanticMessage<'T>`.
* [ ] Implement **Constraint Enforcement** middleware in the Kernel (reject messages that violate constraints).
* [ ] Update Evolution Engine to use speech acts (Curriculum → Request, Executor → Inform).
* [ ] Add telemetry/logging for all agent interactions by intent type

**Acceptance Criteria:**

* All agent messages have explicit `Intent`
* Logs show clear interaction patterns: "Curriculum ASKed → Executor TOLD"
* Tests validate routing by intent

#### Phase 6.3: Semantic Fan-out Limiter (Prevent Task Explosion)

**Priority**: High (v2.1)

**Rationale**: Without fan-out limiting, a planner can generate 50 subtasks → 500 sub-subtasks → bankruptcy.

**Research Pattern**: "Semantic Fan-out Limiter" from backpressure analysis.

**Tasks:**

* [ ] Add `Score: float` field to `TaskDefinition`
* [ ] Implement `scoreTask` function in Curriculum Agent:

  ```fsharp
  let scoreTask (task: TaskDefinition) : float =
      // Score based on:
      // - Novelty (not similar to past tasks)
      // - Difficulty alignment (not too easy/hard)
      // - Budget efficiency (expected tokens vs. value)
  ```

* [ ] Add `selectTopK` in `generateTask`:

  ```fsharp
  let selectTopK (tasks: TaskDefinition list) (k: int) =
      tasks
      |> List.sortByDescending (fun t -> t.Score)
      |> List.truncate k
  ```

* [ ] Add `--max-subtasks` CLI flag (default: 5)

* [ ] Log when tasks are pruned: "Generated 12 subtasks, selected top 5 by score"

**Acceptance Criteria:**

* Curriculum Agent never generates more than K subtasks per level
* Tests verify scoring and selection logic
* Pruned tasks are logged with scores

#### Phase 6.4: Adaptive Reflection (Stop When Converged)

**Priority**: Medium (v2.1)

**Rationale**: Reflection improves quality but has diminishing returns. Stop when improvement < epsilon.

**Research Pattern**: "Adaptive Reflection Loop" from backpressure analysis.

**Tasks:**

* [ ] Add `reflectOnTask` function to Executor Agent
* [ ] Implement `measureImprovement`:

  ```fsharp
  let measureImprovement (before: TaskResult) (after: TaskResult) : float =
      // Options:
      // 1. LLM judges quality delta
      // 2. Test pass rate delta
      // 3. Code complexity reduction
  ```

* [ ] Implement `reflectUntilConvergence`:

  ```fsharp
  let rec reflectUntilConvergence (state: TaskResult) (maxReflections: int) (budget: BudgetGovernor) =
      task {
          if maxReflections = 0 || not (budget.TryConsume(500, 1)) then
              return state
          else
              let! newState = reflectOnce state
              let improvement = measureImprovement state newState
              if improvement < 0.05 then  // < 5% improvement
                  return newState
              else
                  return! reflectUntilConvergence newState (maxReflections - 1) budget
      }
  ```

* [ ] Add to Evolution Engine `executeTask`

* [ ] Add `--max-reflections` CLI flag (default: 2)

**Acceptance Criteria:**

* Reflection stops when improvement plateaus
* Budget prevents runaway reflection loops
* Logs show: "Reflection 1: +12% improvement, Reflection 2: +3% improvement, stopped"

#### Phase 6.5: Agentic Interfaces (Soft Semantic Contracts)

**Priority**: Critical (Immediate)

**Rationale**: Move from rigid contracts to soft semantic interfaces that support partial failure, capability-based routing, and probabilistic outcomes.

**Reference**: `docs/2_Analysis/Architecture/agentic_interfaces.md`

**Tasks:**

##### Phase 6.5.1: Core Types (Hard Shell)

* [ ] Define `PartialFailure` DU in `Tars.Core/Domain.fs`

**Acceptance Criteria:**

* All agent execution returns `ExecutionOutcome<'T>` instead of raw results
* Partial failures are captured as warnings, not lost
* Agents can be selected by capability matching
* Computation expression allows linear workflow composition
* Tests validate that PartialSuccess correctly accumulates warnings

---

## 🛠️ Detailed Task Breakdown

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

* [ ] **2.2 Memory Grid**: Simple Persistence (SQLite/File)
  * *Decision*: Start simple (SQLite/File) before adding complex Vector DBs.
  * *Future*: Evaluate ChromaDB/GAM only when scale demands it.
* [ ] **2.4 Internal Knowledge Graph**: Graphiti-style internal graph
  * *Decision*: Drop external Triple Stores. Use internal graph for beliefs/lineage.

### Phase 3: The Body (Partial)

**Completed:**

* [x] **3.1 Terminal UI**: Spectre.Console with splash screen, spinner, interactive chat

**Deferred:**

* [ ] **3.2 MCP Client**: Model Context Protocol for external tools
  * *Decision*: Phase 3.2 deferred to v2.2
  * *Research*: MCP + OpenAI-compatible stack is emerging standard
* [ ] **3.3 Tool Registry**: SkillRegistry with hardcoded tools
  * *Status*: Basic `ToolRegistry` implemented in `Tars.Tools`
  * *Next*: Add circuit breakers (Phase 6.5)
* [ ] **3.4 Cost Budget**: TokenAccountant middleware
  * *Status*: Merged into Phase 6.1 (BudgetGovernor)

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

1. Token Budget Governor ← **Critical**
2. Semantic Fan-out Limiter ← **Critical**
3. Adaptive Reflection Loop
4. Context Compaction Pipeline ← v2.2
5. Uncertainty-Gated Planner ← v2.2
6. Consensus Circuit Breaker ← v2.2
7. Tool Circuit Breaker ← v2.2
8. Semantic Watchdog ← v2.2

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

* [x] Phase 4: Evolution Loop ✅
* [x] Phase 5: Metascript Engine ✅
* [ ] Phase 6.1-6.4: Cognitive Architecture 🚧

**Exit Criteria:**

1. `tars evolve` runs 10 generations without budget overrun
2. `tars run <script>` executes complex workflows with budgets
3. All tests pass (`dotnet test`)
4. Documentation complete for all implemented features

### v2.1 Beta (Next Milestone)

* [ ] Phase 2.2: Persistent Memory (ChromaDB)

* [ ] Phase 3.2: MCP Client
* [ ] Phase 6.5-6.8: Advanced backpressure patterns
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
3. ⏳ `ChatGPT-K-theory and TARS query.md` → Future (theoretical foundations)
4. ⏳ `Grothendieck Groups and their Application to AI.md` → Future (category theory)

### Key Insights Applied

* **Backpressure Patterns**: 12 design patterns → 4 implemented in Phase 6

* **Emerging Standards**: OpenAI-compatible REST + MCP + JSON-RPC → roadmap alignment
* **Semantic Load**: Tokens/calls/time as first-class resources → `BudgetGovernor`
* **Speech Acts**: ASK/TELL/PROPOSE → `AgentIntent` DU

### Research-Driven Decisions

1. **Budget Governor** (from Pattern #1: Token Budget Governor)
2. **Fan-out Limiter** (from Pattern #2: Semantic Fan-out Limiter)
3. **Adaptive Reflection** (from Pattern #3: Adaptive Reflection Loop)
4. **Agent Protocol** (from Multi-agent protocols analysis)

---

## 🚀 Next Actions

### Immediate (Tonight/Tomorrow)

1. Review and approve this plan
2. Optional: Start Phase 6.1 (Budget Governor)

### Short-Term (This Week)

1. Implement Phase 6.1 (Budget Governor)
2. Implement Phase 6.2 (Agent Speech Acts)
3. Run benchmarks: Evolution Loop with budgets

### Medium-Term (This Month)

1. Complete Phase 6.3-6.4
2. Benchmark against baselines
3. Write v2.0 Alpha release notes

### Long-Term (Q1 2026)

1. Phase 2.2: Memory Grid (ChromaDB or GAM)
2. Phase 3.2: MCP Client
3. v2.1 Beta release

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

#### 2.4 Graph Memory

* **Action**: Research/Evaluate Graphiti.

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
