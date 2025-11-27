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

## 📅 Phased Roadmap

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

**Goal:** Enable reasoning and state persistence.

* [x] **2.1 LLM Integration**: Implement `Tars.Llm` (Ollama & vLLM support).
* [ ] **2.2 Memory Grid**: Set up ChromaDB (Docker) and simple `VectorStore` client.
* [x] **2.3 Grammar Engine (Minimal)**: "Garden Shed" grammar to parse simple goals -> F# AST.
* [ ] **2.4 Graph Memory**: Evaluate **Graphiti** (via AutoGen) for long-term memory.

### Phase 3: The Body (Interface & Tools)

**Goal:** Allow TARS to interact with the world.

* [x] **3.1 Terminal UI**: Build the interactive CLI using **Spectre.Console**.
  * Added "Wow" splash screen and interactive chat with spinner.
* [ ] **3.2 MCP Client**: Implement basic JSON-RPC client for external tools.
* [ ] **3.3 Tool Registry**: Create `SkillRegistry` to load standard tools.
* [ ] **3.4 Cost Budget**: Implement `TokenAccountant` middleware.

### Phase 4: The Soul (Evolution & Autonomy)

**Goal:** Implement the Agent0-inspired Co-Evolution Loop.

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

* [x] **5.1 Metascript DSL**: Define the schema for `.tars` files (JSON).
* [x] **5.2 Workflow Engine**: Implement the engine to parse and execute these scripts.
* [x] **5.3 CLI Integration**: `tars run <script>` to execute workflows.

### Phase 6: Cognitive Architecture (Backpressure & Resource Control)

**Goal:** Implement AI-native backpressure patterns to prevent cognitive overload.

**Based on:** Research synthesis from `docs/__research` (Multi-agent protocols + Backpressure patterns)

* [ ] **6.1 Budget Governor**: Implement `BudgetGovernor` for token/time/cost limits
  * Add to `EvolutionContext` and `MetascriptContext`
  * Enforce `TryConsume` before LLM calls
* [ ] **6.2 Agent Speech Acts**: Formalize `AgentIntent` DU (ASK, TELL, PROPOSE, etc.)
  * Update `AgentMessage` to include `Intent` and `ContentType`
  * Implement in `Tars.Kernel`
* [ ] **6.3 Semantic Fan-out Limiter**: Cap Curriculum Agent's task generation to top-K
  * Add scoring to `TaskDefinition`
  * Select top-K subtasks by priority
* [ ] **6.4 Adaptive Reflection**: Stop reflection when improvement < epsilon
  * Track quality delta between reflection rounds
  * Implement `reflectUntilConvergence` in Evolution Engine

## 🛠️ Detailed Task Breakdown

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
