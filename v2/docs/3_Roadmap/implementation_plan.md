# TARS v2 Implementation Plan

**Date:** November 23, 2025
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

* [ ] `dotnet run --project src/Tars.Interface.Cli -- demo-ping` works.
* [ ] Kernel spins up `EventBus`.
* [ ] Demo agent subscribes, receives message, and logs it.
* [ ] **Golden Run:** 1 test that runs CLI, captures trace, and replays it.

* [ ] **1.1 Project Setup**: Initialize `Tars.sln` with F# structure (Kernel, Core, Interface).
* [ ] **1.2 Tars.Kernel**: Implement `EventBus` (System.Threading.Channels) and `IAgent`.
* [ ] **1.3 Docker Sandbox**: Create `tars-sandbox` image (read-only fs, no network).
* [ ] **1.4 Security Core**: Implement `CredentialVault` and `SandboxedProcess`.

### Phase 2: The Brain (Inference & Memory)

**Goal:** Enable reasoning and state persistence.

* [ ] **2.1 Semantic Kernel Integration**: Implement `ICognitiveProvider` (Single model, one prompt).
* [ ] **2.2 Memory Grid**: Set up ChromaDB (Docker) and simple `VectorStore` client.
* [ ] **2.3 Grammar Engine (Minimal)**: "Garden Shed" grammar to parse simple goals -> F# AST.
* [ ] **2.4 Graph Memory**: Evaluate **Graphiti** (via AutoGen) for long-term memory.

### Phase 3: The Body (Interface & Tools)

**Goal:** Allow TARS to interact with the world.

* [ ] **3.1 Terminal UI**: Build the interactive CLI using **Spectre.Console**.
* [ ] **3.2 MCP Client**: Implement basic JSON-RPC client for external tools.
* [ ] **3.3 Tool Registry**: Create `SkillRegistry` to load standard tools.
* [ ] **3.4 Cost Budget**: Implement `TokenAccountant` middleware.

### Phase 4: The Soul (Agents & Autonomy)

**Goal:** Bring the system to life with autonomous agents.

* [ ] **4.1 AutoGen Bridge**: Create the Python-F# bridge for multi-agent orchestration.
* [ ] **4.2 Agent Personas**: Define "Consultant" and "Architect" agents.
* [ ] **4.3 Golden Run Testing**: Expand trace recorder for complex agent behaviors.
* [ ] **4.4 Self-Correction Loop**: Implement the "Code-Execute-Repair" loop.

---

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

#### 2.1 Semantic Kernel

* **Action**: Implement `Tars.Cortex.Inference`.
* **Scope**: Single `ICognitiveProvider` wrapping Semantic Kernel.
* **Test**: `tars ask "Hello"` returns a response from LLM.

#### 2.2 Memory Grid

* **Action**: `docker-compose up chroma`.
* **Code**: Simple `VectorStore` client (Add/Search).
* **Scope**: One collection, basic cosine similarity.

#### 2.3 Grammar Engine (Minimal)

* **Action**: Implement `Tars.Cortex.Grammar`.
* **Scope**: Parse a simple `.trsx` file with 2-3 block types (Goal, Task).
* **Output**: Generate strongly-typed F# AST messages.

#### 2.4 Graph Memory

* **Action**: Research/Evaluate Graphiti.

---

## 🚀 First Coding Session Checklist

1. [ ] Create `Tars.sln` and projects.
2. [ ] Implement `EventBus` and `IAgent`.
3. [ ] Create `DemoAgent` (logs to console).
4. [ ] Implement `tars demo-ping` CLI command.
5. [ ] Write 1 Golden Run test for `demo-ping`.
