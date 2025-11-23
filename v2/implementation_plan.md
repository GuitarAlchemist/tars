# TARS v2 Implementation Plan

**Date:** November 23, 2025
**Status:** Approved
**Goal:** Build TARS v2.0 - A Modular, Secure, and Agentic AI System.

---

## 📅 Phased Roadmap

### Phase 1: The Foundation (Kernel & Security)

**Goal:** Establish the secure runtime environment and core message passing.

- [ ] **1.1 Project Setup**: Initialize `Tars.sln` with F# structure (Kernel, Core, Interface).
- [ ] **1.2 Docker Sandbox**: Create `tars-sandbox` image with read-only filesystem and network isolation.
- [ ] **1.3 Tars.Kernel**: Implement the `EventBus` (Channels) and `IAgent` interface.
- [ ] **1.4 Security Core**: Implement `CredentialVault` (Env vars) and `FilesystemPolicy`.

### Phase 2: The Brain (Inference & Memory)

**Goal:** Enable reasoning and state persistence.

- [ ] **2.1 Semantic Kernel Integration**: Implement `ICognitiveProvider` using Microsoft.SemanticKernel.
- [ ] **2.2 Memory Grid**: Set up ChromaDB (Docker) and implement `VectorStore` client.
- [ ] **2.3 Grammar Engine**: Port `GrammarDistillation` from v1 to `Tars.Cortex.Grammar`.
- [ ] **2.4 Graph Memory**: Port `BeliefGraph` logic to `Tars.Memory.Graph`.

### Phase 3: The Body (Interface & Tools)

**Goal:** Allow TARS to interact with the world and the user.

- [ ] **3.1 Terminal UI**: Build the interactive CLI using **Spectre.Console**.
- [ ] **3.2 MCP Client**: Implement the Model Context Protocol client to consume tools.
- [ ] **3.3 Tool Registry**: Create `SkillRegistry` to load standard tools (Filesystem, Git).
- [ ] **3.4 Cost Budget**: Implement `TokenAccountant` middleware and budget enforcement.

### Phase 4: The Soul (Agents & Autonomy)

**Goal:** Bring the system to life with autonomous agents.

- [ ] **4.1 AutoGen Bridge**: Create the Python-F# bridge for multi-agent orchestration.
- [ ] **4.2 Agent Personas**: Define "Consultant" (Code Reviewer) and "Architect" (Planner) agents.
- [ ] **4.3 Golden Run Testing**: Implement the trace recorder/replayer for behavior testing.
- [ ] **4.4 Self-Correction Loop**: Implement the "Code-Execute-Repair" loop.

---

## 🛠️ Detailed Task Breakdown

### Phase 1: The Foundation

#### 1.1 Project Structure

* **Action**: Create `src/` and `tests/` directories.
- **Projects**:
  - `src/Tars.Kernel` (F# Lib): Core interfaces and EventBus.
  - `src/Tars.Core` (F# Lib): Domain logic and entities.
  - `src/Tars.Interface.Cli` (F# Exe): The entry point.
- **Dependencies**: `Microsoft.Extensions.Hosting`, `Serilog`.

#### 1.2 Docker Sandbox

* **Action**: Create `docker/sandbox/Dockerfile`.
- **Specs**: Python 3.11, non-root user `tars`, no network by default.
- **Output**: A buildable Docker image `tars/sandbox:latest`.

#### 1.3 Tars.Kernel

* **Action**: Implement `EventBus` using `System.Threading.Channels`.
- **Interfaces**: `IMessage`, `IAgent`, `ISkill`.
- **Tests**: Unit tests for message routing.

#### 1.4 Security Core

* **Action**: Implement `Tars.Security`.
- **Features**:
  - `CredentialVault`: Abstraction over Env Vars / Key Vault.
  - `SandboxedProcess`: Wrapper to run commands inside Docker.

---

### Phase 2: The Brain

#### 2.1 Semantic Kernel

* **Action**: Implement `Tars.Cortex.Inference`.
- **Integration**: Use `Microsoft.SemanticKernel` for LLM connectivity.
- **Features**: Support for OpenAI (GPT-4o) and Ollama (Local).

#### 2.2 Memory Grid

* **Action**: Create `docker-compose.yml` with ChromaDB.
- **Code**: Implement `Tars.Memory.Vector` using a C# Chroma client.
- **Migration**: Port `VectorDocument` types from v1.

#### 2.3 Grammar Engine

* **Action**: Port `TarsEngine.GrammarDistillation` to `Tars.Cortex.Grammar`.
- **Refactor**: Decouple from specific LLM implementations.

---

### Phase 3: The Body

#### 3.1 Terminal UI (TUI)

* **Action**: Build the main loop in `Tars.Interface.Cli`.
- **Library**: `Spectre.Console`.
- **Screens**: Dashboard, Chat, Task View.

#### 3.2 MCP Client

* **Action**: Implement `Tars.Integration.Mcp`.
- **Protocol**: Implement JSON-RPC over Stdio/SSE.
- **Features**: `CallTool`, `ListTools`, `ReadResource`.

#### 3.3 Cost Budget

* **Action**: Implement `Tars.Kernel.Budget`.
- **Logic**: Track token usage per request, enforce `MaxDollars` limit.

---

### Phase 4: The Soul

#### 4.1 AutoGen Bridge

* **Action**: Implement `Tars.Agents.AutoGen`.
- **Strategy**: Use `Python.NET` or a sidecar process to run AutoGen agents.
- **Goal**: Allow F# Kernel to orchestrate Python AutoGen agents.

#### 4.3 Golden Run Testing

* **Action**: Implement `Tars.Testing.Golden`.
- **Features**: Record agent traces to JSON, replay and assert actions.

---

## 🚀 Getting Started (for Developers)

1. **Clone & Init**:

    ```bash
    git checkout -b v2-dev
    dotnet new sln -n Tars
    ```

2. **Start Infrastructure**:

    ```bash
    docker-compose up -d chroma
    ```

3. **Run CLI**:

    ```bash
    dotnet run --project src/Tars.Interface.Cli
    ```
