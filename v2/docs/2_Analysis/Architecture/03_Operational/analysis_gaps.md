# TARS V2 Analysis Gaps & Operational Requirements

**Date:** November 22, 2025
**Status:** Analysis
**Context:** Identifying critical operational aspects missing from the current architectural documentation.

---

## Executive Summary

While TARS v2 has strong **technical architecture** coverage (Agent Runtime, Memory, Tools), we are missing critical **operational requirements** needed for production deployment. This document addresses six key gaps that must be resolved before implementation.

---

## 1. Security & Sandboxing

### 🔒 The Problem

TARS executes **arbitrary code** (Python, F#, shell commands) on behalf of users. Without proper sandboxing, a malicious prompt or compromised agent could:

- Delete files (`rm -rf /`)
- Exfiltrate credentials (read `.env`, `~/.aws/credentials`)
- Make unauthorized API calls
- Install malware

### ✅ Recommendation

| Layer | Technology | Purpose |
| :--- | :--- | :--- |
| **L1: Process Isolation** | **Docker** | Each agent runs in a disposable container with no network access by default. |
| **L2: Filesystem Limits** | **Bind Mounts (Read-Only)** | Agent can only write to `/workspace`. System files are read-only. |
| **L3: Network Policy** | **Firewall Rules** | Outbound connections only to approved domains (e.g., `api.openai.com`). |
| **L4: Credential Vault** | **HashiCorp Vault** or **Azure Key Vault** | API keys are injected as env vars, never stored in code. |
| **L5: Code Review Gate** | **Human-in-the-Loop** | For "Architect" mode (self-modification), all generated F# code requires user approval before compilation. |

**Implementation Priority:** **CRITICAL**. Build this *before* the first agent execution.

---

## 2. Testing Strategy

### 🧪 The Challenge

Traditional unit tests are insufficient for agentic systems. **How do you test intelligence?**

### ✅ Testing Pyramid for TARS

```
              ┌─────────────────────┐
              │  Scenario Tests     │  ← "Can TARS fix a failing build?"
              │  (End-to-End)       │
              └─────────────────────┘
                     ▲
                     │
            ┌────────────────────┐
            │  Agent Behavior    │  ← "Does the Coder agent request tests?"
            │  Tests             │
            └────────────────────┘
                     ▲
                     │
          ┌──────────────────────┐
          │  Integration Tests   │  ← "Does AutoGen talk to MCP servers?"
          └──────────────────────┘
                     ▲
                     │
       ┌─────────────────────────┐
       │  Unit Tests             │  ← "Does VectorStore.Search() work?"
       │  (Pure F# Functions)    │
       └─────────────────────────┘
```

#### Layer 1: Unit Tests (Standard)

- Test pure functions (e.g., `BeliefGraph.addNode`, `VectorStore.cosineSimilarity`).
- **Framework:** xUnit + FsCheck (property-based testing).

#### Layer 2: Integration Tests

- Test MCP client → MCP server communication.
- Test AutoGen agent → F# Kernel message passing.
- **Framework:** Docker Compose + TestContainers.

#### Layer 3: Agent Behavior Tests

- **Concept:** "Behavior-Driven Development for Agents."
- **Example Test:**

  ```gherkin
  Given a repository with a failing test
  When TARS is asked to "Fix the build"
  Then TARS should:
    1. Run the test suite
    2. Read the error message
    3. Modify the code
    4. Re-run tests until green
  ```

- **Challenge:** Non-deterministic. Solution: **Golden Runs** (record successful agent traces, replay as regression tests).

#### Layer 4: Scenario Tests (Smoke Tests)

- Full end-to-end workflows (e.g., "Generate a REST API from a schema").
- **Frequency:** Nightly (too expensive for CI).

**Key Insight:** Use **Agentic Trace Capture** to create reproducible test cases. If TARS solves a problem once, save the trace and replay it as a regression test.

---

## 3. Deployment & Distribution

### 📦 The Question

**How does a user get TARS running on their machine?**

### ✅ Multi-Tier Distribution Strategy

| User Type | Distribution Method | What They Get |
| :--- | :--- | :--- |
| **Developer (Local)** | **Docker Compose** | `docker-compose up` starts TARS + Chroma + Phoenix locally. |
| **Team (Self-Hosted)** | **Kubernetes Helm Chart** | Deploy to internal k8s cluster with shared memory. |
| **Enterprise** | **Azure Container Apps** | Fully managed, auto-scaling TARS instances. |

#### Option 1: Docker Compose (Recommended for v2.0)

**File:** `docker-compose.yml`

```yaml
services:
  tars:
    image: ghcr.io/tars/tars:v2.0
    volumes:
      - ./workspace:/workspace
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - chroma
      - phoenix
  
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
  
  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
```

**User Experience:**

1. `git clone https://github.com/user/tars && cd tars`
2. `cp .env.example .env` (add API keys)
3. `docker-compose up`
4. Open `http://localhost:8080` (TARS UI)

#### Option 2: .NET Global Tool (For F# Micro-Kernel Only)

```bash
dotnet tool install -g Tars.Cli
tars init --mode consultant
```

**Decision:** Start with **Docker Compose**. Add Helm chart in v2.1.

---

## 4. Cost Management

### 💰 The Risk

Without guardrails, a runaway agent loop could cost **$1000+ in a single session** (e.g., infinite retries with GPT-4).

### ✅ Cost Control Mechanisms

#### Layer 1: Pre-Session Budget

```fsharp
type Budget = {
    MaxTokens: int
    MaxDollars: decimal
    TimeoutMinutes: int
}

let defaultBudget = {
    MaxTokens = 100_000  // ~$0.50 for GPT-4o-mini
    MaxDollars = 5.0m
    TimeoutMinutes = 30
}
```

#### Layer 2: Token Accounting

- Every LLM call logs: `ModelName`, `InputTokens`, `OutputTokens`, `EstimatedCost`.
- TARS maintains a **running total** per session.
- **Kill Switch:** When `totalCost >= budget.MaxDollars * 0.9`, switch to "Low Power Mode" (cheaper model, shorter prompts).

#### Layer 3: Model Fallback Strategy

| Budget Used | Model |
| :--- | :--- |
| 0-50% | GPT-4o / Claude 3.5 Sonnet (Primary) |
| 50-80% | GPT-4o-mini (Fallback) |
| 80-95% | **Caution Mode**: Agent asks user to approve expensive calls |
| 95-100% | **Read-Only Mode**: Agent can only read, not write code |

#### Layer 4: Dashboard

- Real-time cost tracker in the UI.
- Alert when approaching budget: *"⚠️ 80% of budget used. Continue?"*

**Implementation:** Add `Tars.Kernel.Budget` module with `ITokenAccountant` interface.

---

## 5. UI/UX Design

### 🎨 The Gap

The architecture mentions "CLI/TUI/API" but provides no:

- User flows
- Interface mockups
- Interaction patterns

### ✅ Recommended UI Strategy

#### Primary Interface: **Terminal UI (TUI)**

- **Why?** Developers already live in the terminal. TARS should too.
- **Technology:** [Spectre.Console](https://spectreconsole.net/) (F#-friendly .NET library).

**Example Interaction:**

```
┌─────────────────────────────────────────────┐
│ TARS v2.0 - Consultant Mode                │
├─────────────────────────────────────────────┤
│ > Fix the failing test in UserService.cs   │
│                                             │
│ 🤔 Analyzing...                             │
│ ├─ Running test suite... ✓                 │
│ ├─ Found error: NullReferenceException     │
│ ├─ Reading UserService.cs... ✓             │
│ └─ Generating fix...                       │
│                                             │
│ 💡 Proposed change:                         │
│ ┌─────────────────────────────────────────┐ │
│ │ - var user = GetUser();                 │ │
│ │ + var user = GetUser() ?? new User();   │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ [A]pply  [R]eject  [E]xplain                │
└─────────────────────────────────────────────┘
```

#### Secondary Interface: **Web UI (Optional)**

- For non-technical stakeholders (e.g., managers reviewing agent activity).
- **Technology:** Blazor Server (shares F# backend).
- **Features:** View agent traces, approve self-modifications, export reports.

#### Tertiary Interface: **VS Code Extension**

- Inline TARS suggestions in the editor.
- **Technology:** Language Server Protocol (LSP).

**Recommendation:** Build **TUI first** (v2.0), Web UI later (v2.1).

---

## 6. Error Handling & Recovery

### ⚠️ The Scenarios

1. **Agent Crash**: AutoGen raises an unhandled exception.
2. **Network Failure**: MCP server is unreachable.
3. **GPU Unavailable**: CUDA kernels fail to load.
4. **Disk Full**: Vector store can't persist embeddings.

### ✅ Recovery Strategies

#### Pattern 1: Graceful Degradation

```fsharp
match tryLoadCudaKernels() with
| Ok kernels -> 
    log.Info "Using GPU acceleration"
    kernels
| Error _ -> 
    log.Warn "CUDA unavailable, falling back to CPU"
    CpuKernels()
```

#### Pattern 2: Session Persistence

- Every 5 minutes, TARS saves its state to `~/.tars/sessions/{session_id}.json`.
- Contains: Agent context, conversation history, intermediate results.
- **Recovery:** `tars resume {session_id}` restarts from last checkpoint.

#### Pattern 3: Circuit Breaker (for MCP Servers)

```fsharp
type CircuitState = Closed | Open | HalfOpen

let mcpCircuitBreaker = {
    State = Closed
    FailureThreshold = 3
    ResetTimeout = 60<seconds>
}

// If GitHub MCP fails 3 times, skip it for 60s
```

#### Pattern 4: Human Escalation

- If TARS encounters an unrecoverable error, it should:
  1. Save state
  2. Log full stack trace
  3. **Ask the user**: *"I'm stuck. Here's what happened: [error]. Should I: (A) Retry, (B) Skip this step, (C) Abort?"*

**Key Principle:** **Never fail silently.** Always surface errors to the user with actionable options.

---

## Prioritization Matrix

| Gap | Risk if Ignored | Effort | Priority |
| :--- | :--- | :--- | :--- |
| **Security** | **CRITICAL** (Data loss, credential leak) | High | **P0** |
| **Testing** | High (Unstable releases) | Medium | **P0** |
| **Deployment** | Medium (Poor UX) | Low | **P1** |
| **Cost Management** | **HIGH** (Financial risk) | Low | **P0** |
| **UI/UX** | Low (Usable via API) | Medium | **P1** |
| **Error Handling** | Medium (Poor reliability) | Low | **P1** |

---

## Next Steps

1. **Immediate (Before Implementation):**
   - Create `security_model.md` with detailed sandboxing architecture
   - Create `testing_strategy.md` with test case examples
   - Create `cost_budget.md` with token accounting design

2. **Phase 1 (v2.0):**
   - Implement Docker-based sandboxing
   - Add token accounting to F# Micro-Kernel
   - Build TUI with Spectre.Console

3. **Phase 2 (v2.1):**
   - Add Web UI for observability
   - Kubernetes Helm chart
   - Advanced testing (Golden Runs)

---
## Implementation & Feature Gaps

This section outlines the identified design gaps and areas for future development in the TARS project, based on an analysis of the existing documentation.

### 1. Core Autonomous Capabilities

*   **Autonomous Code Modification Loop:** The documentation explicitly states that the "autonomous code modification loop" is a missing component required to reach "Tier 2" autonomy. This is a significant gap, as it's the core of the self-improvement capability.
*   **Advanced Agent Collaboration:** While a multi-agent system is in place, the "Path to Tier 3" mentions the need for "multi-agent cross-validation" and "recursive self-improvement," suggesting that the current agent collaboration is not yet at its full potential.

### 2. Feature Completeness

*   **Plugin System:** The `README.md` lists a "Plugin System" as a future direction, which would allow for third-party extensions. The design and implementation of this.
*   **Multiple LLM Providers:** The project aims to integrate with multiple LLM providers, but the current implementation seems to be focused on Ollama.
*   **Web UI:** A web-based user interface is mentioned as a future goal, but it does not seem to be implemented yet.

### 3. Performance and Optimization

*   **CUDA Vector Store:** The `AGENTS.md` file indicates that the CUDA vector store is missing several key features for optimal performance, including "optimized kernels, GPU top-k, batching," "FP16 storage," and "multi-GPU support."

### 4. Lack of Specifics in Documentation

*   While the documentation is extensive, some areas lack specific implementation details. For example, the exact mechanisms for "meta-cognitive awareness" and "autonomous goal setting" in "Tier 3" are not fully elaborated, which may indicate that these are still in the conceptual phase.

---

## Revision History

| Date          | Author      | Description                                                                                             |
| :------------ | :---------- | :------------------------------------------------------------------------------------------------------ |
| Nov 22, 2025  | Gemini      | Merged `DESIGN_GAPS.md` into this document to create a comprehensive list of operational and feature gaps. |