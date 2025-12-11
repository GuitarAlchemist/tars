# TARS v2 - Thinking, Acting, Reasoning System

[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com/)
[![F#](https://img.shields.io/badge/F%23-Functional-378BBA)](https://fsharp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **A modular, multi-agent AI framework built in F# with advanced RAG, budget governance, and epistemic verification.**

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TARS v2 Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│  Interface Layer                                                    │
│  ┌──────────────┐                                                   │
│  │ Tars.CLI     │  Chat, Ask, Diag, Run commands                   │
│  └──────────────┘                                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Agent Layer                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ AgentWorkflow│  │ Patterns     │  │ Budget       │              │
│  │ (CE Builder) │  │ CoT/ReAct    │  │ Governance   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│  Intelligence Layer (Cortex)                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ VectorStores │  │ KnowledgeGraph│ │ Epistemic    │              │
│  │ SQLite/ANN   │  │ Multi-hop    │  │ Governor     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Chunking     │  │ Inference    │  │ GraphAnalyzer│              │
│  │ Strategies   │  │ Engine       │  │ K-Theory     │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│  Execution Layer                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Tars.Graph   │  │ Tars.Kernel  │  │ Metascript   │              │
│  │ DAG Executor │  │ Event Bus    │  │ DSL Engine   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
├─────────────────────────────────────────────────────────────────────┤
│  Foundation Layer                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Tars.Llm     │  │ Tars.Tools   │  │ Tars.Sandbox │              │
│  │ Ollama/vLLM  │  │ Tool Registry│  │ Docker Exec  │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- [Ollama](https://ollama.ai/) (for LLM inference)

### Build & Run

```powershell
# Clone and build
cd v2
dotnet build

# Run diagnostics
dotnet run --project src/Tars.Interface.Cli -- diag

# Start chat (requires Ollama running)
.\start-chat.ps1
# Or manually:
dotnet run --project src/Tars.Interface.Cli -- chat
```

### Run Tests

```powershell
dotnet test --filter Category!=Slow
# If you hit file locks on Windows, run: scripts/kill-testhost.ps1
# Offline eval sanity (records metrics): dotnet test --filter OfflineEvalTests
```

## 📦 Modules

| Module | Description |
|--------|-------------|
| **Tars.Core** | Domain types, AgentWorkflow CE, Budget, Patterns (CoT, ReAct) |
| **Tars.Cortex** | Vector stores, Knowledge graph, Chunking, Epistemic governor |
| **Tars.Llm** | LLM clients (Ollama, vLLM, OpenAI-compatible) |
| **Tars.Graph** | Agent execution DAG with state machine |
| **Tars.Kernel** | Event bus, Safety gates, Governance |
| **Tars.Metascript** | DSL parser, templates, validation |
| **Tars.Tools** | Tool registry and standard tools |
| **Tars.Sandbox** | Docker-based code execution |
| **Tars.Security** | Security policies and validation |
| **Tars.Evolution** | Self-improvement protocols |
| **Tars.Connectors** | External service connectors |
| **Tars.Interface.Cli** | Command-line interface |

## ✨ Key Features

### Agent Workflow Computation Expression

```fsharp
let myWorkflow = agent {
    let! data = fetchData ()
    do! AgentWorkflow.checkBudget { Cost.Zero with Tokens = 100<token> }
    let! result = processWithLlm data
    return result
}
```

### Multi-Dimensional Budget Governance

```fsharp
let budget = { Budget.Infinite with 
    MaxTokens = Some 10000<token>
    MaxMoney = Some 1.0m<usd> }
let governor = BudgetGovernor(budget)
```

### Circuit Combinators

```fsharp
// Transform: map over workflow results
let doubled = transform ((*) 2) computeValue

// Stabilize: smooth rapid changes
let stable = stabilize 0.7 volatileSignal

// Grounded: verify with epistemic governor  
let verified = grounded verifyFact claim
```

### Advanced RAG Pipeline

- Query expansion, Multi-hop retrieval, Metadata filtering
- Semantic chunking, Time decay scoring, Cross-encoder reranking
- Answer attribution, Retrieval metrics, Fallback chains

## 📖 Documentation

- [Chat Quick Start](README_CHAT.md)
- [Metascript Specification](docs/Architecture/Metascript_Specification.md)
- [Implementation Roadmap](docs/3_Roadmap/implementation_plan.md)

## 🧪 Test Coverage

- **255 unit tests** covering all major modules (0 skipped)
- **Real LLM integration tests** with Ollama (Agent, CoT, ReAct, Plan&Execute)
- **4 cognitive evals** for behavioral testing (memory, budget, compression, watchdog)
- **8 cognitive patterns** fully implemented and tested

## 🎉 v2.0-alpha Release

**Version:** v2.0-alpha | **Tests:** 255 passing | **Patterns:** 8/8 complete

Key features:

- Token Budget Governor with multi-dimensional governance
- All 8 cognitive design patterns (SemanticWatchdog, UncertaintyGatedPlanner, etc.)
- MCP Server mode for IDE integration
- Temporal Knowledge Graph with community detection

See [Release Notes](docs/RELEASE_NOTES_v2.0-alpha.md) for details.

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details.
