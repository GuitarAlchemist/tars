# TARS v2 - Project Documentation

## Vision

> **LLMs as stochastic generators + Symbolic systems as memory, law, and self-control.**
>
> You're not building a bigger brain. You're building a system that remembers being wrong.
> That's the only kind of intelligence that scales without breaking.

**Reference:** [Architectural Vision](docs/1_Vision/architectural_vision.md)

## Overview

**TARS v2** is the current major iteration of the Tars autonomous reasoning system. It features a modular, event-driven architecture built on **.NET 10 (F#)**, implementing the synthesis of neural imagination and symbolic memory for durable intelligence.

## Key Features

* **Core Architecture**: Hexagonal/Clean Architecture separating Domain, Core, and Interfaces.
* **Symbolic Knowledge Ledger**: Event-sourced beliefs with provenance tracking (Phase 9).
* **Graphiti Knowledge Graph**: A temporal, episodic memory system (`TemporalKnowledgeGraph`).
* **Metascript Engine**: A recursive, macro-capable workflow engine for defining agent behaviors.
* **Resilience & Monitoring**: Functional circuit breakers, retry policies, and metrics.
* **Dynamic Tooling**: A robust tool registry supporting **124+ tools**, including:
  - Web tools (fetch_webpage, search_web, fetch_wikipedia, fetch_arxiv)
  - Code analysis (analyze_file_complexity, find_code_smells)
  - Graph operations (graph_add_node, graph_get_neighborhood, graph_find_contradictions)
  - Research tools (fetch_arxiv, search_semantic_scholar)

## Cognitive Architecture

| Component | Role | Analogy |
|-----------|------|---------|
| LLMs | Propose (fast, fuzzy, creative) | Cortex |
| Symbolic Ledger | Decide + Remember | Hippocampus + Law |
| Plans | Future commitments | Prefrontal cortex |
| Contradictions | Error signals that drive learning | Pain signals |
| Versioning | Memory of self | Autobiographical memory |

## Project Structure

* `src/Tars.Core`: Domain logic, entities, and core abstractions (Resilience, Metrics, KnowledgeGraph).
* `src/Tars.Metascript`: The workflow engine and DSL parser.
* `src/Tars.Tools`: A comprehensive library of agent capabilities (124+ tools).
* `src/Tars.Cortex`: Advanced cognitive modules (Vector Store, Grammar Distillation).
* `src/Tars.Interface.Cli`: The command-line interface and entry point.
* `src/Tars.Interface.Ui`: The Bolero/Blazor web UI.
* `grammars/`: Centralized grammar definitions (e.g., `cortex.ebnf`).
* `docs/1_Vision/`: Architectural vision and philosophy.

## Getting Started

### Prerequisites

* .NET 10 SDK
* Docker (for sandbox tools)
* Ollama or llama.cpp (for local LLM inference)

### Quick Start

```bash
# One-line setup (Windows)
irm https://raw.githubusercontent.com/GuitarAlchemist/tars/v2/scripts/setup-tars.ps1 | iex

# Or manual build
dotnet build Tars.sln
```

### Run

```bash
dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- --help
```

### Run Demos

* `tars demo-rag`
* `tars macro-demo`
* `tars evolve --max-iterations 3`

## Development Status

**Current Phase**: Phase 7 (Production Hardening) - In Progress

See [`task.md`](task.md) for current progress and [`docs/3_Roadmap/1_Plans/implementation_plan.md`](docs/3_Roadmap/1_Plans/implementation_plan.md) for the full roadmap.

### Roadmap

| Phase | Name | Status |
|-------|------|--------|
| 1-6 | Foundation, Brain, Body, Metascript, Cognitive | ✅ Complete |
| 7 | Production Hardening | 🚧 In Progress |
| 8 | Advanced Prompting Techniques | 🔜 Planned |
| **9** | **Symbolic Knowledge & Free Skills** | 🔜 Planned |
| **10** | **3D Knowledge Graph Visualization** | 🔜 Planned |

### Recent Highlights (December 2025)

- ✅ **124 tools** - Comprehensive agent toolkit
- ✅ **llama.cpp integration** - 75-97 tok/s local inference
- ✅ **Phase 9 roadmap** - Symbolic Knowledge Ledger, Internet Ingestion
- ✅ **Phase 10 roadmap** - 3D Knowledge Graph with Three.js
- ✅ **Architectural Vision** - Core thesis documented
