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
* `tars know status --pg` - Knowledge Ledger status
* `tars know fetch "Topic" --pg` - Fetch from Wikipedia
* `tars know propose "Topic" --pg` - Extract beliefs via LLM

## Development Status

**Current Phase**: Phase 13 (Neuro-Symbolic Foundations) - ✅ **COMPLETE!**

See [`task.md`](task.md) for current progress and [`docs/3_Roadmap/1_Plans/implementation_plan.md`](docs/3_Roadmap/1_Plans/implementation_plan.md) for the full roadmap.

### Roadmap

| Phase | Name | Status |
|-------|------|--------|
| 1-6 | Foundation, Brain, Body, Metascript, Cognitive | ✅ Complete |
| 7 | Production Hardening | 🚧 In Progress |
| 8 | Advanced Prompting Techniques | 🚧 Partial (GoT implemented) |
| **9** | **Symbolic Knowledge & Internet Ingestion** | 🚧 In Progress |
| **10** | **3D Knowledge Graph Visualization** | 🔜 Planned |
| 11 | Cognitive Grounding & Production Intelligence | 🔜 Planned |
| 12 | Web of Things Integration | 🔜 Planned |
| **13** | **Neuro-Symbolic Foundations** 🆕 | ✅ **COMPLETE** (Dec 2024) |
| **14** | **Agent Constitutions** 🆕 | 🔜 Planned (Q2 2025) |
| **15** | **Symbolic Reflection** 🆕 | 🔜 Planned (Q3 2025) |
| **16** | **Context Engineering & Validation** 🆕 | 🔜 Planned (Q4 2025) |

### Recent Highlights (December 2024)

- ✅ **Phase 13 Complete: Neuro-Symbolic Foundations** - 950+ lines of production code!
  - Symbolic Invariants System (6 types, continuous scoring)
  - Constraint Scoring Engine (Logic Tensor Network-style)
  - Neural-Symbolic Feedback Loop (THE KEY INNOVATION!)
  - 19/19 tests passing
  - Agent selection biasing, prompt shaping, mutation filtering
- ✅ **Neuro-Symbolic AI Roadmap** - Comprehensive 3-phase plan (8,500+ lines docs)
- ✅ **Evolution Engine Fixes** - All 7 critical issues resolved (JSON parsing, JSONB, success criteria, graph persistence, memory overflow)
- ✅ **Puzzle Demo System** - 9 diverse AI reasoning benchmarks (River Crossing, Logic Grids, Math, Cryptarithmetic, etc.)
- ✅ **Comprehensive Test Suite** - Regression protection for all evolution fixes
- ✅ **Agent Memory Truncation** - Prevents HTTP 400 errors from oversized requests
- ✅ **Internet Ingestion Pipeline** - Fetch Wikipedia, extract triples via LLM
- ✅ **124 tools** - Comprehensive agent toolkit
- ✅ **llama.cpp integration** - 75-97 tok/s local inference with 32K context support
- ✅ **Docker Compose** - PostgreSQL with pgvector included
- ✅ **Architectural Vision** - Core thesis documented
