# TARS v2 - Project Documentation

## Overview

**TARS v2** is the current major iteration of the Tars autonomous reasoning system. It features a modular, event-driven architecture built on **.NET 10 (F#)**, implementing advanced patterns for agency, memory, and self-evolution.

## Key Features

* **Core Architecture**: Hexagonal/Clean Architecture separating Domain, Core, and Interfaces.
* **Graphiti Knowledge Graph**: A temporal, episodic memory system (`TemporalKnowledgeGraph`) replacing the legacy graph.
* **Metascript Engine**: A recursive, macro-capable workflow engine for defining agent behaviors.
* **Resilience & Monitoring**: Functional circuit breakers, retry policies, and metrics.
* **Dynamic Tooling**: A robust tool registry supporting 75+ tools, including system access, code analysis, and git integration.

## Project Structure

* `src/Tars.Core`: Domain logic, entities, and core abstractions (Resilience, Metrics, KnowledgeGraph).
* `src/Tars.Metascript`: The workflow engine and DSL parser.
* `src/Tars.Tools`: A comprehensive library of agent capabilities.
* `src/Tars.Cortex`: Advanced cognitive modules (Vector Store, Grammar Distillation).
* `src/Tars.Interface.Cli`: The command-line interface and entry point.
* `grammars/`: Centralized grammar definitions (e.g., `cortex.ebnf`).

## Getting Started

### Prerequisites

* .NET 10 SDK
* Docker (for sandbox tools)

### Build

```bash
dotnet build Tars.sln
```

### Run

```bash
dotnet run --project src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- --help
```

### Run Demos

* `tars demo-rag`
* `tars macro-demo`

## Development Status

Active development. See `task.md` for current progress and `implementation_plan.md` for upcoming features.
