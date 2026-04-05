# TARS - Thinking, Acting, Reasoning System

[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com/)
[![F#](https://img.shields.io/badge/F%23-Functional-378BBA)](https://fsharp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-819%20passing-brightgreen)]()

A modular, self-improving AI agent framework built in F#. Combines neuro-symbolic reasoning, multi-agent orchestration, probabilistic grammars, and a closed-loop evolution pipeline.

> *LLMs as stochastic generators + Symbolic systems as memory, law, and self-control.*

---

## Quick Start

The active project lives in [`v2/`](./v2/). All development happens there.

```bash
cd v2
dotnet build
dotnet test          # 819 tests passing
```

Interactive chat:

```bash
dotnet run --project src/Tars.Interface.Cli -- chat
```

Agent-based reasoning:

```bash
dotnet run --project src/Tars.Interface.Cli -- agent run "Explain photosynthesis step by step"
```

Self-improvement loop:

```bash
dotnet run --project src/Tars.Interface.Cli -- evolve --loop 3 --benchmark
```

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              CLI / MCP Server            │
                    └──────────┬──────────────┬───────────────┘
                               │              │
                    ┌──────────▼──────┐  ┌────▼───────────────┐
                    │  Agent Cortex   │  │   WoT DSL Engine   │
                    │  (orchestrator) │  │  (.wot.trsx files)  │
                    └──────┬──────────┘  └────────┬───────────┘
                           │                      │
          ┌────────────────┼──────────────────────┤
          │                │                      │
  ┌───────▼───────┐ ┌─────▼───────┐  ┌───────────▼──────────┐
  │  LLM Factory  │ │  Evolution  │  │  Knowledge / Graph   │
  │  (providers)  │ │  (promote,  │  │  (vector, temporal,  │
  │               │ │   grammar)  │  │   symbolic)          │
  └───────────────┘ └─────────────┘  └──────────────────────┘
```

| Layer | Projects | Purpose |
|-------|----------|---------|
| **Core** | Kernel, Core, Security, Llm | Abstractions, event bus, LLM providers, credentials |
| **Intelligence** | Cortex, Evolution, DSL, Symbolic | Agent brain, promotion pipeline, WoT compiler, reflection |
| **Knowledge** | Knowledge, Graph | Vector store, temporal knowledge graph, ledger |
| **Interface** | CLI, UI, MCP Server | Commands, Blazor dashboard, tool server |
| **Infra** | Tools, Connectors, Sandbox | 90+ tools, external APIs, Docker sandboxing |

---

## MCP Server (150+ tools)

TARS exposes a Model Context Protocol server with 150+ tools for use in Claude Code, VS Code, and other MCP clients.

```bash
dotnet run --project src/Tars.Interface.Cli -- mcp server
```

| Category | Tools | Highlights |
|----------|-------|------------|
| **Code & F#** | ~30 | `read_code`, `fsharp_compile`, `analyze_code`, `search_codebase`, `refactor_*` |
| **Knowledge Graph** | ~12 | `graph_add_node`, `graph_query`, `temporal_detect_contradictions` |
| **Agent & Model** | ~15 | `list_agents`, `delegate_task`, `switch_model`, `recommend_model` |
| **Claude Code Bridge** | 5 | `tars_compile_plan`, `tars_execute_step`, `tars_validate_step`, `tars_memory_op`, `tars_complete_plan` |
| **Probabilistic Grammar** | 4 | `grammar_weights`, `grammar_update`, `grammar_evolve`, `grammar_search` |
| **Pattern Library** | 7 | `list_patterns`, `suggest_pattern`, `run_promotion_pipeline`, `promotion_status` |
| **GA Trace Bridge** | 4 | `ingest_ga_traces`, `ga_trace_stats`, `promotion_index`, `export_insights` |
| **MCP Management** | 4 | `list_mcp_servers`, `configure_mcp_server`, `search_mcp_servers` |
| **File, Web, Git, Utils** | 60+ | `http_get`, `git_diff`, `run_command`, `search_web`, `fetch_arxiv` |

---

## CLI Commands

```bash
cd v2
dotnet run --project src/Tars.Interface.Cli -- <command>
```

| Command | Purpose |
|---------|---------|
| `chat` | Interactive LLM chat session |
| `agent run "<goal>"` | Run a WoT agent with multi-agent orchestration |
| `evolve [--loop N] [--benchmark]` | Self-improvement loop with optional benchmarking |
| `benchmark code [run\|status\|report]` | 19 curated F# coding challenges across 4 difficulty tiers |
| `promote [status\|lineage\|run\|report]` | 7-step promotion pipeline (Inspect-Extract-Classify-Propose-Validate-Persist-Govern) |
| `grammar [weights\|evolve\|search]` | Probabilistic grammar management with Bayesian updates |
| `wot <file.wot.trsx>` | Execute a Workflow-of-Thought file |
| `mcp server` | Start MCP server (stdio) |
| `diag [reasoning\|health]` | Diagnostics and health checks |
| `config [get\|set]` | Configuration management |

---

## Cross-Repo Ecosystem

Three repositories connected via MCP federation and filesystem bridges:

| Repo | Language | Role | Link |
|------|----------|------|------|
| **TARS** | F# | Neuro-symbolic agent system | *this repo* |
| **[ix](https://github.com/GuitarAlchemist/ix)** | Rust | 39 ML tools (stats, neural nets, optimization, game theory) | MCP federation |
| **[GA](https://github.com/GuitarAlchemist/ga)** | C# | Music theory domain + chatbot | MCP + trace bridge |

All governed by the [Demerzel](https://github.com/GuitarAlchemist/Demerzel) constitution (11 articles, 12 personas, tetravalent logic).

### ix ML Integration

```bash
# One-call ML pipeline: load CSV → preprocess → train → evaluate → persist
ix_ml_pipeline

# 39 algorithm tools via MCP
ix_ml_predict, ix_statistics_*, ix_optimization_*, ix_neural_*, ix_game_theory_*
```

### GA Trace Bridge

TARS discovers patterns from Guitar Alchemist orchestrator traces (`~/.ga/traces/`) and promotes them through the evolution pipeline. 5 pattern families (21 artifacts) seeded from static code analysis.

---

## Self-Improvement Loop

TARS has a closed self-improvement loop:

1. **Evolve** — Run tasks, observe outcomes
2. **Extract** — Identify recurring patterns from execution traces
3. **Promote** — 7-step pipeline: Inspect → Extract → Classify → Propose → Validate → Persist → Govern
4. **Index** — Bayesian-weighted ranking persisted to `~/.tars/promotion/index.json`
5. **Select** — PatternSelector reads index, context-gated boost for next execution
6. **Repeat** — `tars evolve --loop N` runs N full cycles back-to-back

Patterns climb: *Implementation → Helper → Builder → DslClause → GrammarRule*

---

## Documentation

See **[v2/README.md](./v2/README.md)** for full documentation including architecture details, configuration, and development guides.

---

## Active Boundaries

| Area | Status | Description |
|------|--------|-------------|
| **`v2/`** | **Active** | F# neuro-symbolic agent framework — all new development happens here |
| **`v2/agents/`** | **Active** | Declarative agent definitions (Markdown + YAML frontmatter) |
| **`v2/grammars/`** | **Active** | EBNF grammars for constrained decoding |
| **`v2/puzzles/`** | **Active** | WoT puzzle benchmarks |
| **`v2/docs/`** | **Active** | Architecture, roadmap, plans |
| **`v1/`** | **Legacy** | Original C#/.NET implementation — retained for reference, not maintained |
| **`archive/docs/`** | **Archived** | 119 historical reports, session summaries, and analysis docs moved from root |
| **`archive/scripts/`** | **Archived** | 245 legacy PowerShell and F# scripts moved from root |
| **`governance/`** | **Active** | Demerzel submodule — constitutions, policies, schemas |

**CI target:** `v2/` only (`working-directory: v2` in `dotnet.yml`). Archived content does not affect builds.

## Repository Layout

| Directory | Description |
|-----------|-------------|
| **[v2/](./v2/)** | Active project — F# neuro-symbolic agent framework |
| **[v2/agents/](./v2/agents/)** | Declarative agent definitions (Markdown + YAML frontmatter) |
| **[v2/grammars/](./v2/grammars/)** | EBNF grammars for constrained decoding |
| **[v2/puzzles/](./v2/puzzles/)** | WoT puzzle benchmarks |
| **[v2/docs/](./v2/docs/)** | Architecture, roadmap, plans |
| **[archive/](./archive/)** | Legacy docs and scripts moved from root |
| **[v1/](./v1/)** | Legacy C# projects — archived, not maintained |

---

## License

MIT License — see [LICENSE](./LICENSE) for details.
