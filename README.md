# TARS — Thinking, Acting, Reasoning System

[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com/)
[![F#](https://img.shields.io/badge/F%23-Functional-378BBA)](https://fsharp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-~820%20passing-brightgreen)]()

Modular, self-improving F# agent framework — neuro-symbolic reasoning, multi-agent orchestration, probabilistic grammars, closed-loop evolution. Part of a four-repo ecosystem (`tars` + [`ga`](https://github.com/GuitarAlchemist/ga) + [`ix`](https://github.com/GuitarAlchemist/ix) + [`Demerzel`](https://github.com/GuitarAlchemist/Demerzel)).

> *LLMs as stochastic generators + symbolic systems as memory, law, and self-control.*

> **Agent-facing canonical docs:** [`CLAUDE.md`](./CLAUDE.md) (breadcrumb-style). Full v2 docs: [`v2/README.md`](./v2/README.md).

---

## Quick Start

All active development lives in [`v2/`](./v2/).

```bash
cd v2
dotnet build
dotnet test                        # ~820 tests (4 skipped — Docker-gated)
dotnet format --verify-no-changes

# Interactive chat
dotnet run --project src/Tars.Interface.Cli -- chat

# Agent reasoning (Workflow-of-Thought)
dotnet run --project src/Tars.Interface.Cli -- agent run "Explain photosynthesis step by step"

# Self-improvement loop
dotnet run --project src/Tars.Interface.Cli -- evolve --loop 3 --benchmark
```

Repo harness verification: `pwsh Scripts/verify.ps1`.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              CLI / MCP Server            │
                    └──────────┬──────────────┬───────────────┘
                               │              │
                    ┌──────────▼──────┐  ┌────▼───────────────┐
                    │  Agent Cortex   │  │   WoT DSL Engine   │
                    │  (orchestrator) │  │  (.wot.trsx files) │
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
| **Knowledge** | Knowledge, Graph, LinkedData | Vector store, temporal knowledge graph, RDF/SPARQL |
| **Interface** | CLI, UI, MCP Server | Commands, Blazor dashboard, tool server |
| **Infra** | Tools, Connectors, Sandbox, Migrations | 90+ tools, external APIs, Docker sandboxing |

Full project map and conventions: [`v2/README.md`](./v2/README.md). LLM access **always** through `LlmFactory.create(logger)`; WoT (`.wot.trsx`) is the primary DSL; Metascript is **frozen**.

---

## MCP Server (150+ tools)

TARS exposes a Model Context Protocol server for Claude Code, VS Code, and other MCP clients.

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

Four sibling repositories connected via MCP federation, filesystem bridges, and the Demerzel governance constitution:

| Repo | Language | Role | Bridge |
|------|----------|------|--------|
| **TARS** | F# | Neuro-symbolic agent system, **cross-model theory validator** | *this repo* |
| **[ga](https://github.com/GuitarAlchemist/ga)** | C# / F# DSL | Music theory domain + agentic chatbot | MCP + `~/.ga/traces/` |
| **[ix](https://github.com/GuitarAlchemist/ix)** | Rust | 39 ML tools (stats, neural nets, optimization, game theory) | MCP federation |
| **[Demerzel](https://github.com/GuitarAlchemist/Demerzel)** | Governance | 11-article epistemic constitution, ACP server, tribunal | Submodule at `governance/demerzel/` |

In the ecosystem, **TARS is the F# theory validator** — used cross-model to validate music-theory hypotheses originating from `ga`. Cross-repo contract drafts under `governance/demerzel/docs/contracts/v0.1.x` are explicitly **not frozen** until the owning plan's Phase 4 milestone.

### ix ML Integration

```bash
# One-call ML pipeline: load CSV → preprocess → train → evaluate → persist
ix_ml_pipeline

# 39 algorithm tools via MCP
ix_ml_predict, ix_statistics_*, ix_optimization_*, ix_neural_*, ix_game_theory_*
```

### GA Trace Bridge

TARS discovers patterns from Guitar Alchemist orchestrator traces (`~/.ga/traces/`) and promotes them through the evolution pipeline. 5 pattern families (21 artifacts) seeded from static code analysis. New: `Tars.Evolution/ChatbotClaimsBridge.fs` extracts skill-routing claims from the `ga` chatbot for cross-model validation.

---

## Self-Improvement Loop

TARS has a closed self-improvement loop:

1. **Evolve** — run tasks, observe outcomes
2. **Extract** — identify recurring patterns from execution traces
3. **Promote** — 7-step pipeline: Inspect → Extract → Classify → Propose → Validate → Persist → Govern
4. **Index** — Bayesian-weighted ranking persisted to `~/.tars/promotion/index.json`
5. **Select** — `PatternSelector` reads index, context-gated boost for next execution
6. **Repeat** — `tars evolve --loop N` runs N full cycles back-to-back

Patterns climb: *Implementation → Helper → Builder → DslClause → GrammarRule*.

---

## AI agent integrations

Issues labeled `ready-for-agent` are automatically delegated to Jules via `.github/workflows/jules-auto-delegate.yml`.

---

## AI discipline (Karpathy + Cherny)

Every code-touching turn applies four rules: **think before coding · simplicity first · surgical changes · goal-driven execution**. Session continuity uses the Cherny pattern:

- `/digest` — captures session state (cursor, in-flight work, hypotheses, success criteria) to `state/digests/latest.md`. Auto-fallback via `.claude/hooks/precompact-digest.ps1`; auto-injected on next session via `.claude/hooks/sessionstart-digest.ps1`.
- `/learnings` — captures surprises to `docs/solutions/<category>/<date>-<topic>.md`.
- `/correct` — turns user corrections into permanent rules in [`CLAUDE.md`](./CLAUDE.md)'s **Session-learned rules**.

CI enforces hook integrity via [`.github/workflows/karpathy-cherny-discipline.yml`](./.github/workflows/karpathy-cherny-discipline.yml).

---

## Quality cadence

Golden artifacts under `v2/baselines/**` are schema-pinned via `v2/baselines/_schema.json` — drift triggers a CI failure ([PR #21](https://github.com/GuitarAlchemist/tars/pull/21)). Tribunal verdicts dispatch to the Demerzel constitutional tribunal via [`.github/workflows/qa-verdict-dispatch.yml`](./.github/workflows/qa-verdict-dispatch.yml) ([PR #22](https://github.com/GuitarAlchemist/tars/pull/22)). Agent risk surfaces through `.github/workflows/agent-blackbox.yml` against `agent-blackbox.policy.json`.

---

## Repository Layout

| Area | Status | Description |
|------|--------|-------------|
| **[v2/](./v2/)** | **Active** | F# neuro-symbolic agent framework — all new development happens here |
| **[v2/agents/](./v2/agents/)** | Active | Declarative agent definitions (Markdown + YAML frontmatter) |
| **[v2/grammars/](./v2/grammars/)** | Active | EBNF grammars for constrained decoding |
| **[v2/baselines/](./v2/baselines/)** | Active | Schema-pinned golden artifacts (validated in CI) |
| **[v2/puzzles/](./v2/puzzles/)** | Active | WoT puzzle benchmarks |
| **[v2/docs/](./v2/docs/)** | Active | Architecture, roadmap, plans |
| **[governance/](./governance/)** | Active | Demerzel submodule — constitution, policies, schemas |
| **[state/](./state/)** | Active | Digests, quality snapshots, beliefs, PDCA, knowledge |
| **[v1/](./v1/)** | Legacy | Original C#/.NET implementation — retained for reference, not maintained |
| **[archive/](./archive/)** | Archived | 119 historical reports + 245 legacy scripts moved from root |

**CI target:** `v2/` only (`working-directory: v2` in `dotnet.yml`). Archived content does not affect builds.

---

## License

MIT — see [LICENSE](./LICENSE).
