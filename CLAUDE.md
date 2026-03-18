# TARS — Neuro-Symbolic Self-Improving Agent System

## Build & Test

```bash
cd v2
dotnet build          # build everything
dotnet test           # run all tests (~820 tests, 4 skipped for Docker)
```

Working directory is **`v2/`**, not the repo root.
Solution: `v2/Tars.sln` — target framework: `net10.0`.

## Project Layout

```
v2/
├── src/
│   ├── Tars.Kernel/          # Core abstractions, event bus
│   ├── Tars.Core/            # Domain types, workflow engine
│   ├── Tars.DSL/             # WoT DSL parser & compiler (.wot.trsx)
│   ├── Tars.Llm/             # LLM providers, constrained decoding
│   ├── Tars.Cortex/          # Agent brain — PatternSelector, WoT agent
│   ├── Tars.Evolution/       # Promotion pipeline, grammar governor, breeding
│   ├── Tars.Tools/           # Tool registry, MCP tool adapter
│   ├── Tars.Knowledge/       # Vector store, knowledge ledger
│   ├── Tars.Graph/           # Knowledge graph, temporal graph
│   ├── Tars.Symbolic/        # Symbolic reflection, LPA
│   ├── Tars.Security/        # Credential vault, policy engine
│   ├── Tars.Connectors/      # External service connectors
│   ├── Tars.Sandbox/         # Docker sandboxed execution
│   ├── Tars.Metascript/      # Legacy scripting (frozen — no new features)
│   ├── Tars.Interface.Cli/   # CLI entry point + all commands
│   └── Tars.Interface.Ui/    # Blazor web UI (experimental)
├── tests/Tars.Tests/         # Single test project, xUnit
├── agents/                   # Declarative agent definitions (.md)
├── grammars/                 # EBNF grammars for constrained decoding
├── puzzles/                  # WoT puzzle benchmarks
└── docs/                     # Architecture, roadmap, plans
```

## Key Conventions

- **F# throughout** — functional-first, immutable types, `Result<>` for errors
- **LLM access**: Always use `LlmFactory.create(logger)` — never instantiate `DefaultLlmService` directly
- **CLI commands**: Each in `src/Tars.Interface.Cli/Commands/`, routed from `Program.fs`
- **Agent definitions**: Markdown + YAML frontmatter in `agents/*.md`, loaded by `AgentRegistry`
- **Tests**: All in `tests/Tars.Tests/`, fixtures via `Content` copy in `.fsproj`
- **Warnings as errors**: `TreatWarningsAsErrors=true` (NU1608 exempted in `Directory.Build.props`)
- **DSL**: WoT (.wot.trsx) is the primary workflow language; Metascript is frozen
- **MCP tools**: Pattern tools in `McpPatternResources.fs`, grammar in `McpGrammarTools.fs`, GA trace in `McpGaTraceBridge.fs`

## CLI Quick Reference

```bash
dotnet run --project src/Tars.Interface.Cli -- <command>
```

| Command | Purpose |
|---------|---------|
| `agent run` | Run a WoT agent with MAF orchestration |
| `evolve [--loop N] [--benchmark]` | Self-improvement loop |
| `benchmark code [run\|status\|report]` | F# coding benchmarks (19 problems) |
| `promote [status\|lineage\|run\|report]` | Promotion pipeline |
| `grammar [weights\|evolve\|search]` | Probabilistic grammar management |
| `mcp server` | Start MCP server (stdio) |
| `diag [reasoning\|health]` | Diagnostics |
| `chat` | Interactive chat |
| `wot <file.wot.trsx>` | Execute a WoT workflow |

## Cross-Repo Ecosystem

Three repos connected via MCP/CLI bridges:
- **TARS** (this repo) — agent system
- **ix** (`~/source/repos/ix`) — Rust MCTS/skill engine
- **GA** (`~/source/repos/ga`) — Guitar Alchemist music theory

Filesystem bridges: `~/.tars/promotion/index.json`, `~/.ga/traces/`, `~/.ga/agents/`

## Known Issues

- NU1608: FSharp.Core version mismatch (10.0.101 vs 10.0.104) — suppressed, harmless
- WDAC: Windows Application Control may block Tars.Tools.dll — tests guarded with `requireTools()`
- `diag reasoning` crashes without Postgres — `evolve` handles it gracefully
- Tars.LSP removed from solution (OmniSharp .NET 10 compat) — project files still on disk


<!-- BEGIN DEMERZEL GOVERNANCE -->
# Demerzel Governance Integration

This repo participates in the Demerzel governance framework.

## Governance Framework

All agents in this repo are governed by the Demerzel constitutional hierarchy:

- **Root constitution:** governance/demerzel/constitutions/asimov.constitution.md (Articles 0-5: Laws of Robotics + LawZero principles)
- **Governance coordinator:** Demerzel (see governance/demerzel/constitutions/demerzel-mandate.md)
- **Operational ethics:** governance/demerzel/constitutions/default.constitution.md (Articles 1-7)
- **Harm taxonomy:** governance/demerzel/constitutions/harm-taxonomy.md

## Policy Compliance

Agents must comply with all Demerzel policies:

- **Alignment:** Verify actions serve user intent (confidence thresholds: 0.9 autonomous, 0.7 with note, 0.5 confirm, 0.3 escalate)
- **Rollback:** Revert failed changes automatically; pause autonomous changes after automatic rollback
- **Self-modification:** Never modify constitutional articles, disable audit logging, or remove safety checks
- **Kaizen:** Follow PDCA cycle for improvements; classify as reactive/proactive/innovative before acting
- **Reconnaissance:** Respond to Demerzel reconnaissance requests with belief snapshots and compliance reports
- **Scientific objectivity:** Tag evidence as empirical/inferential/subjective; generator/estimator accountability
- **Streeling:** Accept knowledge transfers from Seldon; report comprehension via belief state assessment

## Galactic Protocol

This repo communicates with Demerzel via the Galactic Protocol:

- **Inbound (from Demerzel):** Governance directives, knowledge packages
- **Outbound (to Demerzel):** Compliance reports, belief snapshots, learning outcomes
- **Message formats:** See governance/demerzel/schemas/contracts/

## Belief State Persistence

This repo maintains a `state/` directory for belief persistence:

- `state/beliefs/` — Tetravalent belief states (*.belief.json)
- `state/pdca/` — PDCA cycle tracking (*.pdca.json)
- `state/knowledge/` — Knowledge transfer records (*.knowledge.json)
- `state/snapshots/` — Belief snapshots for reconnaissance (*.snapshot.json)

File naming: `{date}-{short-description}.{type}.json`

## Agent Requirements

Every persona in this repo must include:

- `affordances` — Explicit list of permitted actions
- `goal_directedness` — One of: none, task-scoped, session-scoped
- `estimator_pairing` — Neutral evaluator persona (typically skeptical-auditor)
- All fields required by governance/demerzel/schemas/persona.schema.json
<!-- END DEMERZEL GOVERNANCE -->
