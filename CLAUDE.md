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
