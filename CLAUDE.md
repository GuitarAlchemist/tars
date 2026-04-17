# TARS — Neuro-Symbolic Self-Improving Agent System

F# agent system with WoT DSL, evolution pipeline, and MCP tool surface. Part of the GuitarAlchemist ecosystem (tars + ix + ga + Demerzel).

## Build & Test

```bash
cd v2                 # working directory is v2/, not repo root
dotnet build          # full build
dotnet test           # ~820 tests (4 skipped for Docker)
```

Solution: `v2/Tars.sln`, target: `net10.0`.

## Layout

Projects under `v2/src/`. Full layout in `README.md`. Key projects:

- `Tars.Kernel` / `Tars.Core` — abstractions, workflow engine
- `Tars.DSL` — WoT DSL parser (`.wot.trsx`)
- `Tars.Cortex` / `Tars.Evolution` — agent brain, promotion pipeline
- `Tars.Llm` — LLM providers, constrained decoding
- `Tars.Tools` — tool registry, MCP adapter
- `Tars.Interface.Cli` — CLI entry point + all commands

Agents in `v2/agents/*.md` (markdown + YAML frontmatter). Grammars in `v2/grammars/`.

## Conventions

- **F# throughout** — functional-first, immutable types, `Result<>` for errors.
- **LLM access**: always use `LlmFactory.create(logger)`. Never instantiate `DefaultLlmService` directly.
- **Warnings as errors**: `TreatWarningsAsErrors=true` (NU1608 exempted in `Directory.Build.props`).
- **DSL**: WoT (`.wot.trsx`) is primary. Metascript is **frozen** — no new features.
- **Tests**: all in `tests/Tars.Tests/`, xUnit, fixtures via `Content` copy in `.fsproj`.

## CLI

```bash
dotnet run --project src/Tars.Interface.Cli -- <command>
```

Commands: `agent run`, `evolve`, `benchmark code`, `promote`, `grammar`, `mcp server`, `diag`, `chat`, `wot <file>`. See `--help` for each.

## Cross-Repo

Three repos via MCP/CLI bridges: **tars** (this repo, agent system), **ix** (`~/source/repos/ix`, Rust MCTS/skill engine), **ga** (`~/source/repos/ga`, music theory). Filesystem: `~/.tars/promotion/index.json`, `~/.ga/traces/`, `~/.ga/agents/`.

## Known Issues

- NU1608 FSharp.Core version mismatch — suppressed, harmless.
- WDAC may block `Tars.Tools.dll` — tests guarded with `requireTools()`.
- `diag reasoning` needs Postgres (`evolve` handles absence gracefully).
- Tars.LSP removed from solution (OmniSharp .NET 10 compat); project files still on disk.

For governance details, use `demerzel-*` skills.
