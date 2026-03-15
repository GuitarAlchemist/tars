# TARS - Thinking, Acting, Reasoning System

[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com/)
[![F#](https://img.shields.io/badge/F%23-Functional-378BBA)](https://fsharp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-809%20passing-brightgreen)]()

A modular, self-improving AI agent framework built in F#. Combines neuro-symbolic reasoning, multi-agent orchestration, probabilistic grammars, and a closed-loop evolution pipeline.

> *LLMs as stochastic generators + Symbolic systems as memory, law, and self-control.*

---

## Quick Start

The active project lives in [`v2/`](./v2/). All development happens there.

```bash
cd v2
dotnet build
dotnet test
```

For interactive chat:

```bash
cd v2
dotnet run --project src/Tars.Interface.Cli -- chat
```

For agent-based reasoning:

```bash
cd v2
dotnet run --project src/Tars.Interface.Cli -- agent run "Explain photosynthesis step by step"
```

---

## Documentation

See **[v2/README.md](./v2/README.md)** for full documentation including architecture, CLI commands, configuration, and development guides.

---

## Repository Layout

| Directory | Description |
|-----------|-------------|
| **[v2/](./v2/)** | Active project -- F# neuro-symbolic agent framework |
| Root-level C# projects | Legacy, archived, no longer maintained |

> **Note:** Legacy C# projects in the root directory are archived and no longer maintained. All active development targets `v2/`.

---

## License

MIT License -- see [LICENSE](./LICENSE) for details.
