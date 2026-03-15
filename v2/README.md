# TARS v2 - Thinking, Acting, Reasoning System

[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com/)
[![F#](https://img.shields.io/badge/F%23-Functional-378BBA)](https://fsharp.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-809%20passing-brightgreen)]()

A modular, self-improving AI agent framework built in F#. Combines neuro-symbolic reasoning, multi-agent orchestration, probabilistic grammars, and a closed-loop evolution pipeline.

> *LLMs as stochastic generators + Symbolic systems as memory, law, and self-control.*

---

## Architecture

TARS is organized into 5 layers with 16 active projects:

```
Interface        CLI (45+ commands), MCP Server, Web UI (experimental)
                        |
Agent            AgentOrchestrator, PatternSelector, WoT Reasoning
                 (CoT, ReAct, GoT, ToT, WoT patterns)
                        |
Intelligence     VectorStores, KnowledgeGraph, EpistemicGovernor,
(Cortex)         RAG Pipeline, Chunking, Claude Code Bridge
                        |
Evolution        PromotionPipeline, GrammarGovernor, PatternBreeder,
                 MctsBridge, InsightExporter, Cross-Repo Discovery
                        |
Foundation       LLM Abstraction (Ollama/vLLM/Anthropic/llama.cpp),
                 Tool Registry (124+ tools), Docker Sandbox, DSL
```

### Project Map

| Project | Layer | Purpose |
|---------|-------|---------|
| **Tars.Interface.Cli** | Interface | Main executable. 45+ CLI commands |
| **Tars.Core** | Foundation | Domain types, patterns, governance, constitution |
| **Tars.Llm** | Foundation | Multi-backend LLM abstraction + constrained decoding |
| **Tars.Tools** | Foundation | Tool registry, standard tools, MCP resources |
| **Tars.Sandbox** | Foundation | Docker-based sandboxed code execution |
| **Tars.Security** | Foundation | Credential vault, WDAC handling |
| **Tars.Connectors** | Foundation | Ollama, vLLM, OpenWebUI connectors |
| **Tars.Kernel** | Execution | Event bus, agent lifecycle, registry |
| **Tars.Graph** | Execution | DAG executor, pipeline orchestration |
| **Tars.DSL** | Execution | Workflow-of-Thought (WoT) parser and compiler |
| **Tars.Metascript** | Execution | YAML-based scripting (frozen, legacy) |
| **Tars.Cortex** | Intelligence | Vector stores, knowledge graphs, RAG, epistemic gov |
| **Tars.Knowledge** | Intelligence | Knowledge ledger, temporal graphs, belief store |
| **Tars.Symbolic** | Intelligence | Invariants, constraint scoring, neuro-symbolic feedback |
| **Tars.LinkedData** | Intelligence | RDF/SPARQL, semantic web ingestion |
| **Tars.Evolution** | Evolution | Promotion pipeline, grammars, breeding, cross-repo bridge |

**Tests**: `Tars.Tests` — 809 tests (122 test files), all passing.

---

## Quick Start

### Prerequisites

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- [Ollama](https://ollama.ai/) with at least one model (`ollama pull llama3.2:3b`)

### Build and Run

```bash
cd v2
dotnet build
dotnet test

# Interactive chat
dotnet run --project src/Tars.Interface.Cli -- chat

# Single question
dotnet run --project src/Tars.Interface.Cli -- ask "What is the Fibonacci sequence?"

# Agent with WoT reasoning
dotnet run --project src/Tars.Interface.Cli -- agent run "Explain photosynthesis step by step"

# System diagnostics
dotnet run --project src/Tars.Interface.Cli -- diag
```

### LLM Configuration

TARS reads `~/.tars/config.json`. Default: Ollama on `localhost:11434`.

```json
{
  "llm": {
    "backend": "ollama",
    "model": "llama3.2:3b",
    "reasoningModel": "deepseek-r1:8b",
    "embeddingModel": "nomic-embed-text"
  }
}
```

Or configure via CLI:
```bash
dotnet run --project src/Tars.Interface.Cli -- config set llm:model qwen3:14b
dotnet run --project src/Tars.Interface.Cli -- config test
```

---

## CLI Commands

### Core

| Command | Description |
|---------|-------------|
| `chat` | Interactive multi-turn chat |
| `ask <prompt>` | Single-shot question |
| `agent run <goal>` | MAF-orchestrated agent with WoT reasoning |
| `agent chat` | Interactive agent with reasoning loop |
| `diag [--verbose]` | System diagnostics |
| `config [show\|set\|test]` | LLM configuration |

### Reasoning Patterns

| Command | Pattern |
|---------|---------|
| `agent cot <goal>` | Chain of Thought |
| `agent react <goal>` | ReAct (Reason + Act) |
| `agent got <goal>` | Graph of Thoughts |
| `agent tot <goal>` | Tree of Thoughts |
| `agent wot <goal>` | Workflow of Thought |

### Evolution and Self-Improvement

| Command | Description |
|---------|-------------|
| `evolve [--loop N] [--model M]` | Run evolution cycles |
| `promote [status\|lineage\|run\|report]` | Promotion pipeline control |
| `breed [--generations N]` | GA-based pattern optimization |
| `grammar [weights\|evolve\|search]` | Probabilistic grammar management |
| `meta [analyze\|gaps\|clusters\|curriculum]` | Meta-cognitive self-analysis |
| `ralph [start\|stop\|status]` | Iterative self-improvement loop |

### Knowledge and Memory

| Command | Description |
|---------|-------------|
| `know [status\|assert\|query\|fetch]` | Knowledge ledger |
| `memory-add <coll> <id> <text>` | Add to vector memory |
| `memory-search <coll> <text>` | Search vector memory |
| `ingest [--docs PATH]` | Document ingestion |
| `ingest-rdf [--format turtle]` | RDF data ingestion |

### MCP and Tools

| Command | Description |
|---------|-------------|
| `mcp server` | Start MCP server (JSON-RPC 2.0) |
| `mcp [list\|invoke]` | MCP client commands |
| `skill [list\|catalog\|install]` | Skill management |

### Demos

| Command | Description |
|---------|-------------|
| `demo-rag` | RAG pipeline demo |
| `demo-puzzle` | AI reasoning puzzles |
| `demo-escape` | Escape room puzzle solving |

---

## Key Subsystems

### Reasoning Patterns

TARS selects reasoning patterns based on goal analysis, historical outcomes, and promoted cross-repo patterns:

- **ChainOfThought**: Linear step-by-step reasoning
- **ReAct**: Reason-Act loops with tool use
- **GraphOfThoughts**: Multi-path comparison and tradeoff analysis
- **TreeOfThoughts**: Branching exploration with backtracking
- **WorkflowOfThought**: Multi-step pipelines with validation gates

Pattern selection is adaptive: the `HistoryAwareSelector` combines heuristic scores, golden trace history, recorded outcomes, and promotion index boosts.

### Evolution Pipeline

The self-improvement loop runs automatically during `tars evolve`:

```
evolve tasks ──> pattern outcomes ──> promotion pipeline ──> PromotionIndex
     ^                                                           |
     └──── PatternSelector reads index ──> agent execution ──────┘
                                               |
                                          meta-cognitive
                                          gap analysis
                                               |
                                          InsightExporter
                                          (~/.tars/insights/)
```

**Promotion Staircase**: Patterns climb 5 levels through demonstrated value:
1. Implementation (code snippet)
2. Helper (reusable function)
3. Builder (composable abstraction)
4. DslClause (WoT DSL construct)
5. GrammarRule (EBNF production rule)

**Probabilistic Grammars**: Three-force pipeline:
- EBNF constrained decoding (structured LLM output)
- PCFG Bayesian weights (Beta-Binomial updates)
- Replicator dynamics (evolutionary game theory)

### Cross-Repo Ecosystem

TARS operates in a three-repo ecosystem connected via filesystem bridges:

| Repo | Language | Role |
|------|----------|------|
| **TARS** | F# | Self-improving agent framework |
| **Guitar Alchemist** | C# | Music theory AI chatbot |
| **MachinDeOuf** | Rust | ML algorithms (GA, MCTS, PSO) |

Communication:
- **GA --> TARS**: `TraceBridgeHook` writes to `~/.ga/traces/`, `GaTraceBridge` reads
- **TARS --> GA**: `InsightExporter` writes to `~/.tars/insights/`, GA reader consumes
- **MachinDeOuf --> TARS**: `MctsBridge` calls `machin-skill` CLI, falls back to F# MCTS

See [COMPOUND_EVOLUTION.md](docs/COMPOUND_EVOLUTION.md) for the full cross-pollination story.

### MCP Server

TARS exposes tools via [Model Context Protocol](https://modelcontextprotocol.io/):

```bash
dotnet run --project src/Tars.Interface.Cli -- mcp server
```

**12 MCP tools**: 7 pattern tools + 4 grammar tools + 1 insight export tool.

Configure in Claude Code (`.claude/mcp.json`):
```json
{
  "mcpServers": {
    "tars": {
      "command": "dotnet",
      "args": ["run", "--project", "v2/src/Tars.Interface.Cli", "--", "mcp", "server"]
    }
  }
}
```

---

## Project Structure

```
v2/
├── src/
│   ├── Tars.Core/              # Domain types, patterns, governance
│   ├── Tars.Cortex/            # Intelligence: vectors, graphs, RAG, epistemic
│   ├── Tars.DSL/               # WoT DSL parser and compiler
│   ├── Tars.Evolution/         # Promotion pipeline, grammars, breeding
│   ├── Tars.Graph/             # DAG executor
│   ├── Tars.Interface.Cli/     # CLI entry point (45+ commands)
│   ├── Tars.Kernel/            # Event bus, agent lifecycle
│   ├── Tars.Knowledge/         # Knowledge ledger, temporal graphs
│   ├── Tars.LinkedData/        # RDF/SPARQL
│   ├── Tars.Llm/               # LLM abstraction + constrained decoding
│   ├── Tars.Metascript/        # YAML scripting (frozen)
│   ├── Tars.Connectors/        # External service connectors
│   ├── Tars.Sandbox/           # Docker sandboxed execution
│   ├── Tars.Security/          # Credential vault
│   ├── Tars.Symbolic/          # Neuro-symbolic feedback
│   └── Tars.Tools/             # Tool registry (124+ tools)
├── tests/
│   └── Tars.Tests/             # 809 tests across 122 files
├── grammars/                   # EBNF + JSON schema grammars
├── docs/                       # Architecture, roadmap, research
├── scripts/                    # Setup and utility scripts
└── _archive/                   # Parked build logs, debug output, session notes
```

### Key Files

| File | Purpose |
|------|---------|
| `src/Tars.Cortex/PatternSelector.fs` | Adaptive pattern selection with promotion boost |
| `src/Tars.Cortex/AgentOrchestrator.fs` | Multi-agent routing, pipeline, fan-out |
| `src/Tars.Evolution/PromotionPipeline.fs` | 7-step compound engineering loop |
| `src/Tars.Evolution/GrammarGovernor.fs` | 8-criteria promotion gating |
| `src/Tars.Evolution/Engine.fs` | Evolution engine (curriculum, execution, analysis) |
| `src/Tars.Evolution/InsightExporter.fs` | Cross-repo insight bridge |
| `src/Tars.Llm/ConstrainedDecoding.fs` | EBNF/JSON grammar enforcement |
| `src/Tars.DSL/Wot/WotParser.fs` | WoT DSL surface grammar |
| `src/Tars.Interface.Cli/Program.fs` | CLI entry point and command routing |

---

## Configuration

### Persisted State

| Path | Content |
|------|---------|
| `~/.tars/config.json` | LLM configuration |
| `~/.tars/pattern_outcomes.json` | Pattern selection history |
| `~/.tars/promotion/index.json` | Ranked promoted patterns |
| `~/.tars/insights/latest.json` | Meta-cognitive insight snapshot |
| `~/.tars/skills/` | Installed agent skills |
| `~/.tars/golden_traces/` | Validated reasoning traces |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI backend |
| `ANTHROPIC_API_KEY` | Anthropic/Claude backend |
| `GOOGLE_API_KEY` | Google Gemini backend |

---

## Development

### Building

```bash
cd v2
dotnet build           # Build all projects
dotnet test            # Run all 809 tests
dotnet test --filter "ClassName~PromotionIndex"  # Run specific tests
```

### Adding a CLI Command

1. Create `src/Tars.Interface.Cli/Commands/YourCommand.fs`
2. Add to `Tars.Interface.Cli.fsproj` compile order
3. Add match case in `src/Tars.Interface.Cli/Program.fs`

### Adding an MCP Tool

1. Define tool function in `src/Tars.Evolution/McpGaTraceBridge.fs` (or new file)
2. Add `Tool` record to `createTools()` list
3. Wire into `src/Tars.Interface.Cli/Commands/McpServer.fs`

### DSL Strategy

- **WoT DSL** (`src/Tars.DSL/`): Primary workflow language. Active development.
- **Metascript** (`src/Tars.Metascript/`): YAML-based scripting. Frozen — no new features.

See [docs/2_Architecture/DSL_Strategy.md](docs/2_Architecture/DSL_Strategy.md).

---

## Documentation

| Document | Content |
|----------|---------|
| [QUICKSTART.md](QUICKSTART.md) | Setup checklist and quick reference |
| [docs/TARS_ARCHITECTURE.md](docs/TARS_ARCHITECTURE.md) | Deep architectural reference |
| [docs/ZERO_TO_HERO.md](docs/ZERO_TO_HERO.md) | F# learning guide for TARS contributors |
| [docs/COMPOUND_EVOLUTION.md](docs/COMPOUND_EVOLUTION.md) | Cross-repo compound engineering story |
| [docs/2_Architecture/DSL_Strategy.md](docs/2_Architecture/DSL_Strategy.md) | WoT vs Metascript decision |
| [docs/SETUP.md](docs/SETUP.md) | Detailed installation instructions |
| [docs/MCP_SETUP_GUIDE.md](docs/MCP_SETUP_GUIDE.md) | MCP server configuration |

---

## License

MIT License - see [LICENSE](../LICENSE) for details.
