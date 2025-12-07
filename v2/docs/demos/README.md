# TARS v2 Demos

This directory contains documentation for all TARS v2 demonstration commands.

## Available Demos

| Demo | Command | Description |
|------|---------|-------------|
| [Ping](demo_ping.md) | `tars demo-ping` | Basic agent communication test |
| [RAG](demo_rag.md) | `tars demo-rag` | Advanced RAG capabilities (14 scenarios) |
| [Macro](demo_macro.md) | `tars macro-demo` | Evolutive grammar and workflow macros |
| [Evolve](demo_evolve.md) | `tars evolve` | Self-improvement evolution engine |

## Running Demos

```bash
# Build first
dotnet build

# Run any demo
dotnet run --project src/Tars.Interface.Cli -- <demo-command>
```

## Demo Categories

### Core Functionality

- `demo-ping`: Verifies agent lifecycle and messaging

### Knowledge & Retrieval

- `demo-rag`: Demonstrates vector search, hybrid retrieval, reranking

### Advanced Features

- `macro-demo`: Workflow composition via macros
- `evolve`: Autonomous self-improvement loop
