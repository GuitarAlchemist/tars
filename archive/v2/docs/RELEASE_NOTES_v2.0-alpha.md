# TARS v2.0-alpha Release Notes

**Release Date:** December 2025  
**Version:** 2.0.0-alpha

---

## 🎯 Overview

TARS v2.0-alpha represents a complete architectural rewrite of the TARS autonomous reasoning system. Built on .NET 10 (F#), this release establishes the foundation for a modular, event-driven AI agent framework with advanced cognitive capabilities.

---

## ✨ Key Features

### Core Architecture

- **Hexagonal/Clean Architecture** separating Domain, Core, and Interfaces
- **Event-driven messaging** via `EventBus` with backpressure support
- **Docker Sandbox** for secure code execution
- **Security Core** with `CredentialVault` and `FilesystemPolicy`

### Brain (Phase 2)

- **LLM Integration**: Multi-provider support (Ollama, vLLM, OpenAI-compatible)
- **Vector Stores**: InMemory, SQLite, ChromaDB implementations
- **Knowledge Graph**: Temporal graph with bi-temporal tracking
- **Entity Resolution**: Deduplication via embeddings
- **Fact Extraction**: LLM-powered relationship extraction

### Body (Phase 3)

- **75+ Tools**: System access, code analysis, git integration, MCP support
- **Terminal UI**: Spectre.Console with interactive chat
- **MCP Client**: Model Context Protocol for external tool integration

### Soul (Phase 4)

- **Evolution Engine**: Self-improving curriculum with task generation
- **Reflection Agent**: Adaptive feedback loops
- **Workflow Optimizer**: Runtime performance tuning

### Mind (Phase 5)

- **Metascript DSL**: JSON-based workflow definitions
- **Workflow Engine**: Variable resolution, parallel execution
- **Macro System**: Reusable workflow templates

### Cognitive Architecture (Phase 6)

- **Budget Governor**: Token and cost control
- **Semantic Speech Acts**: FIPA-ACL inspired messaging
- **Circuit Flow Control**: BoundedChannels, BufferAgents, Gates
- **Epistemic Governor**: Belief verification and principle extraction
- **Context Compression**: LLM-powered summarization

---

## 🖥️ CLI Commands

| Command | Description |
|---------|-------------|
| `tars chat` | Interactive AI chat with RAG |
| `tars evolve` | Run the self-evolution loop |
| `tars run <script>` | Execute a Metascript workflow |
| `tars mcp` | Connect to MCP servers |
| `tars demo-ping` | Verify core system |
| `tars smem query` | Query semantic memory |

---

## 📊 Test Coverage

- **240+ tests passing**
- Circuit Flow Control fully tested
- Golden Run integration tests
- Epistemic Governor validation

---

## 🔧 Requirements

- .NET 10 SDK
- Docker (for sandbox tools)
- Ollama (for local LLM)

---

## 🚀 Getting Started

```bash
# Build
dotnet build Tars.sln

# Run tests
dotnet test Tars.sln

# Start chat
dotnet run --project src/Tars.Interface.Cli -- chat

# Run evolution
dotnet run --project src/Tars.Interface.Cli -- evolve --demo
```

---

## 📋 Known Issues

- FSharp.Core version warning (NU1608) - cosmetic only
- Golden Run test skipped in CI (SemanticKernelProvider dependency)

---

## 🔮 Roadmap to v2.1-beta

- [ ] Persistent ChromaDB integration
- [ ] Enhanced MCP server capabilities  
- [ ] JetBrains IDE plugin
- [ ] Benchmarks on standard tasks (HotpotQA, LoCoMo)
