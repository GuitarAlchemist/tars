# TARS - The Automated Reasoning System

> **🚀 TARS v2 is currently in active development!**

## What is TARS?

TARS (The Automated Reasoning System & AI Inference Engine) is a high-performance, F#-based AI system designed for autonomous reasoning, self-improvement, and multi-agent collaboration.

---

## Repository Structure

This repository contains two major versions:

### 📁 **[v1/](./v1/)** - TARS v1 (Legacy)

- Production-ready AI inference engine (Ollama-compatible)
- CUDA-accelerated vector operations
- Multi-agent research system
- Grammar distillation and FLUX metascripts

**Status:** Feature-complete, in maintenance mode.  
**Documentation:** See [v1/README.md](./v1/README.md)

---

### 📁 **[v2/](./v2/)** - TARS v2 (Active Development) ⭐

**Current Focus:** Comprehensive architecture and operational design phase.

TARS v2 represents a complete reimagining of the system with:

- **Micro-Kernel Architecture** (F# core)
- **Model Context Protocol (MCP)** for tool integration
- **AutoGen + Semantic Kernel** for multi-agent orchestration
- **Local-first** approach (Docker, Chroma, self-hosted tools)
- **Enterprise-grade security** (sandboxing, vault, HITL)
- **Advanced testing** (Golden Runs for agent behavior)
- **Cost management** (token accounting, budget enforcement)

**Status:** Analysis phase complete, implementation planning in progress.  
👉 **[Start here: v2/GEMINI.md](./v2/GEMINI.md)**

#### Key Documentation (v2)

- [Vision & Goals](./v2/docs/0_Vision/goals_and_vision.md)
- [Architecture Overview](./v2/docs/2_Analysis/Architecture/00_Overview/tars_v2_architecture.md)
- [Security Model](./v2/docs/2_Analysis/Architecture/03_Operational/security_model.md)
- [Testing Strategy](./v2/docs/2_Analysis/Architecture/03_Operational/testing_strategy.md)

---

## Quick Start

### For v1 (Production)

```bash
cd v1
docker build -f Dockerfile.ai -t tars-ai:latest .
docker run -d -p 11434:11434 --gpus all tars-ai:latest
```

### For v2 (Development)

```bash
cd v2
# Documentation only at this stage
# See v2/GEMINI.md for development roadmap
```

---

## Contributing

TARS v2 is actively seeking contributions! Areas of focus:

- F# Micro-Kernel implementation
- MCP server integrations
- AutoGen agent definitions
- Testing infrastructure

See [v2/docs/2_Analysis/Architecture/03_Operational/BRIDGING_THE_GAPS.md](./v2/docs/2_Analysis/Architecture/03_Operational/BRIDGING_THE_GAPS.md) for implementation priorities.

---

## License

See [LICENSE](./LICENSE) for details.

---

## Project Status

| Component | v1 | v2 |
|:----------|:---|:---|
| **Status** | ✅ Production | 🚧 Planning |
| **Architecture** | Monolithic | Micro-Kernel |
| **Agent Runtime** | Custom F# | AutoGen + SK |
| **Tool Integration** | Hardcoded | MCP Protocol |
| **Deployment** | Docker | Docker Compose |

For the latest updates, see the [v2 branch](https://github.com/GuitarAlchemist/tars/tree/pre_v2/v2).
