# Tech Stack

## Context

Global tech stack defaults for Agent OS projects, overridable in project-specific `.agent-os/product/tech-stack.md`.

## TARS Project Stack

- **Primary Language**: F# 9.0+ (functional logic, DSL, reasoning)
- **Infrastructure Language**: C# 12+ (.NET 9, CLI, integrations)
- **Framework**: .NET 9.0+
- **Architecture**: Clean Architecture with Dependency Injection
- **Database**: 
  - Primary: SQLite (local development)
  - Production: PostgreSQL 17+ or SQL Server
  - Vector Store: ChromaDB (Docker) or Custom CUDA Vector Store
- **AI/ML Stack**:
  - Local LLM: Ollama
  - GPU Acceleration: CUDA (WSL for compilation)
  - Vector Operations: Custom CUDA kernels
  - Inference Engine: Custom FLUX engine
- **UI Framework**: Elmish (Model-View-Update pattern)
- **CLI Framework**: Spectre.Console
- **Testing**: 
  - Unit Tests: xUnit
  - Coverage Target: 80%+
  - Approach: TDD preferred
- **Package Management**: 
  - .NET: dotnet CLI
  - F#: Paket (if needed)
- **Build System**: .NET SDK build system
- **Version Control**: Git with conventional commits
- **Containerization**: Docker with docker-compose
- **Development Environment**: 
  - IDE: VS Code or Visual Studio
  - Extensions: Ionide for F#
- **Code Quality**:
  - Linting: FSharpLint
  - Formatting: Fantomas
  - Analysis: Treat FS0988 warnings as errors
- **Documentation**: Markdown with auto-generation
- **Deployment**: Local-first, Docker containers
- **Monitoring**: Built-in diagnostics and metrics

## TARS-Specific Preferences

- **Metascript Language**: FLUX (multi-modal with Wolfram/Julia support)
- **Type System**: Advanced typing (AGDA dependent types, IDRIS linear types, LEAN refinement types)
- **Agent Architecture**: Multi-agent with specialized teams
- **Memory Systems**: Episodic, semantic, and procedural memory
- **Self-Improvement**: Autonomous reasoning and code generation
- **Performance**: CUDA-accelerated with 184M+ searches/second target
- **Standards**: Zero tolerance for simulations/placeholders - real implementations only
