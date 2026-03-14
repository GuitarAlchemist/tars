# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **AgentWorkflow Computation Expression** (`Tars.Core`)
  - `agent { }` builder for composable, cancellable workflows
  - Automatic cancellation checking and warning accumulation
  - Circuit combinators: `transform`, `stabilize`, `forwardOnly`, `grounded`
  - Budget integration with `checkBudget` helper
  - Retry with exponential backoff via `retryWithBackoff`

- **Multi-Dimensional Budget System** (`Tars.Core`)
  - 12+ cost dimensions: tokens, money, duration, RAM, VRAM, CPU, attention, etc.
  - Thread-safe `BudgetGovernor` with `TryConsume`, `CanAfford`, `Allocate`
  - F# units of measure for type-safe resource tracking
  - `IsCritical` threshold detection

- **Agentic Patterns** (`Tars.Core`)
  - Chain of Thought (CoT) - linear reasoning chains
  - ReAct - reason-act-observe loops
  - Plan & Execute - plan generation with step execution
  - Based on Wei et al. (2022) and Yao et al. (2022)

- **K-Theory Graph Analysis** (`Tars.Cortex`)
  - Cycle detection using kernel computation
  - Cyclomatic complexity analysis
  - Adjacency matrix representation

- **Epistemic Governor** (`Tars.Cortex`)
  - Truth verification for agent claims
  - Belief tracking with confidence scores
  - Integration with AgentWorkflow via `grounded` combinator

- **Advanced RAG Features** (`Tars.Cortex`)
  - Query expansion, multi-hop retrieval, metadata filtering
  - Semantic chunking, sentence window, parent document retrieval
  - Time decay scoring, cross-encoder reranking
  - Answer attribution, retrieval metrics, fallback chains
  - Reciprocal Rank Fusion (RRF) for combining retrievers

- **Vector Store Implementations**
  - `InMemoryVectorStore` - thread-safe with SIMD similarity
  - `SqliteVectorStore` - persistent storage with SQLite
  - `AnnVectorStore` - approximate nearest neighbor (stub)

- **Chunking Strategies** (`Tars.Cortex`)
  - Fixed size, sentence-based, paragraph-based
  - Sliding window with overlap
  - Semantic chunking (concept-based)

- **Comprehensive Test Suite**
  - 159+ unit tests across all modules
  - 10 integration test skeletons for LLM testing
  - Edge case coverage for AgentWorkflow, Budget, Graph, Patterns

- **Feature Demos**
  - `TarsV2FeaturesDemo.fsx` - showcases Budget, K-Theory, Circuit Combinators
  - `SafetyGateDemo.fsx` - safety gate functionality

- **XML Documentation**
  - Comprehensive docs for AgentWorkflow, Budget, Patterns
  - GraphAnalyzer K-Theory documentation
  - EpistemicGovernor interface docs

### Changed

- Improved `tars diag` command with clear progress messages and timeout indicators
- Enhanced error messages throughout the codebase

## [0.1.0] - 2025-11-26

### Added

- Initial v2 codebase with core architecture ([96599b8](https://github.com/GuitarAlchemist/tars/commit/96599b8))
  - Tars.Core: Domain types and core abstractions
  - Tars.Kernel: Event bus and kernel interfaces
  - Tars.Graph: Agent graph execution engine
  - Tars.Connectors: LLM and external service connectors
  - Tars.Cortex: Grammar and DSL components
  - Tars.Evolution: Evolution engine and protocols
  - Tars.Sandbox: Docker-based sandbox execution
  - Tars.Security: Security components
  - Tars.Llm: LLM client implementations (Ollama)
  - Tars.Interface.Cli: Command-line interface with Chat and Ask commands

### Fixed

- Fixed ping demo functionality ([9973aef](https://github.com/GuitarAlchemist/tars/commit/9973aef))

### Changed

- Moved implementation plan to `docs/3_Roadmap` folder ([55c8439](https://github.com/GuitarAlchemist/tars/commit/55c8439))
- Updated implementation plan with Graphiti evaluation for hybrid memory approach ([ebfda6d](https://github.com/GuitarAlchemist/tars/commit/ebfda6d))
- Added master implementation plan with 4-phase roadmap (Foundation, Brain, Body, Soul) ([6e62c0c](https://github.com/GuitarAlchemist/tars/commit/6e62c0c))

[0.1.0]: https://github.com/GuitarAlchemist/tars/releases/tag/v0.1.0
