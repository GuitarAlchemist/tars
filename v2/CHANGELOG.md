# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
