# TARS Release Notes

This document provides a history of TARS releases, including new features, improvements, and bug fixes.

## Version 0.1.0 (March 29, 2025)

### New Features

- **Self-Improvement System**
  - Added code analysis capabilities
  - Implemented pattern recognition for common code issues
  - Added improvement proposal generation
  - Implemented auto-accept option for proposals
  - Created learning database for recording improvements and feedback

- **Master Control Program (MCP)**
  - Implemented MCP interface
  - Added support for automatic code generation
  - Implemented triple-quoted syntax for code blocks
  - Added terminal command execution without permission prompts
  - Integrated with Augment Code

- **Hugging Face Integration**
  - Implemented model search and discovery
  - Added model downloading capabilities
  - Created conversion to Ollama format
  - Added CLI commands for Hugging Face operations

- **Language Specifications**
  - Implemented EBNF generation for TARS DSL
  - Added BNF generation capabilities
  - Created JSON schema generation
  - Added markdown documentation generation
  - Implemented CLI commands for language specifications

- **Documentation Explorer**
  - Added `docs-explore` CLI command for exploring documentation
  - Implemented search functionality to find relevant documentation
  - Added support for viewing specific documentation files
  - Created a service for parsing and displaying markdown content

### Improvements

- Made Ollama setup messages more concise
- Enhanced error handling in API communication
- Improved command-line interface with better help messages
- Added comprehensive documentation

### Bug Fixes

- Fixed JSON escaping in API communication with Ollama
- Resolved issues with file path handling in Windows
- Fixed memory leaks in long-running processes
- Addressed concurrency issues in parallel processing

## Planned for Future Releases

### Version 0.2.0

- Multi-Agent Framework
- Web Interface
- IDE Integration
- Advanced Learning System

### Version 0.3.0

- Collaborative Development
- Domain-Specific Adaptation
- REST API
- WebSocket API
