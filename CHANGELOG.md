# Changelog

All notable changes to the TARS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added - TARS Engine API Injection (MAJOR FEATURE)
- **Complete TARS Engine API Injection for F# and C#**
  - Implemented comprehensive ITarsEngineApi interface with 8 core services
  - Added TarsEngineApiImpl with real working functionality (no simulations)
  - Created thread-safe TarsApiRegistry for global API access
  - Built TarsExecutionContext with security and resource management
  - Implemented 3-tier security policies (Restrictive, Standard, Unrestricted)

- **API Services Implemented**
  - VectorStore API: Search, Add, Delete, GetSimilar, CreateIndex, GetIndexInfo
  - LLM Service API: Complete, Chat, Embed, ListModels, SetParameters
  - Agent Coordination API: Spawn, SendMessage, GetStatus, Terminate, ListActive, Broadcast
  - File System API: ReadFile, WriteFile, ListFiles, CreateDirectory, GetMetadata, Exists
  - Execution Context API: LogEvent, StartTrace, EndTrace, AddMetadata, GetContext

- **Multi-Language Integration**
  - F# Native Integration: Direct object references, async workflows, computation expressions
  - C# Interop Integration: Task-based async/await, LINQ operations, exception handling
  - Zero overhead native .NET integration with full type safety
  - Real-time execution monitoring and comprehensive tracing

- **Security and Resource Management**
  - Comprehensive sandboxing with granular API permissions
  - Resource limits: Memory, CPU, network, file operations, LLM requests
  - Network access control with domain allowlisting
  - File system access control with path-based restrictions
  - Execution timeouts and automatic cleanup

- **Production Features**
  - Thread-safe concurrent access with proper synchronization
  - Comprehensive error handling and exception propagation
  - Real-time tracing with detailed execution logs
  - Performance monitoring with sub-millisecond timing
  - Enterprise-grade security and compliance features

- **Demonstration Metascripts**
  - Created comprehensive F# and C# API usage examples
  - Real working demonstrations with actual API calls
  - Multi-language integration scenarios
  - Security and resource management examples

- MCP Integration for TARS/VSCode/Augment Collaboration
  - Added Server-Sent Events (SSE) support for real-time communication
  - Implemented structured message formats for knowledge transfer and code improvement
  - Created collaboration workflows for code improvement, knowledge extraction, and self-improvement
  - Added progress reporting capabilities
  - Implemented feedback loops for continuous improvement
- VS Code Integration
  - Added automatic TARS MCP server startup
  - Created VS Code tasks for managing the TARS MCP server
  - Added launch configurations for debugging TARS
  - Created documentation for the VS Code integration

### Changed
- Enhanced MetascriptService to support TARS Engine API injection
- Updated project structure to include new API infrastructure
- Improved metascript parsing to handle complex F# code blocks
- Enhanced the McpService to better handle knowledge transfer
- Updated the TarsMcpService to support VS Code Agent Mode
- Improved the CollaborationService to handle the three-way collaboration

### Fixed
- Fixed metascript parser to properly handle nested braces in F# blocks
- Resolved compilation errors in TarsExecutionContext implementation
- Fixed vector store search operations to handle empty result sets
- Fixed an issue with the CollaborationService where an extra closing brace was causing compilation errors
- Fixed string interpolation in metascript templates

## [0.1.0] - 2025-04-05

### Added
- Initial release of TARS
- Basic CLI functionality
- DSL processing capabilities
- Metascript execution
- Self-improvement features
