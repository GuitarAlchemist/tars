# Changelog

All notable changes to the TARS project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
- Enhanced the McpService to better handle knowledge transfer
- Updated the TarsMcpService to support VS Code Agent Mode
- Improved the CollaborationService to handle the three-way collaboration

### Fixed
- Fixed an issue with the CollaborationService where an extra closing brace was causing compilation errors
- Fixed string interpolation in metascript templates

## [0.1.0] - 2025-04-05

### Added
- Initial release of TARS
- Basic CLI functionality
- DSL processing capabilities
- Metascript execution
- Self-improvement features
