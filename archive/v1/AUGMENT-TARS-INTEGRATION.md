# Augment Code Integration with TARS

This document explains how to integrate Augment Code with TARS using the Model Context Protocol (MCP).

## Overview

TARS provides a Model Context Protocol (MCP) server that allows Augment Code to interact with TARS capabilities, including:

- Knowledge extraction from documentation
- Knowledge application to improve code files
- Running knowledge improvement cycles
- Generating retroaction reports
- Text generation with Ollama models
- Self-improvement capabilities

## Prerequisites

1. TARS MCP server running on http://localhost:8999/
2. Augment Code with MCP client capabilities

## Starting the TARS MCP Server

To start the TARS MCP server, run:

```powershell
.\tarscli.cmd mcp start
```

## Integration Methods

### 1. Direct MCP Requests

Augment Code can send MCP requests directly to the TARS MCP server:

```javascript
const response = await fetch('http://localhost:8999/', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    action: 'knowledge',
    operation: 'extract',
    filePath: 'docs/Explorations/v1/Chats/ChatGPT-TARS Project Implications.md',
    model: 'llama3'
  })
});

const result = await response.json();
```

### 2. Using the Integration Library

Alternatively, Augment Code can use the provided integration library:

```javascript
const tars = require('./augment-tars-integration');

// Extract knowledge from a file
const knowledge = await tars.extractKnowledge('docs/Explorations/v1/Chats/ChatGPT-TARS Project Implications.md');

// Apply knowledge to improve a file
const result = await tars.applyKnowledge('TarsCli/Services/DslService.cs');

// Run a knowledge improvement cycle
const cycleResult = await tars.runKnowledgeImprovementCycle(
  'docs/Explorations/v1/Chats',
  'TarsCli/Services',
  '*.cs',
  'llama3'
);
```

## Available Operations

### Knowledge Operations

- `extract`: Extract knowledge from a file
- `apply`: Apply knowledge to improve a file
- `report`: Generate a knowledge report
- `metascript`: Generate a knowledge metascript
- `cycle`: Run a knowledge improvement cycle
- `retroaction`: Generate a retroaction report
- `list`: List all knowledge items

### Ollama Operations

- `generate`: Generate text with a model
- `models`: Get available models

### Self-Improvement Operations

- `start`: Start self-improvement
- `status`: Get self-improvement status
- `stop`: Stop self-improvement

## Example: Collaborative Workflow

Here's an example of how Augment Code and TARS can collaborate:

1. Augment Code identifies a file that needs improvement
2. Augment Code requests TARS to extract knowledge from relevant exploration files
3. TARS extracts knowledge and shares it with Augment Code
4. Augment Code analyzes the knowledge and suggests improvements
5. Augment Code requests TARS to apply the improvements
6. TARS applies the improvements and reports back to Augment Code
7. Augment Code reviews the improvements and provides feedback
8. TARS learns from the feedback and improves its knowledge base

## Testing the Integration

Several test scripts are provided to test the integration:

- `test-mcp-connection.ps1`: Test the connection to the TARS MCP server
- `test-knowledge-extraction.ps1`: Test knowledge extraction
- `test-knowledge-application.ps1`: Test knowledge application
- `test-knowledge-cycle.ps1`: Test running a knowledge improvement cycle
- `test-retroaction-report.ps1`: Test generating a retroaction report

## Configuration

The integration can be configured using the `augment-tars-config.json` file, which specifies:

- The MCP server URL
- Available capabilities and operations
- Parameters for each operation

## Troubleshooting

If you encounter issues with the integration:

1. Ensure the TARS MCP server is running
2. Check the TARS logs for error messages
3. Verify that the MCP request format is correct
4. Ensure the file paths are correct and accessible

## Further Development

The integration can be extended to support additional TARS capabilities:

- Speech generation and recognition
- Slack integration for notifications
- Advanced self-improvement capabilities
- Custom metascript generation and execution
