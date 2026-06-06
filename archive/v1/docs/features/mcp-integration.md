# MCP Integration for TARS/VSCode/Augment Collaboration

This document provides a detailed overview of the Model Context Protocol (MCP) integration between TARS, VSCode Agent Mode, and Augment Code.

## Overview

The collaboration between TARS, VSCode Agent Mode, and Augment Code creates a powerful AI-assisted development environment where each component plays to its strengths:

1. **TARS**: Provides domain-specific capabilities like metascript processing, DSL handling, and self-improvement
2. **VSCode Agent Mode**: Provides the user interface and autonomous agent capabilities within the editor
3. **Augment Code**: Offers deep codebase understanding and specialized code generation

## Architecture

The collaboration is built on the Model Context Protocol (MCP), which enables AI models to interact with external tools and services through a unified interface.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  VS Code with   │◄───►│  Augment Code   │◄───►│    TARS CLI     │
│   Agent Mode    │     │                 │     │                 │
│                 │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                     Model Context Protocol                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### TARS MCP Server

The TARS MCP server is the central component of the integration. It:

1. Listens for MCP requests on `http://localhost:9000/`
2. Processes requests from VSCode Agent Mode and Augment Code
3. Executes TARS-specific operations
4. Broadcasts events to connected clients using Server-Sent Events (SSE)

### VSCode Agent Mode

VSCode Agent Mode is a feature of VSCode that enables AI models to interact with the editor. It:

1. Connects to the TARS MCP server
2. Sends requests to the TARS MCP server
3. Processes responses from the TARS MCP server
4. Applies changes to the codebase

### Augment Code

Augment Code is an AI pair programming tool that understands your codebase. It:

1. Connects to the TARS MCP server
2. Analyzes the codebase
3. Suggests improvements
4. Generates code

## Collaboration Workflows

The collaboration supports several workflows:

### Code Improvement Workflow

1. TARS analyzes the codebase to identify areas for improvement
2. Augment Code suggests specific improvements based on the analysis
3. VSCode Agent Mode applies the changes to the codebase
4. TARS verifies the improvements

### Knowledge Extraction Workflow

1. TARS extracts knowledge from the codebase or documentation
2. Augment Code enhances the knowledge with additional context
3. TARS integrates the knowledge into its knowledge base

### Self-Improvement Workflow

1. TARS identifies areas for self-improvement
2. Augment Code analyzes code quality and suggests improvements
3. TARS generates an improvement plan
4. VSCode Agent Mode applies the improvements

## Message Types

The collaboration uses several message types to communicate between the components:

### Collaboration Message

The primary message type used for communication between components:

```json
{
  "id": "message-id",
  "sender": "tars",
  "recipient": "augment",
  "type": "knowledge",
  "operation": "transfer",
  "content": { ... },
  "metadata": { ... }
}
```

### Progress Information

Used to report progress on operations:

```json
{
  "percentage": 50,
  "current_step": "analyzing_code",
  "total_steps": 4,
  "status_message": "Analyzing code quality"
}
```

### Improvement Feedback

Used to provide feedback on improvements:

```json
{
  "accepted": true,
  "reason": "The improvement enhances performance",
  "suggestions": [
    "Consider using async/await for better performance"
  ]
}
```

## Implementation Details

### Server-Sent Events (SSE)

The TARS MCP server uses Server-Sent Events (SSE) to push events to connected clients in real-time. This enables:

1. Real-time progress updates
2. Immediate notification of completed operations
3. Continuous feedback during long-running operations

### Dynamic Workflow Execution

The collaboration supports dynamic workflow execution, where:

1. Workflows are defined at runtime
2. Steps can be added, removed, or modified
3. Different components can coordinate different workflows
4. Workflows can be paused, resumed, or cancelled

### Knowledge Transfer

The collaboration enables knowledge transfer between components:

1. TARS can extract knowledge from the codebase or documentation
2. Augment Code can enhance the knowledge with additional context
3. VSCode Agent Mode can apply the knowledge to the codebase

## Configuration

The collaboration is configured through several files:

### `.vscode/settings.json`

Configures VS Code settings, including Agent Mode:

```json
{
  "chat.agent.enabled": true,
  "augment.advanced": {
    "agentMode": true,
    "mcpServers": [
      {
        "name": "tars",
        "url": "http://localhost:9000/"
      }
    ]
  },
  "task.runOnStartup": {
    "task": "Ensure TARS MCP Server",
    "enabled": true
  }
}
```

### `.vscode/tasks.json`

Defines tasks for building TARS and managing the TARS MCP server:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Ensure TARS MCP Server",
      "type": "shell",
      "command": "${workspaceFolder}/ensure-tars-mcp.cmd",
      "presentation": {
        "reveal": "silent",
        "panel": "dedicated",
        "close": true
      },
      "problemMatcher": []
    },
    // ...
  ]
}
```

### `tars-augment-vscode-config.json`

Defines the collaboration configuration:

```json
{
  "collaboration": {
    "enabled": true,
    "components": {
      "vscode": {
        "role": "user_interface",
        "capabilities": ["file_editing", "terminal_execution", "agent_coordination"]
      },
      "augment": {
        "role": "code_understanding",
        "capabilities": ["codebase_analysis", "code_generation", "refactoring"]
      },
      "tars": {
        "role": "specialized_processing",
        "capabilities": ["metascript_execution", "dsl_processing", "self_improvement"]
      }
    },
    "workflows": [
      {
        "name": "code_improvement",
        "coordinator": "vscode",
        "steps": [
          { "component": "vscode", "action": "get_user_request" },
          { "component": "augment", "action": "analyze_codebase_context" },
          { "component": "tars", "action": "generate_metascript" },
          { "component": "tars", "action": "execute_metascript" },
          { "component": "vscode", "action": "apply_changes" }
        ]
      },
      // ...
    ]
  }
}
```

## Getting Started

To get started with the integration, see [VS Code Integration](../vscode-integration.md).

## Troubleshooting

If you encounter issues with the integration:

1. **Check the TARS MCP Server Status**:
   ```
   .\tarscli.cmd mcp status
   ```

2. **Restart the TARS MCP Server**:
   ```
   .\tarscli.cmd mcp stop
   .\tarscli.cmd mcp start
   ```

3. **Check the TARS MCP Server Logs**:
   ```
   cat tars-mcp.log
   ```

4. **Verify the VS Code Settings**:
   - Make sure `chat.agent.enabled` is set to `true` in `.vscode/settings.json`
   - Make sure the MCP server URL is correct in `.vscode/mcp.json`

## Future Enhancements

Planned enhancements for the integration include:

1. **Enhanced Knowledge Transfer**: Improve the knowledge transfer between components
2. **More Sophisticated Workflows**: Add more sophisticated workflows for specific tasks
3. **Better Progress Reporting**: Enhance progress reporting with more detailed information
4. **Improved Error Handling**: Add better error handling and recovery mechanisms
5. **Extended Capabilities**: Add more capabilities to each component
