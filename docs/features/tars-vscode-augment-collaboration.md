# TARS-VSCode-Augment Collaboration

This document describes the collaboration between TARS, VSCode Agent Mode, and Augment Code.

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

### Knowledge Transfer Message

Used to transfer knowledge between components:

```json
{
  "type": "knowledge_transfer",
  "source": "tars",
  "target": "augment",
  "knowledge_type": "code_analysis",
  "content": { ... },
  "metadata": { ... }
}
```

### Code Improvement Message

Used to suggest code improvements:

```json
{
  "type": "code_improvement",
  "file_path": "TarsCli/Services/TarsMcpService.cs",
  "line_start": 100,
  "line_end": 120,
  "original_code": "...",
  "improved_code": "...",
  "improvement_type": "performance",
  "explanation": "...",
  "confidence": 0.9
}
```

### Progress Report Message

Used to report progress on operations:

```json
{
  "type": "progress_report",
  "operation": "workflow_code_improvement",
  "progress_percentage": 50,
  "status": "in_progress",
  "details": "Analyzing code quality",
  "estimated_completion_time": 1625097600
}
```

### System Handoff Message

Used to hand off control between components:

```json
{
  "type": "system_handoff",
  "from_system": "tars",
  "to_system": "augment",
  "context": { ... },
  "action_requested": "analyze_code_quality",
  "priority": "high"
}
```

### Feedback Message

Used to provide feedback on operations:

```json
{
  "type": "feedback",
  "feedback_type": "code_improvement",
  "target_message_id": "...",
  "rating": 4,
  "comments": "...",
  "suggestions": [ ... ]
}
```

## Using the Collaboration

### Starting the Collaboration

1. Start the TARS MCP server:
   ```
   tarscli mcp start
   ```

2. Enable the collaboration:
   ```
   tarscli mcp collaborate start
   ```

3. In VSCode, enable Agent Mode in settings:
   ```json
   "chat.agent.enabled": true
   ```

### Testing the Collaboration

You can test the collaboration using the `test-collaboration` command:

```
tarscli mcp test-collaboration --workflow code_improvement
```

This will initiate a code improvement workflow and display the results.

### Using VSCode Agent Mode

Once the collaboration is set up, you can use it in VSCode Agent Mode:

1. Open the Chat view in VSCode (Ctrl+Alt+I)
2. Select "Agent" mode from the dropdown
3. Type a request like "Improve the error handling in the TarsMcpService class"

VSCode Agent Mode will use the collaboration to analyze the code, suggest improvements, and apply the changes.

## Security Considerations

The collaboration gives AI models significant access to your system. TARS implements several security measures:

1. **Local-only access**: The MCP server only listens on localhost
2. **Explicit permissions**: Certain operations require explicit user permission
3. **Audit logging**: All operations are logged for audit purposes

## Troubleshooting

If you encounter issues with the collaboration:

1. Check that the TARS MCP server is running
2. Verify that VSCode Agent Mode is enabled
3. Check the TARS logs for any errors
4. Restart the collaboration with `tarscli mcp collaborate start`
