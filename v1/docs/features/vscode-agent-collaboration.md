# VSCode Agent Mode Collaboration with TARS and Augment Code

This document explains how to use the collaboration feature between TARS, Augment Code, and VSCode Agent Mode.

## Overview

The collaboration feature enables three powerful AI systems to work together:

1. **VSCode Agent Mode**: Provides the user interface and autonomous agent capabilities within the editor
2. **Augment Code**: Offers deep codebase understanding and specialized code generation
3. **TARS CLI**: Contributes domain-specific capabilities like metascript processing, DSL handling, and self-improvement

This integration creates a comprehensive AI-assisted development environment where each component plays to its strengths.

## Prerequisites

- VSCode with Agent Mode enabled
- Augment Code extension installed in VSCode
- TARS CLI installed and configured

## Setting Up the Collaboration

### 1. Start the TARS MCP Server

First, start the TARS MCP server:

```bash
tarscli mcp start
```

### 2. Enable the Collaboration

Enable the collaboration between TARS, Augment Code, and VSCode:

```bash
tarscli mcp collaborate start
```

### 3. Configure VSCode for Agent Mode

1. Open VSCode Settings (Ctrl+, or Cmd+, on Mac)
2. Search for "chat.agent.enabled"
3. Check the box to enable it
4. Alternatively, add this to your settings.json:
   ```json
   "chat.agent.enabled": true
   ```

### 4. Verify the Configuration

Verify that the collaboration is properly configured:

```bash
tarscli mcp collaborate status
```

## Using the Collaboration

Once the collaboration is set up, you can use it in the following ways:

### 1. VSCode Agent Mode

1. Open the Chat view in VSCode (Ctrl+Alt+I)
2. Select "Agent" mode from the dropdown
3. Type your request
4. VSCode will use your TARS MCP server to execute tasks

### 2. TARS-Specific Commands

You can use TARS-specific commands in VSCode Agent Mode:

- `#vscode_agent execute_metascript`: Execute a TARS metascript
- `#vscode_agent analyze_codebase`: Analyze the codebase structure and quality
- `#vscode_agent generate_metascript`: Generate a TARS metascript for a specific task

### 3. Collaboration Workflows

The collaboration supports several workflows:

#### Code Generation Workflow

1. User requests a new feature in VSCode Agent Mode
2. VSCode Agent coordinates the workflow
3. Augment Code analyzes the codebase to understand context
4. TARS generates a metascript for the feature
5. TARS executes the metascript to generate code
6. VSCode Agent applies the changes and shows them to the user

#### Self-Improvement Workflow

1. TARS identifies areas for improvement in the codebase
2. Augment Code analyzes code quality and suggests improvements
3. TARS generates an improvement plan
4. VSCode Agent applies the improvements and shows them to the user

#### Learning and Documentation Workflow

1. User asks for help understanding a complex part of the codebase
2. Augment Code analyzes the code structure
3. TARS generates explanations and documentation
4. VSCode Agent presents the information to the user

## Configuration

The collaboration is configured in the `tars-augment-vscode-config.json` file. You can modify this file to customize the collaboration behavior.

## Troubleshooting

If you encounter issues with the collaboration:

1. Check that the TARS MCP server is running
2. Verify that VSCode Agent Mode is enabled
3. Check the TARS logs for any errors
4. Restart the collaboration with `tarscli mcp collaborate start`

## Advanced Usage

For advanced usage, you can customize the collaboration by modifying the `tars-augment-vscode-config.json` file. This file defines the roles, capabilities, and workflows for each component in the collaboration.
