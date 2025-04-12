# VS Code Integration with TARS

This document describes how to use VS Code with TARS, including the integration with the TARS MCP server and Agent Mode.

## Prerequisites

- VS Code
- .NET 9.0 SDK
- TARS codebase

## Setup

The TARS repository includes VS Code configuration files that make it easy to work with TARS in VS Code:

- `.vscode/settings.json`: Configures VS Code settings, including Agent Mode
- `.vscode/tasks.json`: Defines tasks for building TARS and managing the TARS MCP server
- `.vscode/launch.json`: Defines launch configurations for debugging TARS
- `.vscode/mcp.json`: Configures the MCP server connection for Agent Mode
- `.vscode/extensions.json`: Recommends extensions for working with TARS

## Starting the TARS MCP Server

There are several ways to start the TARS MCP server:

1. **Automatic Start**: VS Code will automatically start the TARS MCP server when you open the TARS project, thanks to the `task.runOnStartup` setting in `.vscode/settings.json`.

2. **Using Tasks**: You can manually start the TARS MCP server using VS Code tasks:
   - Press `Ctrl+Shift+P` to open the Command Palette
   - Type "Tasks: Run Task" and press Enter
   - Select "Start TARS MCP Server"

3. **Using the Command Line**: You can start the TARS MCP server from the command line:
   ```
   .\tarscli.cmd mcp start
   ```

4. **Using the Launch Configuration**: You can start the TARS MCP server in debug mode:
   - Press `F5` to open the Debug view
   - Select "TARS MCP Server" from the dropdown
   - Press `F5` again to start debugging

## Using Agent Mode

Once the TARS MCP server is running, you can use Agent Mode in VS Code:

1. Open the Chat view in VS Code (Ctrl+Alt+I)
2. Select "Agent" mode from the dropdown at the top of the Chat view
3. Type a command like "Analyze the TARS codebase structure"

VS Code Agent Mode will use the TARS MCP server to execute the command.

## Using Augment Agent from TARS through VS Code

TARS includes a demo command that shows how to use Augment Agent through VS Code:

```bash
tarscli demo augment-vscode-demo
```

This command will:
1. Start the TARS MCP server
2. Enable collaboration between TARS, VS Code, and Augment Agent
3. Open VS Code and guide you through enabling Agent Mode
4. Show you how to use Augment Agent through VS Code

For more details, see [Augment VS Code Integration](features/augment-vscode-integration.md).

## Available Commands

Here are some commands you can try in Agent Mode:

- **Code Analysis**: "Analyze the TARS codebase structure"
- **Code Generation**: "Generate a metascript for analyzing code quality"
- **Code Improvement**: "Improve the error handling in the TarsMcpService class"
- **Knowledge Extraction**: "Extract knowledge from the TARS documentation"
- **Self-Improvement**: "Identify areas for self-improvement in the TARS codebase"

## Troubleshooting

If you encounter issues with the VS Code integration:

1. **Check the TARS MCP Server Status**:
   - Press `Ctrl+Shift+P` to open the Command Palette
   - Type "Tasks: Run Task" and press Enter
   - Select "Ensure TARS MCP Server"

2. **Restart the TARS MCP Server**:
   - Press `Ctrl+Shift+P` to open the Command Palette
   - Type "Tasks: Run Task" and press Enter
   - Select "Stop TARS MCP Server"
   - Then select "Start TARS MCP Server"

3. **Check the TARS MCP Server Logs**:
   - Open the `tars-mcp.log` file in VS Code

4. **Reload VS Code**:
   - Press `Ctrl+Shift+P` to open the Command Palette
   - Type "Reload Window" and press Enter

## Advanced Configuration

You can customize the VS Code integration by editing the configuration files:

- `.vscode/settings.json`: VS Code settings
- `.vscode/tasks.json`: VS Code tasks
- `.vscode/launch.json`: VS Code launch configurations
- `.vscode/mcp.json`: MCP server configuration

For example, you can change the MCP server port by editing the `url` property in `.vscode/mcp.json`.
