# Using Augment Agent from TARS through VS Code

This guide explains how to use Augment Agent from TARS through VS Code, enabling a powerful collaboration between these AI systems.

## Prerequisites

1. TARS CLI installed
2. VS Code installed
3. Augment Code extension installed in VS Code

## Running the Demo

TARS includes a demo command that shows how to use Augment Agent through VS Code:

```bash
tarscli demo augment-vscode-demo
```

By default, the demo will use the task "Analyze the codebase and suggest improvements". You can specify a different task using the `--task` option:

```bash
tarscli demo augment-vscode-demo --task "Implement a WebGPU renderer for a 3D scene"
```

## What the Demo Does

When you run this command, TARS will:

1. Start the TARS MCP (Model Context Protocol) server
2. Enable collaboration between TARS, VS Code, and Augment Agent
3. Open VS Code and guide you through enabling Agent Mode
4. Show you how to use Augment Agent through VS Code

## Step-by-Step Process

The demo will guide you through the following steps:

1. **Start the MCP Server**: TARS will start the MCP server, which acts as the central hub for communication between TARS, VS Code, and Augment Agent.

2. **Enable Collaboration**: TARS will enable collaboration between the three systems, allowing them to work together seamlessly.

3. **Set Up VS Code**: TARS will open VS Code and guide you through enabling Agent Mode:
   - Open VS Code Settings (Ctrl+,)
   - Search for 'chat.agent.enabled'
   - Check the box to enable it
   - Open the Chat view (Ctrl+Alt+I)
   - Select 'Agent' mode from the dropdown

4. **Use Augment Agent**: Once VS Code is set up, you can type your task in the Chat view, and VS Code Agent Mode will use the TARS MCP server to execute the task, collaborating with Augment Agent to provide enhanced capabilities.

## TARS-Specific Commands in VS Code Agent Mode

You can also use TARS-specific commands in VS Code Agent Mode:

- `#vscode_agent execute_metascript`: Execute a TARS metascript
- `#vscode_agent analyze_codebase`: Analyze the codebase structure and quality
- `#vscode_agent generate_metascript`: Generate a TARS metascript for a specific task

## How It Works

The integration works through the Model Context Protocol (MCP), which enables AI models to interact with external tools and services through a unified interface.

1. **TARS MCP Server**: Acts as the central hub for communication between TARS, VS Code, and Augment Agent
2. **VS Code Agent Mode**: Provides the user interface and autonomous agent capabilities within the editor
3. **Augment Agent**: Offers deep codebase understanding and specialized code generation

When you make a request in VS Code Agent Mode, it's sent to the TARS MCP server, which coordinates with Augment Agent to fulfill the request. The result is then displayed in VS Code.

## Architecture Diagram

```
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│             │      │             │      │             │
│  VS Code    │◄────►│  TARS MCP   │◄────►│  Augment    │
│  Agent Mode │      │  Server     │      │  Agent      │
│             │      │             │      │             │
└─────────────┘      └─────────────┘      └─────────────┘
       ▲                    ▲                    ▲
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                     ┌──────┴──────┐
                     │             │
                     │  User       │
                     │             │
                     └─────────────┘
```

## Example Use Cases

1. **Code Generation**:
   ```
   Generate a WebGPU renderer for a 3D scene with dynamic lighting
   ```

2. **Code Analysis**:
   ```
   Analyze the error handling in the TarsMcpService class and suggest improvements
   ```

3. **Metascript Generation**:
   ```
   #vscode_agent generate_metascript "Create a DSL for defining UI components"
   ```

4. **Codebase Analysis**:
   ```
   #vscode_agent analyze_codebase "Find all usages of the Result monad"
   ```

## Manual Setup (Without Using the Demo)

If you prefer to set up the integration manually, you can follow these steps:

1. **Start the TARS MCP server**:
   ```bash
   tarscli mcp start
   ```

2. **Enable collaboration**:
   ```bash
   tarscli mcp collaborate start
   ```

3. **Configure VS Code**:
   - Open VS Code Settings (Ctrl+,)
   - Search for 'chat.agent.enabled'
   - Check the box to enable it
   - Open the Chat view (Ctrl+Alt+I)
   - Select 'Agent' mode from the dropdown

## Using the VS Code Control Command

TARS also provides a direct VS Code control command that can be used to interact with VS Code programmatically:

```bash
tarscli vscode augment "Analyze the codebase and suggest improvements"
```

This command will:
1. Start the MCP server
2. Enable collaboration
3. Open VS Code and guide you through enabling Agent Mode
4. Show you how to use Augment Agent with the specified task

## Troubleshooting

If you encounter issues with the integration:

1. Check that the TARS MCP server is running (`tarscli mcp status`)
2. Verify that VS Code Agent Mode is enabled
3. Check the TARS logs for any errors
4. Restart the collaboration with `tarscli mcp collaborate start`

## Security Considerations

The collaboration gives AI models significant access to your system. TARS implements several security measures:

1. **Local-only access**: The MCP server only listens on localhost
2. **Explicit permissions**: Certain operations require explicit user permission
3. **Audit logging**: All operations are logged for audit purposes

## Conclusion

The integration between TARS, VS Code, and Augment Agent creates a powerful AI-assisted development environment where each component plays to its strengths. By following this guide, you can leverage this integration to enhance your development workflow.
