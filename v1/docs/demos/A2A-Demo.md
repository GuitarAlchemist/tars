# A2A Protocol Demo

This document describes the A2A (Agent-to-Agent) protocol demonstration for TARS.

## Overview

The A2A protocol demo showcases TARS's implementation of the Agent-to-Agent protocol, which enables interoperability between different AI agents. The demo demonstrates how TARS can expose its capabilities through the A2A protocol and communicate with other A2A-compatible agents.

## Running the Demo

There are several ways to run the A2A protocol demo:

### Using the Demo Scripts

The simplest way to run the demo is to use the provided demo scripts:

#### PowerShell Script

```powershell
.\Scripts\Demo-A2A.ps1
```

#### Batch File

```batch
.\Scripts\Demo-A2A.bat
```

### Using the TARS CLI

You can also run the A2A demo directly through the TARS CLI:

```bash
tarscli demo a2a-demo
```

## Demo Sections

The A2A protocol demo includes the following sections:

1. **A2A Server** - Start the A2A server
2. **Agent Card** - Get the TARS agent card
3. **Code Generation Skill** - Send a code generation task
4. **Code Analysis Skill** - Send a code analysis task
5. **Knowledge Extraction Skill** - Send a knowledge extraction task (in script version)
6. **Self Improvement Skill** - Send a self improvement task (in script version)
7. **MCP Bridge** - Use A2A through MCP
8. **Stopping the Server** - Stop the A2A server

## Manual Testing

You can also manually test the A2A protocol using the TARS CLI commands:

```bash
# Start the A2A server
tarscli a2a start

# Get the agent card
tarscli a2a get-agent-card --agent-url http://localhost:8998/

# Send a task
tarscli a2a send --agent-url http://localhost:8998/ --message "Generate a C# class for a Customer entity" --skill-id code_generation

# Stop the A2A server
tarscli a2a stop
```

## Integration with MCP

The A2A protocol is integrated with the Model Context Protocol (MCP) through a bridge that allows:

- MCP clients to interact with A2A agents
- A2A clients to interact with MCP services

You can test this integration using the MCP execute command:

```bash
tarscli mcp execute --action a2a --operation send_task --agent_url http://localhost:8998/ --content "Generate a simple logging class in C#" --skill_id code_generation
```

## Further Information

For more information about the A2A protocol implementation in TARS, see the [A2A Protocol Documentation](../A2A-Protocol.md).
