# A2A Protocol Support in TARS

TARS now supports the Agent-to-Agent (A2A) protocol, which enables interoperability between different AI agents. This document explains how to use the A2A protocol with TARS.

## What is A2A?

The Agent-to-Agent (A2A) protocol is an open standard developed by Google to enable interoperability between AI agents. It provides a standardized way for agents to communicate with each other, share capabilities, and collaborate on tasks.

A2A uses JSON-RPC over HTTP for communication and defines a set of standard message formats and endpoints. The protocol includes support for:

- Agent discovery through agent cards
- Task submission and retrieval
- Streaming responses
- Push notifications
- State transition history

For more information, see the [official A2A repository](https://github.com/google/A2A).

## TARS A2A Server

TARS includes an A2A server that allows other agents to interact with TARS capabilities. The server exposes TARS skills through the A2A protocol.

### Starting the A2A Server

To start the A2A server, use the following command:

```bash
tarscli a2a start
```

This will start the A2A server on the configured host and port (default: `http://localhost:8998/`).

### Stopping the A2A Server

To stop the A2A server, use the following command:

```bash
tarscli a2a stop
```

## TARS A2A Client

TARS also includes an A2A client that allows TARS to interact with other A2A-compatible agents.

### Sending a Task to an A2A Agent

To send a task to an A2A agent, use the following command:

```bash
tarscli a2a send --agent-url "http://localhost:8998/" --message "Generate a C# class for a customer entity" --skill-id "code_generation"
```

This will send a task to the specified agent and display the response.

### Getting a Task from an A2A Agent

To get the status and result of a task, use the following command:

```bash
tarscli a2a get --agent-url "http://localhost:8998/" --task-id "task-123"
```

### Canceling a Task

To cancel a task, use the following command:

```bash
tarscli a2a cancel --agent-url "http://localhost:8998/" --task-id "task-123"
```

### Getting an Agent Card

To get information about an A2A agent, use the following command:

```bash
tarscli a2a get-agent-card --agent-url "http://localhost:8998/"
```

This will display the agent's capabilities, skills, and other information.

## TARS A2A Skills

TARS exposes the following skills through the A2A protocol:

1. **Code Generation** (`code_generation`): Generate code based on natural language descriptions.
2. **Code Analysis** (`code_analysis`): Analyze code for quality, complexity, and issues.
3. **Metascript Execution** (`metascript_execution`): Execute TARS metascripts.
4. **Knowledge Extraction** (`knowledge_extraction`): Extract knowledge from documents and code.
5. **Self Improvement** (`self_improvement`): Improve TARS capabilities through self-analysis.

## Integration with MCP

TARS integrates the A2A protocol with the Model Context Protocol (MCP) through a bridge that allows:

- MCP clients to interact with A2A agents
- A2A clients to interact with MCP services

This integration enables seamless collaboration between TARS, Augment Code, and other A2A-compatible agents.

### Using A2A through MCP

To use A2A through MCP, you can send a request to the MCP server with the `a2a` action:

```json
{
  "action": "a2a",
  "operation": "send_task",
  "agent_url": "http://localhost:8998/",
  "content": "Generate a C# class for a customer entity",
  "skill_id": "code_generation"
}
```

## Configuration

The A2A server can be configured in the `appsettings.json` file:

```json
{
  "Tars": {
    "A2A": {
      "Host": "localhost",
      "Port": 8998,
      "Enabled": true
    }
  }
}
```

## Examples

### Example 1: Code Generation

```bash
tarscli a2a send --agent-url "http://localhost:8998/" --message "Generate a C# class for a customer entity with properties for ID, Name, Email, and Address" --skill-id "code_generation"
```

### Example 2: Code Analysis

```bash
tarscli a2a send --agent-url "http://localhost:8998/" --message "Analyze this code for potential issues: public void ProcessData(string data) { var result = data.Split(','); Console.WriteLine(result[0]); }" --skill-id "code_analysis"
```

### Example 3: Metascript Execution

```bash
tarscli a2a send --agent-url "http://localhost:8998/" --message "Execute a metascript to analyze the project structure and suggest improvements" --skill-id "metascript_execution"
```

## Troubleshooting

If you encounter issues with the A2A protocol, check the following:

1. Make sure the A2A server is running (`tarscli a2a start`).
2. Verify the agent URL is correct.
3. Check the logs for error messages.
4. Ensure the skill ID is valid for the agent you're communicating with.

## References

- [A2A Protocol GitHub Repository](https://github.com/google/A2A)
- [A2A Protocol Documentation](https://google.github.io/A2A/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
