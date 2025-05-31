# TARS A2A Protocol Implementation

This directory contains the implementation of the Agent-to-Agent (A2A) protocol for TARS.

## Overview

The A2A protocol is an open standard developed by Google to enable interoperability between AI agents. This implementation allows TARS to communicate with other A2A-compatible agents and expose its capabilities through the A2A protocol.

## Components

- **Types.cs**: Core types for the A2A protocol, including agent cards, tasks, messages, and JSON-RPC types.
- **Client.cs**: Client for interacting with A2A protocol servers.
- **Server.cs**: Server for implementing the A2A protocol.
- **CardResolver.cs**: Utility for resolving agent cards from URLs.

## Usage

### Client

```csharp
// Create a client for an A2A agent
var client = new A2AClient("http://localhost:8998/");

// Create a message
var message = new Message
{
    Role = "user",
    Parts = new List<Part>
    {
        new TextPart
        {
            Text = "Generate a C# class for a customer entity"
        }
    },
    Metadata = new Dictionary<string, object>
    {
        { "skillId", "code_generation" }
    }
};

// Send a task
var task = await client.SendTaskAsync(message);
Console.WriteLine($"Task ID: {task.TaskId}");
Console.WriteLine($"Task Status: {task.Status}");

// Get the response
if (task.Messages.Count > 1)
{
    var responseMessage = task.Messages[1];
    if (responseMessage.Parts.Count > 0 && responseMessage.Parts[0] is TextPart textPart)
    {
        Console.WriteLine($"Response: {textPart.Text}");
    }
}
```

### Server

```csharp
// Create an agent card
var agentCard = new AgentCard
{
    Name = "TARS Agent",
    Description = "TARS Agent with A2A protocol support",
    Url = "http://localhost:8998/",
    // ... other properties
};

// Create a server
var server = new A2AServer(agentCard, logger);

// Register a task handler
server.RegisterTaskHandler("code_generation", async (message, cancellationToken) =>
{
    // Process the message and return a task
    return new Task
    {
        TaskId = Guid.NewGuid().ToString(),
        Status = TaskStatus.Completed,
        Messages = new List<Message>
        {
            message,
            new Message
            {
                Role = "agent",
                Parts = new List<Part>
                {
                    new TextPart
                    {
                        Text = "Generated code: ..."
                    }
                }
            }
        }
    };
});

// Start the server
server.Start();
```

## Integration with MCP

The A2A protocol is integrated with the Model Context Protocol (MCP) through a bridge that allows:

- MCP clients to interact with A2A agents
- A2A clients to interact with MCP services

This integration enables seamless collaboration between TARS, Augment Code, and other A2A-compatible agents.

## References

- [A2A Protocol GitHub Repository](https://github.com/google/A2A)
- [A2A Protocol Documentation](https://google.github.io/A2A/)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
