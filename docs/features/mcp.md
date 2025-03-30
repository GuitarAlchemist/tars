# Model Context Protocol (MCP)

TARS implements the Model Context Protocol (MCP) as defined by Anthropic, allowing seamless integration between TARS and other AI tools like Augment Code. This document provides a detailed overview of the MCP implementation in TARS, its capabilities, and how to use it effectively.

## What is MCP?

The Model Context Protocol (MCP) is a standard for communication between AI models and tools. It allows AI models to:

1. **Execute commands** on the user's system
2. **Generate and save code** without requiring manual confirmation
3. **Access system information** and capabilities
4. **Interact with other AI tools and services**

MCP enables AI models to be more helpful by giving them controlled access to your system's capabilities.

## TARS MCP Implementation

TARS implements MCP as both a client and a server:

1. **MCP Server**: TARS can run as an MCP server, allowing other tools like Augment Code to access TARS capabilities
2. **MCP Client**: TARS can act as an MCP client, sending requests to other MCP servers

## Available MCP Actions

TARS MCP server supports the following actions:

- **execute**: Execute terminal commands
- **code**: Generate and save code
- **status**: Get system status
- **tars**: Execute TARS-specific operations
- **ollama**: Execute Ollama operations
- **self-improve**: Execute self-improvement operations
- **slack**: Execute Slack operations
- **speech**: Execute speech operations

## Using TARS MCP

### Starting the MCP Server

To start the TARS MCP server:

```bash
tarscli mcp start
```

This will start the MCP server on the configured port (default: 8999).

### Checking MCP Status

To check the status of the MCP server:

```bash
tarscli mcp status
```

This will show the server URL, configuration settings, and available actions.

### Configuring MCP

To configure the MCP server:

```bash
tarscli mcp configure --port 8999 --auto-execute --auto-code
```

Options:
- `--port`: Port for the MCP server (default: 8999)
- `--auto-execute`: Enable auto-execution of commands
- `--auto-code`: Enable auto-code generation

### Integrating with Augment Code

To configure Augment Code to use TARS:

```bash
tarscli mcp augment
```

This will update your VS Code settings to include TARS as an MCP server for Augment Code.

After configuration, you can use TARS from Augment Code by:

1. Starting the MCP server with `tarscli mcp start`
2. In VS Code, using the command `@tars` to interact with TARS

### Viewing Conversation History

TARS logs all MCP conversations, including those with Augment Code. To view the conversation history:

```bash
tarscli mcp conversations --source augment --count 5
```

Options:
- `--source`: Filter conversations by source (e.g., 'augment')
- `--count`: Number of conversations to show (default: 10)
- `--open`: Open the conversation log in the default browser

To open the full conversation log in your default browser:

```bash
tarscli mcp conversations --open
```

This is particularly useful for reviewing interactions with Augment Code and understanding how TARS is being used.

## MCP Architecture

The TARS MCP implementation consists of the following components:

### McpService

The `McpService` is the core MCP server implementation. It:

- Listens for HTTP requests on the configured port
- Parses MCP requests and routes them to the appropriate handlers
- Executes actions and returns responses
- Provides a client interface for sending requests to other MCP servers

### TarsMcpService

The `TarsMcpService` extends the MCP server with TARS-specific capabilities:

- Ollama integration for text generation
- Self-improvement capabilities
- Slack integration for notifications
- Speech synthesis for text-to-speech

## MCP Security Considerations

The MCP protocol gives AI models significant access to your system. TARS implements several security measures:

1. **Auto-execute setting**: Commands are only executed if auto-execute is enabled
2. **Auto-code setting**: Code is only generated and saved if auto-code is enabled
3. **Local-only server**: The MCP server only listens on localhost, preventing remote access
4. **Explicit configuration**: Security settings must be explicitly enabled

## Example MCP Requests

Here are some example MCP requests that can be sent to the TARS MCP server:

### Execute a Command

```json
{
  "action": "execute",
  "command": "echo Hello, World!"
}
```

### Generate Code

```json
{
  "action": "code",
  "filePath": "path/to/file.cs",
  "content": "public class MyClass { }"
}
```

### Get System Status

```json
{
  "action": "status"
}
```

### Use TARS Capabilities

```json
{
  "action": "tars",
  "operation": "capabilities"
}
```

### Generate Text with Ollama

```json
{
  "action": "ollama",
  "operation": "generate",
  "prompt": "Write a haiku about AI",
  "model": "llama3"
}
```

### Start Self-Improvement

```json
{
  "action": "self-improve",
  "operation": "start",
  "duration": 60,
  "autoAccept": true
}
```

### Send a Slack Announcement

```json
{
  "action": "slack",
  "operation": "announce",
  "title": "New Feature",
  "message": "TARS now supports MCP!"
}
```

### Speak Text

```json
{
  "action": "speech",
  "operation": "speak",
  "text": "Hello, I am TARS",
  "language": "en"
}
```

### Error Handling

The MCP provides detailed error messages:

```bash
tarscli mcp execute "non_existent_command"
```

Output:
```
Error executing command: non_existent_command
Command not found: non_existent_command
Exit code: 127
```

## Security Considerations

### Command Execution

The MCP executes commands with the same permissions as the user running TARS. Be careful when:

1. Executing commands that modify the system
2. Running commands with elevated privileges
3. Executing commands from untrusted sources

### Code Generation

When generating code:

1. Always review generated code before using it in production
2. Be cautious with code that handles sensitive data
3. Test generated code thoroughly

### Integration Points

When integrating with external systems:

1. Use secure communication channels
2. Validate and sanitize inputs
3. Limit access to necessary resources only

## Configuration

The MCP can be configured through the `appsettings.json` file:

```json
{
  "Mcp": {
    "EnabledCommands": ["code", "triple-code", "execute", "augment"],
    "MaxExecutionTime": 60,
    "WorkingDirectory": "/path/to/working/directory",
    "AugmentCode": {
      "Endpoint": "http://localhost:8080",
      "ApiKey": "your-api-key"
    }
  }
}
```

## Best Practices

### When to Use the MCP

The MCP is most effective for:

1. **Repetitive Tasks**: Automating routine development tasks
2. **Code Generation**: Creating boilerplate code
3. **System Integration**: Connecting with external tools and services
4. **Workflow Automation**: Streamlining development workflows

### When Not to Use the MCP

The MCP may not be suitable for:

1. **Security-Critical Operations**: Operations that require careful review
2. **Complex Decision Making**: Tasks that require nuanced human judgment
3. **Untrusted Environments**: Environments where command execution could be exploited

### Tips for Effective Use

1. **Start Simple**: Begin with basic commands and gradually increase complexity
2. **Use Version Control**: Always work in a version-controlled environment
3. **Test Commands**: Test commands in a safe environment before using them in production
4. **Document Workflows**: Document MCP workflows for future reference
5. **Monitor Execution**: Keep an eye on MCP activities, especially in automated scenarios

## Future Enhancements

The MCP is continuously evolving. Planned enhancements include:

1. **Learning Capabilities**: Enabling the MCP to learn from past commands
2. **Integration with Self-Improvement**: Combining MCP with the self-improvement system
3. **Multi-step Operations**: Support for complex, multi-step operations
4. **Context-aware Command Execution**: Smarter command execution based on context
5. **Enhanced Security**: Additional security features for safer command execution

## Extending MCP

You can extend TARS MCP by registering new handlers for custom actions. This allows you to add new capabilities to TARS that can be accessed through the MCP protocol.

To register a new handler, use the `RegisterHandler` method of the `McpService` class:

```csharp
mcpService.RegisterHandler("custom-action", async (request) =>
{
    // Handle the request
    return JsonSerializer.SerializeToElement(new { success = true, result = "Custom action executed" });
});
```

## Troubleshooting

### MCP Server Not Starting

If the MCP server fails to start, check:

- The port is not already in use by another application
- You have sufficient permissions to start a server
- The configuration in `appsettings.json` is valid

### Commands Not Executing

If commands are not executing, check:

- Auto-execute is enabled in the configuration
- The command is valid and can be executed on your system
- You have sufficient permissions to execute the command

### Augment Integration Not Working

If Augment Code integration is not working, check:

- The MCP server is running (`tarscli mcp start`)
- The VS Code settings have been updated with the correct MCP server URL
- You're using the correct syntax in Augment Code (`@tars <command>`)
