# Master Control Program (MCP)

The Master Control Program (MCP) is a core component of TARS that enables autonomous operation and integration with external systems. This document provides a detailed overview of the MCP, its capabilities, and how to use it effectively.

## Overview

The MCP is designed to:

1. **Execute commands** without requiring manual confirmation
2. **Generate code** based on natural language descriptions
3. **Integrate with external systems** like Augment Code
4. **Automate workflows** for increased productivity

![MCP Architecture](../images/mcp_architecture.svg)

## Key Features

### Automatic Code Generation

The MCP can generate code without requiring manual confirmation for each action:

```bash
tarscli mcp code path/to/file.cs "public class MyClass { }"
```

This will create or update the specified file with the provided code.

### Triple-Quoted Syntax

For multi-line code blocks, the MCP supports triple-quoted syntax:

```bash
tarscli mcp triple-code path/to/file.cs """
using System;

public class Program
{
    public static void Main()
    {
        Console.WriteLine("Hello, World!");
    }
}
"""
```

This makes it easier to generate complex code structures with proper formatting.

### Terminal Command Execution

The MCP can execute terminal commands without requiring permission prompts:

```bash
tarscli mcp execute "echo Hello, World!"
```

This enables automation of command-line tasks and integration with external tools.

### Augment Code Integration

The MCP integrates with Augment Code, allowing for enhanced code generation and analysis:

```bash
tarscli mcp augment sqlite uvx --args mcp-server-sqlite --db-path /path/to/test.db
```

## Architecture

The MCP is implemented as a controller with several services:

### McpController

The `McpController` is the main entry point for MCP commands. It:

1. Parses and validates commands
2. Routes commands to the appropriate handlers
3. Manages execution context and state

### EnhancedMcpService

The `EnhancedMcpService` provides advanced MCP capabilities, including:

1. Code generation with context awareness
2. Terminal command execution with output capture
3. Integration with external systems

### McpCommandHandler

The `McpCommandHandler` processes specific MCP commands, such as:

1. `code`: Generate code
2. `triple-code`: Generate multi-line code
3. `execute`: Execute terminal commands
4. `augment`: Integrate with Augment Code

## Using the MCP

### Basic Code Generation

To generate code for a file:

```bash
tarscli mcp code path/to/file.cs "public class MyClass { }"
```

This will:
1. Create the file if it doesn't exist
2. Write the provided code to the file
3. Display a success message

### Multi-line Code Generation

To generate multi-line code:

```bash
tarscli mcp triple-code path/to/file.cs """
using System;
using System.Collections.Generic;

namespace MyNamespace
{
    public class MyClass
    {
        public void MyMethod()
        {
            Console.WriteLine("Hello, World!");
        }
    }
}
"""
```

This will:
1. Create the file if it doesn't exist
2. Write the provided multi-line code to the file
3. Display a success message

### Terminal Command Execution

To execute a terminal command:

```bash
tarscli mcp execute "git status"
```

This will:
1. Execute the command in the terminal
2. Capture and display the output
3. Return the exit code

For more complex commands:

```bash
tarscli mcp execute "for i in {1..5}; do echo $i; done"
```

### Augment Code Integration

To integrate with Augment Code:

```bash
tarscli mcp augment sqlite uvx --args mcp-server-sqlite --db-path /path/to/test.db
```

This will:
1. Connect to the Augment Code server
2. Execute the specified command
3. Return the results

## Advanced Usage

### Command Chaining

You can chain multiple MCP commands together:

```bash
tarscli mcp execute "mkdir -p src/models" && \
tarscli mcp triple-code src/models/user.cs """
using System;

namespace Models
{
    public class User
    {
        public int Id { get; set; }
        public string Name { get; set; }
        public string Email { get; set; }
    }
}
"""
```

### Environment Variables

The MCP respects environment variables:

```bash
export PROJECT_ROOT=/path/to/project
tarscli mcp execute "cd $PROJECT_ROOT && ls -la"
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
