# Master Control Program (MCP) Integration with Augment Code

This document provides detailed information about the integration between TARS and Augment Code through the Master Control Program (MCP).

## Overview

The Master Control Program (MCP) serves as a bridge between TARS and external tools, particularly Augment Code. It enables autonomous operation by allowing TARS to:

1. Generate code without requiring manual confirmation
2. Execute terminal commands without permission prompts
3. Configure and interact with Augment Code MCP servers

## Key Features

### Automatic Code Generation

TARS can now generate code files without requiring manual confirmation from the user. This is particularly useful for:

- Creating boilerplate code
- Implementing standard patterns
- Generating code based on specifications
- Refactoring existing code

Example:
```bash
dotnet run --project TarsCli/TarsCli.csproj -- mcp code path/to/file.cs "public class MyClass { }"
```

### Triple-Quoted Syntax

TARS supports triple-quoted syntax (`"""..."""`) for multi-line code blocks, making it easier to generate complex code structures:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- mcp code path/to/file.cs -triple-quoted """using System;

public class Program
{
    public static void Main()
    {
        Console.WriteLine("Hello, World!");
    }
}"""
```

### Terminal Command Execution

TARS can execute terminal commands without requiring permission prompts, enabling automated workflows:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- mcp execute "echo Hello, World!"
```

### Augment Code Integration

TARS can configure and interact with Augment Code MCP servers:

```bash
dotnet run --project TarsCli/TarsCli.csproj -- mcp augment sqlite uvx --args mcp-server-sqlite --db-path /path/to/test.db
```

## Implementation Details

### Configuration

The MCP integration is configured through the `appsettings.json` file:

```json
{
  "Tars": {
    "Mcp": {
      "AutoExecuteEnabled": true,
      "AutoExecuteCommands": true,
      "AutoCodeGeneration": true
    }
  }
}
```

### Architecture

The MCP integration consists of several components:

1. **McpController**: Handles MCP commands and routes them to the appropriate handlers
2. **EnhancedMcpService**: Provides enhanced MCP functionality for code generation and command execution
3. **TripleQuotedArgumentParser**: Parses command-line arguments with triple-quoted syntax

### Security Considerations

The MCP integration executes commands and generates code without requiring permission prompts. This is powerful but comes with security implications:

- Only enable auto-execution in trusted environments
- Be cautious about the commands being executed
- Review generated code before using it in production

## Integration with Augment Code

TARS integrates with Augment Code by:

1. Configuring Augment Code MCP servers
2. Sending commands to Augment Code through the MCP
3. Receiving responses from Augment Code

### Setting Up Augment Code Integration

1. Install Augment Code in Visual Studio Code
2. Configure an MCP server in TARS:
   ```bash
   dotnet run --project TarsCli/TarsCli.csproj -- mcp augment tars-mcp uvx --args mcp-server-tars
   ```
3. Use the MCP to interact with Augment Code

## Examples

### Generate a C# Class

```bash
dotnet run --project TarsCli/TarsCli.csproj -- mcp code MyClass.cs -triple-quoted """using System;

namespace MyNamespace
{
    public class MyClass
    {
        public string Name { get; set; }
        public int Age { get; set; }
        
        public MyClass(string name, int age)
        {
            Name = name;
            Age = age;
        }
        
        public override string ToString()
        {
            return $"{Name} ({Age})";
        }
    }
}"""
```

### Execute a Build Command

```bash
dotnet run --project TarsCli/TarsCli.csproj -- mcp execute "dotnet build MyProject.csproj"
```

### Configure Augment Code MCP Server

```bash
dotnet run --project TarsCli/TarsCli.csproj -- mcp augment tars-mcp uvx --args mcp-server-tars --project-path /path/to/project
```

## Future Enhancements

Planned enhancements for the MCP integration include:

1. **Enhanced Security**: More granular control over command execution
2. **Workflow Integration**: Integration with TARS workflows
3. **Plugin System**: Support for custom MCP commands through plugins
4. **UI Integration**: Visual interface for MCP commands
