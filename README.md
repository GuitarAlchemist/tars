# TARS Project

<div align="center">
  <img src="docs/images/tars_logo.svg" alt="TARS Logo" width="200"/>
  <br/>
  <img src="docs/images/tars_architecture.svg" alt="TARS Architecture" width="600"/>
  <br/>
  <img src="docs/images/tars_fractal.svg" alt="TARS Bifurcation" width="600"/>
  <p><i>For more information about these images, see <a href="docs/IMAGES.md">IMAGES.md</a></i></p>
</div>

## Overview

This repository contains the TARS (Transformative Autonomous Reasoning System) project, a powerful AI-driven development and automation system.

## Project Structure

- **TarsEngine** - Core C# engine for system operations
- **TarsEngineFSharp** - F# implementation of core algorithms
- **TarsEngine.Interfaces** - Shared interfaces for system components
- **TarsEngine.SelfImprovement** - F# module for self-improvement capabilities
- **TarsCli** - Command-line interface for interacting with TARS
- **TarsApp** - Main application with UI components
- **TarsCli.Tests** - Test suite for the CLI components
- **Experiments** - Various experimental implementations

## Features

### Master Control Program (MCP) Integration

TARS now includes a powerful Master Control Program (MCP) that enables autonomous operation and integration with Augment Code. Key features include:

- **Automatic Code Generation**: Generate code without requiring manual confirmation
- **Triple-Quoted Syntax**: Use `"""..."""` syntax for multi-line code blocks
- **Terminal Command Execution**: Execute terminal commands without permission prompts
- **Augment Code Integration**: Configure and interact with Augment Code MCP servers

### Learning and Education Features

TARS includes comprehensive learning and education features to help users master new skills:

- **Learning Plans**: Generate personalized learning plans with customizable goals, skill levels, and preferences
- **Course Generation**: Create structured courses with lessons, exercises, quizzes, and assessments
- **Tutorial Organization**: Manage and categorize tutorials with different difficulty levels and prerequisites
- **Demo Mode**: Showcase all TARS capabilities through interactive demonstrations

### Hugging Face Integration

TARS can now browse, download, and install the best coding LLMs from Hugging Face:

- **Model Discovery**: Search and find the best coding models on Hugging Face
- **Automatic Installation**: Download and convert models to work with Ollama
- **Seamless Integration**: Use Hugging Face models with all TARS commands
- **Model Management**: List and manage your installed models

[View Hugging Face integration documentation](docs/HUGGINGFACE_INTEGRATION.md)

### Deep Thinking and Exploration

TARS can generate in-depth explorations on complex topics:

- **Topic Exploration**: Generate detailed analyses on any topic
- **Related Topics**: Discover and explore related concepts
- **Version Evolution**: Build on previous explorations to deepen understanding
- **Consolidated Organization**: Explorations are organized by topic for easy reference

### Text-to-Speech Capabilities

TARS now includes text-to-speech functionality:

- **Multiple Voices**: Choose from various voice options
- **Language Support**: Support for multiple languages
- **Voice Cloning**: Clone voices from audio samples
- **Streaming Audio**: Direct audio streaming without intermediate files

### Self-Improvement Capabilities

TARS includes advanced self-improvement capabilities that allow it to analyze, improve, and learn from code:

#### Recent Progress

- **First Analysis Success**: Successfully analyzed test code and identified multiple issues
- **Pattern Recognition**: Implemented detection for magic numbers, inefficient string operations, and more
- **Learning Database**: Created a system to record improvements and feedback
- **JSON Escaping Fix**: Resolved API communication issues with Ollama
- **Console Capture**: Added ability to capture console output and use it to improve code
- **ANSI Escape Sequence Handling**: Improved handling of ANSI escape sequences in console output

[View detailed progress tracking](docs/PROGRESS.md) | [Technical documentation](docs/SELF_IMPROVEMENT.md)

#### Core Features

- **Code Analysis**: Analyze code for potential improvements
- **Improvement Proposals**: Generate proposals for code improvements
- **Self-Rewriting**: Automatically implement approved improvements
- **Learning System**: Track improvement history and learn from past changes
- **Console Output Analysis**: Capture and analyze console output to identify and fix issues

### Agent Coordination System

TARS implements a multi-agent coordination system:

- **Agent Roles**: Specialized agents for planning, coding, reviewing, and execution
- **Workflow Engine**: Coordinate multiple agents to complete complex tasks
- **Communication Protocol**: Structured message passing between agents
- **Custom Agent Configurations**: Define custom agent roles and capabilities

### Template System

TARS includes a flexible template system for session management:

- **Session Templates**: Create and manage templates for new sessions
- **Custom Templates**: Define custom templates for specific use cases
- **Variable Substitution**: Use variables in templates for dynamic content

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tars.git

# Build the solution
dotnet build

# Run the CLI
dotnet run --project TarsCli/TarsCli.csproj
```

### Basic Usage

```bash
# Initialize a new TARS session
dotnet run --project TarsCli/TarsCli.csproj -- init my-session

# Run a plan in a session
dotnet run --project TarsCli/TarsCli.csproj -- run --session my-session --plan template.fsx

# Analyze a file for improvements
dotnet run --project TarsCli/TarsCli.csproj -- self-analyze --file path/to/file.cs --model llama3

# Capture console output and use it to improve code
dotnet run --project TarsCli/TarsCli.csproj -- console-capture --start
# Run your commands that produce output
dotnet run --project TarsCli/TarsCli.csproj -- console-capture --stop
dotnet run --project TarsCli/TarsCli.csproj -- console-capture --analyze path/to/file.cs --apply
```

### MCP Commands

```bash
# Execute a terminal command without permission
dotnet run --project TarsCli/TarsCli.csproj -- mcp execute "echo Hello, World!"

# Generate code with triple-quoted syntax
dotnet run --project TarsCli/TarsCli.csproj -- mcp code path/to/file.cs -triple-quoted """using System;

public class Program
{
    public static void Main()
    {
        Console.WriteLine(\"Hello, World!\");
    }
}"""

# Configure Augment Code MCP server
dotnet run --project TarsCli/TarsCli.csproj -- mcp augment sqlite uvx --args mcp-server-sqlite --db-path /path/to/test.db
```

### Multi-Agent Workflows

```bash
# Run a multi-agent workflow for a task
dotnet run --project TarsCli/TarsCli.csproj -- workflow --task "Create a simple web API in C#"
```

## Development

TARS is designed with a hybrid approach:

- **F#** for core engine components and algorithms
- **C#** for CLI and application interfaces

This combination provides both functional programming benefits and strong integration with .NET ecosystem.

### Testing

```bash
# Run the test suite
dotnet test TarsCli.Tests/TarsCli.Tests.csproj
```

## Future Directions

- **Plugin System**: Support for custom extensions and third-party integrations
- **Multiple LLM Providers**: Integration with various AI providers
- **Web UI**: Browser-based interface for managing TARS sessions
- **Collaborative Workflows**: Support for multi-user collaboration