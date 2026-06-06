# TARS Technical Architecture

This document provides a detailed overview of the TARS architecture, including its components, interactions, and design principles.

## System Overview

TARS is built using a modular, service-oriented architecture that enables flexibility, extensibility, and maintainability. The system is primarily implemented in F# for core engine components and C# for CLI and application interfaces.

![TARS Architecture](images/tars_architecture.svg)

## Core Components

### TarsEngine

The `TarsEngine` is the heart of the system, providing the core reasoning, analysis, and improvement capabilities. It is implemented in F# to leverage the language's strong type system, pattern matching, and functional programming features.

Key modules include:

- **TarsEngine.Interfaces**: Shared interfaces for system components
- **TarsEngine.SelfImprovement**: F# module for self-improvement capabilities
- **TarsEngineFSharp**: Core F# implementation of the engine

### TarsCli

The `TarsCli` is a command-line interface for interacting with TARS. It provides commands for various operations, including code analysis, improvement, and language model management.

Key components include:

- **Command Handlers**: Handlers for CLI commands
- **Services**: Services for various functionalities
- **MCP**: Master Control Program for autonomous operation

### TarsApp

The `TarsApp` is a web application that provides a graphical interface for interacting with TARS. It is built using Blazor WebAssembly for a responsive, client-side experience.

Key components include:

- **Components**: Reusable UI components
- **Pages**: Application pages for different functionalities
- **Services**: Client-side services for interacting with the TARS API

## Service Architecture

TARS follows a service-oriented architecture, with each service responsible for a specific aspect of functionality:

### OllamaService

Handles interaction with the Ollama API for local language model inference.

```csharp
public class OllamaService
{
    // Generate text using the Ollama API
    public async Task<string> GenerateCompletion(string prompt, string model)
    {
        // Implementation details...
    }
    
    // Other methods...
}
```

### OllamaSetupService

Manages the setup and configuration of Ollama, including model installation and verification.

```csharp
public class OllamaSetupService
{
    // Check if Ollama is properly set up
    public async Task<bool> CheckOllamaSetupAsync()
    {
        // Implementation details...
    }
    
    // Other methods...
}
```

### SelfImprovementService

Provides self-improvement capabilities, including code analysis, improvement proposals, and automatic rewriting.

```csharp
public class SelfImprovementService
{
    // Analyze a file for potential improvements
    public async Task<bool> AnalyzeFile(string filePath, string model)
    {
        // Implementation details...
    }
    
    // Propose improvements for a file
    public async Task<bool> ProposeImprovement(string filePath, string model, bool autoAccept = false)
    {
        // Implementation details...
    }
    
    // Other methods...
}
```

### HuggingFaceService

Handles interaction with the Hugging Face API for model discovery, downloading, and installation.

```csharp
public class HuggingFaceService
{
    // Search for models on Hugging Face
    public async Task<List<HuggingFaceModel>> SearchModelsAsync(string query, string task = "text-generation", int limit = 10)
    {
        // Implementation details...
    }
    
    // Other methods...
}
```

### LanguageSpecificationService

Generates language specifications and documentation for the TARS DSL.

```csharp
public class LanguageSpecificationService
{
    // Generate EBNF specification for TARS DSL
    public async Task<string> GenerateEbnfAsync()
    {
        // Implementation details...
    }
    
    // Other methods...
}
```

## Data Flow

The data flow in TARS follows a typical request-response pattern:

1. **User Input**: The user interacts with TARS through the CLI or web interface
2. **Command Processing**: The input is processed by the appropriate command handler
3. **Service Invocation**: The command handler invokes the relevant service(s)
4. **Engine Processing**: The service interacts with the TARS engine for processing
5. **Response Generation**: The engine generates a response
6. **Output Formatting**: The response is formatted for presentation to the user
7. **User Feedback**: The user provides feedback, which is used for learning and improvement

## Design Principles

TARS is designed according to several key principles:

### 1. Modularity

TARS is built as a collection of loosely coupled modules, each with a specific responsibility. This enables flexibility, reusability, and maintainability.

### 2. Functional Core, Imperative Shell

TARS follows the "functional core, imperative shell" pattern, with a pure functional core (implemented in F#) surrounded by an imperative shell (implemented in C#) for interaction with the outside world.

### 3. Dependency Injection

TARS uses dependency injection to manage dependencies between components, making the system more testable, maintainable, and flexible.

### 4. Command-Query Separation

TARS follows the command-query separation principle, with clear separation between commands (which change state) and queries (which return data).

### 5. Progressive Enhancement

TARS is designed to work with varying levels of AI capability, from basic rule-based analysis to advanced language model-powered reasoning.

## Technology Stack

TARS is built using a modern technology stack:

- **Languages**: F#, C#
- **Frameworks**: .NET 9, ASP.NET Core, Blazor WebAssembly
- **Libraries**: FParsec (for parsing), MudBlazor (for UI), Newtonsoft.Json (for JSON handling)
- **Tools**: Ollama (for local language model inference), Hugging Face (for model discovery and download)
- **Infrastructure**: GitHub Actions (for CI/CD), Docker (for containerization)

## Extensibility

TARS is designed to be extensible in several ways:

### 1. New Commands

New CLI commands can be added by creating a new command handler and registering it with the command line parser.

### 2. New Services

New services can be added by creating a new service class and registering it with the dependency injection container.

### 3. New Language Models

New language models can be added by installing them through Ollama or downloading them from Hugging Face.

### 4. New Analysis Patterns

New code analysis patterns can be added to the pattern recognition system in the `TarsEngine.SelfImprovement` module.

## Future Architecture

The future architecture of TARS will include:

1. **Multi-Agent Framework**: A framework for coordinating multiple specialized agents
2. **Distributed Processing**: Support for distributed processing of large codebases
3. **Advanced Learning System**: A more sophisticated learning system with reinforcement learning and transfer learning
4. **IDE Integration**: Deeper integration with popular IDEs
5. **Collaborative Development**: Support for collaborative development with multiple users
