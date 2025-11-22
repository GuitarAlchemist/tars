# Code Style Guide

## Context

TARS-specific code style rules for Agent OS integration.

## General Formatting

### Indentation
- **F#**: Use 4 spaces for indentation (F# standard)
- **C#**: Use 4 spaces for indentation (.NET standard)
- **FLUX Metascripts**: Use 2 spaces for indentation
- Never use tabs - always spaces
- Maintain consistent indentation throughout files

### Naming Conventions

#### F# Conventions
- **Functions and Values**: Use camelCase (e.g., `calculateTotal`, `userProfile`)
- **Types and Modules**: Use PascalCase (e.g., `UserProfile`, `PaymentProcessor`)
- **Constants**: Use PascalCase (e.g., `MaxRetryCount`)
- **Record Fields**: Use PascalCase (e.g., `{ Name = "value"; Age = 25 }`)

#### C# Conventions
- **Methods and Properties**: Use PascalCase (e.g., `CalculateTotal`, `UserProfile`)
- **Variables and Parameters**: Use camelCase (e.g., `userProfile`, `calculateTotal`)
- **Constants**: Use PascalCase (e.g., `MaxRetryCount`)
- **Interfaces**: Prefix with 'I' (e.g., `IUserService`)

#### FLUX Metascript Conventions
- **Variables**: Use snake_case (e.g., `user_profile`, `calculation_result`)
- **Actions**: Use UPPER_CASE (e.g., `EXECUTE`, `ANALYZE`)
- **Agents**: Use PascalCase (e.g., `ReasoningAgent`, `CodeGenerator`)

### Code Organization

#### F# File Structure
```fsharp
namespace TarsEngine.FSharp.Core.ModuleName

open System
open Microsoft.Extensions.Logging
// ... other opens

/// Module documentation
module ModuleName =
    
    /// Type definitions first
    type SomeType = {
        Property1: string
        Property2: int
    }
    
    /// Private functions
    let private helperFunction x = x + 1
    
    /// Public functions
    let publicFunction input =
        // Implementation
        input |> helperFunction
```

#### C# File Structure
```csharp
using System;
using Microsoft.Extensions.Logging;
// ... other usings

namespace TarsEngine.FSharp.Core.ModuleName
{
    /// <summary>
    /// Class documentation
    /// </summary>
    public class SomeClass : ISomeInterface
    {
        private readonly ILogger<SomeClass> _logger;
        
        public SomeClass(ILogger<SomeClass> logger)
        {
            _logger = logger;
        }
        
        public async Task<Result> DoSomethingAsync()
        {
            // Implementation
        }
    }
}
```

### Comments and Documentation

#### F# Documentation
- Use `///` for XML documentation
- Document public functions and types
- Explain complex algorithms and business logic
- Use `//` for inline comments

#### C# Documentation
- Use `/// <summary>` XML documentation
- Document all public members
- Use `//` for inline comments
- Explain the "why" behind implementation choices

#### FLUX Metascript Documentation
- Use `//` for comments
- Document agent capabilities and objectives
- Explain complex reasoning chains

### Error Handling

#### F# Error Handling
- Prefer `Result<'T, 'Error>` over exceptions
- Use `AsyncResult` for async operations
- Chain operations with `|>` and `>>=`
- Handle all error cases explicitly

#### C# Error Handling
- Use exceptions for exceptional cases
- Prefer `Result<T>` pattern when available
- Always handle async operations properly
- Log errors with appropriate context

### TARS-Specific Standards

#### Quality Requirements
- **Zero tolerance for simulations/placeholders**
- All implementations must be real and functional
- Test coverage minimum 80%
- All FS0988 warnings treated as fatal errors

#### Performance Standards
- CUDA operations must show real GPU acceleration
- Vector operations target 184M+ searches/second
- Memory usage should be monitored and optimized
- Async operations preferred for I/O

#### Architecture Patterns
- Follow Clean Architecture principles
- Use dependency injection for all services
- Implement proper separation of concerns
- Prefer functional programming in F# modules
- Use Elmish/MVU for UI components
