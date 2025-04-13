param (
    [Parameter(Mandatory=$true)]
    [string]$ExplorationFile,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputPath,
    
    [Parameter(Mandatory=$false)]
    [string]$Language = "csharp",
    
    [Parameter(Mandatory=$false)]
    [string]$ArchitectureStyle = "clean-architecture",
    
    [Parameter(Mandatory=$false)]
    [string]$DesignPatterns = "repository,cqrs,mediator",
    
    [Parameter(Mandatory=$false)]
    [string]$Framework = "dotnet-core",
    
    [Parameter(Mandatory=$false)]
    [string]$Model = "llama3",
    
    [Parameter(Mandatory=$false)]
    [switch]$Verbose = $false
)

# Function to display colored text
function Write-ColoredText {
    param (
        [string]$Text,
        [string]$Color = "White"
    )
    
    Write-Host $Text -ForegroundColor $Color
}

# Function to log verbose information
function Write-VerboseLog {
    param (
        [string]$Text
    )
    
    if ($Verbose) {
        Write-ColoredText "VERBOSE: $Text" "DarkGray"
    }
}

# Check if exploration file exists
if (-not (Test-Path $ExplorationFile)) {
    Write-ColoredText "Error: Exploration file not found: $ExplorationFile" "Red"
    exit 1
}

# Determine output path
if (-not $OutputPath) {
    $directory = [System.IO.Path]::GetDirectoryName($ExplorationFile)
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($ExplorationFile)
    $OutputPath = Join-Path $directory "$filename-generated"
}

# Create output directory if it doesn't exist
if (-not (Test-Path $OutputPath)) {
    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null
    Write-VerboseLog "Created output directory: $OutputPath"
}

# Read the exploration file
Write-ColoredText "Reading exploration file: $ExplorationFile" "Cyan"
$explorationContent = Get-Content -Path $ExplorationFile -Raw
Write-VerboseLog "Exploration content loaded: $(($explorationContent -split "`n").Length) lines"

# Generate language-specific prompt
function Get-LanguageSpecificPrompt {
    param (
        [string]$Language
    )
    
    switch ($Language.ToLower()) {
        "csharp" {
            return @"
For C# code:
- Use C# 10.0 features where appropriate
- Follow Microsoft's C# coding conventions
- Use LINQ for collection operations
- Use async/await for asynchronous operations
- Use nullable reference types
- Use pattern matching where appropriate
- Prefer expression-bodied members for simple methods
- Use string interpolation instead of string.Format
- Add XML documentation comments for public members
- Follow the principle of minimal API surface
- Use dependency injection for services
- Implement proper exception handling
- Use ILogger for logging
"@
        }
        "fsharp" {
            return @"
For F# code:
- Use F# 6.0 features where appropriate
- Follow F# coding conventions
- Use functional programming patterns
- Use discriminated unions for modeling domain concepts
- Use computation expressions for complex workflows
- Use pattern matching extensively
- Use railway-oriented programming for error handling
- Use type providers where appropriate
- Prefer immutable data structures
- Use active patterns for complex matching scenarios
- Use modules to organize related functions
- Use records for data structures
"@
        }
        "typescript" {
            return @"
For TypeScript code:
- Use TypeScript 4.5+ features where appropriate
- Follow the TypeScript coding guidelines
- Use interfaces for defining contracts
- Use type guards for runtime type checking
- Use generics for reusable components
- Use async/await for asynchronous operations
- Use optional chaining and nullish coalescing
- Use ES6+ features like destructuring, spread operator
- Use enums for related constants
- Use readonly for immutable properties
- Use namespaces to organize related code
- Use decorators for metadata programming
"@
        }
        "python" {
            return @"
For Python code:
- Use Python 3.9+ features where appropriate
- Follow PEP 8 style guide
- Use type hints for better IDE support
- Use dataclasses for data containers
- Use async/await for asynchronous operations
- Use context managers for resource management
- Use list/dict comprehensions for concise code
- Use generators for memory-efficient iteration
- Use proper exception handling with specific exceptions
- Use docstrings for documentation
- Use virtual environments for dependency management
- Follow the principle of duck typing
"@
        }
        default {
            return "Use modern language features and follow best practices for code organization and style."
        }
    }
}

# Generate architecture-specific prompt
function Get-ArchitectureSpecificPrompt {
    param (
        [string]$ArchitectureStyle
    )
    
    switch ($ArchitectureStyle.ToLower()) {
        "clean-architecture" {
            return @"
For Clean Architecture:
- Organize code into layers: Domain, Application, Infrastructure, and Presentation
- Domain layer should contain entities, value objects, and domain services
- Application layer should contain use cases, DTOs, and interfaces
- Infrastructure layer should contain implementations of interfaces defined in the application layer
- Presentation layer should contain controllers, views, and view models
- Dependencies should point inward, with the domain layer having no dependencies
- Use dependency injection to maintain the dependency rule
- Use the repository pattern for data access
- Use the unit of work pattern for transaction management
"@
        }
        "hexagonal-architecture" {
            return @"
For Hexagonal Architecture (Ports and Adapters):
- Organize code into core domain and adapters
- Core domain should contain business logic and port interfaces
- Adapters should implement ports and connect to external systems
- Primary adapters drive the application (e.g., API controllers)
- Secondary adapters are driven by the application (e.g., repositories)
- Dependencies point inward to the core domain
- Use dependency injection to maintain the dependency rule
- Ensure the domain is isolated from infrastructure concerns
"@
        }
        "microservices" {
            return @"
For Microservices Architecture:
- Design services around business capabilities
- Each service should have its own database
- Services should communicate through well-defined APIs
- Implement service discovery and load balancing
- Use the API Gateway pattern for client communication
- Implement the Circuit Breaker pattern for resilience
- Use event-driven communication for asynchronous processes
- Implement distributed tracing for monitoring
- Use containerization for deployment
- Implement health checks for each service
"@
        }
        "event-sourcing" {
            return @"
For Event Sourcing Architecture:
- Store state changes as a sequence of events
- Rebuild state by replaying events
- Use event stores for persistence
- Implement projections for read models
- Use CQRS to separate read and write operations
- Implement event handlers for side effects
- Use snapshots for performance optimization
- Ensure events are immutable and append-only
- Design events to be meaningful business occurrences
"@
        }
        default {
            return "Organize code using appropriate architectural patterns and principles."
        }
    }
}

# Generate design pattern-specific prompt
function Get-DesignPatternPrompt {
    param (
        [string]$DesignPatterns
    )
    
    $patterns = $DesignPatterns -split ','
    $promptParts = @()
    
    foreach ($pattern in $patterns) {
        switch ($pattern.Trim().ToLower()) {
            "repository" {
                $promptParts += @"
Repository Pattern:
- Create repository interfaces in the domain layer
- Implement repositories in the infrastructure layer
- Use the Unit of Work pattern for transaction management
- Repositories should return domain entities, not DTOs
- Use specification pattern for complex queries
"@
            }
            "cqrs" {
                $promptParts += @"
CQRS Pattern:
- Separate command (write) and query (read) operations
- Commands should modify state but return minimal data
- Queries should return data but not modify state
- Use different models for commands and queries
- Consider using different data stores for reads and writes
"@
            }
            "mediator" {
                $promptParts += @"
Mediator Pattern:
- Use a mediator to decouple components
- Commands and queries should be handled by the mediator
- Implement handlers for each command and query
- Use the mediator for cross-cutting concerns like validation and logging
- Consider using a library like MediatR
"@
            }
            "factory" {
                $promptParts += @"
Factory Pattern:
- Use factories to create complex objects
- Hide creation logic from clients
- Consider using abstract factories for families of objects
- Use factory methods for creating objects in base classes
"@
            }
            "strategy" {
                $promptParts += @"
Strategy Pattern:
- Define a family of algorithms
- Encapsulate each algorithm in a separate class
- Make the algorithms interchangeable
- Use dependency injection to provide strategies
"@
            }
            "observer" {
                $promptParts += @"
Observer Pattern:
- Define a one-to-many dependency between objects
- When one object changes state, all dependents are notified
- Use events and event handlers for implementation
- Consider using a message bus for loosely coupled communication
"@
            }
            "decorator" {
                $promptParts += @"
Decorator Pattern:
- Attach additional responsibilities to objects dynamically
- Provide a flexible alternative to subclassing
- Decorators should have the same interface as the objects they decorate
- Use for cross-cutting concerns like logging, caching, or authorization
"@
            }
        }
    }
    
    return $promptParts -join "`n`n"
}

# Generate framework-specific prompt
function Get-FrameworkSpecificPrompt {
    param (
        [string]$Framework
    )
    
    switch ($Framework.ToLower()) {
        "dotnet-core" {
            return @"
For .NET Core:
- Use the latest .NET Core version (6.0+)
- Use dependency injection with IServiceCollection
- Use configuration with IConfiguration and appsettings.json
- Use middleware for cross-cutting concerns
- Use options pattern for configuration
- Use ILogger for logging
- Use Entity Framework Core for data access
- Use ASP.NET Core for web APIs
- Use health checks for monitoring
- Use background services for background processing
"@
        }
        "spring-boot" {
            return @"
For Spring Boot:
- Use the latest Spring Boot version
- Use dependency injection with @Autowired
- Use @Configuration for configuration
- Use @RestController for REST APIs
- Use Spring Data for data access
- Use @Transactional for transaction management
- Use Spring Security for authentication and authorization
- Use Spring Actuator for monitoring
- Use Spring Cloud for microservices
"@
        }
        "react" {
            return @"
For React:
- Use functional components with hooks
- Use React Router for routing
- Use Redux or Context API for state management
- Use React Query for data fetching
- Use styled-components or CSS modules for styling
- Use React Testing Library for testing
- Use React.memo for performance optimization
- Use React.lazy and Suspense for code splitting
- Use custom hooks for reusable logic
"@
        }
        "angular" {
            return @"
For Angular:
- Use the latest Angular version
- Use Angular CLI for project scaffolding
- Use Angular modules for organization
- Use services for business logic and data access
- Use RxJS for reactive programming
- Use Angular Material for UI components
- Use NgRx for state management
- Use Angular forms for form handling
- Use Angular routing for navigation
"@
        }
        default {
            return "Use appropriate framework features and follow framework-specific best practices."
        }
    }
}

# Create the prompt
Write-ColoredText "Generating prompt for code generation..." "Cyan"

$languagePrompt = Get-LanguageSpecificPrompt -Language $Language
$architecturePrompt = Get-ArchitectureSpecificPrompt -ArchitectureStyle $ArchitectureStyle
$designPatternPrompt = Get-DesignPatternPrompt -DesignPatterns $DesignPatterns
$frameworkPrompt = Get-FrameworkSpecificPrompt -Framework $Framework

$prompt = @"
You are an expert software developer specializing in $Language development. Your task is to generate high-quality code based on the following exploration transcript.

The exploration describes requirements, ideas, and concepts that need to be implemented. You should generate complete, working code that implements these concepts.

# Language Guidelines
$languagePrompt

# Architecture Guidelines
$architecturePrompt

# Design Pattern Guidelines
$designPatternPrompt

# Framework Guidelines
$frameworkPrompt

# General Guidelines
- Generate complete, working code that can be compiled and run
- Include proper error handling, validation, and logging
- Write clean, maintainable code with appropriate comments
- Follow SOLID principles and other best practices
- Consider performance, security, and scalability
- Include unit tests where appropriate
- Organize code in a logical directory structure
- Include necessary configuration files
- Provide clear instructions for running the code

# Exploration Transcript
$explorationContent

Now, generate the code that implements the concepts described in this exploration. Provide a complete solution with all necessary files and directory structure.
"@

# Create a temporary file for the prompt
$promptFile = [System.IO.Path]::GetTempFileName()
$prompt | Set-Content -Path $promptFile -Force
Write-VerboseLog "Prompt saved to temporary file: $promptFile"

# Generate code using the prompt
Write-ColoredText "Generating code using $Model model..." "Cyan"

try {
    # Call TARS CLI to generate code
    $cliOutput = dotnet run --project TarsCli/TarsCli.csproj -- generate --prompt-file $promptFile --model $Model
    
    # Extract the generated code
    $generatedCode = $cliOutput -join "`n"
    
    # Save the generated code to the output directory
    $outputFile = Join-Path $OutputPath "README.md"
    @"
# Generated Code

This code was generated based on the exploration file: $ExplorationFile

## Generation Parameters
- Language: $Language
- Architecture: $ArchitectureStyle
- Design Patterns: $DesignPatterns
- Framework: $Framework
- Model: $Model

## Generated Code Overview

The following code was generated based on the exploration:

```
$generatedCode
```

## Next Steps
1. Review the generated code
2. Make any necessary adjustments
3. Run the code to verify functionality
4. Write additional tests as needed
"@ | Set-Content -Path $outputFile -Force
    
    Write-ColoredText "Generated code saved to: $OutputPath" "Green"
    Write-ColoredText "README file created at: $outputFile" "Green"
}
catch {
    Write-ColoredText "Error generating code: $_" "Red"
    exit 1
}
finally {
    # Clean up
    Remove-Item -Path $promptFile -Force
    Write-VerboseLog "Temporary prompt file removed"
}

Write-ColoredText "Code generation completed successfully!" "Green"
