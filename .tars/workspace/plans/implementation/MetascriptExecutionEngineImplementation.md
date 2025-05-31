# MetascriptExecutionEngine Implementation Plan

## Overview
This document outlines the implementation plan for the MetascriptExecutionEngine.cs file and its dependencies. We'll use monads for null handling and implement mock versions of the required classes.

## Required Model Classes

### 1. MetascriptType.cs (Enum)
```csharp
namespace TarsEngine.Models;

/// <summary>
/// Types of metascripts
/// </summary>
public enum MetascriptType
{
    /// <summary>
    /// Self-improvement metascript
    /// </summary>
    SelfImprovement,
    
    /// <summary>
    /// Code generation metascript
    /// </summary>
    CodeGeneration,
    
    /// <summary>
    /// Documentation generation metascript
    /// </summary>
    DocumentationGeneration,
    
    /// <summary>
    /// Test generation metascript
    /// </summary>
    TestGeneration,
    
    /// <summary>
    /// Refactoring metascript
    /// </summary>
    Refactoring,
    
    /// <summary>
    /// Optimization metascript
    /// </summary>
    Optimization,
    
    /// <summary>
    /// Bug fixing metascript
    /// </summary>
    BugFixing,
    
    /// <summary>
    /// Feature implementation metascript
    /// </summary>
    FeatureImplementation,
    
    /// <summary>
    /// Learning metascript
    /// </summary>
    Learning,
    
    /// <summary>
    /// Analysis metascript
    /// </summary>
    Analysis,
    
    /// <summary>
    /// Custom metascript
    /// </summary>
    Custom
}
```

### 2. MetascriptExecutionStatus.cs (Enum)
```csharp
namespace TarsEngine.Models;

/// <summary>
/// Execution statuses for metascripts
/// </summary>
public enum MetascriptExecutionStatus
{
    /// <summary>
    /// Pending execution
    /// </summary>
    Pending,
    
    /// <summary>
    /// In progress
    /// </summary>
    InProgress,
    
    /// <summary>
    /// Completed successfully
    /// </summary>
    Completed,
    
    /// <summary>
    /// Failed
    /// </summary>
    Failed,
    
    /// <summary>
    /// Cancelled
    /// </summary>
    Cancelled,
    
    /// <summary>
    /// Paused
    /// </summary>
    Paused,
    
    /// <summary>
    /// Scheduled
    /// </summary>
    Scheduled,
    
    /// <summary>
    /// Waiting for input
    /// </summary>
    WaitingForInput,
    
    /// <summary>
    /// Waiting for resources
    /// </summary>
    WaitingForResources,
    
    /// <summary>
    /// Waiting for approval
    /// </summary>
    WaitingForApproval
}
```

### 3. Metascript.cs
```csharp
namespace TarsEngine.Models;

/// <summary>
/// Represents a metascript
/// </summary>
public class Metascript
{
    /// <summary>
    /// Gets or sets the metascript ID
    /// </summary>
    public string Id { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the metascript name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the metascript description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the metascript type
    /// </summary>
    public MetascriptType Type { get; set; }
    
    /// <summary>
    /// Gets or sets the metascript content
    /// </summary>
    public string Content { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the metascript version
    /// </summary>
    public string Version { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the metascript author
    /// </summary>
    public string Author { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the metascript creation timestamp
    /// </summary>
    public DateTime CreationTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the metascript last modified timestamp
    /// </summary>
    public DateTime LastModifiedTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the metascript tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the metascript dependencies
    /// </summary>
    public List<string> Dependencies { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the metascript parameters
    /// </summary>
    public Dictionary<string, string> Parameters { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the metascript input schema
    /// </summary>
    public string? InputSchema { get; set; }
    
    /// <summary>
    /// Gets or sets the metascript output schema
    /// </summary>
    public string? OutputSchema { get; set; }
    
    /// <summary>
    /// Gets or sets the metascript file path
    /// </summary>
    public string? FilePath { get; set; }
}
```

### 4. MetascriptExecutionContext.cs
```csharp
namespace TarsEngine.Models;

/// <summary>
/// Represents a metascript execution context
/// </summary>
public class MetascriptExecutionContext
{
    /// <summary>
    /// Gets or sets the context ID
    /// </summary>
    public string Id { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the metascript
    /// </summary>
    public Metascript Metascript { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution status
    /// </summary>
    public MetascriptExecutionStatus Status { get; set; }
    
    /// <summary>
    /// Gets or sets the execution start timestamp
    /// </summary>
    public DateTime? StartTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the execution end timestamp
    /// </summary>
    public DateTime? EndTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the execution duration in seconds
    /// </summary>
    public double? DurationSeconds { get; set; }
    
    /// <summary>
    /// Gets or sets the execution input
    /// </summary>
    public Dictionary<string, object> Input { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution output
    /// </summary>
    public Dictionary<string, object> Output { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution logs
    /// </summary>
    public List<string> Logs { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution errors
    /// </summary>
    public List<string> Errors { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution warnings
    /// </summary>
    public List<string> Warnings { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution progress
    /// </summary>
    public double Progress { get; set; }
    
    /// <summary>
    /// Gets or sets the execution parent context ID
    /// </summary>
    public string? ParentContextId { get; set; }
    
    /// <summary>
    /// Gets or sets the execution child context IDs
    /// </summary>
    public List<string> ChildContextIds { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution user
    /// </summary>
    public string? User { get; set; }
    
    /// <summary>
    /// Gets or sets the execution environment
    /// </summary>
    public string? Environment { get; set; }
    
    /// <summary>
    /// Gets or sets the execution tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
}
```

### 5. MetascriptExecutionResult.cs
```csharp
namespace TarsEngine.Models;

/// <summary>
/// Represents a metascript execution result
/// </summary>
public class MetascriptExecutionResult
{
    /// <summary>
    /// Gets or sets the result ID
    /// </summary>
    public string Id { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the execution context
    /// </summary>
    public MetascriptExecutionContext Context { get; set; } = new();
    
    /// <summary>
    /// Gets or sets whether the execution was successful
    /// </summary>
    public bool IsSuccess { get; set; }
    
    /// <summary>
    /// Gets or sets the execution output
    /// </summary>
    public Dictionary<string, object> Output { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution error message
    /// </summary>
    public string? ErrorMessage { get; set; }
    
    /// <summary>
    /// Gets or sets the execution error details
    /// </summary>
    public string? ErrorDetails { get; set; }
    
    /// <summary>
    /// Gets or sets the execution logs
    /// </summary>
    public List<string> Logs { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution artifacts
    /// </summary>
    public List<MetascriptArtifact> Artifacts { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution metrics
    /// </summary>
    public Dictionary<string, double> Metrics { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the execution summary
    /// </summary>
    public string? Summary { get; set; }
    
    /// <summary>
    /// Gets or sets the execution insights
    /// </summary>
    public List<string> Insights { get; set; } = new();
}
```

### 6. MetascriptArtifact.cs
```csharp
namespace TarsEngine.Models;

/// <summary>
/// Represents a metascript artifact
/// </summary>
public class MetascriptArtifact
{
    /// <summary>
    /// Gets or sets the artifact ID
    /// </summary>
    public string Id { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the artifact name
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the artifact description
    /// </summary>
    public string Description { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the artifact type
    /// </summary>
    public string Type { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the artifact content
    /// </summary>
    public string Content { get; set; } = string.Empty;
    
    /// <summary>
    /// Gets or sets the artifact file path
    /// </summary>
    public string? FilePath { get; set; }
    
    /// <summary>
    /// Gets or sets the artifact creation timestamp
    /// </summary>
    public DateTime CreationTimestamp { get; set; }
    
    /// <summary>
    /// Gets or sets the artifact size in bytes
    /// </summary>
    public long SizeBytes { get; set; }
    
    /// <summary>
    /// Gets or sets the artifact metadata
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the artifact tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
}
```

### 7. MetascriptExecutionOptions.cs
```csharp
namespace TarsEngine.Models;

/// <summary>
/// Options for metascript execution
/// </summary>
public class MetascriptExecutionOptions
{
    /// <summary>
    /// Gets or sets the execution timeout in seconds
    /// </summary>
    public double? TimeoutSeconds { get; set; }
    
    /// <summary>
    /// Gets or sets whether to execute in dry run mode
    /// </summary>
    public bool DryRun { get; set; }
    
    /// <summary>
    /// Gets or sets whether to execute in verbose mode
    /// </summary>
    public bool Verbose { get; set; }
    
    /// <summary>
    /// Gets or sets whether to execute in debug mode
    /// </summary>
    public bool Debug { get; set; }
    
    /// <summary>
    /// Gets or sets whether to execute in silent mode
    /// </summary>
    public bool Silent { get; set; }
    
    /// <summary>
    /// Gets or sets whether to execute in interactive mode
    /// </summary>
    public bool Interactive { get; set; }
    
    /// <summary>
    /// Gets or sets whether to execute in force mode
    /// </summary>
    public bool Force { get; set; }
    
    /// <summary>
    /// Gets or sets whether to execute in safe mode
    /// </summary>
    public bool SafeMode { get; set; }
    
    /// <summary>
    /// Gets or sets whether to execute in parallel mode
    /// </summary>
    public bool Parallel { get; set; }
    
    /// <summary>
    /// Gets or sets the maximum number of parallel executions
    /// </summary>
    public int? MaxParallelExecutions { get; set; }
    
    /// <summary>
    /// Gets or sets whether to continue on error
    /// </summary>
    public bool ContinueOnError { get; set; }
    
    /// <summary>
    /// Gets or sets whether to validate input
    /// </summary>
    public bool ValidateInput { get; set; } = true;
    
    /// <summary>
    /// Gets or sets whether to validate output
    /// </summary>
    public bool ValidateOutput { get; set; } = true;
    
    /// <summary>
    /// Gets or sets whether to generate artifacts
    /// </summary>
    public bool GenerateArtifacts { get; set; } = true;
    
    /// <summary>
    /// Gets or sets whether to generate metrics
    /// </summary>
    public bool GenerateMetrics { get; set; } = true;
    
    /// <summary>
    /// Gets or sets whether to generate insights
    /// </summary>
    public bool GenerateInsights { get; set; } = true;
    
    /// <summary>
    /// Gets or sets whether to generate summary
    /// </summary>
    public bool GenerateSummary { get; set; } = true;
    
    /// <summary>
    /// Gets or sets the environment variables
    /// </summary>
    public Dictionary<string, string> EnvironmentVariables { get; set; } = new();
    
    /// <summary>
    /// Gets or sets the working directory
    /// </summary>
    public string? WorkingDirectory { get; set; }
    
    /// <summary>
    /// Gets or sets the user
    /// </summary>
    public string? User { get; set; }
    
    /// <summary>
    /// Gets or sets the environment
    /// </summary>
    public string? Environment { get; set; }
    
    /// <summary>
    /// Gets or sets the tags
    /// </summary>
    public List<string> Tags { get; set; } = new();
}
```

## Required Service Interface

### 1. IMetascriptExecutionEngine.cs
```csharp
namespace TarsEngine.Models;

using TarsEngine.Monads;

/// <summary>
/// Interface for the metascript execution engine
/// </summary>
public interface IMetascriptExecutionEngine
{
    /// <summary>
    /// Executes a metascript
    /// </summary>
    /// <param name="metascript">The metascript</param>
    /// <param name="input">The input</param>
    /// <param name="options">The execution options</param>
    /// <returns>A result containing the execution result or an error</returns>
    Task<Result<MetascriptExecutionResult, string>> ExecuteAsync(Metascript metascript, Dictionary<string, object>? input = null, MetascriptExecutionOptions? options = null);
    
    /// <summary>
    /// Executes a metascript by ID
    /// </summary>
    /// <param name="metascriptId">The metascript ID</param>
    /// <param name="input">The input</param>
    /// <param name="options">The execution options</param>
    /// <returns>A result containing the execution result or an error</returns>
    Task<Result<MetascriptExecutionResult, string>> ExecuteByIdAsync(string metascriptId, Dictionary<string, object>? input = null, MetascriptExecutionOptions? options = null);
    
    /// <summary>
    /// Executes a metascript by file path
    /// </summary>
    /// <param name="filePath">The file path</param>
    /// <param name="input">The input</param>
    /// <param name="options">The execution options</param>
    /// <returns>A result containing the execution result or an error</returns>
    Task<Result<MetascriptExecutionResult, string>> ExecuteByFilePathAsync(string filePath, Dictionary<string, object>? input = null, MetascriptExecutionOptions? options = null);
    
    /// <summary>
    /// Executes a metascript by content
    /// </summary>
    /// <param name="content">The content</param>
    /// <param name="input">The input</param>
    /// <param name="options">The execution options</param>
    /// <returns>A result containing the execution result or an error</returns>
    Task<Result<MetascriptExecutionResult, string>> ExecuteByContentAsync(string content, Dictionary<string, object>? input = null, MetascriptExecutionOptions? options = null);
    
    /// <summary>
    /// Gets an execution context by ID
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>A result containing the execution context or an error</returns>
    Task<Result<MetascriptExecutionContext, string>> GetExecutionContextByIdAsync(string contextId);
    
    /// <summary>
    /// Gets execution contexts by status
    /// </summary>
    /// <param name="status">The status</param>
    /// <returns>A result containing the execution contexts or an error</returns>
    Task<Result<List<MetascriptExecutionContext>, string>> GetExecutionContextsByStatusAsync(MetascriptExecutionStatus status);
    
    /// <summary>
    /// Gets execution contexts by metascript ID
    /// </summary>
    /// <param name="metascriptId">The metascript ID</param>
    /// <returns>A result containing the execution contexts or an error</returns>
    Task<Result<List<MetascriptExecutionContext>, string>> GetExecutionContextsByMetascriptIdAsync(string metascriptId);
    
    /// <summary>
    /// Gets execution contexts by user
    /// </summary>
    /// <param name="user">The user</param>
    /// <returns>A result containing the execution contexts or an error</returns>
    Task<Result<List<MetascriptExecutionContext>, string>> GetExecutionContextsByUserAsync(string user);
    
    /// <summary>
    /// Gets execution contexts by environment
    /// </summary>
    /// <param name="environment">The environment</param>
    /// <returns>A result containing the execution contexts or an error</returns>
    Task<Result<List<MetascriptExecutionContext>, string>> GetExecutionContextsByEnvironmentAsync(string environment);
    
    /// <summary>
    /// Gets execution contexts by tag
    /// </summary>
    /// <param name="tag">The tag</param>
    /// <returns>A result containing the execution contexts or an error</returns>
    Task<Result<List<MetascriptExecutionContext>, string>> GetExecutionContextsByTagAsync(string tag);
    
    /// <summary>
    /// Gets execution contexts by time range
    /// </summary>
    /// <param name="startTimestamp">The start timestamp</param>
    /// <param name="endTimestamp">The end timestamp</param>
    /// <returns>A result containing the execution contexts or an error</returns>
    Task<Result<List<MetascriptExecutionContext>, string>> GetExecutionContextsByTimeRangeAsync(DateTime startTimestamp, DateTime endTimestamp);
    
    /// <summary>
    /// Gets an execution result by ID
    /// </summary>
    /// <param name="resultId">The result ID</param>
    /// <returns>A result containing the execution result or an error</returns>
    Task<Result<MetascriptExecutionResult, string>> GetExecutionResultByIdAsync(string resultId);
    
    /// <summary>
    /// Gets execution results by context ID
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>A result containing the execution results or an error</returns>
    Task<Result<List<MetascriptExecutionResult>, string>> GetExecutionResultsByContextIdAsync(string contextId);
    
    /// <summary>
    /// Gets execution results by metascript ID
    /// </summary>
    /// <param name="metascriptId">The metascript ID</param>
    /// <returns>A result containing the execution results or an error</returns>
    Task<Result<List<MetascriptExecutionResult>, string>> GetExecutionResultsByMetascriptIdAsync(string metascriptId);
    
    /// <summary>
    /// Gets execution results by success
    /// </summary>
    /// <param name="isSuccess">Whether the execution was successful</param>
    /// <returns>A result containing the execution results or an error</returns>
    Task<Result<List<MetascriptExecutionResult>, string>> GetExecutionResultsBySuccessAsync(bool isSuccess);
    
    /// <summary>
    /// Gets execution results by time range
    /// </summary>
    /// <param name="startTimestamp">The start timestamp</param>
    /// <param name="endTimestamp">The end timestamp</param>
    /// <returns>A result containing the execution results or an error</returns>
    Task<Result<List<MetascriptExecutionResult>, string>> GetExecutionResultsByTimeRangeAsync(DateTime startTimestamp, DateTime endTimestamp);
    
    /// <summary>
    /// Cancels an execution
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>A result containing a boolean indicating whether the execution was cancelled or an error</returns>
    Task<Result<bool, string>> CancelExecutionAsync(string contextId);
    
    /// <summary>
    /// Pauses an execution
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>A result containing a boolean indicating whether the execution was paused or an error</returns>
    Task<Result<bool, string>> PauseExecutionAsync(string contextId);
    
    /// <summary>
    /// Resumes an execution
    /// </summary>
    /// <param name="contextId">The context ID</param>
    /// <returns>A result containing a boolean indicating whether the execution was resumed or an error</returns>
    Task<Result<bool, string>> ResumeExecutionAsync(string contextId);
    
    /// <summary>
    /// Validates a metascript
    /// </summary>
    /// <param name="metascript">The metascript</param>
    /// <returns>A result containing a boolean indicating whether the metascript is valid or an error</returns>
    Task<Result<bool, string>> ValidateMetascriptAsync(Metascript metascript);
    
    /// <summary>
    /// Validates metascript input
    /// </summary>
    /// <param name="metascript">The metascript</param>
    /// <param name="input">The input</param>
    /// <returns>A result containing a boolean indicating whether the input is valid or an error</returns>
    Task<Result<bool, string>> ValidateMetascriptInputAsync(Metascript metascript, Dictionary<string, object> input);
    
    /// <summary>
    /// Validates metascript output
    /// </summary>
    /// <param name="metascript">The metascript</param>
    /// <param name="output">The output</param>
    /// <returns>A result containing a boolean indicating whether the output is valid or an error</returns>
    Task<Result<bool, string>> ValidateMetascriptOutputAsync(Metascript metascript, Dictionary<string, object> output);
}
```

## Implementation Strategy
1. Create all enums first
2. Create all model classes next
3. Create the service interface next
4. Implement the MetascriptExecutionEngine class with AsyncMonad support
5. Add unit tests for each class
6. Document the implementation

## Testing Strategy
1. Create unit tests for each model class
2. Create unit tests for the MetascriptExecutionEngine class
3. Verify that all async methods have proper await operators
4. Verify that all null references are handled using monads
