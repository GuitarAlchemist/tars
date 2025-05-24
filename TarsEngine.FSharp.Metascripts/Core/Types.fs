namespace TarsEngine.FSharp.Metascripts.Core

open System

/// <summary>
/// Core types for metascripts - self-contained engine.
/// </summary>
type Id = string
type Timestamp = DateTime
type Metadata = Map<string, obj>

/// <summary>
/// Represents a metascript category.
/// </summary>
type MetascriptCategory =
    | Core
    | Intelligence
    | ML
    | CodeAnalysis
    | Automation
    | Custom

/// <summary>
/// Represents metascript metadata.
/// </summary>
type MetascriptMetadata = {
    Id: Id
    Name: string
    Description: string
    Author: string
    Version: string
    Category: MetascriptCategory
    Tags: string list
    CreatedAt: Timestamp
    UpdatedAt: Timestamp
    Dependencies: string list
    Parameters: Map<string, string>
}

/// <summary>
/// Represents a metascript source.
/// </summary>
type MetascriptSource = {
    Id: Id
    Name: string
    Content: string
    FilePath: string
    Metadata: MetascriptMetadata
    IsValid: bool
    ValidationErrors: string list
}

/// <summary>
/// Represents metascript execution status.
/// </summary>
type MetascriptExecutionStatus =
    | NotStarted
    | Running
    | Completed
    | Failed
    | Cancelled

/// <summary>
/// Represents metascript execution result.
/// </summary>
type MetascriptExecutionResult = {
    Id: Id
    MetascriptId: Id
    Status: MetascriptExecutionStatus
    Output: string
    Error: string option
    Variables: Map<string, obj>
    ExecutionTime: TimeSpan
    StartTime: Timestamp
    EndTime: Timestamp option
}

/// <summary>
/// Represents metascript statistics.
/// </summary>
type MetascriptStats = {
    TotalMetascripts: int
    ExecutedToday: int
    SuccessRate: float
    AverageExecutionTime: TimeSpan
    MostUsedCategory: MetascriptCategory
    LastExecuted: Timestamp option
}

/// <summary>
/// Represents a registered metascript.
/// </summary>
type RegisteredMetascript = {
    Source: MetascriptSource
    RegistrationTime: Timestamp
    UsageCount: int
    LastUsed: Timestamp option
    IsActive: bool
}

/// <summary>
/// Helper functions for metascript types.
/// </summary>
module MetascriptHelpers =
    /// <summary>
    /// Creates a new metascript metadata.
    /// </summary>
    let createMetadata name description author =
        {
            Id = Guid.NewGuid().ToString()
            Name = name
            Description = description
            Author = author
            Version = "1.0.0"
            Category = Custom
            Tags = []
            CreatedAt = DateTime.UtcNow
            UpdatedAt = DateTime.UtcNow
            Dependencies = []
            Parameters = Map.empty
        }

    /// <summary>
    /// Creates a new metascript source.
    /// </summary>
    let createMetascriptSource name content filePath metadata =
        {
            Id = Guid.NewGuid().ToString()
            Name = name
            Content = content
            FilePath = filePath
            Metadata = metadata
            IsValid = true
            ValidationErrors = []
        }

    /// <summary>
    /// Creates a new execution result.
    /// </summary>
    let createExecutionResult metascriptId status output =
        {
            Id = Guid.NewGuid().ToString()
            MetascriptId = metascriptId
            Status = status
            Output = output
            Error = None
            Variables = Map.empty
            ExecutionTime = TimeSpan.Zero
            StartTime = DateTime.UtcNow
            EndTime = None
        }
