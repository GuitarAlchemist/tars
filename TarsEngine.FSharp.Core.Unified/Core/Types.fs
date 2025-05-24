namespace TarsEngine.FSharp.Core.Unified

open System

/// <summary>
/// Core types for the unified TARS F# engine.
/// </summary>
module Types =
    
    /// <summary>
    /// Represents a unique identifier.
    /// </summary>
    type Id = string
    
    /// <summary>
    /// Represents a timestamp.
    /// </summary>
    type Timestamp = DateTime
    
    /// <summary>
    /// Represents metadata as key-value pairs.
    /// </summary>
    type Metadata = Map<string, obj>
    
    /// <summary>
    /// Represents an error with a message and optional details.
    /// </summary>
    type TarsError = {
        Message: string
        Details: string option
        Timestamp: Timestamp
        ErrorCode: string option
        StackTrace: string option
    }
    
    /// <summary>
    /// Creates a new error.
    /// </summary>
    let createError message details =
        {
            Message = message
            Details = details
            Timestamp = DateTime.UtcNow
            ErrorCode = None
            StackTrace = None
        }
    
    /// <summary>
    /// Creates a new error with error code.
    /// </summary>
    let createErrorWithCode message details errorCode =
        {
            Message = message
            Details = details
            Timestamp = DateTime.UtcNow
            ErrorCode = Some errorCode
            StackTrace = None
        }
    
    /// <summary>
    /// Represents execution context for TARS operations.
    /// </summary>
    type ExecutionContext = {
        Id: Id
        StartTime: Timestamp
        WorkingDirectory: string
        Variables: Map<string, obj>
        Metadata: Metadata
        UserId: string option
        SessionId: string option
    }
    
    /// <summary>
    /// Creates a new execution context.
    /// </summary>
    let createExecutionContext workingDirectory =
        {
            Id = Guid.NewGuid().ToString()
            StartTime = DateTime.UtcNow
            WorkingDirectory = workingDirectory
            Variables = Map.empty
            Metadata = Map.empty
            UserId = None
            SessionId = None
        }
    
    /// <summary>
    /// Represents the status of an operation.
    /// </summary>
    type OperationStatus =
        | NotStarted
        | InProgress
        | Completed
        | Failed
        | Cancelled
    
    /// <summary>
    /// Represents the result of an operation.
    /// </summary>
    type OperationResult<'T> = {
        Status: OperationStatus
        Result: 'T option
        Error: TarsError option
        Duration: TimeSpan
        Metadata: Metadata
    }
    
    /// <summary>
    /// Creates a successful operation result.
    /// </summary>
    let createSuccessResult result duration =
        {
            Status = Completed
            Result = Some result
            Error = None
            Duration = duration
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Creates a failed operation result.
    /// </summary>
    let createFailureResult error duration =
        {
            Status = Failed
            Result = None
            Error = Some error
            Duration = duration
            Metadata = Map.empty
        }
