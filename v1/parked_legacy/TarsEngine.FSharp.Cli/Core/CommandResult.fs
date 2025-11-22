namespace TarsEngine.FSharp.Cli.Core

open System

/// Result of executing a TARS CLI command
type CommandResult = {
    /// Exit code (0 for success, non-zero for failure)
    ExitCode: int
    /// Output message
    Message: string
    /// Additional data returned by the command
    Data: obj option
    /// Execution time
    ExecutionTime: TimeSpan option
    /// Any errors that occurred
    Errors: string list
    /// Warnings generated during execution
    Warnings: string list
}

module CommandResult =
    
    /// Create a successful command result
    let success message = {
        ExitCode = 0
        Message = message
        Data = None
        ExecutionTime = None
        Errors = []
        Warnings = []
    }
    
    /// Create a successful command result with data
    let successWithData message data = {
        ExitCode = 0
        Message = message
        Data = Some data
        ExecutionTime = None
        Errors = []
        Warnings = []
    }
    
    /// Create a failed command result
    let failure message = {
        ExitCode = 1
        Message = message
        Data = None
        ExecutionTime = None
        Errors = [message]
        Warnings = []
    }
    
    /// Create a failed command result with specific exit code
    let failureWithCode exitCode message = {
        ExitCode = exitCode
        Message = message
        Data = None
        ExecutionTime = None
        Errors = [message]
        Warnings = []
    }
    
    /// Add execution time to result
    let withExecutionTime time result = 
        { result with ExecutionTime = Some time }
    
    /// Add data to result
    let withData data result = 
        { result with Data = Some data }
    
    /// Add warning to result
    let addWarning warning result = 
        { result with Warnings = warning :: result.Warnings }
    
    /// Add error to result
    let addError error result = 
        { result with Errors = error :: result.Errors; ExitCode = if result.ExitCode = 0 then 1 else result.ExitCode }
    
    /// Check if result is successful
    let isSuccess result = result.ExitCode = 0
    
    /// Check if result is failure
    let isFailure result = result.ExitCode <> 0
