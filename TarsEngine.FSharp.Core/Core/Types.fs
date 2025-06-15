namespace TarsEngine.FSharp.Core

open System

/// Core types for TARS Engine - Unified and Working
module Types =
    
    /// Execution status for operations
    type ExecutionStatus =
        | Success
        | Failed
        | InProgress
        | Cancelled
    
    /// Execution result with output and metadata
    type ExecutionResult = {
        Status: ExecutionStatus
        Output: string
        Error: string option
        Variables: Map<string, obj>
        ExecutionTime: TimeSpan
    }
    
    /// Agent configuration
    type AgentConfig = {
        Type: string
        Parameters: Map<string, obj>
        ResourceLimits: obj option
    }
    
    /// Simple service interface
    type IService<'T> =
        abstract member ExecuteAsync: 'T -> Async<ExecutionResult>
    
    /// TARS API interface
    type ITarsApi =
        abstract member SearchVector: string * int -> Async<{| Id: int; Content: string; Score: float |} list>
        abstract member AskLlm: string * string -> Async<string>
        abstract member SpawnAgent: string * AgentConfig -> string
        abstract member WriteFile: string * string -> bool
