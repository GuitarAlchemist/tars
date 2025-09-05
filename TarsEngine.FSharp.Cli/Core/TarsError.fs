namespace TarsEngine.FSharp.Cli.Core

open System

/// TARS CLI error types
type TarsError =
    | CommandNotFound of string
    | InvalidArguments of string
    | FileNotFound of string
    | AccessDenied of string
    | NetworkError of string
    | ConfigurationError of string
    | CompilationError of string
    | RuntimeError of string
    | AutonomousExecutionError of string
    | InstructionParsingError of string
    | UnknownError of string * Exception option

module TarsError =
    
    /// Convert error to string message
    let toString = function
        | CommandNotFound cmd -> $"Command not found: {cmd}"
        | InvalidArguments msg -> $"Invalid arguments: {msg}"
        | FileNotFound path -> $"File not found: {path}"
        | AccessDenied resource -> $"Access denied: {resource}"
        | NetworkError msg -> $"Network error: {msg}"
        | ConfigurationError msg -> $"Configuration error: {msg}"
        | CompilationError msg -> $"Compilation error: {msg}"
        | RuntimeError msg -> $"Runtime error: {msg}"
        | AutonomousExecutionError msg -> $"Autonomous execution error: {msg}"
        | InstructionParsingError msg -> $"Instruction parsing error: {msg}"
        | UnknownError (msg, Some ex) -> $"Unknown error: {msg} - {ex.Message}"
        | UnknownError (msg, None) -> $"Unknown error: {msg}"
    
    /// Convert error to exit code
    let toExitCode = function
        | CommandNotFound _ -> 127
        | InvalidArguments _ -> 2
        | FileNotFound _ -> 2
        | AccessDenied _ -> 13
        | NetworkError _ -> 3
        | ConfigurationError _ -> 78
        | CompilationError _ -> 1
        | RuntimeError _ -> 1
        | AutonomousExecutionError _ -> 1
        | InstructionParsingError _ -> 2
        | UnknownError _ -> 1
    
    /// Create error from exception
    let fromException (ex: Exception) =
        match ex with
        | :? System.IO.FileNotFoundException as fnf -> FileNotFound fnf.FileName
        | :? System.UnauthorizedAccessException as uae -> AccessDenied uae.Message
        | :? System.Net.NetworkInformation.NetworkInformationException as nie -> NetworkError nie.Message
        | :? System.ArgumentException as ae -> InvalidArguments ae.Message
        | _ -> UnknownError (ex.Message, Some ex)
    
    /// Create command result from error
    let toCommandResult error =
        CommandResult.failureWithCode (toExitCode error) (toString error)
