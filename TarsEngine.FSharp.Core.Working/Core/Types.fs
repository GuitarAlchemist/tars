namespace TarsEngine.FSharp.Core.Working

open System

/// <summary>
/// Core types for the TARS engine.
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
    }
    
    /// <summary>
    /// Creates a new error.
    /// </summary>
    let createError message details =
        {
            Message = message
            Details = details
            Timestamp = DateTime.UtcNow
        }
