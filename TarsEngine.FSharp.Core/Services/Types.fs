namespace TarsEngine.FSharp.Core.Services

open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks

/// <summary>
/// Represents an improvement result.
/// </summary>
type ImprovementResult = {
    /// <summary>
    /// The capability that was improved.
    /// </summary>
    Capability: string
    
    /// <summary>
    /// The confidence level of the improvement.
    /// </summary>
    Confidence: float
    
    /// <summary>
    /// The source of the improvement.
    /// </summary>
    Source: string
}

/// <summary>
/// Represents a metascript execution result.
/// </summary>
type MetascriptExecutionResult<'T> = {
    /// <summary>
    /// Whether the execution was successful.
    /// </summary>
    Success: bool
    
    /// <summary>
    /// The result of the execution.
    /// </summary>
    Result: 'T option
    
    /// <summary>
    /// The error message, if any.
    /// </summary>
    ErrorMessage: string option
    
    /// <summary>
    /// The execution time in milliseconds.
    /// </summary>
    ExecutionTime: int64
    
    /// <summary>
    /// The logs from the execution.
    /// </summary>
    Logs: string list
    
    /// <summary>
    /// Additional information about the execution.
    /// </summary>
    AdditionalInfo: Map<string, obj>
}

/// <summary>
/// Represents a metascript context.
/// </summary>
type MetascriptContext = {
    /// <summary>
    /// The variables available in the metascript.
    /// </summary>
    Variables: Map<string, obj>
    
    /// <summary>
    /// The functions available in the metascript.
    /// </summary>
    Functions: Map<string, obj -> obj>
    
    /// <summary>
    /// The services available in the metascript.
    /// </summary>
    Services: Map<string, obj>
    
    /// <summary>
    /// The configuration for the metascript.
    /// </summary>
    Configuration: Map<string, string>
    
    /// <summary>
    /// Additional information for the metascript.
    /// </summary>
    AdditionalInfo: Map<string, obj>
}

/// <summary>
/// Represents a generated metascript.
/// </summary>
type GeneratedMetascript = {
    /// <summary>
    /// The name of the metascript.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the metascript.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The content of the metascript.
    /// </summary>
    Content: string
    
    /// <summary>
    /// The language of the metascript.
    /// </summary>
    Language: string
    
    /// <summary>
    /// The category of the metascript.
    /// </summary>
    Category: string
    
    /// <summary>
    /// The tags of the metascript.
    /// </summary>
    Tags: string list
    
    /// <summary>
    /// The author of the metascript.
    /// </summary>
    Author: string
    
    /// <summary>
    /// The creation date of the metascript.
    /// </summary>
    CreationDate: DateTime
    
    /// <summary>
    /// The last modified date of the metascript.
    /// </summary>
    LastModifiedDate: DateTime
    
    /// <summary>
    /// The version of the metascript.
    /// </summary>
    Version: string
    
    /// <summary>
    /// Additional information about the metascript.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a metascript template.
/// </summary>
type MetascriptTemplate = {
    /// <summary>
    /// The name of the template.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the template.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The content of the template.
    /// </summary>
    Content: string
    
    /// <summary>
    /// The language of the template.
    /// </summary>
    Language: string
    
    /// <summary>
    /// The category of the template.
    /// </summary>
    Category: string
    
    /// <summary>
    /// The tags of the template.
    /// </summary>
    Tags: string list
    
    /// <summary>
    /// The placeholders in the template.
    /// </summary>
    Placeholders: Map<string, string>
    
    /// <summary>
    /// Additional information about the template.
    /// </summary>
    AdditionalInfo: Map<string, string>
}
