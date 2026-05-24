namespace TarsEngine.FSharp.Requirements.Models

/// <summary>
/// Enumeration of requirement types for TARS system
/// </summary>
type RequirementType =
    | Functional
    | NonFunctional
    | Performance
    | Security
    | Usability
    | Reliability
    | Scalability
    | Maintainability
    | Compatibility
    | Business

/// <summary>
/// Priority levels for requirements
/// </summary>
type RequirementPriority =
    | Critical = 1
    | High = 2
    | Medium = 3
    | Low = 4

/// <summary>
/// Status of requirement implementation
/// </summary>
type RequirementStatus =
    | Draft
    | Approved
    | InProgress
    | Implemented
    | Testing
    | Verified
    | Rejected
    | Obsolete

/// <summary>
/// Test execution status
/// </summary>
type TestStatus =
    | NotRun
    | Running
    | Passed
    | Failed
    | Skipped
    | Error

module RequirementTypeHelpers =
    
    /// <summary>
    /// Convert requirement type to string
    /// </summary>
    let toString (reqType: RequirementType) =
        match reqType with
        | Functional -> "Functional"
        | NonFunctional -> "Non-Functional"
        | Performance -> "Performance"
        | Security -> "Security"
        | Usability -> "Usability"
        | Reliability -> "Reliability"
        | Scalability -> "Scalability"
        | Maintainability -> "Maintainability"
        | Compatibility -> "Compatibility"
        | Business -> "Business"
    
    /// <summary>
    /// Parse requirement type from string
    /// </summary>
    let fromString (str: string) =
        match str.ToLowerInvariant() with
        | "functional" -> Some Functional
        | "non-functional" | "nonfunctional" -> Some NonFunctional
        | "performance" -> Some Performance
        | "security" -> Some Security
        | "usability" -> Some Usability
        | "reliability" -> Some Reliability
        | "scalability" -> Some Scalability
        | "maintainability" -> Some Maintainability
        | "compatibility" -> Some Compatibility
        | "business" -> Some Business
        | _ -> None
    
    /// <summary>
    /// Get all requirement types
    /// </summary>
    let getAllTypes() =
        [
            Functional
            NonFunctional
            Performance
            Security
            Usability
            Reliability
            Scalability
            Maintainability
            Compatibility
            Business
        ]
    
    /// <summary>
    /// Get priority description
    /// </summary>
    let getPriorityDescription (priority: RequirementPriority) =
        match priority with
        | RequirementPriority.Critical -> "Critical - Must be implemented immediately"
        | RequirementPriority.High -> "High - Should be implemented soon"
        | RequirementPriority.Medium -> "Medium - Normal priority"
        | RequirementPriority.Low -> "Low - Can be deferred"
        | _ -> "Unknown priority"
    
    /// <summary>
    /// Get status description
    /// </summary>
    let getStatusDescription (status: RequirementStatus) =
        match status with
        | Draft -> "Draft - Being written"
        | Approved -> "Approved - Ready for implementation"
        | InProgress -> "In Progress - Being implemented"
        | Implemented -> "Implemented - Code complete"
        | Testing -> "Testing - Under test"
        | Verified -> "Verified - Tested and working"
        | Rejected -> "Rejected - Will not be implemented"
        | Obsolete -> "Obsolete - No longer relevant"
