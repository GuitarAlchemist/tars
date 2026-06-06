namespace TarsEngine.DSL

/// <summary>
/// Represents the severity of a diagnostic message.
/// </summary>
type DiagnosticSeverity =
    /// <summary>
    /// Reports an error, which prevents successful compilation or execution.
    /// </summary>
    | Error = 1
    
    /// <summary>
    /// Reports a warning, which indicates a potential problem but doesn't prevent compilation or execution.
    /// </summary>
    | Warning = 2
    
    /// <summary>
    /// Reports an informational message, which provides additional context but doesn't indicate a problem.
    /// </summary>
    | Information = 3
    
    /// <summary>
    /// Reports a hint, which suggests a potential improvement but doesn't indicate a problem.
    /// </summary>
    | Hint = 4
