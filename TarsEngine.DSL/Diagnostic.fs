namespace TarsEngine.DSL

/// <summary>
/// Represents a diagnostic message, which can be an error, warning, information, or hint.
/// </summary>
type Diagnostic = {
    /// <summary>
    /// The severity of the diagnostic message.
    /// </summary>
    Severity: DiagnosticSeverity
    
    /// <summary>
    /// The warning code of the diagnostic message.
    /// </summary>
    Code: WarningCode
    
    /// <summary>
    /// The message of the diagnostic message.
    /// </summary>
    Message: string
    
    /// <summary>
    /// The line number where the diagnostic message occurred.
    /// </summary>
    Line: int
    
    /// <summary>
    /// The column number where the diagnostic message occurred.
    /// </summary>
    Column: int
    
    /// <summary>
    /// The line content where the diagnostic message occurred.
    /// </summary>
    LineContent: string
    
    /// <summary>
    /// Suggestions for fixing the issue.
    /// </summary>
    Suggestions: string list
}
