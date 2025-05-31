namespace TarsEngine.Models;

/// <summary>
/// Represents the type of a code issue
/// </summary>
public enum CodeIssueType
{
    /// <summary>
    /// Code smell
    /// </summary>
    CodeSmell,

    /// <summary>
    /// Bug
    /// </summary>
    Bug,

    /// <summary>
    /// Vulnerability
    /// </summary>
    Vulnerability,

    /// <summary>
    /// Security hotspot
    /// </summary>
    SecurityHotspot,

    /// <summary>
    /// Performance issue
    /// </summary>
    Performance,

    /// <summary>
    /// Maintainability issue
    /// </summary>
    Maintainability,

    /// <summary>
    /// Design issue
    /// </summary>
    Design,

    /// <summary>
    /// Documentation issue
    /// </summary>
    Documentation,

    /// <summary>
    /// Duplication
    /// </summary>
    Duplication,

    /// <summary>
    /// Complexity
    /// </summary>
    Complexity,

    /// <summary>
    /// Style
    /// </summary>
    Style,

    /// <summary>
    /// Naming
    /// </summary>
    Naming,

    /// <summary>
    /// Unused code
    /// </summary>
    UnusedCode,

    /// <summary>
    /// Dead code
    /// </summary>
    DeadCode,

    /// <summary>
    /// Security issue
    /// </summary>
    Security,

    /// <summary>
    /// Other issue
    /// </summary>
    Other,
    ComplexityIssue
}