namespace TarsEngine.Models;

/// <summary>
/// Represents the type of a validation rule
/// </summary>
public enum ValidationRuleType
{
    /// <summary>
    /// Syntax validation rule
    /// </summary>
    Syntax,

    /// <summary>
    /// Semantic validation rule
    /// </summary>
    Semantic,

    /// <summary>
    /// Compilation validation rule
    /// </summary>
    Compilation,

    /// <summary>
    /// Test validation rule
    /// </summary>
    Test,

    /// <summary>
    /// Performance validation rule
    /// </summary>
    Performance,

    /// <summary>
    /// Security validation rule
    /// </summary>
    Security,

    /// <summary>
    /// Style validation rule
    /// </summary>
    Style,

    /// <summary>
    /// Documentation validation rule
    /// </summary>
    Documentation,

    /// <summary>
    /// Dependency validation rule
    /// </summary>
    Dependency,

    /// <summary>
    /// Compatibility validation rule
    /// </summary>
    Compatibility,

    /// <summary>
    /// Other validation rule type
    /// </summary>
    Other
}
