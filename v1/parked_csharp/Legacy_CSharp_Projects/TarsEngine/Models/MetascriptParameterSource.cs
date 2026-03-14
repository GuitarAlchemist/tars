namespace TarsEngine.Models;

/// <summary>
/// Represents the source of a metascript parameter value
/// </summary>
public enum MetascriptParameterSource
{
    /// <summary>
    /// Parameter value is provided manually
    /// </summary>
    Manual,

    /// <summary>
    /// Parameter value is extracted from a pattern match
    /// </summary>
    PatternMatch,

    /// <summary>
    /// Parameter value is extracted from code analysis
    /// </summary>
    CodeAnalysis,

    /// <summary>
    /// Parameter value is calculated from other parameters
    /// </summary>
    Calculated,

    /// <summary>
    /// Parameter value is provided by the system
    /// </summary>
    System,

    /// <summary>
    /// Parameter value is provided by the user
    /// </summary>
    User,

    /// <summary>
    /// Parameter value is provided by a configuration file
    /// </summary>
    Configuration,

    /// <summary>
    /// Parameter value is provided by an environment variable
    /// </summary>
    Environment,

    /// <summary>
    /// Parameter value is provided by a database
    /// </summary>
    Database,

    /// <summary>
    /// Parameter value is provided by an API
    /// </summary>
    Api,

    /// <summary>
    /// Parameter value is provided by a file
    /// </summary>
    File,

    /// <summary>
    /// Parameter value is provided by a custom source
    /// </summary>
    Custom
}
