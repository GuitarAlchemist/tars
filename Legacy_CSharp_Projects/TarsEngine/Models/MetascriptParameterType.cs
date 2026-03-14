namespace TarsEngine.Models;

/// <summary>
/// Represents the type of a metascript parameter
/// </summary>
public enum MetascriptParameterType
{
    /// <summary>
    /// String parameter
    /// </summary>
    String,

    /// <summary>
    /// Integer parameter
    /// </summary>
    Integer,

    /// <summary>
    /// Float parameter
    /// </summary>
    Float,

    /// <summary>
    /// Boolean parameter
    /// </summary>
    Boolean,

    /// <summary>
    /// Enum parameter (one of a set of allowed values)
    /// </summary>
    Enum,

    /// <summary>
    /// Code parameter (multi-line string with code)
    /// </summary>
    Code,

    /// <summary>
    /// File path parameter
    /// </summary>
    FilePath,

    /// <summary>
    /// Directory path parameter
    /// </summary>
    DirectoryPath,

    /// <summary>
    /// List parameter
    /// </summary>
    List,

    /// <summary>
    /// Dictionary parameter
    /// </summary>
    Dictionary,

    /// <summary>
    /// JSON parameter
    /// </summary>
    Json,

    /// <summary>
    /// XML parameter
    /// </summary>
    Xml,

    /// <summary>
    /// Regular expression parameter
    /// </summary>
    Regex,

    /// <summary>
    /// Date parameter
    /// </summary>
    Date,

    /// <summary>
    /// Time parameter
    /// </summary>
    Time,

    /// <summary>
    /// DateTime parameter
    /// </summary>
    DateTime,

    /// <summary>
    /// TimeSpan parameter
    /// </summary>
    TimeSpan,

    /// <summary>
    /// Color parameter
    /// </summary>
    Color,

    /// <summary>
    /// URL parameter
    /// </summary>
    Url,

    /// <summary>
    /// Email parameter
    /// </summary>
    Email,

    /// <summary>
    /// Phone parameter
    /// </summary>
    Phone,

    /// <summary>
    /// Custom parameter type
    /// </summary>
    Custom
}
