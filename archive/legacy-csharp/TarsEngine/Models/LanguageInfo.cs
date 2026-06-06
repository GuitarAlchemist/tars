namespace TarsEngine.Models;

/// <summary>
/// Represents information about a language used in a project
/// </summary>
public class LanguageInfo
{
    /// <summary>
    /// Gets or sets the language name
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the line count
    /// </summary>
    public int LineCount { get; set; }
}
