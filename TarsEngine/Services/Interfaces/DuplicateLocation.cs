namespace TarsEngine.Services.Interfaces;

/// <summary>
/// Represents a duplicate location in code
/// </summary>
public class DuplicateLocation
{
    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the start line
    /// </summary>
    public int StartLine { get; set; }

    /// <summary>
    /// Gets or sets the end line
    /// </summary>
    public int EndLine { get; set; }
}