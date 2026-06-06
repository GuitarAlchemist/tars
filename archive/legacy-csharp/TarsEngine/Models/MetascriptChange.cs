namespace TarsEngine.Models;

/// <summary>
/// Represents a change made by a metascript
/// </summary>
public class MetascriptChange
{
    /// <summary>
    /// Gets or sets the type of change
    /// </summary>
    public MetascriptChangeType Type { get; set; } = MetascriptChangeType.Modification;

    /// <summary>
    /// Gets or sets the path of the affected file
    /// </summary>
    public string FilePath { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the original content
    /// </summary>
    public string? OriginalContent { get; set; }

    /// <summary>
    /// Gets or sets the new content
    /// </summary>
    public string? NewContent { get; set; }

    /// <summary>
    /// Gets or sets the start line of the change
    /// </summary>
    public int StartLine { get; set; }

    /// <summary>
    /// Gets or sets the end line of the change
    /// </summary>
    public int EndLine { get; set; }

    /// <summary>
    /// Gets or sets the description of the change
    /// </summary>
    public string? Description { get; set; }
}
