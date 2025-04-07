namespace TarsEngine.Models;

/// <summary>
/// Represents a virtual file
/// </summary>
public class VirtualFile
{
    /// <summary>
    /// Gets or sets the file path
    /// </summary>
    public string Path { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the file content
    /// </summary>
    public string Content { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the original file content
    /// </summary>
    public string OriginalContent { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets whether the file is modified
    /// </summary>
    public bool IsModified { get; set; }

    /// <summary>
    /// Gets or sets whether the file is created
    /// </summary>
    public bool IsCreated { get; set; }

    /// <summary>
    /// Gets or sets whether the file is deleted
    /// </summary>
    public bool IsDeleted { get; set; }
}
