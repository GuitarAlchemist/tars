namespace TarsEngine.Models;

/// <summary>
/// Represents a file in a project
/// </summary>
public class ProjectFile
{
    /// <summary>
    /// The name of the file
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// The path to the file
    /// </summary>
    public string Path { get; set; } = string.Empty;
    
    /// <summary>
    /// The type of the file
    /// </summary>
    public string FileType { get; set; } = string.Empty;
    
    /// <summary>
    /// The size of the file in bytes
    /// </summary>
    public long Size { get; set; }
}
