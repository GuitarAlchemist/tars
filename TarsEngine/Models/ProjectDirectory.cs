namespace TarsEngine.Models;

/// <summary>
/// Represents a directory in a project
/// </summary>
public class ProjectDirectory
{
    /// <summary>
    /// The name of the directory
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// The path to the directory
    /// </summary>
    public string Path { get; set; } = string.Empty;
    
    /// <summary>
    /// The list of files in the directory
    /// </summary>
    public List<ProjectFile> Files { get; set; } = new List<ProjectFile>();
    
    /// <summary>
    /// The list of subdirectories in the directory
    /// </summary>
    public List<ProjectDirectory> Subdirectories { get; set; } = new List<ProjectDirectory>();
}
