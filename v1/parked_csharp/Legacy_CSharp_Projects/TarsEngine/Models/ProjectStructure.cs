namespace TarsEngine.Models;

/// <summary>
/// Represents the structure of a project
/// </summary>
public class ProjectStructure
{
    /// <summary>
    /// The name of the project
    /// </summary>
    public string Name { get; set; } = string.Empty;
    
    /// <summary>
    /// The path to the project
    /// </summary>
    public string Path { get; set; } = string.Empty;
    
    /// <summary>
    /// The list of files in the project
    /// </summary>
    public List<ProjectFile> Files { get; set; } = new();
    
    /// <summary>
    /// The list of directories in the project
    /// </summary>
    public List<ProjectDirectory> Directories { get; set; } = new();
    
    /// <summary>
    /// Convert the project structure to a string
    /// </summary>
    /// <returns>A string representation of the project structure</returns>
    public override string ToString()
    {
        var result = new System.Text.StringBuilder();
        
        result.AppendLine($"Project: {Name}");
        result.AppendLine($"Path: {Path}");
        result.AppendLine();
        
        result.AppendLine("Directories:");
        foreach (var directory in Directories)
        {
            result.AppendLine($"- {directory.Name} ({directory.Path})");
        }
        result.AppendLine();
        
        result.AppendLine("Files:");
        foreach (var file in Files)
        {
            result.AppendLine($"- {file.Name} ({file.Path})");
        }
        
        return result.ToString();
    }
}
