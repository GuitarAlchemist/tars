namespace TarsEngine.Models;

/// <summary>
/// Represents a code structure identified during analysis
/// </summary>
public class CodeStructure
{
    /// <summary>
    /// Gets or sets the type of the structure
    /// </summary>
    public StructureType Type { get; set; }

    /// <summary>
    /// Gets or sets the name of the structure
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the location of the structure in the code
    /// </summary>
    public CodeLocation Location { get; set; } = new();

    /// <summary>
    /// Gets or sets the parent structure, if any
    /// </summary>
    public string? ParentName { get; set; }

    /// <summary>
    /// Gets or sets the list of child structure names
    /// </summary>
    public List<string> ChildNames { get; set; } = new();

    /// <summary>
    /// Gets or sets the list of dependencies for this structure
    /// </summary>
    public List<string> Dependencies { get; set; } = new();

    /// <summary>
    /// Gets or sets the complexity score of the structure
    /// </summary>
    public double ComplexityScore { get; set; }

    /// <summary>
    /// Gets or sets the size of the structure (lines of code)
    /// </summary>
    public int Size { get; set; }

    /// <summary>
    /// Gets or sets additional properties of the structure
    /// </summary>
    public Dictionary<string, string> Properties { get; set; } = new();
}