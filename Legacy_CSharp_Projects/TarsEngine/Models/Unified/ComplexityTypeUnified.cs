namespace TarsEngine.Models.Unified;

/// <summary>
/// Represents a unified complexity type that combines all properties from different ComplexityType enums
/// </summary>
public enum ComplexityTypeUnified
{
    /// <summary>
    /// Cyclomatic complexity
    /// </summary>
    Cyclomatic,
        
    /// <summary>
    /// Cognitive complexity
    /// </summary>
    Cognitive,
        
    /// <summary>
    /// Halstead complexity
    /// </summary>
    Halstead,
        
    /// <summary>
    /// Maintainability index
    /// </summary>
    Maintainability,
        
    /// <summary>
    /// Maintainability index (alternative name)
    /// </summary>
    MaintainabilityIndex,
        
    /// <summary>
    /// Structural complexity
    /// </summary>
    Structural,
        
    /// <summary>
    /// Algorithmic complexity
    /// </summary>
    Algorithmic,
        
    /// <summary>
    /// Method length
    /// </summary>
    MethodLength,
        
    /// <summary>
    /// Class length
    /// </summary>
    ClassLength,
        
    /// <summary>
    /// Parameter count
    /// </summary>
    ParameterCount,
        
    /// <summary>
    /// Nesting depth
    /// </summary>
    NestingDepth,
        
    /// <summary>
    /// Other complexity
    /// </summary>
    Other
}