namespace TarsEngine.Unified;

/// <summary>
/// Adapter for converting between different complexity type enums
/// </summary>
public static class ComplexityTypeAdapter
{
    /// <summary>
    /// Converts a string representation of a complexity type to the unified ComplexityType
    /// </summary>
    /// <param name="complexityTypeName">The name of the complexity type</param>
    /// <returns>The unified ComplexityType</returns>
    public static ComplexityType FromString(string complexityTypeName)
    {
        return complexityTypeName.ToLowerInvariant() switch
        {
            "cyclomatic" => ComplexityType.Cyclomatic,
            "cognitive" => ComplexityType.Cognitive,
            "halstead" => ComplexityType.Halstead,
            "maintainability" => ComplexityType.Maintainability,
            "maintainabilityindex" => ComplexityType.MaintainabilityIndex,
            "methodlength" => ComplexityType.MethodLength,
            "classlength" => ComplexityType.ClassLength,
            "parametercount" => ComplexityType.ParameterCount,
            "nestingdepth" => ComplexityType.NestingDepth,
            "structural" => ComplexityType.Structural,
            "algorithmic" => ComplexityType.Algorithmic,
            _ => ComplexityType.Other
        };
    }

    /// <summary>
    /// Converts the unified ComplexityType to a string representation
    /// </summary>
    /// <param name="complexityType">The unified ComplexityType</param>
    /// <returns>The string representation</returns>
    public static string ToString(ComplexityType complexityType)
    {
        return complexityType.ToString();
    }
}