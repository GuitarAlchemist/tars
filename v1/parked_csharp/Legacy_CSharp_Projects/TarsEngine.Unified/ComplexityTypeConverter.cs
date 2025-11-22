namespace TarsEngine.Unified;

/// <summary>
/// Provides conversion methods between different ComplexityType enums
/// </summary>
public static class ComplexityTypeConverter
{
    /// <summary>
    /// Converts from string to ComplexityType
    /// </summary>
    /// <param name="complexityTypeName">The name of the complexity type</param>
    /// <returns>The ComplexityType</returns>
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
    /// Converts from ComplexityType to string
    /// </summary>
    /// <param name="complexityType">The ComplexityType</param>
    /// <returns>The string representation</returns>
    public static string ToString(ComplexityType complexityType)
    {
        return complexityType.ToString();
    }
}