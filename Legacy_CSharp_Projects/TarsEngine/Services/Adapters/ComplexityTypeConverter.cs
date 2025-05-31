using InterfaceComplexityType = TarsEngine.Services.Interfaces.ComplexityType;
using ModelComplexityType = TarsEngine.Models.Metrics.ComplexityType;
using UnifiedComplexityType = TarsEngine.Models.Unified.ComplexityTypeUnified;

namespace TarsEngine.Services.Adapters;

/// <summary>
/// Converter between different ComplexityType enums
/// </summary>
public static class ComplexityTypeConverter
{
    /// <summary>
    /// Converts from Services.Interfaces.ComplexityType to Unified.ComplexityType
    /// </summary>
    public static UnifiedComplexityType ToUnifiedType(this InterfaceComplexityType complexityType)
    {
        return complexityType switch
        {
            InterfaceComplexityType.Cyclomatic => UnifiedComplexityType.Cyclomatic,
            InterfaceComplexityType.Cognitive => UnifiedComplexityType.Cognitive,
            InterfaceComplexityType.Halstead => UnifiedComplexityType.Halstead,
            InterfaceComplexityType.MaintainabilityIndex => UnifiedComplexityType.Maintainability,
            _ => UnifiedComplexityType.Cyclomatic
        };
    }

    /// <summary>
    /// Converts from Models.Metrics.ComplexityType to Unified.ComplexityType
    /// </summary>
    public static UnifiedComplexityType ToUnifiedType(this ModelComplexityType complexityType)
    {
        return complexityType switch
        {
            ModelComplexityType.Cyclomatic => UnifiedComplexityType.Cyclomatic,
            ModelComplexityType.Cognitive => UnifiedComplexityType.Cognitive,
            ModelComplexityType.Halstead => UnifiedComplexityType.Halstead,
            ModelComplexityType.MaintainabilityIndex => UnifiedComplexityType.Maintainability,
            _ => UnifiedComplexityType.Cyclomatic
        };
    }

    /// <summary>
    /// Converts from Unified.ComplexityType to Services.Interfaces.ComplexityType
    /// </summary>
    public static InterfaceComplexityType ToInterfaceType(this UnifiedComplexityType complexityType)
    {
        return complexityType switch
        {
            UnifiedComplexityType.Cyclomatic => InterfaceComplexityType.Cyclomatic,
            UnifiedComplexityType.Cognitive => InterfaceComplexityType.Cognitive,
            UnifiedComplexityType.Halstead => InterfaceComplexityType.Halstead,
            UnifiedComplexityType.Maintainability => InterfaceComplexityType.MaintainabilityIndex,
            _ => InterfaceComplexityType.Cyclomatic
        };
    }

    /// <summary>
    /// Converts from Unified.ComplexityType to Models.Metrics.ComplexityType
    /// </summary>
    public static ModelComplexityType ToModelType(this UnifiedComplexityType complexityType)
    {
        return complexityType switch
        {
            UnifiedComplexityType.Cyclomatic => ModelComplexityType.Cyclomatic,
            UnifiedComplexityType.Cognitive => ModelComplexityType.Cognitive,
            UnifiedComplexityType.Halstead => ModelComplexityType.Halstead,
            UnifiedComplexityType.Maintainability => ModelComplexityType.MaintainabilityIndex,
            _ => ModelComplexityType.Cyclomatic
        };
    }
}