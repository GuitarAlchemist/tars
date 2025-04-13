namespace TarsEngine.Services.Adapters;

/// <summary>
/// Adapter for converting between TarsEngine.Services.Interfaces.KnowledgeType and TarsEngine.Models.KnowledgeType
/// </summary>
public static class KnowledgeTypeAdapter
{
    /// <summary>
    /// Converts from TarsEngine.Services.Interfaces.KnowledgeType to TarsEngine.Models.KnowledgeType
    /// </summary>
    /// <param name="interfaceType">The interface type to convert</param>
    /// <returns>The model type</returns>
    public static TarsEngine.Models.KnowledgeType ToModelType(TarsEngine.Services.Interfaces.KnowledgeType interfaceType)
    {
        return interfaceType switch
        {
            TarsEngine.Services.Interfaces.KnowledgeType.Code => TarsEngine.Models.KnowledgeType.CodePattern,
            TarsEngine.Services.Interfaces.KnowledgeType.Concept => TarsEngine.Models.KnowledgeType.Concept,
            TarsEngine.Services.Interfaces.KnowledgeType.Procedure => TarsEngine.Models.KnowledgeType.Algorithm,
            TarsEngine.Services.Interfaces.KnowledgeType.DesignPattern => TarsEngine.Models.KnowledgeType.DesignPattern,
            TarsEngine.Services.Interfaces.KnowledgeType.BestPractice => TarsEngine.Models.KnowledgeType.BestPractice,
            _ => TarsEngine.Models.KnowledgeType.Unknown
        };
    }

    /// <summary>
    /// Converts from TarsEngine.Models.KnowledgeType to TarsEngine.Services.Interfaces.KnowledgeType
    /// </summary>
    /// <param name="modelType">The model type to convert</param>
    /// <returns>The interface type</returns>
    public static TarsEngine.Services.Interfaces.KnowledgeType ToInterfaceType(TarsEngine.Models.KnowledgeType modelType)
    {
        return modelType switch
        {
            TarsEngine.Models.KnowledgeType.CodePattern => TarsEngine.Services.Interfaces.KnowledgeType.Code,
            TarsEngine.Models.KnowledgeType.Concept => TarsEngine.Services.Interfaces.KnowledgeType.Concept,
            TarsEngine.Models.KnowledgeType.Algorithm => TarsEngine.Services.Interfaces.KnowledgeType.Procedure,
            TarsEngine.Models.KnowledgeType.DesignPattern => TarsEngine.Services.Interfaces.KnowledgeType.DesignPattern,
            TarsEngine.Models.KnowledgeType.BestPractice => TarsEngine.Services.Interfaces.KnowledgeType.BestPractice,
            _ => TarsEngine.Services.Interfaces.KnowledgeType.Fact
        };
    }
}