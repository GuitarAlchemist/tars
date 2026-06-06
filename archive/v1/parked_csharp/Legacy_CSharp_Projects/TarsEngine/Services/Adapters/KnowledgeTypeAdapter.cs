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
    public static TarsEngine.Models.KnowledgeType ToModelType(Interfaces.KnowledgeType interfaceType)
    {
        return interfaceType switch
        {
            Interfaces.KnowledgeType.Code => TarsEngine.Models.KnowledgeType.CodePattern,
            Interfaces.KnowledgeType.Concept => TarsEngine.Models.KnowledgeType.Concept,
            Interfaces.KnowledgeType.Procedure => TarsEngine.Models.KnowledgeType.Algorithm,
            Interfaces.KnowledgeType.DesignPattern => TarsEngine.Models.KnowledgeType.DesignPattern,
            Interfaces.KnowledgeType.BestPractice => TarsEngine.Models.KnowledgeType.BestPractice,
            _ => TarsEngine.Models.KnowledgeType.Unknown
        };
    }

    /// <summary>
    /// Converts from TarsEngine.Models.KnowledgeType to TarsEngine.Services.Interfaces.KnowledgeType
    /// </summary>
    /// <param name="modelType">The model type to convert</param>
    /// <returns>The interface type</returns>
    public static Interfaces.KnowledgeType ToInterfaceType(TarsEngine.Models.KnowledgeType modelType)
    {
        return modelType switch
        {
            TarsEngine.Models.KnowledgeType.CodePattern => Interfaces.KnowledgeType.Code,
            TarsEngine.Models.KnowledgeType.Concept => Interfaces.KnowledgeType.Concept,
            TarsEngine.Models.KnowledgeType.Algorithm => Interfaces.KnowledgeType.Procedure,
            TarsEngine.Models.KnowledgeType.DesignPattern => Interfaces.KnowledgeType.DesignPattern,
            TarsEngine.Models.KnowledgeType.BestPractice => Interfaces.KnowledgeType.BestPractice,
            _ => Interfaces.KnowledgeType.Fact
        };
    }
}