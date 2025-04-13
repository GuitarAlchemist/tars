namespace TarsEngine.Models.Adapters;

/// <summary>
/// Adapter class to resolve ambiguity between TarsEngine.Services.Interfaces.KnowledgeType and TarsEngine.Models.KnowledgeType
/// </summary>
public class KnowledgeTypeAdapter
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;

    // Convert from service interface to model
    public static KnowledgeType ToModel(Services.Interfaces.KnowledgeType type)
    {
        // Map from service interface enum to model enum
        return MapServiceToModel(type);
    }

    // Convert from model to service interface
    public static Services.Interfaces.KnowledgeType ToService(KnowledgeType type)
    {
        // Map from model enum to service interface enum
        return MapModelToService(type);
    }

    private static KnowledgeType MapServiceToModel(Services.Interfaces.KnowledgeType type)
    {
        return type switch
        {
            Services.Interfaces.KnowledgeType.Fact => KnowledgeType.Concept,
            Services.Interfaces.KnowledgeType.Concept => KnowledgeType.Concept,
            Services.Interfaces.KnowledgeType.Procedure => KnowledgeType.Algorithm,
            Services.Interfaces.KnowledgeType.Code => KnowledgeType.CodePattern,
            Services.Interfaces.KnowledgeType.Architecture => KnowledgeType.DesignPattern,
            Services.Interfaces.KnowledgeType.DesignPattern => KnowledgeType.DesignPattern,
            Services.Interfaces.KnowledgeType.BestPractice => KnowledgeType.BestPractice,
            Services.Interfaces.KnowledgeType.UserPreference => KnowledgeType.Resource,
            Services.Interfaces.KnowledgeType.ProjectSpecific => KnowledgeType.Resource,
            _ => KnowledgeType.Unknown
        };
    }

    private static Services.Interfaces.KnowledgeType MapModelToService(KnowledgeType type)
    {
        return type switch
        {
            KnowledgeType.Concept => Services.Interfaces.KnowledgeType.Concept,
            KnowledgeType.CodePattern => Services.Interfaces.KnowledgeType.Code,
            KnowledgeType.Algorithm => Services.Interfaces.KnowledgeType.Procedure,
            KnowledgeType.DesignPattern => Services.Interfaces.KnowledgeType.DesignPattern,
            KnowledgeType.BestPractice => Services.Interfaces.KnowledgeType.BestPractice,
            KnowledgeType.ApiUsage => Services.Interfaces.KnowledgeType.Code,
            KnowledgeType.ErrorPattern => Services.Interfaces.KnowledgeType.Code,
            KnowledgeType.Performance => Services.Interfaces.KnowledgeType.BestPractice,
            KnowledgeType.Security => Services.Interfaces.KnowledgeType.BestPractice,
            KnowledgeType.Testing => Services.Interfaces.KnowledgeType.Procedure,
            KnowledgeType.Insight => Services.Interfaces.KnowledgeType.Fact,
            KnowledgeType.Question => Services.Interfaces.KnowledgeType.Fact,
            KnowledgeType.Answer => Services.Interfaces.KnowledgeType.Fact,
            KnowledgeType.Tool => Services.Interfaces.KnowledgeType.ProjectSpecific,
            KnowledgeType.Resource => Services.Interfaces.KnowledgeType.ProjectSpecific,
            _ => Services.Interfaces.KnowledgeType.Fact
        };
    }
}