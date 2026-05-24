namespace TarsEngine.Utilities;

/// <summary>
/// Provides conversion methods between similar types in different namespaces
/// </summary>
public static class TypeConverters
{
    /// <summary>
    /// Converts from Models.ProgrammingLanguage to Services.ProgrammingLanguage
    /// </summary>
    public static Services.ProgrammingLanguage ToServicesProgrammingLanguage(Services.Models.ProgrammingLanguage language)
    {
        return (Services.ProgrammingLanguage)((int)language);
    }
        
    /// <summary>
    /// Converts from Services.ProgrammingLanguage to Models.ProgrammingLanguage
    /// </summary>
    public static Services.Models.ProgrammingLanguage ToModelsProgrammingLanguage(Services.ProgrammingLanguage language)
    {
        return (Services.Models.ProgrammingLanguage)((int)language);
    }
        
    /// <summary>
    /// Converts from Services.Interfaces.ComplexityType to Models.Metrics.ComplexityType
    /// </summary>
    public static Models.Metrics.ComplexityType ToMetricsComplexityType(Services.Interfaces.ComplexityType complexityType)
    {
        return (Models.Metrics.ComplexityType)((int)complexityType);
    }
        
    /// <summary>
    /// Converts from Models.Metrics.ComplexityType to Services.Interfaces.ComplexityType
    /// </summary>
    public static Services.Interfaces.ComplexityType ToInterfacesComplexityType(Models.Metrics.ComplexityType complexityType)
    {
        return (Services.Interfaces.ComplexityType)((int)complexityType);
    }
        
    /// <summary>
    /// Converts from Services.Interfaces.IssueSeverity to Models.IssueSeverity
    /// </summary>
    public static Models.IssueSeverity ToModelsIssueSeverity(Services.Interfaces.IssueSeverity severity)
    {
        return (Models.IssueSeverity)((int)severity);
    }
        
    /// <summary>
    /// Converts from Models.IssueSeverity to Services.Interfaces.IssueSeverity
    /// </summary>
    public static Services.Interfaces.IssueSeverity ToInterfacesIssueSeverity(Models.IssueSeverity severity)
    {
        return (Services.Interfaces.IssueSeverity)((int)severity);
    }
}