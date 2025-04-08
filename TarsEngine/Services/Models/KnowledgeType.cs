namespace TarsEngine.Services.Models;

/// <summary>
/// Represents the type of knowledge item
/// </summary>
public enum KnowledgeType
{
    /// <summary>
    /// Unknown knowledge type
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// Concept knowledge type
    /// </summary>
    Concept = 1,

    /// <summary>
    /// Code pattern knowledge type
    /// </summary>
    CodePattern = 2,

    /// <summary>
    /// Algorithm knowledge type
    /// </summary>
    Algorithm = 3,

    /// <summary>
    /// Insight knowledge type
    /// </summary>
    Insight = 4,

    /// <summary>
    /// Question knowledge type
    /// </summary>
    Question = 5,

    /// <summary>
    /// Answer knowledge type
    /// </summary>
    Answer = 6,

    /// <summary>
    /// Documentation knowledge type
    /// </summary>
    Documentation = 7,

    /// <summary>
    /// Example knowledge type
    /// </summary>
    Example = 8,

    /// <summary>
    /// Best practice knowledge type
    /// </summary>
    BestPractice = 9,

    /// <summary>
    /// Anti-pattern knowledge type
    /// </summary>
    AntiPattern = 10
}
