namespace TarsEngine.Models;

/// <summary>
/// Represents the type of dependency between improvements
/// </summary>
public enum ImprovementDependencyType
{
    /// <summary>
    /// Source requires target
    /// </summary>
    Requires,

    /// <summary>
    /// Source is related to target
    /// </summary>
    RelatedTo,

    /// <summary>
    /// Source is similar to target
    /// </summary>
    SimilarTo,

    /// <summary>
    /// Source conflicts with target
    /// </summary>
    ConflictsWith,

    /// <summary>
    /// Source enhances target
    /// </summary>
    Enhances,

    /// <summary>
    /// Source is enhanced by target
    /// </summary>
    EnhancedBy,

    /// <summary>
    /// Source is alternative to target
    /// </summary>
    AlternativeTo,

    /// <summary>
    /// Source is part of target
    /// </summary>
    PartOf,

    /// <summary>
    /// Source contains target
    /// </summary>
    Contains,

    /// <summary>
    /// Source is before target
    /// </summary>
    Before,

    /// <summary>
    /// Source is after target
    /// </summary>
    After,

    /// <summary>
    /// Source is blocked by target
    /// </summary>
    BlockedBy,

    /// <summary>
    /// Source blocks target
    /// </summary>
    Blocks,

    /// <summary>
    /// Source is derived from target
    /// </summary>
    DerivedFrom,

    /// <summary>
    /// Source is source of target
    /// </summary>
    SourceOf,

    /// <summary>
    /// Other dependency type
    /// </summary>
    Other
}
