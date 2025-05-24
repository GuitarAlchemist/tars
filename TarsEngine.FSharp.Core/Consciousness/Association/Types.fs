namespace TarsEngine.FSharp.Core.Consciousness.Association

open System
open TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Represents the type of an association.
/// </summary>
type AssociationType =
    | Semantic
    | Causal
    | Temporal
    | Spatial
    | Emotional
    | Conceptual
    | Hierarchical
    | Similarity
    | Contrast
    | Custom of string

/// <summary>
/// Represents the strength of an association.
/// </summary>
type AssociationStrength =
    | Weak
    | Moderate
    | Strong
    | VeryStrong
    | Custom of float

/// <summary>
/// Represents an association between two concepts.
/// </summary>
type Association = {
    /// <summary>
    /// The ID of the association.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The source concept of the association.
    /// </summary>
    Source: string
    
    /// <summary>
    /// The target concept of the association.
    /// </summary>
    Target: string
    
    /// <summary>
    /// The type of the association.
    /// </summary>
    Type: AssociationType
    
    /// <summary>
    /// The strength of the association.
    /// </summary>
    Strength: AssociationStrength
    
    /// <summary>
    /// The description of the association.
    /// </summary>
    Description: string option
    
    /// <summary>
    /// The creation time of the association.
    /// </summary>
    CreationTime: DateTime
    
    /// <summary>
    /// The last activation time of the association.
    /// </summary>
    LastActivationTime: DateTime option
    
    /// <summary>
    /// The activation count of the association.
    /// </summary>
    ActivationCount: int
    
    /// <summary>
    /// The associated emotions of the association.
    /// </summary>
    AssociatedEmotions: Emotion list
    
    /// <summary>
    /// The bidirectional flag of the association.
    /// </summary>
    IsBidirectional: bool
    
    /// <summary>
    /// Additional metadata about the association.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents an association network.
/// </summary>
type AssociationNetwork = {
    /// <summary>
    /// The ID of the network.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The name of the network.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the network.
    /// </summary>
    Description: string option
    
    /// <summary>
    /// The associations in the network.
    /// </summary>
    Associations: Association list
    
    /// <summary>
    /// The creation time of the network.
    /// </summary>
    CreationTime: DateTime
    
    /// <summary>
    /// The last modification time of the network.
    /// </summary>
    LastModificationTime: DateTime
    
    /// <summary>
    /// Additional metadata about the network.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents an association path.
/// </summary>
type AssociationPath = {
    /// <summary>
    /// The source concept of the path.
    /// </summary>
    Source: string
    
    /// <summary>
    /// The target concept of the path.
    /// </summary>
    Target: string
    
    /// <summary>
    /// The associations in the path.
    /// </summary>
    Associations: Association list
    
    /// <summary>
    /// The strength of the path.
    /// </summary>
    Strength: float
    
    /// <summary>
    /// The length of the path.
    /// </summary>
    Length: int
    
    /// <summary>
    /// Additional metadata about the path.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents an association activation.
/// </summary>
type AssociationActivation = {
    /// <summary>
    /// The association that was activated.
    /// </summary>
    Association: Association
    
    /// <summary>
    /// The activation time.
    /// </summary>
    ActivationTime: DateTime
    
    /// <summary>
    /// The activation strength.
    /// </summary>
    ActivationStrength: float
    
    /// <summary>
    /// The context of the activation.
    /// </summary>
    Context: string option
    
    /// <summary>
    /// The trigger of the activation.
    /// </summary>
    Trigger: string option
    
    /// <summary>
    /// Additional metadata about the activation.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents an association query.
/// </summary>
type AssociationQuery = {
    /// <summary>
    /// The source concept of the query.
    /// </summary>
    Source: string option
    
    /// <summary>
    /// The target concept of the query.
    /// </summary>
    Target: string option
    
    /// <summary>
    /// The types of associations to include.
    /// </summary>
    Types: AssociationType list option
    
    /// <summary>
    /// The minimum strength of associations to include.
    /// </summary>
    MinimumStrength: AssociationStrength option
    
    /// <summary>
    /// The maximum path length to consider.
    /// </summary>
    MaxPathLength: int option
    
    /// <summary>
    /// The maximum number of results to return.
    /// </summary>
    MaxResults: int option
    
    /// <summary>
    /// Additional metadata about the query.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents an association query result.
/// </summary>
type AssociationQueryResult = {
    /// <summary>
    /// The query that was executed.
    /// </summary>
    Query: AssociationQuery
    
    /// <summary>
    /// The associations that matched the query.
    /// </summary>
    Associations: Association list
    
    /// <summary>
    /// The paths that matched the query, if applicable.
    /// </summary>
    Paths: AssociationPath list option
    
    /// <summary>
    /// The execution time of the query.
    /// </summary>
    ExecutionTime: TimeSpan
    
    /// <summary>
    /// Additional metadata about the result.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents an association suggestion.
/// </summary>
type AssociationSuggestion = {
    /// <summary>
    /// The source concept of the suggestion.
    /// </summary>
    Source: string
    
    /// <summary>
    /// The target concept of the suggestion.
    /// </summary>
    Target: string
    
    /// <summary>
    /// The type of the suggestion.
    /// </summary>
    Type: AssociationType
    
    /// <summary>
    /// The strength of the suggestion.
    /// </summary>
    Strength: AssociationStrength
    
    /// <summary>
    /// The confidence of the suggestion.
    /// </summary>
    Confidence: float
    
    /// <summary>
    /// The reason for the suggestion.
    /// </summary>
    Reason: string option
    
    /// <summary>
    /// Additional metadata about the suggestion.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents an association learning result.
/// </summary>
type AssociationLearningResult = {
    /// <summary>
    /// The associations that were learned.
    /// </summary>
    LearnedAssociations: Association list
    
    /// <summary>
    /// The associations that were updated.
    /// </summary>
    UpdatedAssociations: Association list
    
    /// <summary>
    /// The associations that were removed.
    /// </summary>
    RemovedAssociations: Association list
    
    /// <summary>
    /// The learning time.
    /// </summary>
    LearningTime: TimeSpan
    
    /// <summary>
    /// Additional metadata about the result.
    /// </summary>
    Metadata: Map<string, obj>
}
