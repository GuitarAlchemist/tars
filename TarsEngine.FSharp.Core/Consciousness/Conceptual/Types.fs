namespace TarsEngine.FSharp.Core.Consciousness.Conceptual

open System
open TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Represents the type of a concept.
/// </summary>
type ConceptType =
    | Entity
    | Action
    | Property
    | Relation
    | Event
    | Abstract
    | Category
    | Process
    | State
    | Custom of string

/// <summary>
/// Represents the complexity of a concept.
/// </summary>
type ConceptComplexity =
    | Simple
    | Moderate
    | Complex
    | VeryComplex
    | Custom of float

/// <summary>
/// Represents a concept.
/// </summary>
type Concept = {
    /// <summary>
    /// The ID of the concept.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The name of the concept.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the concept.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The type of the concept.
    /// </summary>
    Type: ConceptType
    
    /// <summary>
    /// The complexity of the concept.
    /// </summary>
    Complexity: ConceptComplexity
    
    /// <summary>
    /// The creation time of the concept.
    /// </summary>
    CreationTime: DateTime
    
    /// <summary>
    /// The last activation time of the concept.
    /// </summary>
    LastActivationTime: DateTime option
    
    /// <summary>
    /// The activation count of the concept.
    /// </summary>
    ActivationCount: int
    
    /// <summary>
    /// The associated emotions of the concept.
    /// </summary>
    AssociatedEmotions: Emotion list
    
    /// <summary>
    /// The related concepts.
    /// </summary>
    RelatedConcepts: (Guid * float) list
    
    /// <summary>
    /// The attributes of the concept.
    /// </summary>
    Attributes: Map<string, obj>
    
    /// <summary>
    /// The examples of the concept.
    /// </summary>
    Examples: string list
    
    /// <summary>
    /// The tags of the concept.
    /// </summary>
    Tags: string list
    
    /// <summary>
    /// Additional metadata about the concept.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a concept activation.
/// </summary>
type ConceptActivation = {
    /// <summary>
    /// The concept that was activated.
    /// </summary>
    Concept: Concept
    
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
/// Represents a concept hierarchy.
/// </summary>
type ConceptHierarchy = {
    /// <summary>
    /// The ID of the hierarchy.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The name of the hierarchy.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the hierarchy.
    /// </summary>
    Description: string option
    
    /// <summary>
    /// The root concepts of the hierarchy.
    /// </summary>
    RootConcepts: Guid list
    
    /// <summary>
    /// The parent-child relationships in the hierarchy.
    /// </summary>
    ParentChildRelationships: Map<Guid, Guid list>
    
    /// <summary>
    /// The creation time of the hierarchy.
    /// </summary>
    CreationTime: DateTime
    
    /// <summary>
    /// The last modification time of the hierarchy.
    /// </summary>
    LastModificationTime: DateTime
    
    /// <summary>
    /// Additional metadata about the hierarchy.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a concept query.
/// </summary>
type ConceptQuery = {
    /// <summary>
    /// The name pattern to match.
    /// </summary>
    NamePattern: string option
    
    /// <summary>
    /// The types to include.
    /// </summary>
    Types: ConceptType list option
    
    /// <summary>
    /// The tags to include.
    /// </summary>
    Tags: string list option
    
    /// <summary>
    /// The minimum complexity to include.
    /// </summary>
    MinimumComplexity: ConceptComplexity option
    
    /// <summary>
    /// The maximum complexity to include.
    /// </summary>
    MaximumComplexity: ConceptComplexity option
    
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
/// Represents a concept query result.
/// </summary>
type ConceptQueryResult = {
    /// <summary>
    /// The query that was executed.
    /// </summary>
    Query: ConceptQuery
    
    /// <summary>
    /// The concepts that matched the query.
    /// </summary>
    Concepts: Concept list
    
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
/// Represents a concept suggestion.
/// </summary>
type ConceptSuggestion = {
    /// <summary>
    /// The name of the suggestion.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the suggestion.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The type of the suggestion.
    /// </summary>
    Type: ConceptType
    
    /// <summary>
    /// The complexity of the suggestion.
    /// </summary>
    Complexity: ConceptComplexity
    
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
/// Represents a concept learning result.
/// </summary>
type ConceptLearningResult = {
    /// <summary>
    /// The concepts that were learned.
    /// </summary>
    LearnedConcepts: Concept list
    
    /// <summary>
    /// The concepts that were updated.
    /// </summary>
    UpdatedConcepts: Concept list
    
    /// <summary>
    /// The concepts that were removed.
    /// </summary>
    RemovedConcepts: Concept list
    
    /// <summary>
    /// The learning time.
    /// </summary>
    LearningTime: TimeSpan
    
    /// <summary>
    /// Additional metadata about the result.
    /// </summary>
    Metadata: Map<string, obj>
}
