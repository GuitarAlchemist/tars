namespace TarsEngine.FSharp.Core.Consciousness.Decision

open System
open TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Represents the type of a decision.
/// </summary>
type DecisionType =
    | Binary
    | MultipleChoice
    | Ranking
    | Allocation
    | Scheduling
    | Optimization
    | Custom of string

/// <summary>
/// Represents the status of a decision.
/// </summary>
type DecisionStatus =
    | Pending
    | InProgress
    | Completed
    | Cancelled
    | Failed

/// <summary>
/// Represents the priority of a decision.
/// </summary>
type DecisionPriority =
    | Low
    | Medium
    | High
    | Critical
    | Custom of int

/// <summary>
/// Represents a decision option.
/// </summary>
type DecisionOption = {
    /// <summary>
    /// The ID of the option.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The name of the option.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the option.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The pros of the option.
    /// </summary>
    Pros: string list
    
    /// <summary>
    /// The cons of the option.
    /// </summary>
    Cons: string list
    
    /// <summary>
    /// The score of the option.
    /// </summary>
    Score: float option
    
    /// <summary>
    /// The rank of the option.
    /// </summary>
    Rank: int option
    
    /// <summary>
    /// Additional metadata about the option.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a decision criterion.
/// </summary>
type DecisionCriterion = {
    /// <summary>
    /// The ID of the criterion.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The name of the criterion.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the criterion.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The weight of the criterion.
    /// </summary>
    Weight: float
    
    /// <summary>
    /// The scores for each option.
    /// </summary>
    Scores: Map<Guid, float>
    
    /// <summary>
    /// Additional metadata about the criterion.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a decision constraint.
/// </summary>
type DecisionConstraint = {
    /// <summary>
    /// The ID of the constraint.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The name of the constraint.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the constraint.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The type of the constraint.
    /// </summary>
    Type: string
    
    /// <summary>
    /// The value of the constraint.
    /// </summary>
    Value: obj
    
    /// <summary>
    /// Whether the constraint is satisfied.
    /// </summary>
    IsSatisfied: bool
    
    /// <summary>
    /// Additional metadata about the constraint.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a decision.
/// </summary>
type Decision = {
    /// <summary>
    /// The ID of the decision.
    /// </summary>
    Id: Guid
    
    /// <summary>
    /// The name of the decision.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the decision.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The type of the decision.
    /// </summary>
    Type: DecisionType
    
    /// <summary>
    /// The status of the decision.
    /// </summary>
    Status: DecisionStatus
    
    /// <summary>
    /// The priority of the decision.
    /// </summary>
    Priority: DecisionPriority
    
    /// <summary>
    /// The options for the decision.
    /// </summary>
    Options: DecisionOption list
    
    /// <summary>
    /// The criteria for the decision.
    /// </summary>
    Criteria: DecisionCriterion list
    
    /// <summary>
    /// The constraints for the decision.
    /// </summary>
    Constraints: DecisionConstraint list
    
    /// <summary>
    /// The selected option for the decision.
    /// </summary>
    SelectedOption: Guid option
    
    /// <summary>
    /// The creation time of the decision.
    /// </summary>
    CreationTime: DateTime
    
    /// <summary>
    /// The deadline for the decision.
    /// </summary>
    Deadline: DateTime option
    
    /// <summary>
    /// The completion time of the decision.
    /// </summary>
    CompletionTime: DateTime option
    
    /// <summary>
    /// The associated emotions of the decision.
    /// </summary>
    AssociatedEmotions: Emotion list
    
    /// <summary>
    /// The context of the decision.
    /// </summary>
    Context: string option
    
    /// <summary>
    /// The justification for the decision.
    /// </summary>
    Justification: string option
    
    /// <summary>
    /// Additional metadata about the decision.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a decision evaluation.
/// </summary>
type DecisionEvaluation = {
    /// <summary>
    /// The decision that was evaluated.
    /// </summary>
    Decision: Decision
    
    /// <summary>
    /// The evaluation score.
    /// </summary>
    Score: float
    
    /// <summary>
    /// The evaluation time.
    /// </summary>
    EvaluationTime: DateTime
    
    /// <summary>
    /// The strengths of the decision.
    /// </summary>
    Strengths: string list
    
    /// <summary>
    /// The weaknesses of the decision.
    /// </summary>
    Weaknesses: string list
    
    /// <summary>
    /// The opportunities for the decision.
    /// </summary>
    Opportunities: string list
    
    /// <summary>
    /// The threats for the decision.
    /// </summary>
    Threats: string list
    
    /// <summary>
    /// Additional metadata about the evaluation.
    /// </summary>
    Metadata: Map<string, obj>
}

/// <summary>
/// Represents a decision query.
/// </summary>
type DecisionQuery = {
    /// <summary>
    /// The name pattern to match.
    /// </summary>
    NamePattern: string option
    
    /// <summary>
    /// The types to include.
    /// </summary>
    Types: DecisionType list option
    
    /// <summary>
    /// The statuses to include.
    /// </summary>
    Statuses: DecisionStatus list option
    
    /// <summary>
    /// The priorities to include.
    /// </summary>
    Priorities: DecisionPriority list option
    
    /// <summary>
    /// The minimum creation time to include.
    /// </summary>
    MinimumCreationTime: DateTime option
    
    /// <summary>
    /// The maximum creation time to include.
    /// </summary>
    MaximumCreationTime: DateTime option
    
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
/// Represents a decision query result.
/// </summary>
type DecisionQueryResult = {
    /// <summary>
    /// The query that was executed.
    /// </summary>
    Query: DecisionQuery
    
    /// <summary>
    /// The decisions that matched the query.
    /// </summary>
    Decisions: Decision list
    
    /// <summary>
    /// The execution time of the query.
    /// </summary>
    ExecutionTime: TimeSpan
    
    /// <summary>
    /// Additional metadata about the result.
    /// </summary>
    Metadata: Map<string, obj>
}
