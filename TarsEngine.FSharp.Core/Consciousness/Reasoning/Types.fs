namespace TarsEngine.FSharp.Core.Consciousness.Reasoning

open System

/// <summary>
/// Types of intuitions.
/// </summary>
type IntuitionType =
    | PatternRecognition
    | HeuristicReasoning
    | GutFeeling
    | Custom of string

/// <summary>
/// Verification status of an intuition.
/// </summary>
type VerificationStatus =
    | Unverified
    | Verified
    | Falsified
    | PartiallyVerified
    | Inconclusive

/// <summary>
/// Represents a heuristic rule.
/// </summary>
type HeuristicRule = {
    /// <summary>
    /// The unique identifier of the rule.
    /// </summary>
    Id: string
    
    /// <summary>
    /// The name of the rule.
    /// </summary>
    Name: string
    
    /// <summary>
    /// The description of the rule.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The reliability of the rule (0.0 to 1.0).
    /// </summary>
    Reliability: float
    
    /// <summary>
    /// The context of the rule.
    /// </summary>
    Context: string
    
    /// <summary>
    /// The creation timestamp of the rule.
    /// </summary>
    CreationTimestamp: DateTime
    
    /// <summary>
    /// The last used timestamp of the rule.
    /// </summary>
    LastUsedTimestamp: DateTime option
    
    /// <summary>
    /// The usage count of the rule.
    /// </summary>
    UsageCount: int
    
    /// <summary>
    /// The success count of the rule.
    /// </summary>
    SuccessCount: int
    
    /// <summary>
    /// The tags associated with the rule.
    /// </summary>
    Tags: string list
    
    /// <summary>
    /// The examples of the rule.
    /// </summary>
    Examples: string list
    
    /// <summary>
    /// The counter-examples of the rule.
    /// </summary>
    CounterExamples: string list
}

/// <summary>
/// Functions for working with heuristic rules.
/// </summary>
module HeuristicRule =
    /// <summary>
    /// Creates a new heuristic rule with default values.
    /// </summary>
    let create name description reliability context = {
        Id = Guid.NewGuid().ToString()
        Name = name
        Description = description
        Reliability = reliability
        Context = context
        CreationTimestamp = DateTime.UtcNow
        LastUsedTimestamp = None
        UsageCount = 0
        SuccessCount = 0
        Tags = []
        Examples = []
        CounterExamples = []
    }
    
    /// <summary>
    /// Gets the success rate of the rule.
    /// </summary>
    let successRate rule =
        if rule.UsageCount > 0 then
            float rule.SuccessCount / float rule.UsageCount
        else
            0.0
    
    /// <summary>
    /// Records a rule usage.
    /// </summary>
    let recordUsage success rule =
        { rule with 
            UsageCount = rule.UsageCount + 1
            SuccessCount = if success then rule.SuccessCount + 1 else rule.SuccessCount
            LastUsedTimestamp = Some DateTime.UtcNow }
    
    /// <summary>
    /// Adds an example to the rule.
    /// </summary>
    let addExample example isCounterExample rule =
        if isCounterExample then
            { rule with CounterExamples = example :: rule.CounterExamples }
        else
            { rule with Examples = example :: rule.Examples }
    
    /// <summary>
    /// Updates the reliability of the rule.
    /// </summary>
    let updateReliability reliability rule =
        { rule with Reliability = Math.Max(0.0, Math.Min(1.0, reliability)) }

/// <summary>
/// Represents an intuition.
/// </summary>
type Intuition = {
    /// <summary>
    /// The unique identifier of the intuition.
    /// </summary>
    Id: string
    
    /// <summary>
    /// The description of the intuition.
    /// </summary>
    Description: string
    
    /// <summary>
    /// The type of the intuition.
    /// </summary>
    Type: IntuitionType
    
    /// <summary>
    /// The confidence of the intuition (0.0 to 1.0).
    /// </summary>
    Confidence: float
    
    /// <summary>
    /// The timestamp of the intuition.
    /// </summary>
    Timestamp: DateTime
    
    /// <summary>
    /// The context of the intuition.
    /// </summary>
    Context: Map<string, obj>
    
    /// <summary>
    /// The tags associated with the intuition.
    /// </summary>
    Tags: string list
    
    /// <summary>
    /// The source of the intuition.
    /// </summary>
    Source: string
    
    /// <summary>
    /// The verification status of the intuition.
    /// </summary>
    VerificationStatus: VerificationStatus
    
    /// <summary>
    /// The verification timestamp of the intuition.
    /// </summary>
    VerificationTimestamp: DateTime option
    
    /// <summary>
    /// The verification notes of the intuition.
    /// </summary>
    VerificationNotes: string
    
    /// <summary>
    /// The accuracy of the intuition (0.0 to 1.0).
    /// </summary>
    Accuracy: float option
    
    /// <summary>
    /// The impact of the intuition (0.0 to 1.0).
    /// </summary>
    Impact: float
    
    /// <summary>
    /// The explanation of the intuition.
    /// </summary>
    Explanation: string
    
    /// <summary>
    /// The decision of the intuition.
    /// </summary>
    Decision: string
    
    /// <summary>
    /// The selected option of the intuition.
    /// </summary>
    SelectedOption: string
    
    /// <summary>
    /// The options of the intuition.
    /// </summary>
    Options: string list
}

/// <summary>
/// Functions for working with intuitions.
/// </summary>
module Intuition =
    /// <summary>
    /// Creates a new intuition with default values.
    /// </summary>
    let create () = {
        Id = Guid.NewGuid().ToString()
        Description = ""
        Type = IntuitionType.GutFeeling
        Confidence = 0.5
        Timestamp = DateTime.UtcNow
        Context = Map.empty
        Tags = []
        Source = ""
        VerificationStatus = VerificationStatus.Unverified
        VerificationTimestamp = None
        VerificationNotes = ""
        Accuracy = None
        Impact = 0.5
        Explanation = ""
        Decision = ""
        SelectedOption = ""
        Options = []
    }
    
    /// <summary>
    /// Verifies the intuition.
    /// </summary>
    let verify isCorrect accuracy notes intuition =
        { intuition with 
            VerificationStatus = if isCorrect then VerificationStatus.Verified else VerificationStatus.Falsified
            Accuracy = Some accuracy
            VerificationNotes = notes
            VerificationTimestamp = Some DateTime.UtcNow }
    
    /// <summary>
    /// Adds an explanation to the intuition.
    /// </summary>
    let addExplanation explanation intuition =
        { intuition with Explanation = explanation }
    
    /// <summary>
    /// Sets the decision context of the intuition.
    /// </summary>
    let setDecisionContext decision selectedOption options intuition =
        { intuition with 
            Decision = decision
            SelectedOption = selectedOption
            Options = options }
