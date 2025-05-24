namespace TarsEngine.FSharp.Core.TreeOfThought

open System.Collections.Generic

/// <summary>
/// Options for creating a thought tree.
/// </summary>
type TreeCreationOptions() =
    /// <summary>
    /// Gets or sets the approaches to consider.
    /// </summary>
    member val Approaches = List<string>() with get, set
    
    /// <summary>
    /// Gets or sets the evaluations for each approach.
    /// </summary>
    member val ApproachEvaluations = Dictionary<string, EvaluationMetrics>() with get, set
