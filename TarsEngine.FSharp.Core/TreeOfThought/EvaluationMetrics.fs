namespace TarsEngine.FSharp.Core.TreeOfThought

/// <summary>
/// Evaluation metrics for a thought node.
/// </summary>
[<AllowNullLiteral>]
type EvaluationMetrics(correctness: float, efficiency: float, robustness: float, maintainability: float, ?overall: float) =
    /// <summary>
    /// Gets the correctness score.
    /// </summary>
    member val Correctness = correctness with get
    
    /// <summary>
    /// Gets the efficiency score.
    /// </summary>
    member val Efficiency = efficiency with get
    
    /// <summary>
    /// Gets the robustness score.
    /// </summary>
    member val Robustness = robustness with get
    
    /// <summary>
    /// Gets the maintainability score.
    /// </summary>
    member val Maintainability = maintainability with get
    
    /// <summary>
    /// Gets the overall score.
    /// </summary>
    member val Overall = 
        match overall with
        | Some value -> value
        | None -> (correctness + efficiency + robustness + maintainability) / 4.0
        with get
