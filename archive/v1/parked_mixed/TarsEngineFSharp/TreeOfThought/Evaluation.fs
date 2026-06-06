namespace TarsEngineFSharp.TreeOfThought

/// Module containing functions for evaluating thought nodes
module Evaluation =
    open ThoughtNode
    
    /// Normalizes metrics to [0, 1] range
    let normalizeMetrics metrics =
        let normalize value = max 0.0 (min 1.0 value)
        
        { Correctness = normalize metrics.Correctness
          Efficiency = normalize metrics.Efficiency
          Robustness = normalize metrics.Robustness
          Maintainability = normalize metrics.Maintainability
          Overall = normalize metrics.Overall }
    
    /// Compares two metrics
    let compareMetrics metrics1 metrics2 =
        metrics1.Overall - metrics2.Overall
    
    /// Applies a threshold to metrics
    let thresholdMetrics threshold metrics =
        if metrics.Overall < threshold then
            None
        else
            Some metrics
    
    /// Combines multiple metrics
    let combineMetrics metrics =
        if List.isEmpty metrics then
            createMetrics 0.0 0.0 0.0 0.0
        else
            let correctness = metrics |> List.averageBy (fun m -> m.Correctness)
            let efficiency = metrics |> List.averageBy (fun m -> m.Efficiency)
            let robustness = metrics |> List.averageBy (fun m -> m.Robustness)
            let maintainability = metrics |> List.averageBy (fun m -> m.Maintainability)
            
            createMetrics correctness efficiency robustness maintainability
    
    /// Evaluates a thought node based on its children
    let evaluateNodeFromChildren node =
        let childMetrics = 
            node.Children
            |> List.choose (fun child -> child.Evaluation)
        
        if List.isEmpty childMetrics then
            node
        else
            let combinedMetrics = combineMetrics childMetrics
            evaluateNode node combinedMetrics
    
    /// Evaluates a thought tree from bottom to top
    let rec evaluateTree node =
        let evaluatedChildren = 
            node.Children
            |> List.map evaluateTree
        
        let nodeWithEvaluatedChildren = 
            { node with Children = evaluatedChildren }
        
        evaluateNodeFromChildren nodeWithEvaluatedChildren
