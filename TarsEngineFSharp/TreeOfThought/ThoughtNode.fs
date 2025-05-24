namespace TarsEngineFSharp.TreeOfThought

/// Module containing types and functions for working with thought nodes
module ThoughtNode =
    
    /// Represents evaluation metrics for thought nodes
    type EvaluationMetrics = {
        /// Correctness of the solution (0.0 to 1.0)
        Correctness: float
        /// Efficiency of the solution (0.0 to 1.0)
        Efficiency: float
        /// Robustness of the solution (0.0 to 1.0)
        Robustness: float
        /// Maintainability of the solution (0.0 to 1.0)
        Maintainability: float
        /// Overall score (weighted average of all metrics)
        Overall: float
    }
    
    /// Represents a node in a thought tree
    type ThoughtNode = {
        /// The thought content
        Thought: string
        /// Child nodes
        Children: ThoughtNode list
        /// Evaluation metrics
        Evaluation: EvaluationMetrics option
        /// Whether the node has been pruned
        Pruned: bool
        /// Additional metadata
        Metadata: Map<string, obj>
    }
    
    /// Creates a new thought node
    let createNode thought =
        { Thought = thought
          Children = []
          Evaluation = None
          Pruned = false
          Metadata = Map.empty }
    
    /// Adds a child to a node
    let addChild parent child =
        { parent with Children = child :: parent.Children }
    
    /// Evaluates a node with metrics
    let evaluateNode node metrics =
        { node with Evaluation = Some metrics }
    
    /// Marks a node as pruned
    let pruneNode node =
        { node with Pruned = true }
    
    /// Adds metadata to a node
    let addMetadata node key value =
        { node with Metadata = node.Metadata.Add(key, value) }
    
    /// Gets metadata from a node
    let getMetadata<'T> node key =
        match node.Metadata.TryGetValue(key) with
        | true, value -> Some (value :?> 'T)
        | false, _ -> None
    
    /// Creates evaluation metrics with equal weights
    let createMetrics correctness efficiency robustness maintainability =
        { Correctness = correctness
          Efficiency = efficiency
          Robustness = robustness
          Maintainability = maintainability
          Overall = (correctness + efficiency + robustness + maintainability) / 4.0 }
    
    /// Creates evaluation metrics with weighted average
    let createWeightedMetrics correctness efficiency robustness maintainability weights =
        let (wc, we, wr, wm) = weights
        let total = wc + we + wr + wm
        let overall = 
            (correctness * wc + efficiency * we + robustness * wr + maintainability * wm) / total
        
        { Correctness = correctness
          Efficiency = efficiency
          Robustness = robustness
          Maintainability = maintainability
          Overall = overall }
    
    /// Converts a node to a string
    let toString node =
        let evaluationStr = 
            match node.Evaluation with
            | Some eval -> sprintf "Score: %.2f" eval.Overall
            | None -> "Not evaluated"
        
        sprintf "Thought: %s, %s, Pruned: %b, Children: %d" 
            node.Thought evaluationStr node.Pruned node.Children.Length
