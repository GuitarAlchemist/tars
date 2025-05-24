namespace TarsEngine.FSharp.Core.Compatibility

/// Module providing compatibility with the old MetascriptToT module
module MetascriptToTCompat =
    open TarsEngine.FSharp.Core.TreeOfThought
    
    /// Represents evaluation metrics for metascript thought nodes (compatibility with old API)
    type MetascriptEvaluationMetrics = {
        /// Correctness of the metascript (syntax, semantics)
        Correctness: float
        /// Efficiency of the metascript (resource usage, performance)
        Efficiency: float
        /// Robustness of the metascript (error handling, edge cases)
        Robustness: float
        /// Maintainability of the metascript (readability, structure)
        Maintainability: float
        /// Overall score (weighted average of all metrics)
        Overall: float
    }
    
    /// Represents a node in a metascript thought tree (compatibility with old API)
    type MetascriptThoughtNode = {
        /// The thought content
        Thought: string
        /// Child nodes
        Children: MetascriptThoughtNode list
        /// Evaluation metrics
        Evaluation: MetascriptEvaluationMetrics option
        /// Whether the node has been pruned
        Pruned: bool
        /// Additional metadata for metascript-specific information
        Metadata: Map<string, obj>
    }
    
    /// Converts a MetascriptEvaluationMetrics to an EvaluationMetrics
    let toEvaluationMetrics (metrics: MetascriptEvaluationMetrics) : ThoughtNode.EvaluationMetrics =
        { Correctness = metrics.Correctness
          Efficiency = metrics.Efficiency
          Robustness = metrics.Robustness
          Maintainability = metrics.Maintainability
          Overall = metrics.Overall }
    
    /// Converts an EvaluationMetrics to a MetascriptEvaluationMetrics
    let fromEvaluationMetrics (metrics: ThoughtNode.EvaluationMetrics) : MetascriptEvaluationMetrics =
        { Correctness = metrics.Correctness
          Efficiency = metrics.Efficiency
          Robustness = metrics.Robustness
          Maintainability = metrics.Maintainability
          Overall = metrics.Overall }
    
    /// Converts a MetascriptThoughtNode to a ThoughtNode
    let rec toThoughtNode (node: MetascriptThoughtNode) : ThoughtNode.ThoughtNode =
        { Thought = node.Thought
          Children = node.Children |> List.map toThoughtNode
          Evaluation = node.Evaluation |> Option.map toEvaluationMetrics
          Pruned = node.Pruned
          Metadata = node.Metadata }
    
    /// Converts a ThoughtNode to a MetascriptThoughtNode
    let rec fromThoughtNode (node: ThoughtNode.ThoughtNode) : MetascriptThoughtNode =
        { Thought = node.Thought
          Children = node.Children |> List.map fromThoughtNode
          Evaluation = node.Evaluation |> Option.map fromEvaluationMetrics
          Pruned = node.Pruned
          Metadata = node.Metadata }
    
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
    
    /// Gets the score of a node
    let getScore node =
        match node.Evaluation with
        | Some metrics -> metrics.Overall
        | None -> 0.0
    
    /// Creates evaluation metrics with equal weights
    let createMetrics correctness efficiency robustness maintainability =
        { Correctness = correctness
          Efficiency = efficiency
          Robustness = robustness
          Maintainability = maintainability
          Overall = (correctness + efficiency + robustness + maintainability) / 4.0 }
    
    /// Creates evaluation metrics with custom weights
    let createWeightedMetrics correctness efficiency robustness maintainability weights =
        let (wCorrectness, wEfficiency, wRobustness, wMaintainability) = weights
        let totalWeight = wCorrectness + wEfficiency + wRobustness + wMaintainability
        
        if totalWeight <= 0.0 then
            failwith "Total weight must be greater than zero"
        
        let overall =
            (correctness * wCorrectness +
             efficiency * wEfficiency +
             robustness * wRobustness +
             maintainability * wMaintainability) / totalWeight
        
        { Correctness = correctness
          Efficiency = efficiency
          Robustness = robustness
          Maintainability = maintainability
          Overall = overall }
    
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
    
    /// Converts metrics to JSON
    let toJson metrics =
        sprintf """
        {
            "correctness": %.2f,
            "efficiency": %.2f,
            "robustness": %.2f,
            "maintainability": %.2f,
            "overall": %.2f
        }
        """
            metrics.Correctness
            metrics.Efficiency
            metrics.Robustness
            metrics.Maintainability
            metrics.Overall
    
    /// Converts a tree to JSON
    let rec treeToJson node =
        let childrenJson =
            node.Children
            |> List.map treeToJson
            |> String.concat ", "
        
        let evaluationJson =
            match node.Evaluation with
            | Some metrics -> toJson metrics
            | None -> "null"
        
        let metadataJson =
            node.Metadata
            |> Map.toList
            |> List.map (fun (k, v) -> sprintf "\"%s\": \"%s\"" k (v.ToString()))
            |> String.concat ", "
        
        sprintf """
        {
            "thought": "%s",
            "evaluation": %s,
            "pruned": %b,
            "metadata": {%s},
            "children": [%s]
        }
        """
            node.Thought
            evaluationJson
            node.Pruned
            metadataJson
            childrenJson
