namespace TarsEngine.FSharp.Core.Adapters

/// Module containing adapters for metascript types
module MetascriptAdapter =
    open TarsEngine.FSharp.Core.TreeOfThought
    open TarsEngine.FSharp.MetascriptToT
    
    /// Converts a MetascriptEvaluationMetrics to an EvaluationMetrics
    let convertMetricsToCore (metrics: MetascriptEvaluationMetrics) : ThoughtNode.EvaluationMetrics =
        { Correctness = metrics.Correctness
          Efficiency = metrics.Efficiency
          Robustness = metrics.Robustness
          Maintainability = metrics.Maintainability
          Overall = metrics.Overall }
    
    /// Converts an EvaluationMetrics to a MetascriptEvaluationMetrics
    let convertMetricsToMetascript (metrics: ThoughtNode.EvaluationMetrics) : MetascriptEvaluationMetrics =
        { Correctness = metrics.Correctness
          Efficiency = metrics.Efficiency
          Robustness = metrics.Robustness
          Maintainability = metrics.Maintainability
          Overall = metrics.Overall }
    
    /// Converts a MetascriptThoughtNode to a ThoughtNode
    let rec convertNodeToCore (node: MetascriptThoughtNode) : ThoughtNode.ThoughtNode =
        { Thought = node.Thought
          Children = node.Children |> List.map convertNodeToCore
          Evaluation = node.Evaluation |> Option.map convertMetricsToCore
          Pruned = node.Pruned
          Metadata = node.Metadata }
    
    /// Converts a ThoughtNode to a MetascriptThoughtNode
    let rec convertNodeToMetascript (node: ThoughtNode.ThoughtNode) : MetascriptThoughtNode =
        { Thought = node.Thought
          Children = node.Children |> List.map convertNodeToMetascript
          Evaluation = node.Evaluation |> Option.map convertMetricsToMetascript
          Pruned = node.Pruned
          Metadata = node.Metadata }
    
    /// Creates a new MetascriptThoughtNode
    let createMetascriptNode thought =
        let coreNode = ThoughtNode.createNode thought
        convertNodeToMetascript coreNode
    
    /// Adds a child to a MetascriptThoughtNode
    let addChildToMetascriptNode parent child =
        let coreParent = convertNodeToCore parent
        let coreChild = convertNodeToCore child
        let updatedCoreParent = ThoughtNode.addChild coreParent coreChild
        convertNodeToMetascript updatedCoreParent
    
    /// Evaluates a MetascriptThoughtNode with metrics
    let evaluateMetascriptNode node metrics =
        let coreNode = convertNodeToCore node
        let coreMetrics = convertMetricsToCore metrics
        let evaluatedCoreNode = ThoughtNode.evaluateNode coreNode coreMetrics
        convertNodeToMetascript evaluatedCoreNode
    
    /// Marks a MetascriptThoughtNode as pruned
    let pruneMetascriptNode node =
        let coreNode = convertNodeToCore node
        let prunedCoreNode = ThoughtNode.pruneNode coreNode
        convertNodeToMetascript prunedCoreNode
    
    /// Adds metadata to a MetascriptThoughtNode
    let addMetadataToMetascriptNode node key value =
        let coreNode = convertNodeToCore node
        let updatedCoreNode = ThoughtNode.addMetadata coreNode key value
        convertNodeToMetascript updatedCoreNode
    
    /// Gets metadata from a MetascriptThoughtNode
    let getMetadataFromMetascriptNode<'T> node key =
        let coreNode = convertNodeToCore node
        ThoughtNode.getMetadata<'T> coreNode key
    
    /// Gets the score of a MetascriptThoughtNode, or 0.0 if not evaluated
    let getMetascriptNodeScore node =
        let coreNode = convertNodeToCore node
        ThoughtNode.getScore coreNode
