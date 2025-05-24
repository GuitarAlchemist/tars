namespace TarsEngine.FSharp.Core.TreeOfThought

open System.Collections.Generic

/// <summary>
/// Interface for a thought node in a tree-of-thought.
/// </summary>
type IThoughtNode =
    /// <summary>
    /// Gets the thought content.
    /// </summary>
    abstract member Thought : string
    
    /// <summary>
    /// Gets the child nodes.
    /// </summary>
    abstract member Children : IReadOnlyList<IThoughtNode>
    
    /// <summary>
    /// Gets the evaluation metrics.
    /// </summary>
    abstract member Evaluation : EvaluationMetrics option
    
    /// <summary>
    /// Gets a value indicating whether the node has been pruned.
    /// </summary>
    abstract member Pruned : bool
    
    /// <summary>
    /// Gets the metadata.
    /// </summary>
    abstract member Metadata : IReadOnlyDictionary<string, obj>
    
    /// <summary>
    /// Gets the score of the node.
    /// </summary>
    abstract member Score : float
