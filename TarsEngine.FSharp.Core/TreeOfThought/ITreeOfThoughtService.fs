namespace TarsEngine.FSharp.Core.TreeOfThought

open System.Threading.Tasks

/// <summary>
/// Interface for Tree-of-Thought service.
/// </summary>
type ITreeOfThoughtService =
    /// <summary>
    /// Creates a thought tree for the specified problem.
    /// </summary>
    /// <param name="problem">The problem to solve.</param>
    /// <param name="options">The options for tree creation.</param>
    /// <returns>The thought tree.</returns>
    abstract member CreateThoughtTreeAsync : problem:string * options:TreeCreationOptions -> Task<IThoughtNode>
    
    /// <summary>
    /// Evaluates a thought node.
    /// </summary>
    /// <param name="node">The node to evaluate.</param>
    /// <param name="metrics">The evaluation metrics.</param>
    /// <returns>The evaluated node.</returns>
    abstract member EvaluateNodeAsync : node:IThoughtNode * metrics:EvaluationMetrics -> Task<IThoughtNode>
    
    /// <summary>
    /// Adds a child to a node.
    /// </summary>
    /// <param name="parent">The parent node.</param>
    /// <param name="childThought">The child thought.</param>
    /// <returns>The updated parent node.</returns>
    abstract member AddChildAsync : parent:IThoughtNode * childThought:string -> Task<IThoughtNode>
    
    /// <summary>
    /// Selects the best node from a thought tree.
    /// </summary>
    /// <param name="root">The root node.</param>
    /// <returns>The best node.</returns>
    abstract member SelectBestNodeAsync : root:IThoughtNode -> Task<IThoughtNode>
    
    /// <summary>
    /// Prunes nodes that don't meet a threshold.
    /// </summary>
    /// <param name="root">The root node.</param>
    /// <param name="threshold">The threshold.</param>
    /// <returns>The pruned tree.</returns>
    abstract member PruneByThresholdAsync : root:IThoughtNode * threshold:float -> Task<IThoughtNode>
    
    /// <summary>
    /// Generates a report for a thought tree.
    /// </summary>
    /// <param name="root">The root node.</param>
    /// <param name="title">The report title.</param>
    /// <returns>The report.</returns>
    abstract member GenerateReportAsync : root:IThoughtNode * title:string -> Task<string>
    
    /// <summary>
    /// Saves a report for a thought tree.
    /// </summary>
    /// <param name="root">The root node.</param>
    /// <param name="title">The report title.</param>
    /// <param name="filePath">The file path.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    abstract member SaveReportAsync : root:IThoughtNode * title:string * filePath:string -> Task
