namespace TarsEngine.FSharp.Core.TreeOfThought

open System
open System.Threading.Tasks
open System.Collections.Generic
open System.IO

/// <summary>
/// Implementation of the ITreeOfThoughtService interface.
/// </summary>
type TreeOfThoughtService() =
    /// <summary>
    /// Creates a thought tree for the specified problem.
    /// </summary>
    /// <param name="problem">The problem to solve.</param>
    /// <param name="options">The options for tree creation.</param>
    /// <returns>The thought tree.</returns>
    member _.CreateThoughtTreeAsync(problem: string, options: TreeCreationOptions) =
        // Create the root node with the problem as the thought
        let rootNode = ThoughtNode.createNode problem
        
        // If approaches are specified, add them as children
        let rootWithChildren =
            if options.Approaches.Count > 0 then
                let children =
                    options.Approaches
                    |> Seq.map ThoughtNode.createNode
                    |> Seq.toList
                
                ThoughtNode.addChildren rootNode children
            else
                rootNode
        
        // If approach evaluations are specified, evaluate the children
        let rootWithEvaluatedChildren =
            if options.ApproachEvaluations.Count > 0 then
                let evaluatedChildren =
                    rootWithChildren.Children
                    |> List.map (fun child ->
                        match options.ApproachEvaluations.TryGetValue(child.Thought) with
                        | true, metrics -> ThoughtNode.evaluateNode child metrics
                        | false, _ -> child)
                
                { rootWithChildren with Children = evaluatedChildren }
            else
                rootWithChildren
        
        // Return the root node
        Task.FromResult(rootWithEvaluatedChildren :> IThoughtNode)
    
    /// <summary>
    /// Evaluates a thought node.
    /// </summary>
    /// <param name="node">The node to evaluate.</param>
    /// <param name="metrics">The evaluation metrics.</param>
    /// <returns>The evaluated node.</returns>
    member _.EvaluateNodeAsync(node: IThoughtNode, metrics: EvaluationMetrics) =
        // Convert the interface to our concrete type
        let concreteNode = node :?> ThoughtNode.ThoughtNode
        
        // Evaluate the node
        let evaluatedNode = ThoughtNode.evaluateNode concreteNode metrics
        
        // Return the evaluated node
        Task.FromResult(evaluatedNode :> IThoughtNode)
    
    /// <summary>
    /// Adds a child to a node.
    /// </summary>
    /// <param name="parent">The parent node.</param>
    /// <param name="childThought">The child thought.</param>
    /// <returns>The updated parent node.</returns>
    member _.AddChildAsync(parent: IThoughtNode, childThought: string) =
        // Convert the interface to our concrete type
        let concreteParent = parent :?> ThoughtNode.ThoughtNode
        
        // Create a new child node
        let childNode = ThoughtNode.createNode childThought
        
        // Add the child to the parent
        let updatedParent = ThoughtNode.addChild concreteParent childNode
        
        // Return the updated parent
        Task.FromResult(updatedParent :> IThoughtNode)
    
    /// <summary>
    /// Selects the best node from a thought tree.
    /// </summary>
    /// <param name="root">The root node.</param>
    /// <returns>The best node.</returns>
    member _.SelectBestNodeAsync(root: IThoughtNode) =
        // Convert the interface to our concrete type
        let concreteRoot = root :?> ThoughtNode.ThoughtNode
        
        // Select the best node
        let bestNode = ThoughtTree.selectBestNode concreteRoot
        
        // Return the best node
        Task.FromResult(bestNode :> IThoughtNode)
    
    /// <summary>
    /// Prunes nodes that don't meet a threshold.
    /// </summary>
    /// <param name="root">The root node.</param>
    /// <param name="threshold">The threshold.</param>
    /// <returns>The pruned tree.</returns>
    member _.PruneByThresholdAsync(root: IThoughtNode, threshold: float) =
        // Convert the interface to our concrete type
        let concreteRoot = root :?> ThoughtNode.ThoughtNode
        
        // Prune the tree
        let prunedTree = ThoughtTree.pruneByThreshold threshold concreteRoot
        
        // Return the pruned tree
        Task.FromResult(prunedTree :> IThoughtNode)
    
    /// <summary>
    /// Generates a report for a thought tree.
    /// </summary>
    /// <param name="root">The root node.</param>
    /// <param name="title">The report title.</param>
    /// <returns>The report.</returns>
    member _.GenerateReportAsync(root: IThoughtNode, title: string) =
        // Convert the interface to our concrete type
        let concreteRoot = root :?> ThoughtNode.ThoughtNode
        
        // Generate the report
        let report = Visualization.toMarkdownReport concreteRoot title
        
        // Return the report
        Task.FromResult(report)
    
    /// <summary>
    /// Saves a report for a thought tree.
    /// </summary>
    /// <param name="root">The root node.</param>
    /// <param name="title">The report title.</param>
    /// <param name="filePath">The file path.</param>
    /// <returns>A task representing the asynchronous operation.</returns>
    member _.SaveReportAsync(root: IThoughtNode, title: string, filePath: string) =
        // Convert the interface to our concrete type
        let concreteRoot = root :?> ThoughtNode.ThoughtNode
        
        // Determine the file format based on the extension
        let extension = Path.GetExtension(filePath).ToLowerInvariant()
        
        // Save the report in the appropriate format
        match extension with
        | ".md" -> Visualization.saveMarkdownReport concreteRoot title filePath
        | ".json" -> Visualization.saveJsonReport concreteRoot filePath
        | ".dot" -> Visualization.saveDotGraph concreteRoot title filePath
        | _ -> Visualization.saveMarkdownReport concreteRoot title filePath
        
        // Return a completed task
        Task.CompletedTask

/// <summary>
/// Implementation of the ITreeOfThoughtService interface.
/// </summary>
type FSharpTreeOfThoughtService() =
    inherit TreeOfThoughtService()
