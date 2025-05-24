namespace TarsEngine.FSharp.Core.TreeOfThought

open System.Collections.Generic

/// <summary>
/// Implementation of the IThoughtNode interface.
/// </summary>
type ThoughtNodeImpl(node: ThoughtNode.ThoughtNode) =
    /// <summary>
    /// Gets the thought content.
    /// </summary>
    member _.Thought = node.Thought
    
    /// <summary>
    /// Gets the child nodes.
    /// </summary>
    member _.Children =
        node.Children
        |> List.map (fun child -> ThoughtNodeImpl(child) :> IThoughtNode)
        |> List.toArray
        |> (fun arr -> arr :> IReadOnlyList<IThoughtNode>)
    
    /// <summary>
    /// Gets the evaluation metrics.
    /// </summary>
    member _.Evaluation = node.Evaluation
    
    /// <summary>
    /// Gets a value indicating whether the node has been pruned.
    /// </summary>
    member _.Pruned = node.Pruned
    
    /// <summary>
    /// Gets the metadata.
    /// </summary>
    member _.Metadata =
        node.Metadata
        |> Map.toSeq
        |> Seq.map (fun (k, v) -> KeyValuePair<string, obj>(k, v))
        |> Dictionary<string, obj>
        |> (fun dict -> dict :> IReadOnlyDictionary<string, obj>)
    
    /// <summary>
    /// Gets the score of the node.
    /// </summary>
    member _.Score = ThoughtNode.getScore node
    
    /// <summary>
    /// Gets the underlying ThoughtNode.
    /// </summary>
    member _.Node = node
    
    interface IThoughtNode with
        member this.Thought = this.Thought
        member this.Children = this.Children
        member this.Evaluation = this.Evaluation
        member this.Pruned = this.Pruned
        member this.Metadata = this.Metadata
        member this.Score = this.Score

/// <summary>
/// Module containing functions for working with ThoughtNodeImpl.
/// </summary>
module ThoughtNodeImpl =
    /// <summary>
    /// Creates a new ThoughtNodeImpl from a ThoughtNode.
    /// </summary>
    /// <param name="node">The ThoughtNode.</param>
    /// <returns>A new ThoughtNodeImpl.</returns>
    let fromThoughtNode (node: ThoughtNode.ThoughtNode) =
        ThoughtNodeImpl(node)
    
    /// <summary>
    /// Gets the underlying ThoughtNode from a ThoughtNodeImpl.
    /// </summary>
    /// <param name="node">The ThoughtNodeImpl.</param>
    /// <returns>The underlying ThoughtNode.</returns>
    let toThoughtNode (node: ThoughtNodeImpl) =
        node.Node
    
    /// <summary>
    /// Gets the underlying ThoughtNode from an IThoughtNode.
    /// </summary>
    /// <param name="node">The IThoughtNode.</param>
    /// <returns>The underlying ThoughtNode.</returns>
    let toThoughtNodeFromInterface (node: IThoughtNode) =
        match node with
        | :? ThoughtNodeImpl as impl -> impl.Node
        | _ -> failwith "Not a ThoughtNodeImpl"
