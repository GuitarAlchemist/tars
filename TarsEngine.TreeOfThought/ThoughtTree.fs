namespace TarsEngine.TreeOfThought

open System
open System.Collections.Generic

/// Represents a tree structure for organizing thoughts and reasoning
type ThoughtTree<'T> = {
    Root: ThoughtNode.ThoughtNode<'T>
    Nodes: Dictionary<string, ThoughtNode.ThoughtNode<'T>>
}

/// Operations for working with thought trees
module ThoughtTree =
    
    /// Create a new empty thought tree
    let create (rootValue: 'T) : ThoughtTree<'T> =
        let rootNode = ThoughtNode.create "root" rootValue
        let nodes = Dictionary<string, ThoughtNode.ThoughtNode<'T>>()
        nodes.["root"] <- rootNode
        {
            Root = rootNode
            Nodes = nodes
        }
    
    /// Add a new thought node to the tree
    let addNode (tree: ThoughtTree<'T>) (parentId: string) (nodeId: string) (value: 'T) : ThoughtTree<'T> =
        let newNode = ThoughtNode.create nodeId value
        
        // Add to parent if it exists
        if tree.Nodes.ContainsKey(parentId) then
            let parent = tree.Nodes.[parentId]
            ThoughtNode.addChild parent newNode |> ignore
        elif parentId = "root" then
            ThoughtNode.addChild tree.Root newNode |> ignore
        
        // Add to nodes dictionary
        tree.Nodes.[nodeId] <- newNode
        tree
    
    /// Find a node by its ID
    let findNode (tree: ThoughtTree<'T>) (nodeId: string) : ThoughtNode.ThoughtNode<'T> option =
        if nodeId = "root" then
            Some tree.Root
        elif tree.Nodes.ContainsKey(nodeId) then
            Some tree.Nodes.[nodeId]
        else
            None
    
    /// Get all leaf nodes (nodes with no children)
    let getLeafNodes (tree: ThoughtTree<'T>) : ThoughtNode.ThoughtNode<'T> list =
        let allNodes = tree.Root :: (tree.Nodes.Values |> Seq.toList)
        allNodes |> List.filter (fun node -> node.Children.Length = 0)
    
    /// Traverse the tree depth-first
    let traverseDepthFirst (tree: ThoughtTree<'T>) (visitor: ThoughtNode.ThoughtNode<'T> -> unit) : unit =
        let rec traverse (node: ThoughtNode.ThoughtNode<'T>) =
            visitor node
            for child in node.Children do
                traverse child
        
        traverse tree.Root
    
    /// Get the path from root to a specific node
    let getPathToNode (tree: ThoughtTree<'T>) (targetId: string) : ThoughtNode.ThoughtNode<'T> list option =
        let rec findPath (current: ThoughtNode.ThoughtNode<'T>) (path: ThoughtNode.ThoughtNode<'T> list) =
            let newPath = current :: path
            if current.Id = targetId then
                Some (List.rev newPath)
            else
                current.Children
                |> Seq.tryPick (fun child -> findPath child newPath)
        
        findPath tree.Root []
    
    /// Calculate the depth of the tree
    let getDepth (tree: ThoughtTree<'T>) : int =
        let rec calculateDepth (node: ThoughtNode.ThoughtNode<'T>) =
            if node.Children.Length = 0 then
                1
            else
                1 + (node.Children |> Seq.map calculateDepth |> Seq.max)
        
        calculateDepth tree.Root
    
    /// Get all nodes at a specific level
    let getNodesAtLevel (tree: ThoughtTree<'T>) (level: int) : ThoughtNode.ThoughtNode<'T> list =
        let rec collectAtLevel (node: ThoughtNode.ThoughtNode<'T>) (currentLevel: int) =
            if currentLevel = level then
                [node]
            elif currentLevel < level then
                node.Children |> Seq.collect (fun child -> collectAtLevel child (currentLevel + 1)) |> Seq.toList
            else
                []
        
        collectAtLevel tree.Root 0
    
    /// Prune nodes that don't meet a condition
    let prune (tree: ThoughtTree<'T>) (predicate: ThoughtNode.ThoughtNode<'T> -> bool) : ThoughtTree<'T> =
        let rec pruneNode (node: ThoughtNode.ThoughtNode<'T>) =
            // First prune children
            let childrenToKeep =
                node.Children
                |> List.filter predicate
                |> List.map pruneNode

            // Create new node with updated children
            { node with Children = childrenToKeep }
        
        let prunedRoot = pruneNode tree.Root
        
        // Rebuild nodes dictionary
        let newNodes = Dictionary<string, ThoughtNode.ThoughtNode<'T>>()
        traverseDepthFirst { tree with Root = prunedRoot } (fun node ->
            if node.Id <> "root" then
                newNodes.[node.Id] <- node
        )
        
        { Root = prunedRoot; Nodes = newNodes }
