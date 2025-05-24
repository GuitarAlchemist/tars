namespace TarsEngine.FSharp.Core.TreeOfThought

/// Module containing functions for working with thought trees
module ThoughtTree =
    open ThoughtNode
    
    /// Gets the depth of a tree
    let rec depth node =
        if List.isEmpty node.Children then
            1
        else
            1 + (node.Children |> List.map depth |> List.max)
    
    /// Gets the breadth of a tree at the given level
    let breadthAtLevel level node =
        let rec breadthAtLevelRec currentLevel currentNode =
            if currentLevel = level then
                1
            elif currentLevel < level then
                currentNode.Children
                |> List.sumBy (breadthAtLevelRec (currentLevel + 1))
            else
                0
        
        breadthAtLevelRec 0 node
    
    /// Gets the maximum breadth of a tree
    let maxBreadth node =
        let d = depth node
        [0..d-1]
        |> List.map (fun level -> breadthAtLevel level node)
        |> List.max
    
    /// Finds a node by its thought content
    let rec findNode thought node =
        if node.Thought = thought then
            Some node
        else
            node.Children
            |> List.tryPick (findNode thought)
    
    /// Finds all nodes that match a predicate
    let rec findNodes predicate node =
        let matches = if predicate node then [node] else []
        let childMatches = 
            node.Children
            |> List.collect (findNodes predicate)
        
        matches @ childMatches
    
    /// Selects the best node based on evaluation
    let rec selectBestNode node =
        let bestChild = 
            node.Children
            |> List.filter (fun child -> not child.Pruned)
            |> List.sortByDescending (fun child -> 
                match child.Evaluation with
                | Some eval -> eval.Overall
                | None -> 0.0)
            |> List.tryHead
        
        match bestChild with
        | Some child -> 
            let bestGrandchild = selectBestNode child
            match (child.Evaluation, bestGrandchild.Evaluation) with
            | (Some childEval, Some grandchildEval) when grandchildEval.Overall > childEval.Overall ->
                bestGrandchild
            | _ -> 
                child
        | None -> 
            node
    
    /// Prunes nodes that don't meet a threshold
    let rec pruneByThreshold threshold node =
        let prunedChildren =
            node.Children
            |> List.map (pruneByThreshold threshold)
            |> List.map (fun child ->
                match child.Evaluation with
                | Some eval when eval.Overall < threshold -> pruneNode child
                | _ -> child)
        
        { node with Children = prunedChildren }
    
    /// Prunes all but the top k nodes at each level
    let pruneBeamSearch k node =
        let rec pruneLevel nodes =
            if List.isEmpty nodes then
                []
            else
                // Sort nodes by evaluation score
                let sortedNodes =
                    nodes
                    |> List.sortByDescending (fun n -> 
                        match n.Evaluation with
                        | Some eval -> eval.Overall
                        | None -> 0.0)
                
                // Keep top k nodes, prune the rest
                let (kept, pruned) = 
                    if List.length sortedNodes <= k then
                        (sortedNodes, [])
                    else
                        let topK = sortedNodes |> List.take k
                        let rest = sortedNodes |> List.skip k |> List.map pruneNode
                        (topK, rest)
                
                // Process children of kept nodes
                let childrenOfKept = 
                    kept
                    |> List.collect (fun n -> n.Children)
                
                // Recursively prune the next level
                let prunedChildren = pruneLevel childrenOfKept
                
                // Update kept nodes with pruned children
                let updatedKept =
                    kept
                    |> List.map (fun n ->
                        let updatedChildren =
                            prunedChildren
                            |> List.filter (fun c -> 
                                n.Children 
                                |> List.exists (fun nc -> nc.Thought = c.Thought))
                        { n with Children = updatedChildren })
                
                // Return all nodes
                updatedKept @ pruned
        
        // Start pruning from the root's children
        let prunedChildren = pruneLevel node.Children
        { node with Children = prunedChildren }
    
    /// Maps a function over all nodes in a tree
    let rec mapTree f node =
        let mappedNode = f node
        let mappedChildren = 
            mappedNode.Children
            |> List.map (mapTree f)
        
        { mappedNode with Children = mappedChildren }
    
    /// Folds a function over all nodes in a tree
    let rec foldTree f state node =
        let newState = f state node
        node.Children
        |> List.fold (foldTree f) newState
    
    /// Counts the number of nodes in a tree
    let countNodes node =
        foldTree (fun count _ -> count + 1) 0 node
    
    /// Counts the number of evaluated nodes in a tree
    let countEvaluatedNodes node =
        foldTree (fun count n -> 
            if n.Evaluation.IsSome then count + 1 else count) 0 node
    
    /// Counts the number of pruned nodes in a tree
    let countPrunedNodes node =
        foldTree (fun count n -> 
            if n.Pruned then count + 1 else count) 0 node
