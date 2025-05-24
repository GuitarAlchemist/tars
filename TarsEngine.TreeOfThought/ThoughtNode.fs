namespace TarsEngine.TreeOfThought

/// Module containing types and functions for working with thought nodes
module ThoughtNode =
    
    /// Represents a node in a thought tree
    type ThoughtNode = {
        /// The thought content
        Thought: string
        /// Child nodes
        Children: ThoughtNode list
        /// Evaluation score (0.0 to 1.0)
        Score: float
        /// Whether the node has been pruned
        Pruned: bool
        /// Additional metadata
        Metadata: Map<string, obj>
    }
    
    /// Creates a new thought node
    let createNode thought =
        { Thought = thought
          Children = []
          Score = 0.0
          Pruned = false
          Metadata = Map.empty }
    
    /// Adds a child to a node
    let addChild parent child =
        { parent with Children = child :: parent.Children }
    
    /// Evaluates a node with a score
    let evaluateNode node score =
        { node with Score = score }
    
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
    
    /// Converts a node to a string
    let toString node =
        sprintf "Thought: %s, Score: %.2f, Pruned: %b, Children: %d" 
            node.Thought node.Score node.Pruned node.Children.Length
