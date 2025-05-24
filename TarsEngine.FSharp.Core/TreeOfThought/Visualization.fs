namespace TarsEngine.FSharp.Core.TreeOfThought

/// Module containing functions for visualizing thought trees
module Visualization =
    open ThoughtNode
    open Newtonsoft.Json
    
    /// Converts a tree to JSON
    let rec toJson node =
        let childrenJson = 
            node.Children
            |> List.map toJson
            |> String.concat ", "
        
        let evaluationJson =
            match node.Evaluation with
            | Some eval -> 
                sprintf """
                "evaluation": {
                    "correctness": %.2f,
                    "efficiency": %.2f,
                    "robustness": %.2f,
                    "maintainability": %.2f,
                    "overall": %.2f
                }"""
                    eval.Correctness
                    eval.Efficiency
                    eval.Robustness
                    eval.Maintainability
                    eval.Overall
            | None -> 
                "\"evaluation\": null"
        
        let metadataJson =
            if Map.isEmpty node.Metadata then
                "\"metadata\": {}"
            else
                let metadataItems =
                    node.Metadata
                    |> Map.toList
                    |> List.map (fun (k, v) -> 
                        sprintf "\"%s\": \"%s\"" k (v.ToString()))
                    |> String.concat ", "
                
                sprintf "\"metadata\": { %s }" metadataItems
        
        sprintf """
        {
            "thought": "%s",
            %s,
            "pruned": %b,
            %s,
            "children": [%s]
        }
        """
            node.Thought
            evaluationJson
            node.Pruned
            metadataJson
            childrenJson
    
    /// Converts a tree to a formatted JSON string
    let toFormattedJson node =
        let json = toJson node
        let parsedJson = JsonConvert.DeserializeObject(json)
        JsonConvert.SerializeObject(parsedJson, Formatting.Indented)
    
    /// Converts a tree to a Markdown representation
    let rec toMarkdown node level =
        let indent = String.replicate level "  "
        let score = 
            match node.Evaluation with
            | Some eval -> sprintf " (Score: %.2f)" eval.Overall
            | None -> ""
        
        let pruned = if node.Pruned then " [PRUNED]" else ""
        
        let nodeMarkdown = sprintf "%s- %s%s%s\n" indent node.Thought score pruned
        
        let childrenMarkdown = 
            node.Children
            |> List.map (fun child -> toMarkdown child (level + 1))
            |> String.concat ""
        
        nodeMarkdown + childrenMarkdown
    
    /// Converts a tree to a Markdown report
    let toMarkdownReport node title =
        let header = sprintf "# %s\n\n" title
        let treeMarkdown = toMarkdown node 0
        let summary = 
            sprintf """
## Summary

- Total nodes: %d
- Evaluated nodes: %d
- Pruned nodes: %d
- Maximum depth: %d
- Maximum breadth: %d

## Tree Structure

%s
"""
                (ThoughtTree.countNodes node)
                (ThoughtTree.countEvaluatedNodes node)
                (ThoughtTree.countPrunedNodes node)
                (ThoughtTree.depth node)
                (ThoughtTree.maxBreadth node)
                treeMarkdown
        
        header + summary
    
    /// Converts a tree to a DOT graph representation for Graphviz
    let toDotGraph node title =
        let sb = System.Text.StringBuilder()
        
        // Add header
        sb.AppendLine(sprintf "digraph \"%s\" {" title) |> ignore
        sb.AppendLine("  node [shape=box, style=filled, fontname=\"Arial\"];") |> ignore
        
        // Add nodes
        let rec addNodes nodeId node =
            let label = 
                match node.Evaluation with
                | Some eval -> sprintf "%s\\nScore: %.2f" node.Thought eval.Overall
                | None -> node.Thought
            
            let color = 
                if node.Pruned then
                    "lightgray"
                else
                    match node.Evaluation with
                    | Some eval ->
                        if eval.Overall > 0.8 then "lightgreen"
                        elif eval.Overall > 0.6 then "lightyellow"
                        elif eval.Overall > 0.4 then "lightblue"
                        else "lightcoral"
                    | None -> "white"
            
            sb.AppendLine(sprintf "  %d [label=\"%s\", fillcolor=\"%s\"];" nodeId label color) |> ignore
            
            // Process children
            let mutable nextId = nodeId + 1
            for child in node.Children do
                sb.AppendLine(sprintf "  %d -> %d;" nodeId nextId) |> ignore
                nextId <- addNodes nextId child
            
            nextId
        
        // Add edges by traversing the tree
        addNodes 0 node |> ignore
        
        // Add footer
        sb.AppendLine("}") |> ignore
        
        sb.ToString()
    
    /// Saves a DOT graph to a file
    let saveDotGraph node title filePath =
        let dotGraph = toDotGraph node title
        System.IO.File.WriteAllText(filePath, dotGraph)
    
    /// Saves a Markdown report to a file
    let saveMarkdownReport node title filePath =
        let report = toMarkdownReport node title
        System.IO.File.WriteAllText(filePath, report)
    
    /// Saves a JSON representation to a file
    let saveJsonReport node filePath =
        let json = toFormattedJson node
        System.IO.File.WriteAllText(filePath, json)
