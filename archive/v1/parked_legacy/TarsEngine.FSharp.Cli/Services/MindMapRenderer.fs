namespace TarsEngine.FSharp.Cli.Services

open System
open System.Text

/// Service for rendering mind maps in various formats
type MindMapRenderer() =
    
    /// Render Markdown mind map with diagrams
    member this.RenderMarkdownMindMap(mindMap: KnowledgeMindMap, centerTopic: string, includeContent: bool, includeMermaid: bool, allKnowledge: LearnedKnowledge list) =
        let sb = StringBuilder()
        
        // Header
        sb.AppendLine("# 🧠 TARS Knowledge Mind Map") |> ignore
        sb.AppendLine() |> ignore
        sb.AppendLine(sprintf "**Center Topic:** %s" centerTopic) |> ignore
        sb.AppendLine(sprintf "**Generated:** %s" (DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC"))) |> ignore
        sb.AppendLine(sprintf "**Total Nodes:** %d | **Max Depth:** %d | **Knowledge Entries:** %d" mindMap.Nodes.Length mindMap.MaxDepth mindMap.TotalKnowledge) |> ignore
        sb.AppendLine() |> ignore
        
        // Mermaid diagram if requested
        if includeMermaid then
            sb.AppendLine("## 📊 Knowledge Graph Visualization") |> ignore
            sb.AppendLine() |> ignore
            sb.AppendLine("```mermaid") |> ignore
            sb.AppendLine("graph TD") |> ignore
            
            // Add center node
            sb.AppendLine(sprintf "    center[\"%s\"]" centerTopic) |> ignore
            sb.AppendLine("    center:::centerNode") |> ignore
            
            // Add all nodes and edges
            for node in mindMap.Nodes do
                if node.Level > 0 then
                    let nodeId = node.Id.Replace("-", "_").Replace(" ", "_")
                    let confidencePercent = int (node.Confidence * 100.0)
                    sb.AppendLine(sprintf "    %s[\"%s<br/>📊 %d%% confidence<br/>📚 %d items\"]" 
                        nodeId node.Topic confidencePercent node.Knowledge.Length) |> ignore
                    
                    // Add styling based on confidence
                    let styleClass = 
                        if node.Confidence > 0.8 then "highConf"
                        elif node.Confidence > 0.6 then "medConf"
                        else "lowConf"
                    sb.AppendLine(sprintf "    %s:::%s" nodeId styleClass) |> ignore
            
            // Add edges
            for edge in mindMap.Edges do
                let fromId = edge.From.Replace("-", "_").Replace(" ", "_")
                let toId = edge.To.Replace("-", "_").Replace(" ", "_")
                let strengthPercent = int (edge.Strength * 100.0)
                sb.AppendLine(sprintf "    %s -->|%d%%| %s" fromId strengthPercent toId) |> ignore
            
            // Add styling
            sb.AppendLine() |> ignore
            sb.AppendLine("    classDef centerNode fill:#ff6b6b,stroke:#333,stroke-width:3px,color:#fff") |> ignore
            sb.AppendLine("    classDef highConf fill:#51cf66,stroke:#333,stroke-width:2px") |> ignore
            sb.AppendLine("    classDef medConf fill:#ffd43b,stroke:#333,stroke-width:2px") |> ignore
            sb.AppendLine("    classDef lowConf fill:#ff8787,stroke:#333,stroke-width:2px") |> ignore
            sb.AppendLine("```") |> ignore
            sb.AppendLine() |> ignore
        
        // Knowledge hierarchy
        sb.AppendLine("## 🌳 Knowledge Hierarchy") |> ignore
        sb.AppendLine() |> ignore
        
        let nodesByLevel = 
            mindMap.Nodes 
            |> List.groupBy (fun n -> n.Level)
            |> List.sortBy fst
        
        for (level, nodes) in nodesByLevel do
            if level = 0 then
                sb.AppendLine(sprintf "### 🎯 %s (Center)" centerTopic) |> ignore
                sb.AppendLine() |> ignore
                let centerKnowledge = nodes |> List.head |> fun n -> n.Knowledge
                sb.AppendLine(sprintf "- **Knowledge Items:** %d" centerKnowledge.Length) |> ignore
                sb.AppendLine(sprintf "- **Average Confidence:** %.1f%%" (centerKnowledge |> List.averageBy (fun k -> k.Confidence) |> fun x -> x * 100.0)) |> ignore
                sb.AppendLine() |> ignore
            else
                sb.AppendLine(sprintf "### Level %d" level) |> ignore
                sb.AppendLine() |> ignore
                
                for node in nodes |> List.sortByDescending (fun n -> n.Confidence) do
                    let confidencePercent = node.Confidence * 100.0
                    let confidenceEmoji = 
                        if node.Confidence > 0.8 then "🟢"
                        elif node.Confidence > 0.6 then "🟡"
                        else "🔴"
                    
                    sb.AppendLine(sprintf "#### %s %s" confidenceEmoji node.Topic) |> ignore
                    sb.AppendLine() |> ignore
                    sb.AppendLine(sprintf "- **Confidence:** %.1f%%" confidencePercent) |> ignore
                    sb.AppendLine(sprintf "- **Knowledge Items:** %d" node.Knowledge.Length) |> ignore
                    sb.AppendLine(sprintf "- **Connection Strength:** %.1f%%" (node.ConnectionStrength * 100.0)) |> ignore
                    
                    // Top tags
                    let topTags = 
                        node.Knowledge 
                        |> List.collect (fun k -> k.Tags)
                        |> List.groupBy id
                        |> List.map (fun (tag, items) -> (tag, items.Length))
                        |> List.sortByDescending snd
                        |> List.take (min 5 (node.Knowledge |> List.collect (fun k -> k.Tags) |> List.distinct |> List.length))
                    
                    if not topTags.IsEmpty then
                        sb.AppendLine("- **Top Tags:**") |> ignore
                        for (tag, count) in topTags do
                            sb.AppendLine(sprintf "  - `%s` (%d)" tag count) |> ignore
                    
                    // Include content if requested
                    if includeContent && not node.Knowledge.IsEmpty then
                        sb.AppendLine("- **Knowledge Details:**") |> ignore
                        for knowledge in node.Knowledge |> List.take (min 3 node.Knowledge.Length) do
                            let preview = 
                                if knowledge.Content.Length > 150 then
                                    knowledge.Content.Substring(0, 150) + "..."
                                else knowledge.Content
                            sb.AppendLine(sprintf "  - **%s:** %s" knowledge.Topic preview) |> ignore
                            sb.AppendLine(sprintf "    - *Source:* %s" knowledge.Source) |> ignore
                            sb.AppendLine(sprintf "    - *Learned:* %s" (knowledge.LearnedAt.ToString("yyyy-MM-dd"))) |> ignore
                    
                    sb.AppendLine() |> ignore
        
        sb.ToString()
    
    /// Render mind map as JSON for API consumption
    member this.RenderJsonMindMap(mindMap: KnowledgeMindMap) =
        // TODO: Implement JSON serialization
        sprintf """
{
  "centerTopic": "%s",
  "totalNodes": %d,
  "maxDepth": %d,
  "totalKnowledge": %d,
  "nodes": [
    %s
  ],
  "edges": [
    %s
  ]
}
""" 
            mindMap.CenterTopic 
            mindMap.Nodes.Length 
            mindMap.MaxDepth 
            mindMap.TotalKnowledge
            (mindMap.Nodes |> List.map (fun n -> sprintf """{"id": "%s", "topic": "%s", "level": %d, "confidence": %.2f}""" n.Id n.Topic n.Level n.Confidence) |> String.concat ",\n    ")
            (mindMap.Edges |> List.map (fun e -> sprintf """{"from": "%s", "to": "%s", "strength": %.2f}""" e.From e.To e.Strength) |> String.concat ",\n    ")
    
    /// Render mind map as DOT format for Graphviz
    member this.RenderDotMindMap(mindMap: KnowledgeMindMap) =
        let sb = StringBuilder()
        
        sb.AppendLine("digraph KnowledgeMindMap {") |> ignore
        sb.AppendLine("  rankdir=TB;") |> ignore
        sb.AppendLine("  node [shape=box, style=rounded];") |> ignore
        sb.AppendLine() |> ignore
        
        // Add nodes
        for node in mindMap.Nodes do
            let color = 
                if node.Confidence > 0.8 then "lightgreen"
                elif node.Confidence > 0.6 then "lightyellow"
                else "lightcoral"
            
            sb.AppendLine(sprintf "  \"%s\" [label=\"%s\\n%.1f%% confidence\\n%d items\", fillcolor=%s, style=filled];" 
                node.Id node.Topic (node.Confidence * 100.0) node.Knowledge.Length color) |> ignore
        
        sb.AppendLine() |> ignore
        
        // Add edges
        for edge in mindMap.Edges do
            let weight = edge.Strength * 10.0 |> int |> max 1
            sb.AppendLine(sprintf "  \"%s\" -> \"%s\" [weight=%d, label=\"%.2f\"];" 
                edge.From edge.To weight edge.Strength) |> ignore
        
        sb.AppendLine("}") |> ignore
        
        sb.ToString()
    
    /// Export mind map to various formats
    member this.ExportMindMap(mindMap: KnowledgeMindMap, format: string, filePath: string) =
        let content = 
            match format.ToLowerInvariant() with
            | "json" -> this.RenderJsonMindMap(mindMap)
            | "dot" | "graphviz" -> this.RenderDotMindMap(mindMap)
            | "markdown" | "md" -> this.RenderMarkdownMindMap(mindMap, mindMap.CenterTopic, true, true, [])
            | _ -> failwith $"Unsupported format: {format}"
        
        System.IO.File.WriteAllText(filePath, content)
        filePath
