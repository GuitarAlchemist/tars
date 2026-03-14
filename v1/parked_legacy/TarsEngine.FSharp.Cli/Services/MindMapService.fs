namespace TarsEngine.FSharp.Cli.Services

open System
open System.Text
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Services.RDF

/// Service for generating knowledge mind maps
type MindMapService(logger: ILogger<MindMapService>, rdfClient: IRdfClient option) =

    let renderer = new MindMapRenderer()
    
    /// Generate ASCII mind map for CLI display
    member this.GenerateAsciiMindMap(allKnowledge: LearnedKnowledge list, centralTopic: string option, maxDepth: int, maxNodes: int) =
        async {
            logger.LogInformation("🧠 MIND MAP: Generating ASCII mind map for knowledge visualization")
            
            // Determine central topic
            let centerTopic = 
                match centralTopic with
                | Some topic -> topic
                | None -> 
                    // Find most connected topic
                    allKnowledge
                    |> List.groupBy (fun k -> k.Tags |> List.tryHead |> Option.defaultValue "General")
                    |> List.maxBy (fun (_, items) -> items.Length)
                    |> fst
            
            // Build knowledge graph
            let knowledgeGraph = this.BuildKnowledgeGraph(allKnowledge, centerTopic, maxDepth, maxNodes)
            
            // Generate ASCII representation
            let asciiMap = this.RenderAsciiMindMap(knowledgeGraph, centerTopic)
            
            logger.LogInformation("✅ MIND MAP: Generated ASCII mind map with {NodeCount} nodes", knowledgeGraph.Nodes.Length)
            return asciiMap
        }

    /// Generate detailed Markdown mind map with diagrams
    member this.GenerateMarkdownMindMap(allKnowledge: LearnedKnowledge list, centralTopic: string option, includeContent: bool, includeMermaid: bool) =
        async {
            logger.LogInformation("📝 MIND MAP: Generating Markdown mind map with diagrams")
            
            // Determine central topic
            let centerTopic = 
                match centralTopic with
                | Some topic -> topic
                | None -> "TARS Knowledge Base"
            
            // Build comprehensive knowledge graph
            let knowledgeGraph = this.BuildKnowledgeGraph(allKnowledge, centerTopic, 5, 50)
            
            // Generate Markdown content
            let markdown = renderer.RenderMarkdownMindMap(knowledgeGraph, centerTopic, includeContent, includeMermaid, allKnowledge)
            
            logger.LogInformation("✅ MIND MAP: Generated Markdown mind map with {NodeCount} nodes", knowledgeGraph.Nodes.Length)
            return markdown
        }

    /// Build knowledge graph structure using RDF relationships when available
    member private this.BuildKnowledgeGraph(allKnowledge: LearnedKnowledge list, centerTopic: string, maxDepth: int, maxNodes: int) =
        let nodes = System.Collections.Generic.List<MindMapNode>()
        let edges = System.Collections.Generic.List<MindMapEdge>()
        let processedTopics = System.Collections.Generic.HashSet<string>()
        
        // Add center node
        let centerNode = {
            Id = "center"
            Topic = centerTopic
            Level = 0
            Knowledge = allKnowledge |> List.filter (fun k -> k.Tags |> List.exists (fun tag -> tag.ToLowerInvariant().Contains(centerTopic.ToLowerInvariant())))
            Confidence = 1.0
            ConnectionStrength = 1.0
        }
        nodes.Add(centerNode)
        processedTopics.Add(centerTopic.ToLowerInvariant()) |> ignore
        
        // Build graph level by level
        let rec buildLevel currentLevel nodeQueue =
            if currentLevel >= maxDepth || nodes.Count >= maxNodes || nodeQueue |> List.isEmpty then
                ()
            else
                let nextQueue = System.Collections.Generic.List<string * LearnedKnowledge list>()
                
                for (parentTopic : string, parentKnowledge : LearnedKnowledge list) in nodeQueue do
                    // Find related knowledge through RDF relationships, tags, and content similarity
                    let relatedKnowledge =
                        async {
                            // First try to get RDF-based relationships if RDF client is available
                            let! rdfRelated = 
                                match rdfClient with
                                | Some client ->
                                    async {
                                        try
                                            let parentUri = sprintf "http://tars.ai/ontology#knowledge/%s" (parentTopic.Replace(" ", "_"))
                                            let! result = client.GetRelatedKnowledge(parentUri) |> Async.AwaitTask
                                            match result with
                                            | Ok relatedUris -> 
                                                logger.LogInformation("🔗 RDF: Found {Count} related knowledge entries via RDF", relatedUris.Length)
                                                return relatedUris |> List.map (fun r -> r.Topic)
                                            | Error _ -> return []
                                        with
                                        | ex ->
                                            logger.LogWarning(ex, "⚠️ RDF: Failed to query related knowledge")
                                            return []
                                    }
                                | None -> async { return [] }
                            
                            // Combine RDF relationships with traditional tag/content matching
                            let traditionalRelated =
                                allKnowledge
                                |> List.filter (fun k ->
                                    not (processedTopics.Contains(k.Topic.ToLowerInvariant() : string)) &&
                                    (parentKnowledge |> List.exists (fun pk ->
                                        k.Tags |> List.exists (fun tag -> pk.Tags |> List.contains tag) ||
                                        let kContent : string = k.Content.ToLowerInvariant()
                                        let kTopic : string = k.Topic.ToLowerInvariant()
                                        let parentTopicLower : string = parentTopic.ToLowerInvariant()
                                        kContent.Contains(parentTopicLower) ||
                                        parentTopicLower.Contains(kTopic) ||
                                        rdfRelated |> List.contains k.Topic))) // Include RDF-discovered relationships
                                |> List.groupBy (fun k -> k.Topic)
                                |> fun grouped ->
                                    let maxTake = min 5 (maxNodes - nodes.Count)
                                    if grouped.Length <= maxTake then grouped
                                    else grouped |> List.take maxTake
                            
                            return traditionalRelated
                        } |> Async.RunSynchronously

                    for (topic, knowledgeGroup) in relatedKnowledge do
                        if not (processedTopics.Contains(topic.ToLowerInvariant() : string)) then
                            let avgConfidence = knowledgeGroup |> List.averageBy (fun k -> k.Confidence)
                            let connectionStrength = 
                                knowledgeGroup 
                                |> List.sumBy (fun k -> 
                                    let tagOverlap = 
                                        parentKnowledge 
                                        |> List.sumBy (fun pk -> 
                                            k.Tags |> List.filter (fun tag -> pk.Tags |> List.contains tag) |> List.length)
                                    float tagOverlap / float (max 1 k.Tags.Length))
                                |> fun sum -> sum / float knowledgeGroup.Length
                            
                            let node = {
                                Id = sprintf "node_%d_%s" currentLevel (topic.Replace(" ", "_"))
                                Topic = topic
                                Level = currentLevel
                                Knowledge = knowledgeGroup
                                Confidence = avgConfidence
                                ConnectionStrength = connectionStrength
                            }
                            nodes.Add(node)
                            processedTopics.Add(topic.ToLowerInvariant()) |> ignore
                            
                            // Add edge from parent
                            let parentNodeId = 
                                if currentLevel = 1 then "center"
                                else sprintf "node_%d_%s" (currentLevel - 1) ((parentTopic : string).Replace(" ", "_"))
                            
                            let edge = {
                                From = parentNodeId
                                To = node.Id
                                Strength = connectionStrength
                                RelationType = "related_to"
                            }
                            edges.Add(edge)
                            
                            nextQueue.Add((topic, knowledgeGroup))
                
                buildLevel (currentLevel + 1) (nextQueue |> Seq.toList)
        
        buildLevel 1 [(centerTopic, centerNode.Knowledge)]
        
        {
            CenterTopic = centerTopic
            Nodes = nodes |> Seq.toList
            Edges = edges |> Seq.toList
            MaxDepth = maxDepth
            TotalKnowledge = allKnowledge.Length
        }

    /// Render ASCII mind map for CLI display
    member private this.RenderAsciiMindMap(mindMap: KnowledgeMindMap, centerTopic: string) =
        let sb = StringBuilder()
        
        // Header
        sb.AppendLine("╔══════════════════════════════════════════════════════════════════════════════╗") |> ignore
        sb.AppendLine(sprintf "║                           🧠 TARS KNOWLEDGE MIND MAP 🧠                      ║") |> ignore
        sb.AppendLine("╠══════════════════════════════════════════════════════════════════════════════╣") |> ignore
        sb.AppendLine(sprintf "║ Center Topic: %-62s ║" centerTopic) |> ignore
        sb.AppendLine(sprintf "║ Total Nodes: %-3d | Max Depth: %-2d | Knowledge Entries: %-4d           ║" mindMap.Nodes.Length mindMap.MaxDepth mindMap.TotalKnowledge) |> ignore
        sb.AppendLine("╚══════════════════════════════════════════════════════════════════════════════╝") |> ignore
        sb.AppendLine() |> ignore
        
        // Group nodes by level
        let nodesByLevel = 
            mindMap.Nodes 
            |> List.groupBy (fun n -> n.Level)
            |> List.sortBy fst
        
        // Render each level
        for (level, nodes) in nodesByLevel do
            if level = 0 then
                // Center node
                sb.AppendLine(sprintf "                                🎯 %s" centerTopic) |> ignore
                sb.AppendLine("                                      │") |> ignore
            else
                // Calculate indentation and spacing
                let indent = String.replicate (level * 4) " "
                let connector = if level = 1 then "├─" else "└─"
                
                sb.AppendLine(sprintf "%s%s Level %d:" indent connector level) |> ignore
                
                for (i, node) in nodes |> List.mapi (fun i n -> (i, n)) do
                    let isLast = i = nodes.Length - 1
                    let nodeConnector = if isLast then "└─" else "├─"
                    let confidenceBar = this.CreateConfidenceBar(node.Confidence)
                    let knowledgeCount = node.Knowledge.Length
                    
                    sb.AppendLine(sprintf "%s  %s 📚 %s %s (%d items)" 
                        indent nodeConnector node.Topic confidenceBar knowledgeCount) |> ignore
                    
                    // Show top tags for this node
                    let topTags = 
                        node.Knowledge 
                        |> List.collect (fun k -> k.Tags)
                        |> List.groupBy id
                        |> List.map (fun (tag, items) -> (tag, items.Length))
                        |> List.sortByDescending snd
                        |> fun tagList ->
                            let maxTake = min 3 (node.Knowledge |> List.collect (fun k -> k.Tags) |> List.distinct |> List.length)
                            if tagList.Length <= maxTake then tagList
                            else tagList |> List.take maxTake
                        |> List.map fst
                    
                    if not topTags.IsEmpty then
                        let tagStr = String.concat ", " topTags
                        sb.AppendLine(sprintf "%s     🏷️  %s" indent tagStr) |> ignore
        
        sb.AppendLine() |> ignore
        sb.AppendLine("Legend: 🎯 Center Topic | 📚 Knowledge Node | 🏷️ Tags") |> ignore
        sb.AppendLine("Confidence: ████████ = High | ████░░░░ = Medium | ██░░░░░░ = Low") |> ignore
        
        sb.ToString()

    /// Create confidence visualization bar
    member private this.CreateConfidenceBar(confidence: float) =
        let barLength = 8
        let filledLength = int (confidence * float barLength)
        let filled = String.replicate filledLength "█"
        let empty = String.replicate (barLength - filledLength) "░"
        sprintf "[%s%s]" filled empty
