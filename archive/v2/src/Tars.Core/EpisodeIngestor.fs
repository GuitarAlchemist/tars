namespace Tars.Core

/// <summary>
/// Handles logical ingestion of Episodes into the Temporal Knowledge Graph.
/// Extracts entities and facts from raw episodic data.
/// </summary>
module EpisodeIngestor =
    
    /// Extract facts and nodes from an episode and add them to the graph.
    let ingestEpisode (graph: TemporalKnowledgeGraph.TemporalGraph) (episode: Episode) =
        // 1. Add Episode as a foundational node
        let episodeEntity = EpisodeE episode
        let episodeId = graph.AddNode(episodeEntity)
        
        // 2. Extract facts based on type
        match episode with
        | AgentInteraction(agentId, input, output, ts) ->
            // Use common names for physical concepts in the graph
            let agent = ConceptE { Name = $"Agent:{agentId}"; Description = "Agent actor"; RelatedConcepts = [] }
            graph.AddFact(Contains(agent, episodeEntity)) |> ignore
            
        | BeliefUpdate(agentId, statement, confidence, ts) ->
            let newBelief = AgentBeliefE {
                Statement = statement
                Confidence = confidence
                DerivedFrom = [episodeId]
                AgentId = agentId
                ValidFrom = ts
                InvalidAt = None
            }
            
            // Evolution detection: Look for a belief with the same statement by the same agent
            let existing = 
                graph.GetAllNodes() 
                |> List.tryFind (function 
                    | AgentBeliefE b when b.Statement = statement && b.AgentId = agentId -> true 
                    | _ -> false)
            
            match existing with
            | Some oldBelief ->
                graph.AddFact(EvolvedFrom(newBelief, oldBelief, "Updated belief confidence or provenance")) |> ignore
            | None ->
                graph.AddFact(BelongsTo(newBelief, $"agent:{agentId}")) |> ignore
                
        | CodeChange(file, diff, author, ts) ->
            let fileEntity = FileE file
            let authorEntity = ConceptE { Name = $"Author:{author}"; Description = "Code contributor"; RelatedConcepts = [] }
            
            graph.AddFact(DerivedFrom(fileEntity, episodeEntity)) |> ignore
            graph.AddFact(Contains(authorEntity, episodeEntity)) |> ignore
            
        | ToolCall(name, args, result, ts) ->
            let toolEntity = ConceptE { Name = $"Tool:{name}"; Description = "External tool capability"; RelatedConcepts = [] }
            graph.AddFact(DerivedFrom(episodeEntity, toolEntity)) |> ignore
            
        | Reflection(agentId, content, ts) ->
            let agent = ConceptE { Name = $"Agent:{agentId}"; Description = "Agent actor"; RelatedConcepts = [] }
            graph.AddFact(Contains(agent, episodeEntity)) |> ignore
            
        | UserMessage(content, metadata, ts) ->
            // User messages are evidence
            metadata |> Map.iter (fun key value ->
                let metaConcept = ConceptE { Name = $"{key}:{value}"; Description = "Message metadata"; RelatedConcepts = [] }
                graph.AddFact(SimilarTo(episodeEntity, metaConcept, 1.0)) |> ignore
            )
            
        | PatternDetected(pType, location, details, ts) ->
            let pattern = CodePatternE {
                Name = pType
                Category = Structural // Default
                Signature = details
                Occurrences = 1
                FirstSeen = ts
                LastSeen = ts
            }
            let locEntity = ConceptE { Name = location; Description = "Pattern location"; RelatedConcepts = [] }
            graph.AddFact(Contains(locEntity, episodeEntity)) |> ignore
            graph.AddFact(DerivedFrom(pattern, episodeEntity)) |> ignore

        | CognitiveStateUpdate(runId, mode, entropy, stability, ts) ->
            let runEntity = ConceptE { Name = $"Run:{runId}"; Description = "WoT Execution Run"; RelatedConcepts = [] }
            let modeEntity = ConceptE { Name = $"Mode:{mode}"; Description = "Cognitive Mode"; RelatedConcepts = [] }
            
            // Connect episode to Run and Mode
            graph.AddFact(Contains(runEntity, episodeEntity)) |> ignore
            graph.AddFact(DerivedFrom(episodeEntity, modeEntity)) |> ignore // Use DerivedFrom to link state to mode


        episodeId
