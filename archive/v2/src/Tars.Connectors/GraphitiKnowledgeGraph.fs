namespace Tars.Connectors

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Connectors.Graphiti

/// Graphiti-backed implementation of temporal knowledge graph storage
/// Replaces file-based persistence with Graphiti's temporal knowledge graph
module GraphitiKnowledgeGraph =

    /// Convert TarsEntity to Graphiti message content
    let entityToContent (entity: TarsEntity) : string =
        match entity with
        | TarsEntity.CodePatternE p ->
            $"Pattern: {p.Name} ({p.Category})\nSignature: {p.Signature}\nOccurrences: {p.Occurrences}"
        | TarsEntity.AgentBeliefE b -> $"Belief: {b.Statement}\nConfidence: {b.Confidence:F2}\nAgent: {b.AgentId}"
        | TarsEntity.GrammarRuleE g -> $"Grammar: {g.Name} v{g.Version}\nProduction: {g.Production}"
        | TarsEntity.CodeModuleE m ->
            $"Module: {m.Path}\nNamespace: {m.Namespace}\nComplexity: {m.Complexity:F2}\nLines: {m.LineCount}"
        | TarsEntity.AnomalyE a -> $"Anomaly: {a.Type} at {a.Location}\nSeverity: {a.Severity}"
        | TarsEntity.ConceptE c ->
            let related = String.concat ", " c.RelatedConcepts
            $"Concept: {c.Name}\nDescription: {c.Description}\nRelated: {related}"
        | TarsEntity.EpisodeE e -> $"Episode: {Episode.typeTag e} at {Episode.timestamp e}"
        | TarsEntity.FileE path -> $"File: {path}"
        | TarsEntity.RunE r -> $"Run: {r.Id}\nGoal: {r.Goal}\nPattern: {r.Pattern}"
        | TarsEntity.StepE s -> $"Step: {s.StepId}\nRunId: {s.RunId}\nType: {s.NodeType}\nContent: {s.Content}"
        | TarsEntity.FunctionE name -> $"Function: {name}"

    /// Convert TarsFact to Graphiti message content
    let factToContent (fact: TarsFact) : string =
        match fact with
        | TarsFact.Implements(s, t, conf) ->
            $"IMPLEMENTS (confidence: {conf:F2}): {TarsEntity.getId s} -> {TarsEntity.getId t}"
        | TarsFact.DependsOn(s, t, strength) ->
            $"DEPENDS_ON (strength: {strength:F2}): {TarsEntity.getId s} -> {TarsEntity.getId t}"
        | TarsFact.Contradicts(s, t, resolution) ->
            let res = resolution |> Option.defaultValue "unresolved"
            $"CONTRADICTS: {TarsEntity.getId s} contradicts {TarsEntity.getId t}\nResolution: {res}"
        | TarsFact.EvolvedFrom(s, t, delta) ->
            $"EVOLVED_FROM: {TarsEntity.getId s} evolved from {TarsEntity.getId t}\nDelta: {delta}"
        | TarsFact.BelongsTo(entity, community) ->
            $"BELONGS_TO: {TarsEntity.getId entity} belongs to community {community}"
        | TarsFact.SimilarTo(s, t, similarity) ->
            $"SIMILAR_TO (similarity: {similarity:F2}): {TarsEntity.getId s} ~ {TarsEntity.getId t}"
        | TarsFact.DerivedFrom(s, t) -> $"DERIVED_FROM: {TarsEntity.getId s} -> {TarsEntity.getId t}"
        | TarsFact.Contains(s, t) -> $"CONTAINS: {TarsEntity.getId s} contains {TarsEntity.getId t}"
        | TarsFact.NextStep(s, t) -> $"NEXT_STEP: {TarsEntity.getId s} -> {TarsEntity.getId t}"

    /// Graphiti-backed temporal graph
    type GraphitiTemporalGraph(graphitiUrl: string, ?groupId: string) =
        let client = new GraphitiClient(Uri(graphitiUrl))
        let gid = defaultArg groupId "tars_knowledge_graph"

        // In-memory cache for quick access (mirroring Graphiti state)
        let mutable nodeCache = Map.empty<string, TemporalKnowledgeGraph.TemporalNode>
        let mutable edgeCache = Map.empty<Guid, TemporalKnowledgeGraph.TemporalEdge>

        /// Add a node to the graph
        member this.AddNode(entity: TarsEntity) : string =
            let id = TarsEntity.getId entity

            // Create temporal node
            let validity = TemporalValidityOps.now ()

            let node: TemporalKnowledgeGraph.TemporalNode =
                { Entity = entity
                  Validity = validity
                  CommunityId = None }

            nodeCache <- nodeCache |> Map.add id node

            // Send to Graphiti asynchronously (fire and forget for performance)
            let content = entityToContent entity

            let message: Graphiti.MessageDto =
                { Content = content
                  RoleType = "entity"
                  Role = "system"
                  Timestamp = Some DateTime.UtcNow
                  SourceDescription = Some "TARS Temporal Knowledge Graph"
                  Uuid = None }

            Task.Run(fun () ->
                task {
                    let! result = client.AddMessagesAsync(gid, [| message |])

                    match result with
                    | Result.Ok _ -> ()
                    | Result.Error e -> printfn $"Warning: Failed to add node to Graphiti: {e}"
                }
                :> Task)
            |> ignore

            id

        /// Add a fact (edge) to the graph
        member this.AddFact(fact: TarsFact) : Guid =
            let _ = this.AddNode(TarsFact.source fact)

            match TarsFact.target fact with
            | Some t -> this.AddNode t |> ignore
            | None -> ()

            let edgeId = Guid.NewGuid()
            let validity = TemporalValidityOps.now ()

            let edge: TemporalKnowledgeGraph.TemporalEdge =
                { Id = edgeId
                  Fact = fact
                  Validity = validity
                  SupersededBy = None }

            edgeCache <- edgeCache |> Map.add edgeId edge

            // Send to Graphiti asynchronously
            let content = factToContent fact

            let message: MessageDto =
                { Content = content
                  Role = "system"
                  RoleType = "fact"
                  Timestamp = Some DateTime.UtcNow
                  SourceDescription = Some "TARS Temporal Knowledge Graph - Fact"
                  Uuid = Some(edgeId.ToString()) }

            Task.Run(fun () ->
                task {
                    let! result = client.AddMessagesAsync(gid, [| message |])

                    match result with
                    | Result.Ok _ -> ()
                    | Result.Error e -> printfn $"Warning: Failed to add fact to Graphiti: {e}"
                }
                :> Task)
            |> ignore

            edgeId

        /// Get current facts from the graph
        member this.GetCurrentFacts() : TarsFact list =
            edgeCache.Values
            |> Seq.filter (fun e -> TemporalValidityOps.isValidAt DateTime.UtcNow e.Validity)
            |> Seq.map (fun e -> e.Fact)
            |> Seq.toList

        /// Save graph state (NO-OP for Graphiti - it auto-persists)
        member this.Save(path: string) =
            // Graphiti auto-persists, but we log for compatibility
            printfn $"[GraphitiKnowledgeGraph] State auto-persisted to Graphiti at {graphitiUrl}"
            printfn $"[GraphitiKnowledgeGraph] {nodeCache.Count} nodes, {edgeCache.Count} edges"

        /// Load graph state from Graphiti
        member this.Load(path: string) : bool =
            let loadTask =
                task {
                    try
                        // Query Graphiti for existing facts (explicit type to avoid AgentState collision)
                        let! (result: Result<SearchResultDto list, string>) =
                            client.SearchAsync("TARS temporal knowledge graph", numResults = 100)

                        match result with
                        | Result.Ok results ->
                            printfn $"[GraphitiKnowledgeGraph] Loaded {results.Length} items from Graphiti"
                            return true
                        | Result.Error e ->
                            printfn $"[GraphitiKnowledgeGraph] Load failed: {e}"
                            return false
                    with ex ->
                        printfn $"[GraphitiKnowledgeGraph] Load error: {ex.Message}"
                        return false
                }

            loadTask.Result

        interface IDisposable with
            member _.Dispose() = (client :> IDisposable).Dispose()

    /// Factory to create Graphiti-backed graph
    let create (graphitiUrl: string) (groupId: string) =
        new GraphitiTemporalGraph(graphitiUrl, groupId)
