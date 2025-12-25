namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

/// Entity and fact extraction types
module EntityExtractor =

    /// Result of entity extraction
    type ExtractionResult =
        { Entities: TarsEntity list
          Facts: TarsFact list
          Confidence: float
          ExtractedAt: DateTime }

    /// Error during extraction
    type ExtractionError =
        | LlmError of message: string
        | ParseError of message: string
        | EmptyContent

    /// Create an empty extraction result
    let empty () : ExtractionResult =
        { Entities = []
          Facts = []
          Confidence = 0.0
          ExtractedAt = DateTime.UtcNow }

    /// Create a successful result with entities
    let success (entities: TarsEntity list) (facts: TarsFact list) : ExtractionResult =
        { Entities = entities
          Facts = facts
          Confidence = 0.8
          ExtractedAt = DateTime.UtcNow }

    /// Merge multiple extraction results
    let mergeResults (results: ExtractionResult list) : ExtractionResult =
        let allEntities = results |> List.collect (fun r -> r.Entities)
        let allFacts = results |> List.collect (fun r -> r.Facts)

        let avgConfidence =
            if results.IsEmpty then
                0.0
            else
                results |> List.averageBy (fun r -> r.Confidence)

        { Entities = allEntities
          Facts = allFacts
          Confidence = avgConfidence
          ExtractedAt = DateTime.UtcNow }


/// Entity resolution for deduplicating and merging entities
module EntityResolver =

    open EntityExtractor

    /// Resolution strategy for handling duplicate entities
    type ResolutionStrategy =
        | KeepFirst
        | KeepLast
        | MergeByConfidence

    /// Get canonical ID for an entity (for deduplication)
    let getCanonicalId (entity: TarsEntity) : string = TarsEntity.getId entity

    /// Merge two similar entities based on type
    let private mergeEntities (e1: TarsEntity) (e2: TarsEntity) : TarsEntity =
        match e1, e2 with
        | CodePatternE p1, CodePatternE p2 ->
            CodePatternE
                { p1 with
                    Occurrences = p1.Occurrences + p2.Occurrences
                    LastSeen = max p1.LastSeen p2.LastSeen
                    FirstSeen = min p1.FirstSeen p2.FirstSeen }
        | AgentBeliefE b1, AgentBeliefE b2 -> if b1.Confidence >= b2.Confidence then e1 else e2
        | ConceptE c1, ConceptE c2 ->
            ConceptE
                { c1 with
                    RelatedConcepts = (c1.RelatedConcepts @ c2.RelatedConcepts) |> List.distinct
                    Description =
                        if String.IsNullOrEmpty c1.Description then
                            c2.Description
                        else
                            c1.Description }
        | CodeModuleE m1, CodeModuleE m2 ->
            CodeModuleE
                { m1 with
                    Dependencies = (m1.Dependencies @ m2.Dependencies) |> List.distinct
                    Complexity = max m1.Complexity m2.Complexity
                    LineCount = max m1.LineCount m2.LineCount }
        | AnomalyE a1, AnomalyE a2 ->
            // Keep the most severe anomaly
            if a1.Severity >= a2.Severity then e1 else e2
        | GrammarRuleE g1, GrammarRuleE g2 ->
            GrammarRuleE
                { g1 with
                    Examples = (g1.Examples @ g2.Examples) |> List.distinct }
        | _, _ -> e1 // Different types - keep first

    /// Get representative text for an entity to use in semantic embedding
    let getRepresentativeText =
        function
        | ConceptE c ->
            if String.IsNullOrEmpty c.Description then
                c.Name
            else
                $"{c.Name}: {c.Description}"
        | AgentBeliefE b -> b.Statement
        | CodePatternE p -> $"{p.Name} ({p.Category})"
        | CodeModuleE m -> m.Path
        | AnomalyE a -> $"{a.Location}: {a.Type}"
        | GrammarRuleE g -> g.Name
        | FileE p -> p
        | FunctionE n -> n
        | EpisodeE e -> Episode.typeTag e

    /// Resolve/deduplicate a list of entities using exact string matching
    let resolveEntities (strategy: ResolutionStrategy) (entities: TarsEntity list) : TarsEntity list =
        entities
        |> List.groupBy getCanonicalId
        |> List.choose (fun (_, group) ->
            match group with
            | [] -> None
            | [ single ] -> Some single
            | first :: rest ->
                match strategy with
                | KeepFirst -> Some first
                | KeepLast -> Some(List.last group)
                | MergeByConfidence -> Some(List.fold mergeEntities first rest))

    /// Resolve/deduplicate a list of entities using neural-symbolic semantic clustering
    let resolveEntitiesSemanticAsync
        (llm: ILlmService)
        (threshold: float)
        (strategy: ResolutionStrategy)
        (entities: TarsEntity list)
        : Task<TarsEntity list> =
        task {
            if entities.IsEmpty then
                return []
            else
                // 1. Generate embeddings for all entities
                let! entityEmbeddings =
                    entities
                    |> List.map (fun e ->
                        task {
                            let text = getRepresentativeText e
                            let! emb = llm.EmbedAsync text
                            return (e, emb)
                        })
                    |> Task.WhenAll

                // 2. Cluster entities based on semantic similarity
                let mutable clusters: (TarsEntity * float32[]) list list = []

                for (entity, emb) in entityEmbeddings do
                    let mutable assigned = false

                    let updatedClusters =
                        clusters
                        |> List.map (fun cluster ->
                            let (representative, representativeEmb) = cluster.Head
                            let sim = MetricSpace.cosineSimilarity emb representativeEmb

                            if float sim >= threshold then
                                assigned <- true
                                (entity, emb) :: cluster
                            else
                                cluster)

                    clusters <- updatedClusters

                    if not assigned then
                        clusters <- [ (entity, emb) ] :: clusters

                // 3. Resolve each cluster into a single entity
                return
                    clusters
                    |> List.choose (fun cluster ->
                        let ents = cluster |> List.map fst

                        match ents with
                        | [] -> None
                        | [ single ] -> Some single
                        | first :: rest ->
                            match strategy with
                            | KeepFirst -> Some first
                            | KeepLast -> Some(List.last ents)
                            | MergeByConfidence -> Some(List.fold mergeEntities first rest))
        }

    /// Update fact source entity to resolved entity
    let private updateFactSource (newSource: TarsEntity) (fact: Tars.Core.TarsFact) : Tars.Core.TarsFact =
        match fact with
        | Tars.Core.DependsOn(_, target, strength) -> Tars.Core.DependsOn(newSource, target, strength)
        | Tars.Core.Implements(_, target, confidence) -> Tars.Core.Implements(newSource, target, confidence)
        | Tars.Core.Contradicts(_, target, resolution) -> Tars.Core.Contradicts(newSource, target, resolution)
        | Tars.Core.EvolvedFrom(_, target, delta) -> Tars.Core.EvolvedFrom(newSource, target, delta)
        | Tars.Core.BelongsTo(_, communityId) -> Tars.Core.BelongsTo(newSource, communityId)
        | Tars.Core.SimilarTo(_, target, similarity) -> Tars.Core.SimilarTo(newSource, target, similarity)
        | Tars.Core.DerivedFrom(_, target) -> Tars.Core.DerivedFrom(newSource, target)
        | Tars.Core.Contains(_, target) -> Tars.Core.Contains(newSource, target)

    /// Resolve facts after entity resolution (update references)
    let resolveFacts (entityMap: Map<string, TarsEntity>) (facts: TarsFact list) : TarsFact list =
        let resolveEntity (entity: TarsEntity) =
            let id = getCanonicalId entity
            entityMap |> Map.tryFind id |> Option.defaultValue entity

        facts
        |> List.map (fun fact ->
            let newSource = TarsFact.source fact |> resolveEntity
            updateFactSource newSource fact)

    /// Full resolution of an extraction result
    let resolveExtractionResult (strategy: ResolutionStrategy) (result: ExtractionResult) : ExtractionResult =
        let resolvedEntities = resolveEntities strategy result.Entities

        let entityMap =
            resolvedEntities |> List.map (fun e -> (getCanonicalId e, e)) |> Map.ofList

        let resolvedFacts = resolveFacts entityMap result.Facts

        { result with
            Entities = resolvedEntities
            Facts = resolvedFacts }

    /// Find similar entities that might be duplicates
    let findPotentialDuplicates (entities: TarsEntity list) : (TarsEntity * TarsEntity) list =
        entities
        |> List.groupBy getCanonicalId
        |> List.filter (fun (_, group) -> List.length group > 1)
        |> List.collect (fun (_, group) ->
            match group with
            | []
            | [ _ ] -> []
            | first :: rest -> rest |> List.map (fun e -> (first, e)))


/// Fact extraction from entities
module FactExtractor =

    open System.Text.Json
    open System.Text.Json.Serialization
    open System.Threading.Tasks
    open EntityExtractor
    open Tars.Llm

    /// Prompt templates for fact extraction
    module Prompts =

        let factExtractionSystem =
            """You are a relationship extraction system for a software development knowledge graph.
Given a list of entities and their context, identify relationships between them.

Return a JSON array of facts:
[
  {"type": "depends_on", "source": "entity_name", "target": "entity_name", "strength": 0.0-1.0},
  {"type": "implements", "source": "entity_name", "target": "entity_name", "confidence": 0.0-1.0},
  {"type": "similar_to", "source": "entity_name", "target": "entity_name", "similarity": 0.0-1.0},
  {"type": "contradicts", "source": "entity_name", "target": "entity_name", "resolution": "explanation"},
  {"type": "evolved_from", "source": "entity_name", "target": "entity_name", "delta": "changes"},
  {"type": "derived_from", "source": "entity_name", "target": "entity_name"}
]

Relationship types:
- depends_on: Source depends on/requires target
- implements: Source implements/realizes target concept
- similar_to: Source and target are conceptually similar
- contradicts: Source and target are in conflict
- evolved_from: Source evolved/changed from target
- derived_from: Source is derived/based on target

Only include relationships clearly supported by the content. Be conservative."""

        let factExtractionUser (entities: string) (context: string) =
            $"""Extract relationships between these entities:

Entities:
{entities}

Context:
{context}

Return valid JSON array only."""

    /// JSON DTO for parsing LLM output
    [<CLIMutable>]
    type FactDto =
        { [<JsonPropertyName("type")>]
          Type: string
          [<JsonPropertyName("source")>]
          Source: string
          [<JsonPropertyName("target")>]
          Target: string
          [<JsonPropertyName("strength")>]
          Strength: Nullable<float>
          [<JsonPropertyName("confidence")>]
          Confidence: Nullable<float>
          [<JsonPropertyName("similarity")>]
          Similarity: Nullable<float>
          [<JsonPropertyName("resolution")>]
          Resolution: string
          [<JsonPropertyName("delta")>]
          Delta: string }

    let private jsonOptions =
        let opts = JsonSerializerOptions()
        opts.PropertyNameCaseInsensitive <- true
        opts.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase
        opts

    /// Format entities for the prompt
    let private formatEntities (entities: TarsEntity list) : string =
        entities
        |> List.map (fun e ->
            match e with
            | Tars.Core.ConceptE c -> $"- Concept: {c.Name} - {c.Description}"
            | Tars.Core.CodePatternE p -> $"- Pattern: {p.Name} ({p.Category})"
            | Tars.Core.AgentBeliefE b -> $"- Belief: {b.Statement} (confidence: {b.Confidence})"
            | Tars.Core.CodeModuleE m -> $"- Module: {m.Path}"
            | Tars.Core.AnomalyE a -> $"- Anomaly: {a.Location} ({a.Severity})"
            | Tars.Core.GrammarRuleE g -> $"- Grammar: {g.Name}"
            | Tars.Core.EpisodeE e -> $"- Episode: {TarsEntity.getId (Tars.Core.EpisodeE e)}"
            | Tars.Core.FileE p -> $"- File: {p}"
            | Tars.Core.FunctionE f -> $"- Function: {f}")
        |> String.concat "\n"

    /// Create placeholder entity for fact reference
    let private createPlaceholder (name: string) : TarsEntity =
        Tars.Core.ConceptE
            { Name = name
              Description = ""
              RelatedConcepts = [] }

    /// Convert DTO to TarsFact
    let private dtoToFact (entityMap: Map<string, TarsEntity>) (dto: FactDto) : Tars.Core.TarsFact option =
        let getEntity name =
            entityMap |> Map.tryFind name |> Option.defaultValue (createPlaceholder name)

        if String.IsNullOrEmpty dto.Source || String.IsNullOrEmpty dto.Target then
            None
        else
            let source = getEntity dto.Source
            let target = getEntity dto.Target
            let factType = if isNull dto.Type then "" else dto.Type.ToLowerInvariant()

            match factType with
            | "depends_on" ->
                let strength = if dto.Strength.HasValue then dto.Strength.Value else 0.5
                Some(Tars.Core.DependsOn(source, target, strength))
            | "implements" ->
                let conf =
                    if dto.Confidence.HasValue then
                        dto.Confidence.Value
                    else
                        0.5

                Some(Tars.Core.Implements(source, target, conf))
            | "similar_to" ->
                let sim =
                    if dto.Similarity.HasValue then
                        dto.Similarity.Value
                    else
                        0.5

                Some(Tars.Core.SimilarTo(source, target, sim))
            | "contradicts" ->
                let resolution = if isNull dto.Resolution then None else Some dto.Resolution
                Some(Tars.Core.Contradicts(source, target, resolution))
            | "evolved_from" ->
                let delta = if isNull dto.Delta then "" else dto.Delta
                Some(Tars.Core.EvolvedFrom(source, target, delta))
            | "derived_from" -> Some(Tars.Core.DerivedFrom(source, target))
            | _ -> None

    /// Extract JSON from LLM response (handles markdown code blocks)
    let private extractJson (text: string) : string =
        let text = text.Trim()

        if text.StartsWith("```json") then
            text.Substring(7).TrimStart().TrimEnd('`').Trim()
        elif text.StartsWith("```") then
            text.Substring(3).TrimStart().TrimEnd('`').Trim()
        else
            text

    /// Build entity name map for fact resolution
    let private buildEntityMap (entities: TarsEntity list) : Map<string, TarsEntity> =
        entities
        |> List.choose (fun e ->
            let name =
                match e with
                | Tars.Core.ConceptE c -> Some c.Name
                | Tars.Core.CodePatternE p -> Some p.Name
                | Tars.Core.AgentBeliefE b -> Some b.Statement
                | Tars.Core.CodeModuleE m -> Some m.Path
                | Tars.Core.AnomalyE a -> Some a.Location
                | Tars.Core.GrammarRuleE g -> Some g.Name
                | Tars.Core.EpisodeE e -> Some("Episode " + TarsEntity.getId (Tars.Core.EpisodeE e))
                | Tars.Core.FileE p -> Some p
                | Tars.Core.FunctionE f -> Some f

            name |> Option.map (fun n -> (n, e)))
        |> Map.ofList

    /// Result of fact extraction
    type FactExtractionResult =
        | FactSuccess of Tars.Core.TarsFact list
        | FactFailure of ExtractionError

    /// Extract facts from entities using LLM
    let extractFactsAsync
        (llmService: ILlmService)
        (entities: TarsEntity list)
        (context: string)
        : Task<FactExtractionResult> =
        task {
            if entities.IsEmpty then
                return FactSuccess []
            else
                let entityStr = formatEntities entities

                let request: LlmRequest =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt = Some Prompts.factExtractionSystem
                      MaxTokens = Some 1500
                      Temperature = Some 0.1
                      Stop = []
                      Messages =
                        [ { Role = Role.User
                            Content = Prompts.factExtractionUser entityStr context } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = true
                      Seed = None

                      ContextWindow = None }

                try
                    let! response = llmService.CompleteAsync(request)
                    let jsonText = extractJson response.Text

                    try
                        let dtos = JsonSerializer.Deserialize<FactDto array>(jsonText, jsonOptions)
                        let entityMap = buildEntityMap entities

                        let facts =
                            if isNull dtos then
                                []
                            else
                                dtos |> Array.choose (dtoToFact entityMap) |> Array.toList

                        return FactSuccess facts
                    with ex ->
                        return FactFailure(ParseError ex.Message)
                with ex ->
                    return FactFailure(LlmError ex.Message)
        }

    let private canonicalId (entity: TarsEntity) = TarsEntity.getId entity

    let private matchesPair (lhs: TarsEntity) (rhs: TarsEntity) (fact: TarsFact) =
        let lhsId = canonicalId lhs
        let rhsId = canonicalId rhs
        let sourceId = canonicalId (TarsFact.source fact)

        match TarsFact.target fact with
        | Some target ->
            let targetId = canonicalId target
            (sourceId = lhsId && targetId = rhsId) || (sourceId = rhsId && targetId = lhsId)
        | None -> false

    /// Extract relationships between the provided entities using the LLM
    let suggestRelationshipsAsync
        (llmService: ILlmService)
        (context: string)
        (e1: TarsEntity)
        (e2: TarsEntity)
        : Task<FactExtractionResult> =
        task {
            let! extraction = extractFactsAsync llmService [ e1; e2 ] context

            match extraction with
            | FactSuccess facts ->
                let filtered = facts |> List.filter (matchesPair e1 e2)
                return FactSuccess filtered
            | failure -> return failure
        }
