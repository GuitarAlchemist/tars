namespace Tars.Cortex

open System
open Tars.Core

/// Entity and fact extraction types
module EntityExtractor =

    /// Result of entity extraction
    type ExtractionResult = {
        Entities: TarsEntity list
        Facts: TarsFact list
        Confidence: float
        ExtractedAt: DateTime
    }
    
    /// Error during extraction
    type ExtractionError =
        | LlmError of message: string
        | ParseError of message: string
        | EmptyContent
    
    /// Create an empty extraction result
    let empty () : ExtractionResult = {
        Entities = []
        Facts = []
        Confidence = 0.0
        ExtractedAt = DateTime.UtcNow
    }
    
    /// Create a successful result with entities
    let success (entities: TarsEntity list) (facts: TarsFact list) : ExtractionResult = {
        Entities = entities
        Facts = facts
        Confidence = 0.8
        ExtractedAt = DateTime.UtcNow
    }
    
    /// Merge multiple extraction results
    let mergeResults (results: ExtractionResult list) : ExtractionResult =
        let allEntities = results |> List.collect (fun r -> r.Entities)
        let allFacts = results |> List.collect (fun r -> r.Facts)
        let avgConfidence = 
            if results.IsEmpty then 0.0
            else results |> List.averageBy (fun r -> r.Confidence)
        {
            Entities = allEntities
            Facts = allFacts
            Confidence = avgConfidence
            ExtractedAt = DateTime.UtcNow
        }


/// Entity resolution for deduplicating and merging entities
module EntityResolver =
    
    open EntityExtractor
    
    /// Resolution strategy for handling duplicate entities
    type ResolutionStrategy =
        | KeepFirst
        | KeepLast  
        | MergeByConfidence
    
    /// Get canonical ID for an entity (for deduplication)
    let getCanonicalId (entity: TarsEntity) : string =
        TarsEntity.getId entity
    
    /// Merge two similar entities based on type
    let private mergeEntities (e1: TarsEntity) (e2: TarsEntity) : TarsEntity =
        match e1, e2 with
        | CodePatternE p1, CodePatternE p2 ->
            CodePatternE {
                p1 with 
                    Occurrences = p1.Occurrences + p2.Occurrences
                    LastSeen = max p1.LastSeen p2.LastSeen
                    FirstSeen = min p1.FirstSeen p2.FirstSeen
            }
        | AgentBeliefE b1, AgentBeliefE b2 ->
            if b1.Confidence >= b2.Confidence then e1 else e2
        | ConceptE c1, ConceptE c2 ->
            ConceptE {
                c1 with 
                    RelatedConcepts = (c1.RelatedConcepts @ c2.RelatedConcepts) |> List.distinct
                    Description = if String.IsNullOrEmpty c1.Description then c2.Description else c1.Description
            }
        | CodeModuleE m1, CodeModuleE m2 ->
            CodeModuleE {
                m1 with
                    Dependencies = (m1.Dependencies @ m2.Dependencies) |> List.distinct
                    Complexity = max m1.Complexity m2.Complexity
                    LineCount = max m1.LineCount m2.LineCount
            }
        | AnomalyE a1, AnomalyE a2 ->
            // Keep the most severe anomaly
            if a1.Severity >= a2.Severity then e1 else e2
        | GrammarRuleE g1, GrammarRuleE g2 ->
            GrammarRuleE {
                g1 with
                    Examples = (g1.Examples @ g2.Examples) |> List.distinct
            }
        | _, _ -> e1 // Different types - keep first
    
    /// Resolve/deduplicate a list of entities
    let resolveEntities (strategy: ResolutionStrategy) (entities: TarsEntity list) : TarsEntity list =
        entities 
        |> List.groupBy getCanonicalId
        |> List.choose (fun (_, group) ->
            match group with
            | [] -> None
            | [single] -> Some single
            | first :: rest ->
                match strategy with
                | KeepFirst -> Some first
                | KeepLast -> Some (List.last group)
                | MergeByConfidence -> Some (List.fold mergeEntities first rest))
    
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
    
    /// Resolve facts after entity resolution (update references)  
    let resolveFacts (entityMap: Map<string, TarsEntity>) (facts: TarsFact list) : TarsFact list =
        let resolveEntity (entity: TarsEntity) =
            let id = getCanonicalId entity
            entityMap |> Map.tryFind id |> Option.defaultValue entity
        
        facts |> List.map (fun fact ->
            let newSource = TarsFact.source fact |> resolveEntity
            updateFactSource newSource fact)
    
    /// Full resolution of an extraction result
    let resolveExtractionResult (strategy: ResolutionStrategy) (result: ExtractionResult) : ExtractionResult =
        let resolvedEntities = resolveEntities strategy result.Entities
        let entityMap = 
            resolvedEntities 
            |> List.map (fun e -> (getCanonicalId e, e)) 
            |> Map.ofList
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
            | [] | [_] -> []
            | first :: rest -> rest |> List.map (fun e -> (first, e)))
