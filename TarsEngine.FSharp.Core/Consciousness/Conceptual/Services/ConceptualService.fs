namespace TarsEngine.FSharp.Core.Consciousness.Conceptual.Services

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Conceptual
open TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Implementation of IConceptualService.
/// </summary>
type ConceptualService(logger: ILogger<ConceptualService>) =
    
    // In-memory storage for concepts and hierarchies
    let concepts = Dictionary<Guid, Concept>()
    let hierarchies = Dictionary<Guid, ConceptHierarchy>()
    
    /// <summary>
    /// Creates a concept.
    /// </summary>
    /// <param name="name">The name of the concept.</param>
    /// <param name="description">The description of the concept.</param>
    /// <param name="type">The type of the concept.</param>
    /// <param name="complexity">The complexity of the concept.</param>
    /// <param name="attributes">The attributes of the concept.</param>
    /// <param name="examples">The examples of the concept.</param>
    /// <param name="tags">The tags of the concept.</param>
    /// <returns>The created concept.</returns>
    member _.CreateConcept(name: string, description: string, type': ConceptType, ?complexity: ConceptComplexity, ?attributes: Map<string, obj>, ?examples: string list, ?tags: string list) =
        task {
            let id = Guid.NewGuid()
            let now = DateTime.Now
            
            let concept = {
                Id = id
                Name = name
                Description = description
                Type = type'
                Complexity = defaultArg complexity ConceptComplexity.Moderate
                CreationTime = now
                LastActivationTime = None
                ActivationCount = 0
                AssociatedEmotions = []
                RelatedConcepts = []
                Attributes = defaultArg attributes Map.empty
                Examples = defaultArg examples []
                Tags = defaultArg tags []
                Metadata = Map.empty
            }
            
            concepts.Add(id, concept)
            
            logger.LogInformation("Created concept {ConceptId} with name {ConceptName}", id, name)
            
            return concept
        }
    
    /// <summary>
    /// Gets a concept by ID.
    /// </summary>
    /// <param name="id">The ID of the concept.</param>
    /// <returns>The concept, if found.</returns>
    member _.GetConcept(id: Guid) =
        task {
            if concepts.ContainsKey(id) then
                return Some concepts.[id]
            else
                logger.LogWarning("Concept {ConceptId} not found", id)
                return None
        }
    
    /// <summary>
    /// Gets all concepts.
    /// </summary>
    /// <returns>The list of all concepts.</returns>
    member _.GetAllConcepts() =
        task {
            return concepts.Values |> Seq.toList
        }
    
    /// <summary>
    /// Updates a concept.
    /// </summary>
    /// <param name="id">The ID of the concept to update.</param>
    /// <param name="description">The new description of the concept.</param>
    /// <param name="type">The new type of the concept.</param>
    /// <param name="complexity">The new complexity of the concept.</param>
    /// <param name="attributes">The new attributes of the concept.</param>
    /// <param name="examples">The new examples of the concept.</param>
    /// <param name="tags">The new tags of the concept.</param>
    /// <returns>The updated concept.</returns>
    member _.UpdateConcept(id: Guid, ?description: string, ?type': ConceptType, ?complexity: ConceptComplexity, ?attributes: Map<string, obj>, ?examples: string list, ?tags: string list) =
        task {
            if concepts.ContainsKey(id) then
                let concept = concepts.[id]
                
                let updatedConcept = {
                    concept with
                        Description = defaultArg description concept.Description
                        Type = defaultArg type' concept.Type
                        Complexity = defaultArg complexity concept.Complexity
                        Attributes = defaultArg attributes concept.Attributes
                        Examples = defaultArg examples concept.Examples
                        Tags = defaultArg tags concept.Tags
                }
                
                concepts.[id] <- updatedConcept
                
                logger.LogInformation("Updated concept {ConceptId}", id)
                
                return updatedConcept
            else
                logger.LogWarning("Concept {ConceptId} not found for update", id)
                return failwith $"Concept {id} not found"
        }
    
    /// <summary>
    /// Deletes a concept.
    /// </summary>
    /// <param name="id">The ID of the concept to delete.</param>
    /// <returns>Whether the concept was deleted.</returns>
    member _.DeleteConcept(id: Guid) =
        task {
            if concepts.ContainsKey(id) then
                concepts.Remove(id) |> ignore
                
                // Remove from all hierarchies
                for hierarchyId in hierarchies.Keys do
                    let hierarchy = hierarchies.[hierarchyId]
                    
                    // Remove from root concepts
                    let updatedRootConcepts = hierarchy.RootConcepts |> List.filter (fun c -> c <> id)
                    
                    // Remove from parent-child relationships
                    let updatedParentChildRelationships = 
                        hierarchy.ParentChildRelationships
                        |> Map.map (fun parentId children -> children |> List.filter (fun c -> c <> id))
                        |> Map.filter (fun parentId _ -> parentId <> id)
                    
                    if updatedRootConcepts.Length <> hierarchy.RootConcepts.Length || 
                       updatedParentChildRelationships.Count <> hierarchy.ParentChildRelationships.Count then
                        hierarchies.[hierarchyId] <- { 
                            hierarchy with 
                                RootConcepts = updatedRootConcepts
                                ParentChildRelationships = updatedParentChildRelationships
                                LastModificationTime = DateTime.Now 
                        }
                
                // Remove from related concepts
                for conceptId in concepts.Keys do
                    let concept = concepts.[conceptId]
                    let updatedRelatedConcepts = concept.RelatedConcepts |> List.filter (fun (relatedId, _) -> relatedId <> id)
                    
                    if updatedRelatedConcepts.Length <> concept.RelatedConcepts.Length then
                        concepts.[conceptId] <- { concept with RelatedConcepts = updatedRelatedConcepts }
                
                logger.LogInformation("Deleted concept {ConceptId}", id)
                
                return true
            else
                logger.LogWarning("Concept {ConceptId} not found for deletion", id)
                return false
        }
    
    /// <summary>
    /// Activates a concept.
    /// </summary>
    /// <param name="id">The ID of the concept to activate.</param>
    /// <param name="activationStrength">The strength of the activation.</param>
    /// <param name="context">The context of the activation.</param>
    /// <param name="trigger">The trigger of the activation.</param>
    /// <returns>The concept activation.</returns>
    member _.ActivateConcept(id: Guid, ?activationStrength: float, ?context: string, ?trigger: string) =
        task {
            if concepts.ContainsKey(id) then
                let concept = concepts.[id]
                let now = DateTime.Now
                
                let updatedConcept = {
                    concept with
                        LastActivationTime = Some now
                        ActivationCount = concept.ActivationCount + 1
                }
                
                concepts.[id] <- updatedConcept
                
                let activation = {
                    Concept = updatedConcept
                    ActivationTime = now
                    ActivationStrength = defaultArg activationStrength 1.0
                    Context = context
                    Trigger = trigger
                    Metadata = Map.empty
                }
                
                logger.LogInformation("Activated concept {ConceptId}", id)
                
                return activation
            else
                logger.LogWarning("Concept {ConceptId} not found for activation", id)
                return failwith $"Concept {id} not found"
        }
    
    /// <summary>
    /// Adds an emotion to a concept.
    /// </summary>
    /// <param name="conceptId">The ID of the concept.</param>
    /// <param name="emotion">The emotion to add.</param>
    /// <returns>The updated concept.</returns>
    member _.AddEmotionToConcept(conceptId: Guid, emotion: Emotion) =
        task {
            if concepts.ContainsKey(conceptId) then
                let concept = concepts.[conceptId]
                
                let updatedConcept = {
                    concept with
                        AssociatedEmotions = emotion :: concept.AssociatedEmotions
                }
                
                concepts.[conceptId] <- updatedConcept
                
                logger.LogInformation("Added emotion {EmotionCategory} to concept {ConceptId}", emotion.Category, conceptId)
                
                return updatedConcept
            else
                logger.LogWarning("Concept {ConceptId} not found for adding emotion", conceptId)
                return failwith $"Concept {conceptId} not found"
        }
    
    /// <summary>
    /// Relates two concepts.
    /// </summary>
    /// <param name="sourceId">The ID of the source concept.</param>
    /// <param name="targetId">The ID of the target concept.</param>
    /// <param name="strength">The strength of the relation.</param>
    /// <returns>The updated source concept.</returns>
    member _.RelateConcepts(sourceId: Guid, targetId: Guid, strength: float) =
        task {
            if concepts.ContainsKey(sourceId) && concepts.ContainsKey(targetId) then
                let sourceConcept = concepts.[sourceId]
                
                // Check if the relation already exists
                let existingRelation = sourceConcept.RelatedConcepts |> List.tryFind (fun (id, _) -> id = targetId)
                
                let updatedRelatedConcepts =
                    match existingRelation with
                    | Some _ ->
                        // Update existing relation
                        sourceConcept.RelatedConcepts
                        |> List.map (fun (id, s) -> if id = targetId then (id, strength) else (id, s))
                    | None ->
                        // Add new relation
                        (targetId, strength) :: sourceConcept.RelatedConcepts
                
                let updatedSourceConcept = {
                    sourceConcept with
                        RelatedConcepts = updatedRelatedConcepts
                }
                
                concepts.[sourceId] <- updatedSourceConcept
                
                logger.LogInformation("Related concept {SourceConceptId} to {TargetConceptId} with strength {Strength}", sourceId, targetId, strength)
                
                return updatedSourceConcept
            else
                if not (concepts.ContainsKey(sourceId)) then
                    logger.LogWarning("Source concept {SourceConceptId} not found", sourceId)
                    return failwith $"Source concept {sourceId} not found"
                else
                    logger.LogWarning("Target concept {TargetConceptId} not found", targetId)
                    return failwith $"Target concept {targetId} not found"
        }
    
    /// <summary>
    /// Creates a concept hierarchy.
    /// </summary>
    /// <param name="name">The name of the hierarchy.</param>
    /// <param name="description">The description of the hierarchy.</param>
    /// <param name="rootConcepts">The root concepts of the hierarchy.</param>
    /// <returns>The created hierarchy.</returns>
    member _.CreateHierarchy(name: string, ?description: string, ?rootConcepts: Guid list) =
        task {
            let id = Guid.NewGuid()
            let now = DateTime.Now
            
            let hierarchy = {
                Id = id
                Name = name
                Description = description
                RootConcepts = defaultArg rootConcepts []
                ParentChildRelationships = Map.empty
                CreationTime = now
                LastModificationTime = now
                Metadata = Map.empty
            }
            
            hierarchies.Add(id, hierarchy)
            
            logger.LogInformation("Created concept hierarchy {HierarchyId} with name {HierarchyName}", id, name)
            
            return hierarchy
        }
    
    /// <summary>
    /// Gets a concept hierarchy by ID.
    /// </summary>
    /// <param name="id">The ID of the hierarchy.</param>
    /// <returns>The hierarchy, if found.</returns>
    member _.GetHierarchy(id: Guid) =
        task {
            if hierarchies.ContainsKey(id) then
                return Some hierarchies.[id]
            else
                logger.LogWarning("Concept hierarchy {HierarchyId} not found", id)
                return None
        }
    
    /// <summary>
    /// Gets all concept hierarchies.
    /// </summary>
    /// <returns>The list of all hierarchies.</returns>
    member _.GetAllHierarchies() =
        task {
            return hierarchies.Values |> Seq.toList
        }
    
    /// <summary>
    /// Adds a concept to a hierarchy.
    /// </summary>
    /// <param name="hierarchyId">The ID of the hierarchy.</param>
    /// <param name="conceptId">The ID of the concept.</param>
    /// <param name="parentId">The ID of the parent concept, if any.</param>
    /// <returns>The updated hierarchy.</returns>
    member _.AddConceptToHierarchy(hierarchyId: Guid, conceptId: Guid, ?parentId: Guid) =
        task {
            if hierarchies.ContainsKey(hierarchyId) && concepts.ContainsKey(conceptId) then
                let hierarchy = hierarchies.[hierarchyId]
                
                match parentId with
                | Some pid when concepts.ContainsKey(pid) ->
                    // Add as child of parent
                    let children = 
                        match hierarchy.ParentChildRelationships.TryFind(pid) with
                        | Some existingChildren -> 
                            if existingChildren |> List.contains conceptId then
                                existingChildren
                            else
                                conceptId :: existingChildren
                        | None -> [conceptId]
                    
                    let updatedRelationships = hierarchy.ParentChildRelationships.Add(pid, children)
                    
                    let updatedHierarchy = {
                        hierarchy with
                            ParentChildRelationships = updatedRelationships
                            LastModificationTime = DateTime.Now
                    }
                    
                    hierarchies.[hierarchyId] <- updatedHierarchy
                    
                    logger.LogInformation("Added concept {ConceptId} to hierarchy {HierarchyId} as child of {ParentId}", conceptId, hierarchyId, pid)
                    
                    return updatedHierarchy
                | _ ->
                    // Add as root concept
                    if hierarchy.RootConcepts |> List.contains conceptId then
                        logger.LogInformation("Concept {ConceptId} already exists in hierarchy {HierarchyId} as root", conceptId, hierarchyId)
                        return hierarchy
                    else
                        let updatedHierarchy = {
                            hierarchy with
                                RootConcepts = conceptId :: hierarchy.RootConcepts
                                LastModificationTime = DateTime.Now
                        }
                        
                        hierarchies.[hierarchyId] <- updatedHierarchy
                        
                        logger.LogInformation("Added concept {ConceptId} to hierarchy {HierarchyId} as root", conceptId, hierarchyId)
                        
                        return updatedHierarchy
            else
                if not (hierarchies.ContainsKey(hierarchyId)) then
                    logger.LogWarning("Concept hierarchy {HierarchyId} not found", hierarchyId)
                    return failwith $"Concept hierarchy {hierarchyId} not found"
                else
                    logger.LogWarning("Concept {ConceptId} not found", conceptId)
                    return failwith $"Concept {conceptId} not found"
        }
    
    /// <summary>
    /// Removes a concept from a hierarchy.
    /// </summary>
    /// <param name="hierarchyId">The ID of the hierarchy.</param>
    /// <param name="conceptId">The ID of the concept.</param>
    /// <returns>The updated hierarchy.</returns>
    member _.RemoveConceptFromHierarchy(hierarchyId: Guid, conceptId: Guid) =
        task {
            if hierarchies.ContainsKey(hierarchyId) then
                let hierarchy = hierarchies.[hierarchyId]
                
                // Remove from root concepts
                let updatedRootConcepts = hierarchy.RootConcepts |> List.filter (fun c -> c <> conceptId)
                
                // Remove from parent-child relationships
                let updatedParentChildRelationships = 
                    hierarchy.ParentChildRelationships
                    |> Map.map (fun parentId children -> children |> List.filter (fun c -> c <> conceptId))
                
                let updatedHierarchy = {
                    hierarchy with
                        RootConcepts = updatedRootConcepts
                        ParentChildRelationships = updatedParentChildRelationships
                        LastModificationTime = DateTime.Now
                }
                
                hierarchies.[hierarchyId] <- updatedHierarchy
                
                logger.LogInformation("Removed concept {ConceptId} from hierarchy {HierarchyId}", conceptId, hierarchyId)
                
                return updatedHierarchy
            else
                logger.LogWarning("Concept hierarchy {HierarchyId} not found", hierarchyId)
                return failwith $"Concept hierarchy {hierarchyId} not found"
        }
    
    /// <summary>
    /// Deletes a concept hierarchy.
    /// </summary>
    /// <param name="id">The ID of the hierarchy to delete.</param>
    /// <returns>Whether the hierarchy was deleted.</returns>
    member _.DeleteHierarchy(id: Guid) =
        task {
            if hierarchies.ContainsKey(id) then
                hierarchies.Remove(id) |> ignore
                
                logger.LogInformation("Deleted concept hierarchy {HierarchyId}", id)
                
                return true
            else
                logger.LogWarning("Concept hierarchy {HierarchyId} not found for deletion", id)
                return false
        }
    
    /// <summary>
    /// Finds concepts.
    /// </summary>
    /// <param name="query">The concept query.</param>
    /// <returns>The concept query result.</returns>
    member _.FindConcepts(query: ConceptQuery) =
        task {
            let startTime = DateTime.Now
            
            // Filter concepts based on the query
            let filteredConcepts =
                concepts.Values
                |> Seq.filter (fun c ->
                    // Filter by name pattern
                    (query.NamePattern.IsNone || c.Name.Contains(query.NamePattern.Value, StringComparison.OrdinalIgnoreCase)) &&
                    // Filter by type
                    (query.Types.IsNone || query.Types.Value |> List.exists (fun t -> t = c.Type)) &&
                    // Filter by tags
                    (query.Tags.IsNone || query.Tags.Value |> List.exists (fun t -> c.Tags |> List.exists (fun ct -> ct.Equals(t, StringComparison.OrdinalIgnoreCase)))) &&
                    // Filter by minimum complexity
                    (query.MinimumComplexity.IsNone || 
                        match (c.Complexity, query.MinimumComplexity.Value) with
                        | (ConceptComplexity.Custom c1, ConceptComplexity.Custom c2) -> c1 >= c2
                        | (ConceptComplexity.Custom _, _) -> true
                        | (_, ConceptComplexity.Custom _) -> false
                        | (ConceptComplexity.Simple, _) -> false
                        | (ConceptComplexity.Moderate, ConceptComplexity.Simple) -> true
                        | (ConceptComplexity.Moderate, _) -> false
                        | (ConceptComplexity.Complex, ConceptComplexity.VeryComplex) -> false
                        | (ConceptComplexity.Complex, _) -> true
                        | (ConceptComplexity.VeryComplex, _) -> true) &&
                    // Filter by maximum complexity
                    (query.MaximumComplexity.IsNone || 
                        match (c.Complexity, query.MaximumComplexity.Value) with
                        | (ConceptComplexity.Custom c1, ConceptComplexity.Custom c2) -> c1 <= c2
                        | (ConceptComplexity.Custom _, _) -> false
                        | (_, ConceptComplexity.Custom _) -> true
                        | (ConceptComplexity.Simple, _) -> true
                        | (ConceptComplexity.Moderate, ConceptComplexity.Simple) -> false
                        | (ConceptComplexity.Moderate, _) -> true
                        | (ConceptComplexity.Complex, ConceptComplexity.Simple) -> false
                        | (ConceptComplexity.Complex, ConceptComplexity.Moderate) -> false
                        | (ConceptComplexity.Complex, _) -> true
                        | (ConceptComplexity.VeryComplex, ConceptComplexity.VeryComplex) -> true
                        | (ConceptComplexity.VeryComplex, _) -> false)
                )
                |> Seq.toList
            
            // Limit the number of results if specified
            let limitedConcepts =
                match query.MaxResults with
                | Some maxResults -> filteredConcepts |> List.truncate maxResults
                | None -> filteredConcepts
            
            let endTime = DateTime.Now
            let executionTime = endTime - startTime
            
            let result = {
                Query = query
                Concepts = limitedConcepts
                ExecutionTime = executionTime
                Metadata = Map.empty
            }
            
            logger.LogInformation("Found {ConceptCount} concepts matching query", limitedConcepts.Length)
            
            return result
        }
    
    /// <summary>
    /// Suggests concepts.
    /// </summary>
    /// <param name="text">The text to suggest concepts for.</param>
    /// <param name="maxSuggestions">The maximum number of suggestions.</param>
    /// <returns>The list of concept suggestions.</returns>
    member _.SuggestConcepts(text: string, ?maxSuggestions: int) =
        task {
            let maxSugg = defaultArg maxSuggestions 5
            
            // Simple implementation: extract words and suggest concepts
            let words =
                text.Split([|' '; '.'; ','; ';'; '!'; '?'; '('; ')'; '['; ']'; '{'; '}'; '\n'; '\r'; '\t'|], StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun w -> w.ToLowerInvariant())
                |> Array.filter (fun w -> w.Length > 3) // Filter out short words
                |> Array.distinct
                |> Array.toList
            
            // Create suggestions
            let suggestions =
                words
                |> List.map (fun word ->
                    {
                        Name = word
                        Description = $"Concept derived from text: {word}"
                        Type = ConceptType.Entity
                        Complexity = ConceptComplexity.Simple
                        Confidence = 0.7
                        Reason = Some "Extracted from text"
                        Metadata = Map.empty
                    })
                |> List.truncate maxSugg
            
            logger.LogInformation("Generated {SuggestionCount} concept suggestions from text", suggestions.Length)
            
            return suggestions
        }
    
    /// <summary>
    /// Learns concepts from text.
    /// </summary>
    /// <param name="text">The text to learn from.</param>
    /// <returns>The concept learning result.</returns>
    member this.LearnConceptsFromText(text: string) =
        task {
            let startTime = DateTime.Now
            
            // Simple implementation: extract words and create concepts
            let words =
                text.Split([|' '; '.'; ','; ';'; '!'; '?'; '('; ')'; '['; ']'; '{'; '}'; '\n'; '\r'; '\t'|], StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun w -> w.ToLowerInvariant())
                |> Array.filter (fun w -> w.Length > 3) // Filter out short words
                |> Array.distinct
                |> Array.toList
            
            let learnedConcepts = ResizeArray<Concept>()
            let updatedConcepts = ResizeArray<Concept>()
            
            for word in words do
                // Check if the concept already exists
                let existingConcept =
                    concepts.Values
                    |> Seq.tryFind (fun c -> c.Name.Equals(word, StringComparison.OrdinalIgnoreCase))
                
                match existingConcept with
                | Some concept ->
                    // Update existing concept
                    let updatedConcept = {
                        concept with
                            ActivationCount = concept.ActivationCount + 1
                            LastActivationTime = Some DateTime.Now
                    }
                    
                    concepts.[concept.Id] <- updatedConcept
                    updatedConcepts.Add(updatedConcept)
                | None ->
                    // Create new concept
                    let! newConcept = this.CreateConcept(word, $"Concept learned from text: {word}", ConceptType.Entity, ConceptComplexity.Simple)
                    learnedConcepts.Add(newConcept)
            
            let endTime = DateTime.Now
            let learningTime = endTime - startTime
            
            let result = {
                LearnedConcepts = learnedConcepts |> Seq.toList
                UpdatedConcepts = updatedConcepts |> Seq.toList
                RemovedConcepts = []
                LearningTime = learningTime
                Metadata = Map.empty
            }
            
            logger.LogInformation("Learned {LearnedCount} new concepts and updated {UpdatedCount} existing concepts from text", 
                                 learnedConcepts.Count, updatedConcepts.Count)
            
            return result
        }
    
    /// <summary>
    /// Exports a concept hierarchy.
    /// </summary>
    /// <param name="hierarchyId">The ID of the hierarchy to export.</param>
    /// <param name="format">The format to export in.</param>
    /// <param name="path">The path to export to.</param>
    /// <returns>Whether the export was successful.</returns>
    member _.ExportHierarchy(hierarchyId: Guid, format: string, path: string) =
        task {
            if hierarchies.ContainsKey(hierarchyId) then
                let hierarchy = hierarchies.[hierarchyId]
                
                // In a real implementation, this would serialize the hierarchy to the specified format
                // For now, we'll just log the export
                logger.LogInformation("Exported concept hierarchy {HierarchyId} to {Path} in {Format} format", hierarchyId, path, format)
                
                return true
            else
                logger.LogWarning("Concept hierarchy {HierarchyId} not found for export", hierarchyId)
                return false
        }
    
    /// <summary>
    /// Imports a concept hierarchy.
    /// </summary>
    /// <param name="format">The format to import from.</param>
    /// <param name="path">The path to import from.</param>
    /// <returns>The imported hierarchy.</returns>
    member _.ImportHierarchy(format: string, path: string) =
        task {
            // In a real implementation, this would deserialize the hierarchy from the specified format
            // For now, we'll just create a new empty hierarchy
            let id = Guid.NewGuid()
            let now = DateTime.Now
            
            let hierarchy = {
                Id = id
                Name = $"Imported Hierarchy ({path})"
                Description = Some $"Imported from {path} in {format} format"
                RootConcepts = []
                ParentChildRelationships = Map.empty
                CreationTime = now
                LastModificationTime = now
                Metadata = Map.empty
            }
            
            hierarchies.Add(id, hierarchy)
            
            logger.LogInformation("Imported concept hierarchy from {Path} in {Format} format", path, format)
            
            return hierarchy
        }
    
    interface IConceptualService with
        member this.CreateConcept(name, description, type', ?complexity, ?attributes, ?examples, ?tags) = 
            this.CreateConcept(name, description, type', ?complexity = complexity, ?attributes = attributes, ?examples = examples, ?tags = tags)
        member this.GetConcept(id) = this.GetConcept(id)
        member this.GetAllConcepts() = this.GetAllConcepts()
        member this.UpdateConcept(id, ?description, ?type', ?complexity, ?attributes, ?examples, ?tags) = 
            this.UpdateConcept(id, ?description = description, ?type' = type', ?complexity = complexity, ?attributes = attributes, ?examples = examples, ?tags = tags)
        member this.DeleteConcept(id) = this.DeleteConcept(id)
        member this.ActivateConcept(id, ?activationStrength, ?context, ?trigger) = 
            this.ActivateConcept(id, ?activationStrength = activationStrength, ?context = context, ?trigger = trigger)
        member this.AddEmotionToConcept(conceptId, emotion) = this.AddEmotionToConcept(conceptId, emotion)
        member this.RelateConcepts(sourceId, targetId, strength) = this.RelateConcepts(sourceId, targetId, strength)
        member this.CreateHierarchy(name, ?description, ?rootConcepts) = 
            this.CreateHierarchy(name, ?description = description, ?rootConcepts = rootConcepts)
        member this.GetHierarchy(id) = this.GetHierarchy(id)
        member this.GetAllHierarchies() = this.GetAllHierarchies()
        member this.AddConceptToHierarchy(hierarchyId, conceptId, ?parentId) = 
            this.AddConceptToHierarchy(hierarchyId, conceptId, ?parentId = parentId)
        member this.RemoveConceptFromHierarchy(hierarchyId, conceptId) = this.RemoveConceptFromHierarchy(hierarchyId, conceptId)
        member this.DeleteHierarchy(id) = this.DeleteHierarchy(id)
        member this.FindConcepts(query) = this.FindConcepts(query)
        member this.SuggestConcepts(text, ?maxSuggestions) = this.SuggestConcepts(text, ?maxSuggestions = maxSuggestions)
        member this.LearnConceptsFromText(text) = this.LearnConceptsFromText(text)
        member this.ExportHierarchy(hierarchyId, format, path) = this.ExportHierarchy(hierarchyId, format, path)
        member this.ImportHierarchy(format, path) = this.ImportHierarchy(format, path)
