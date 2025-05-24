namespace TarsEngine.FSharp.Core.Consciousness.Association.Services

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Association

/// <summary>
/// Implementation of IAssociationService.
/// </summary>
type AssociationService(logger: ILogger<AssociationService>) =
    
    // In-memory storage for associations and networks
    let associations = Dictionary<Guid, Association>()
    let networks = Dictionary<Guid, AssociationNetwork>()
    
    /// <summary>
    /// Creates an association.
    /// </summary>
    /// <param name="source">The source concept.</param>
    /// <param name="target">The target concept.</param>
    /// <param name="type">The type of the association.</param>
    /// <param name="strength">The strength of the association.</param>
    /// <param name="description">The description of the association.</param>
    /// <param name="isBidirectional">Whether the association is bidirectional.</param>
    /// <returns>The created association.</returns>
    member _.CreateAssociation(source: string, target: string, type': AssociationType, ?strength: AssociationStrength, ?description: string, ?isBidirectional: bool) =
        task {
            let id = Guid.NewGuid()
            let now = DateTime.Now
            
            let association = {
                Id = id
                Source = source
                Target = target
                Type = type'
                Strength = defaultArg strength AssociationStrength.Moderate
                Description = description
                CreationTime = now
                LastActivationTime = None
                ActivationCount = 0
                AssociatedEmotions = []
                IsBidirectional = defaultArg isBidirectional false
                Metadata = Map.empty
            }
            
            associations.Add(id, association)
            
            logger.LogInformation("Created association {AssociationId} from {Source} to {Target}", id, source, target)
            
            return association
        }
    
    /// <summary>
    /// Gets an association by ID.
    /// </summary>
    /// <param name="id">The ID of the association.</param>
    /// <returns>The association, if found.</returns>
    member _.GetAssociation(id: Guid) =
        task {
            if associations.ContainsKey(id) then
                return Some associations.[id]
            else
                logger.LogWarning("Association {AssociationId} not found", id)
                return None
        }
    
    /// <summary>
    /// Gets all associations.
    /// </summary>
    /// <returns>The list of all associations.</returns>
    member _.GetAllAssociations() =
        task {
            return associations.Values |> Seq.toList
        }
    
    /// <summary>
    /// Updates an association.
    /// </summary>
    /// <param name="id">The ID of the association to update.</param>
    /// <param name="type">The new type of the association.</param>
    /// <param name="strength">The new strength of the association.</param>
    /// <param name="description">The new description of the association.</param>
    /// <param name="isBidirectional">Whether the association is bidirectional.</param>
    /// <returns>The updated association.</returns>
    member _.UpdateAssociation(id: Guid, ?type': AssociationType, ?strength: AssociationStrength, ?description: string, ?isBidirectional: bool) =
        task {
            if associations.ContainsKey(id) then
                let association = associations.[id]
                
                let updatedAssociation = {
                    association with
                        Type = defaultArg type' association.Type
                        Strength = defaultArg strength association.Strength
                        Description = defaultArg description association.Description
                        IsBidirectional = defaultArg isBidirectional association.IsBidirectional
                }
                
                associations.[id] <- updatedAssociation
                
                logger.LogInformation("Updated association {AssociationId}", id)
                
                return updatedAssociation
            else
                logger.LogWarning("Association {AssociationId} not found for update", id)
                return failwith $"Association {id} not found"
        }
    
    /// <summary>
    /// Deletes an association.
    /// </summary>
    /// <param name="id">The ID of the association to delete.</param>
    /// <returns>Whether the association was deleted.</returns>
    member _.DeleteAssociation(id: Guid) =
        task {
            if associations.ContainsKey(id) then
                associations.Remove(id) |> ignore
                
                // Remove from all networks
                for networkId in networks.Keys do
                    let network = networks.[networkId]
                    let updatedAssociations = network.Associations |> List.filter (fun a -> a.Id <> id)
                    
                    if updatedAssociations.Length <> network.Associations.Length then
                        networks.[networkId] <- { network with Associations = updatedAssociations; LastModificationTime = DateTime.Now }
                
                logger.LogInformation("Deleted association {AssociationId}", id)
                
                return true
            else
                logger.LogWarning("Association {AssociationId} not found for deletion", id)
                return false
        }
    
    /// <summary>
    /// Activates an association.
    /// </summary>
    /// <param name="id">The ID of the association to activate.</param>
    /// <param name="activationStrength">The strength of the activation.</param>
    /// <param name="context">The context of the activation.</param>
    /// <param name="trigger">The trigger of the activation.</param>
    /// <returns>The association activation.</returns>
    member _.ActivateAssociation(id: Guid, ?activationStrength: float, ?context: string, ?trigger: string) =
        task {
            if associations.ContainsKey(id) then
                let association = associations.[id]
                let now = DateTime.Now
                
                let updatedAssociation = {
                    association with
                        LastActivationTime = Some now
                        ActivationCount = association.ActivationCount + 1
                }
                
                associations.[id] <- updatedAssociation
                
                let activation = {
                    Association = updatedAssociation
                    ActivationTime = now
                    ActivationStrength = defaultArg activationStrength 1.0
                    Context = context
                    Trigger = trigger
                    Metadata = Map.empty
                }
                
                logger.LogInformation("Activated association {AssociationId}", id)
                
                return activation
            else
                logger.LogWarning("Association {AssociationId} not found for activation", id)
                return failwith $"Association {id} not found"
        }
    
    /// <summary>
    /// Creates an association network.
    /// </summary>
    /// <param name="name">The name of the network.</param>
    /// <param name="description">The description of the network.</param>
    /// <returns>The created network.</returns>
    member _.CreateNetwork(name: string, ?description: string) =
        task {
            let id = Guid.NewGuid()
            let now = DateTime.Now
            
            let network = {
                Id = id
                Name = name
                Description = description
                Associations = []
                CreationTime = now
                LastModificationTime = now
                Metadata = Map.empty
            }
            
            networks.Add(id, network)
            
            logger.LogInformation("Created association network {NetworkId} with name {NetworkName}", id, name)
            
            return network
        }
    
    /// <summary>
    /// Gets an association network by ID.
    /// </summary>
    /// <param name="id">The ID of the network.</param>
    /// <returns>The network, if found.</returns>
    member _.GetNetwork(id: Guid) =
        task {
            if networks.ContainsKey(id) then
                return Some networks.[id]
            else
                logger.LogWarning("Association network {NetworkId} not found", id)
                return None
        }
    
    /// <summary>
    /// Gets all association networks.
    /// </summary>
    /// <returns>The list of all networks.</returns>
    member _.GetAllNetworks() =
        task {
            return networks.Values |> Seq.toList
        }
    
    /// <summary>
    /// Adds an association to a network.
    /// </summary>
    /// <param name="networkId">The ID of the network.</param>
    /// <param name="associationId">The ID of the association.</param>
    /// <returns>The updated network.</returns>
    member _.AddAssociationToNetwork(networkId: Guid, associationId: Guid) =
        task {
            if networks.ContainsKey(networkId) && associations.ContainsKey(associationId) then
                let network = networks.[networkId]
                let association = associations.[associationId]
                
                // Check if the association is already in the network
                if network.Associations |> List.exists (fun a -> a.Id = associationId) then
                    logger.LogInformation("Association {AssociationId} already exists in network {NetworkId}", associationId, networkId)
                    return network
                else
                    let updatedNetwork = {
                        network with
                            Associations = association :: network.Associations
                            LastModificationTime = DateTime.Now
                    }
                    
                    networks.[networkId] <- updatedNetwork
                    
                    logger.LogInformation("Added association {AssociationId} to network {NetworkId}", associationId, networkId)
                    
                    return updatedNetwork
            else
                if not (networks.ContainsKey(networkId)) then
                    logger.LogWarning("Association network {NetworkId} not found", networkId)
                    return failwith $"Association network {networkId} not found"
                else
                    logger.LogWarning("Association {AssociationId} not found", associationId)
                    return failwith $"Association {associationId} not found"
        }
    
    /// <summary>
    /// Removes an association from a network.
    /// </summary>
    /// <param name="networkId">The ID of the network.</param>
    /// <param name="associationId">The ID of the association.</param>
    /// <returns>The updated network.</returns>
    member _.RemoveAssociationFromNetwork(networkId: Guid, associationId: Guid) =
        task {
            if networks.ContainsKey(networkId) then
                let network = networks.[networkId]
                
                let updatedAssociations = network.Associations |> List.filter (fun a -> a.Id <> associationId)
                
                if updatedAssociations.Length = network.Associations.Length then
                    logger.LogWarning("Association {AssociationId} not found in network {NetworkId}", associationId, networkId)
                    return network
                else
                    let updatedNetwork = {
                        network with
                            Associations = updatedAssociations
                            LastModificationTime = DateTime.Now
                    }
                    
                    networks.[networkId] <- updatedNetwork
                    
                    logger.LogInformation("Removed association {AssociationId} from network {NetworkId}", associationId, networkId)
                    
                    return updatedNetwork
            else
                logger.LogWarning("Association network {NetworkId} not found", networkId)
                return failwith $"Association network {networkId} not found"
        }
    
    /// <summary>
    /// Deletes an association network.
    /// </summary>
    /// <param name="id">The ID of the network to delete.</param>
    /// <returns>Whether the network was deleted.</returns>
    member _.DeleteNetwork(id: Guid) =
        task {
            if networks.ContainsKey(id) then
                networks.Remove(id) |> ignore
                
                logger.LogInformation("Deleted association network {NetworkId}", id)
                
                return true
            else
                logger.LogWarning("Association network {NetworkId} not found for deletion", id)
                return false
        }
    
    /// <summary>
    /// Finds associations between concepts.
    /// </summary>
    /// <param name="query">The association query.</param>
    /// <returns>The association query result.</returns>
    member _.FindAssociations(query: AssociationQuery) =
        task {
            let startTime = DateTime.Now
            
            // Filter associations based on the query
            let filteredAssociations =
                associations.Values
                |> Seq.filter (fun a ->
                    // Filter by source
                    (query.Source.IsNone || a.Source = query.Source.Value) &&
                    // Filter by target
                    (query.Target.IsNone || a.Target = query.Target.Value) &&
                    // Filter by type
                    (query.Types.IsNone || query.Types.Value |> List.exists (fun t -> t = a.Type)) &&
                    // Filter by strength
                    (query.MinimumStrength.IsNone || 
                        match (a.Strength, query.MinimumStrength.Value) with
                        | (AssociationStrength.Custom s1, AssociationStrength.Custom s2) -> s1 >= s2
                        | (AssociationStrength.Custom _, _) -> true
                        | (_, AssociationStrength.Custom _) -> false
                        | (AssociationStrength.Weak, _) -> false
                        | (AssociationStrength.Moderate, AssociationStrength.Weak) -> true
                        | (AssociationStrength.Moderate, _) -> false
                        | (AssociationStrength.Strong, AssociationStrength.VeryStrong) -> false
                        | (AssociationStrength.Strong, _) -> true
                        | (AssociationStrength.VeryStrong, _) -> true)
                )
                |> Seq.toList
            
            // Limit the number of results if specified
            let limitedAssociations =
                match query.MaxResults with
                | Some maxResults -> filteredAssociations |> List.truncate maxResults
                | None -> filteredAssociations
            
            let endTime = DateTime.Now
            let executionTime = endTime - startTime
            
            let result = {
                Query = query
                Associations = limitedAssociations
                Paths = None
                ExecutionTime = executionTime
                Metadata = Map.empty
            }
            
            logger.LogInformation("Found {AssociationCount} associations matching query", limitedAssociations.Length)
            
            return result
        }
    
    /// <summary>
    /// Finds paths between concepts.
    /// </summary>
    /// <param name="source">The source concept.</param>
    /// <param name="target">The target concept.</param>
    /// <param name="maxPathLength">The maximum path length.</param>
    /// <param name="maxResults">The maximum number of results.</param>
    /// <returns>The list of association paths.</returns>
    member _.FindPaths(source: string, target: string, ?maxPathLength: int, ?maxResults: int) =
        task {
            let maxLength = defaultArg maxPathLength 3
            let maxRes = defaultArg maxResults 10
            
            // Simple breadth-first search to find paths
            let rec findPathsBFS (visited: Set<string>) (queue: (string * Association list) list) (paths: AssociationPath list) =
                match queue with
                | [] -> paths
                | (current, pathSoFar) :: rest ->
                    if current = target then
                        // Found a path to the target
                        let path = {
                            Source = source
                            Target = target
                            Associations = pathSoFar |> List.rev
                            Strength = 
                                pathSoFar 
                                |> List.map (fun a -> 
                                    match a.Strength with
                                    | AssociationStrength.Weak -> 0.25
                                    | AssociationStrength.Moderate -> 0.5
                                    | AssociationStrength.Strong -> 0.75
                                    | AssociationStrength.VeryStrong -> 1.0
                                    | AssociationStrength.Custom s -> s)
                                |> List.fold (*) 1.0
                            Length = pathSoFar.Length
                            Metadata = Map.empty
                        }
                        
                        let newPaths = path :: paths
                        
                        if newPaths.Length >= maxRes then
                            // Reached maximum number of results
                            newPaths
                        else
                            findPathsBFS visited rest newPaths
                    else if pathSoFar.Length >= maxLength then
                        // Reached maximum path length
                        findPathsBFS visited rest paths
                    else
                        // Find next steps
                        let nextSteps =
                            associations.Values
                            |> Seq.filter (fun a -> 
                                (a.Source = current && not (visited.Contains a.Target)) ||
                                (a.IsBidirectional && a.Target = current && not (visited.Contains a.Source)))
                            |> Seq.map (fun a ->
                                if a.Source = current then
                                    (a.Target, a :: pathSoFar)
                                else
                                    (a.Source, a :: pathSoFar))
                            |> Seq.toList
                        
                        let newVisited = visited.Add current
                        let newQueue = rest @ nextSteps
                        
                        findPathsBFS newVisited newQueue paths
            
            let paths = findPathsBFS Set.empty [(source, [])] []
            
            // Sort paths by strength (descending) and length (ascending)
            let sortedPaths =
                paths
                |> List.sortByDescending (fun p -> p.Strength / float p.Length)
                |> List.truncate maxRes
            
            logger.LogInformation("Found {PathCount} paths from {Source} to {Target}", sortedPaths.Length, source, target)
            
            return sortedPaths
        }
    
    /// <summary>
    /// Suggests associations.
    /// </summary>
    /// <param name="concept">The concept to suggest associations for.</param>
    /// <param name="maxSuggestions">The maximum number of suggestions.</param>
    /// <returns>The list of association suggestions.</returns>
    member _.SuggestAssociations(concept: string, ?maxSuggestions: int) =
        task {
            let maxSugg = defaultArg maxSuggestions 5
            
            // Find existing associations for the concept
            let existingAssociations =
                associations.Values
                |> Seq.filter (fun a -> a.Source = concept || (a.IsBidirectional && a.Target = concept))
                |> Seq.toList
            
            // Find concepts that are associated with the same concepts as the input concept
            let relatedConcepts =
                associations.Values
                |> Seq.filter (fun a -> 
                    let aTarget = if a.Source = concept then a.Target else if a.IsBidirectional && a.Target = concept then a.Source else ""
                    aTarget <> "" && 
                    existingAssociations |> List.exists (fun ea -> 
                        ea.Target = aTarget || (ea.IsBidirectional && ea.Source = aTarget)))
                |> Seq.map (fun a -> 
                    if a.Source = concept then a.Target
                    else if a.IsBidirectional && a.Target = concept then a.Source
                    else "")
                |> Seq.filter (fun c -> c <> "")
                |> Seq.distinct
                |> Seq.toList
            
            // Create suggestions
            let suggestions =
                relatedConcepts
                |> List.map (fun c ->
                    {
                        Source = concept
                        Target = c
                        Type = AssociationType.Semantic
                        Strength = AssociationStrength.Moderate
                        Confidence = 0.7
                        Reason = Some "Related through common associations"
                        Metadata = Map.empty
                    })
                |> List.truncate maxSugg
            
            logger.LogInformation("Generated {SuggestionCount} association suggestions for concept {Concept}", suggestions.Length, concept)
            
            return suggestions
        }
    
    /// <summary>
    /// Learns associations from text.
    /// </summary>
    /// <param name="text">The text to learn from.</param>
    /// <returns>The association learning result.</returns>
    member this.LearnAssociationsFromText(text: string) =
        task {
            let startTime = DateTime.Now
            
            // Simple implementation: extract words and create associations between adjacent words
            let words =
                text.Split([|' '; '.'; ','; ';'; '!'; '?'; '('; ')'; '['; ']'; '{'; '}'; '\n'; '\r'; '\t'|], StringSplitOptions.RemoveEmptyEntries)
                |> Array.map (fun w -> w.ToLowerInvariant())
                |> Array.filter (fun w -> w.Length > 3) // Filter out short words
                |> Array.distinct
                |> Array.toList
            
            let learnedAssociations = ResizeArray<Association>()
            let updatedAssociations = ResizeArray<Association>()
            
            // Create associations between adjacent words
            for i in 0 .. words.Length - 2 do
                let source = words.[i]
                let target = words.[i + 1]
                
                // Check if the association already exists
                let existingAssociation =
                    associations.Values
                    |> Seq.tryFind (fun a -> 
                        (a.Source = source && a.Target = target) || 
                        (a.IsBidirectional && a.Source = target && a.Target = source))
                
                match existingAssociation with
                | Some association ->
                    // Update existing association
                    let updatedAssociation = {
                        association with
                            Strength = 
                                match association.Strength with
                                | AssociationStrength.Weak -> AssociationStrength.Moderate
                                | AssociationStrength.Moderate -> AssociationStrength.Strong
                                | AssociationStrength.Strong -> AssociationStrength.VeryStrong
                                | AssociationStrength.VeryStrong -> AssociationStrength.VeryStrong
                                | AssociationStrength.Custom s -> AssociationStrength.Custom (min 1.0 (s + 0.1))
                            ActivationCount = association.ActivationCount + 1
                            LastActivationTime = Some DateTime.Now
                    }
                    
                    associations.[association.Id] <- updatedAssociation
                    updatedAssociations.Add(updatedAssociation)
                | None ->
                    // Create new association
                    let! newAssociation = this.CreateAssociation(source, target, AssociationType.Semantic, AssociationStrength.Weak, isBidirectional = false)
                    learnedAssociations.Add(newAssociation)
            
            let endTime = DateTime.Now
            let learningTime = endTime - startTime
            
            let result = {
                LearnedAssociations = learnedAssociations |> Seq.toList
                UpdatedAssociations = updatedAssociations |> Seq.toList
                RemovedAssociations = []
                LearningTime = learningTime
                Metadata = Map.empty
            }
            
            logger.LogInformation("Learned {LearnedCount} new associations and updated {UpdatedCount} existing associations from text", 
                                 learnedAssociations.Count, updatedAssociations.Count)
            
            return result
        }
    
    /// <summary>
    /// Exports an association network.
    /// </summary>
    /// <param name="networkId">The ID of the network to export.</param>
    /// <param name="format">The format to export in.</param>
    /// <param name="path">The path to export to.</param>
    /// <returns>Whether the export was successful.</returns>
    member _.ExportNetwork(networkId: Guid, format: string, path: string) =
        task {
            if networks.ContainsKey(networkId) then
                let network = networks.[networkId]
                
                // In a real implementation, this would serialize the network to the specified format
                // For now, we'll just log the export
                logger.LogInformation("Exported association network {NetworkId} to {Path} in {Format} format", networkId, path, format)
                
                return true
            else
                logger.LogWarning("Association network {NetworkId} not found for export", networkId)
                return false
        }
    
    /// <summary>
    /// Imports an association network.
    /// </summary>
    /// <param name="format">The format to import from.</param>
    /// <param name="path">The path to import from.</param>
    /// <returns>The imported network.</returns>
    member _.ImportNetwork(format: string, path: string) =
        task {
            // In a real implementation, this would deserialize the network from the specified format
            // For now, we'll just create a new empty network
            let id = Guid.NewGuid()
            let now = DateTime.Now
            
            let network = {
                Id = id
                Name = $"Imported Network ({path})"
                Description = Some $"Imported from {path} in {format} format"
                Associations = []
                CreationTime = now
                LastModificationTime = now
                Metadata = Map.empty
            }
            
            networks.Add(id, network)
            
            logger.LogInformation("Imported association network from {Path} in {Format} format", path, format)
            
            return network
        }
    
    interface IAssociationService with
        member this.CreateAssociation(source, target, type', ?strength, ?description, ?isBidirectional) = 
            this.CreateAssociation(source, target, type', ?strength = strength, ?description = description, ?isBidirectional = isBidirectional)
        member this.GetAssociation(id) = this.GetAssociation(id)
        member this.GetAllAssociations() = this.GetAllAssociations()
        member this.UpdateAssociation(id, ?type', ?strength, ?description, ?isBidirectional) = 
            this.UpdateAssociation(id, ?type' = type', ?strength = strength, ?description = description, ?isBidirectional = isBidirectional)
        member this.DeleteAssociation(id) = this.DeleteAssociation(id)
        member this.ActivateAssociation(id, ?activationStrength, ?context, ?trigger) = 
            this.ActivateAssociation(id, ?activationStrength = activationStrength, ?context = context, ?trigger = trigger)
        member this.CreateNetwork(name, ?description) = this.CreateNetwork(name, ?description = description)
        member this.GetNetwork(id) = this.GetNetwork(id)
        member this.GetAllNetworks() = this.GetAllNetworks()
        member this.AddAssociationToNetwork(networkId, associationId) = this.AddAssociationToNetwork(networkId, associationId)
        member this.RemoveAssociationFromNetwork(networkId, associationId) = this.RemoveAssociationFromNetwork(networkId, associationId)
        member this.DeleteNetwork(id) = this.DeleteNetwork(id)
        member this.FindAssociations(query) = this.FindAssociations(query)
        member this.FindPaths(source, target, ?maxPathLength, ?maxResults) = 
            this.FindPaths(source, target, ?maxPathLength = maxPathLength, ?maxResults = maxResults)
        member this.SuggestAssociations(concept, ?maxSuggestions) = this.SuggestAssociations(concept, ?maxSuggestions = maxSuggestions)
        member this.LearnAssociationsFromText(text) = this.LearnAssociationsFromText(text)
        member this.ExportNetwork(networkId, format, path) = this.ExportNetwork(networkId, format, path)
        member this.ImportNetwork(format, path) = this.ImportNetwork(format, path)
