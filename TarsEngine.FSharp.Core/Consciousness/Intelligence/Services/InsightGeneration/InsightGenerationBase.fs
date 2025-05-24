namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.InsightGeneration

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence
open TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

/// <summary>
/// Base implementation of the insight generation capabilities.
/// </summary>
type InsightGenerationBase(logger: ILogger<InsightGenerationBase>) =
    // State variables
    let mutable isInitialized = false
    let mutable isActive = false
    let mutable insightLevel = 0.5 // Starting with moderate insight
    let mutable connectionDiscoveryLevel = 0.6 // Starting with moderate connection discovery
    let mutable problemRestructuringLevel = 0.5 // Starting with moderate problem restructuring
    let mutable incubationLevel = 0.4 // Starting with moderate incubation
    let mutable insights = List.empty<Insight>
    let mutable conceptConnections = Map.empty<string, string list>
    let random = System.Random()
    let mutable lastInsightTime = DateTime.MinValue
    
    /// <summary>
    /// Gets the insight level (0.0 to 1.0).
    /// </summary>
    member _.InsightLevel = insightLevel
    
    /// <summary>
    /// Gets the connection discovery level (0.0 to 1.0).
    /// </summary>
    member _.ConnectionDiscoveryLevel = connectionDiscoveryLevel
    
    /// <summary>
    /// Gets the problem restructuring level (0.0 to 1.0).
    /// </summary>
    member _.ProblemRestructuringLevel = problemRestructuringLevel
    
    /// <summary>
    /// Gets the incubation level (0.0 to 1.0).
    /// </summary>
    member _.IncubationLevel = incubationLevel
    
    /// <summary>
    /// Gets the insights.
    /// </summary>
    member _.Insights = insights
    
    /// <summary>
    /// Gets the concept connections.
    /// </summary>
    member _.ConceptConnections = conceptConnections
    
    /// <summary>
    /// Initializes the insight generation.
    /// </summary>
    /// <returns>True if initialization was successful.</returns>
    member _.InitializeAsync() =
        task {
            try
                logger.LogInformation("Initializing insight generation")
                
                // Initialize state
                isInitialized <- true
                
                logger.LogInformation("Insight generation initialized successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error initializing insight generation")
                return false
        }
    
    /// <summary>
    /// Activates the insight generation.
    /// </summary>
    /// <returns>True if activation was successful.</returns>
    member _.ActivateAsync() =
        task {
            if not isInitialized then
                logger.LogWarning("Cannot activate insight generation: not initialized")
                return false
            
            if isActive then
                logger.LogInformation("Insight generation is already active")
                return true
            
            try
                logger.LogInformation("Activating insight generation")
                
                // Activate state
                isActive <- true
                
                logger.LogInformation("Insight generation activated successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error activating insight generation")
                return false
        }
    
    /// <summary>
    /// Deactivates the insight generation.
    /// </summary>
    /// <returns>True if deactivation was successful.</returns>
    member _.DeactivateAsync() =
        task {
            if not isActive then
                logger.LogInformation("Insight generation is already inactive")
                return true
            
            try
                logger.LogInformation("Deactivating insight generation")
                
                // Deactivate state
                isActive <- false
                
                logger.LogInformation("Insight generation deactivated successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error deactivating insight generation")
                return false
        }
    
    /// <summary>
    /// Updates the insight generation.
    /// </summary>
    /// <returns>True if update was successful.</returns>
    member _.UpdateAsync() =
        task {
            if not isInitialized then
                logger.LogWarning("Cannot update insight generation: not initialized")
                return false
            
            try
                // Gradually increase insight levels over time (very slowly)
                if insightLevel < 0.95 then
                    insightLevel <- insightLevel + 0.0001 * random.NextDouble()
                    insightLevel <- Math.Min(insightLevel, 1.0)
                
                if connectionDiscoveryLevel < 0.95 then
                    connectionDiscoveryLevel <- connectionDiscoveryLevel + 0.0001 * random.NextDouble()
                    connectionDiscoveryLevel <- Math.Min(connectionDiscoveryLevel, 1.0)
                
                if problemRestructuringLevel < 0.95 then
                    problemRestructuringLevel <- problemRestructuringLevel + 0.0001 * random.NextDouble()
                    problemRestructuringLevel <- Math.Min(problemRestructuringLevel, 1.0)
                
                if incubationLevel < 0.95 then
                    incubationLevel <- incubationLevel + 0.0001 * random.NextDouble()
                    incubationLevel <- Math.Min(incubationLevel, 1.0)
                
                return true
            with
            | ex ->
                logger.LogError(ex, "Error updating insight generation")
                return false
        }
    
    /// <summary>
    /// Gets recent insights.
    /// </summary>
    /// <param name="count">The number of insights to get.</param>
    /// <returns>The recent insights.</returns>
    member _.GetRecentInsights(count: int) =
        insights
        |> List.sortByDescending (fun insight -> insight.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets insights by method.
    /// </summary>
    /// <param name="method">The insight generation method.</param>
    /// <param name="count">The number of insights to get.</param>
    /// <returns>The insights generated by the specified method.</returns>
    member _.GetInsightsByMethod(method: InsightGenerationMethod, count: int) =
        insights
        |> List.filter (fun insight -> insight.Method = method)
        |> List.sortByDescending (fun insight -> insight.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets insights by tag.
    /// </summary>
    /// <param name="tag">The tag.</param>
    /// <param name="count">The number of insights to get.</param>
    /// <returns>The insights with the tag.</returns>
    member _.GetInsightsByTag(tag: string, count: int) =
        insights
        |> List.filter (fun insight -> insight.Tags |> List.exists (fun t -> t.Contains(tag, StringComparison.OrdinalIgnoreCase)))
        |> List.sortByDescending (fun insight -> insight.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets the most significant insights.
    /// </summary>
    /// <param name="count">The number of insights to get.</param>
    /// <returns>The most significant insights.</returns>
    member _.GetMostSignificantInsights(count: int) =
        insights
        |> List.sortByDescending (fun insight -> insight.Significance)
        |> List.truncate count
    
    /// <summary>
    /// Adds a concept connection.
    /// </summary>
    /// <param name="concept1">The first concept.</param>
    /// <param name="concept2">The second concept.</param>
    /// <returns>True if the connection was added.</returns>
    member _.AddConceptConnectionAsync(concept1: string, concept2: string) =
        task {
            try
                logger.LogInformation("Adding concept connection between: {Concept1} and {Concept2}", concept1, concept2)
                
                // Normalize concepts
                let normalizedConcept1 = concept1.Trim().ToLowerInvariant()
                let normalizedConcept2 = concept2.Trim().ToLowerInvariant()
                
                // Skip if concepts are the same
                if normalizedConcept1 = normalizedConcept2 then
                    logger.LogWarning("Cannot connect a concept to itself: {Concept}", normalizedConcept1)
                    return false
                
                // Update connections for concept1
                let connections1 = 
                    match Map.tryFind normalizedConcept1 conceptConnections with
                    | Some connections -> 
                        if List.contains normalizedConcept2 connections then
                            connections // Already connected
                        else
                            normalizedConcept2 :: connections
                    | None -> 
                        [normalizedConcept2]
                
                // Update connections for concept2
                let connections2 = 
                    match Map.tryFind normalizedConcept2 conceptConnections with
                    | Some connections -> 
                        if List.contains normalizedConcept1 connections then
                            connections // Already connected
                        else
                            normalizedConcept1 :: connections
                    | None -> 
                        [normalizedConcept1]
                
                // Update the concept connections map
                conceptConnections <- 
                    conceptConnections
                    |> Map.add normalizedConcept1 connections1
                    |> Map.add normalizedConcept2 connections2
                
                logger.LogInformation("Added concept connection between: {Concept1} and {Concept2}", 
                                     normalizedConcept1, normalizedConcept2)
                
                return true
            with
            | ex ->
                logger.LogError(ex, "Error adding concept connection")
                return false
        }
    
    /// <summary>
    /// Gets connected concepts.
    /// </summary>
    /// <param name="concept">The concept.</param>
    /// <returns>The connected concepts.</returns>
    member _.GetConnectedConcepts(concept: string) =
        // Normalize concept
        let normalizedConcept = concept.Trim().ToLowerInvariant()
        
        // Get connections
        match Map.tryFind normalizedConcept conceptConnections with
        | Some connections -> connections
        | None -> []
    
    /// <summary>
    /// Calculates the concept distance.
    /// </summary>
    /// <param name="concept1">The first concept.</param>
    /// <param name="concept2">The second concept.</param>
    /// <returns>The concept distance (0.0 to 1.0).</returns>
    member this.CalculateConceptDistance(concept1: string, concept2: string) =
        // Normalize concepts
        let normalizedConcept1 = concept1.Trim().ToLowerInvariant()
        let normalizedConcept2 = concept2.Trim().ToLowerInvariant()
        
        // If concepts are the same, distance is 0
        if normalizedConcept1 = normalizedConcept2 then
            0.0
        else
            // Breadth-first search to find shortest path
            let rec bfs (queue: (string * int) list) (visited: Set<string>) =
                match queue with
                | [] -> 
                    // No path found, maximum distance
                    1.0
                | (current, distance) :: rest ->
                    if current = normalizedConcept2 then
                        // Path found, calculate normalized distance
                        // Max distance considered is 6 (small world theory)
                        let maxDistance = 6.0
                        Math.Min(float distance / maxDistance, 1.0)
                    else
                        // Get connected concepts not yet visited
                        let connections = 
                            match Map.tryFind current conceptConnections with
                            | Some conns -> conns
                            | None -> []
                        
                        let newConnections = 
                            connections
                            |> List.filter (fun c -> not (Set.contains c visited))
                            |> List.map (fun c -> (c, distance + 1))
                        
                        // Add to visited
                        let newVisited = 
                            newConnections
                            |> List.fold (fun vs (c, _) -> Set.add c vs) visited
                        
                        // Continue search
                        bfs (rest @ newConnections) newVisited
            
            // Start search from concept1
            bfs [(normalizedConcept1, 0)] (Set.singleton normalizedConcept1)
    
    /// <summary>
    /// Adds an insight.
    /// </summary>
    /// <param name="insight">The insight to add.</param>
    member _.AddInsight(insight: Insight) =
        insights <- insight :: insights
        lastInsightTime <- DateTime.UtcNow
    
    /// <summary>
    /// Gets whether the insight generation is initialized.
    /// </summary>
    member _.IsInitialized = isInitialized
    
    /// <summary>
    /// Gets whether the insight generation is active.
    /// </summary>
    member _.IsActive = isActive
