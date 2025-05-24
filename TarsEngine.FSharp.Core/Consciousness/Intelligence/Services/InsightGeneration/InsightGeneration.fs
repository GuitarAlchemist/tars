namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence
open TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.InsightGeneration

/// <summary>
/// Implementation of the insight generation capabilities.
/// </summary>
type InsightGeneration(logger: ILogger<InsightGeneration>) =
    inherit InsightGenerationBase(logger)
    
    let random = System.Random()
    let mutable lastInsightTime = DateTime.MinValue
    
    /// <summary>
    /// Generates an insight.
    /// </summary>
    /// <returns>The generated insight.</returns>
    member this.GenerateInsightAsync() =
        task {
            if not this.IsInitialized || not this.IsActive then
                return None
            
            // Only generate insights periodically
            if (DateTime.UtcNow - lastInsightTime).TotalSeconds < 60 then
                return None
            
            try
                logger.LogDebug("Generating insight")
                
                // Choose a method based on current levels
                let method = 
                    let connectionProb = this.ConnectionDiscoveryLevel * 0.6
                    let restructuringProb = this.ProblemRestructuringLevel * 0.4
                    
                    // Normalize probabilities
                    let total = connectionProb + restructuringProb
                    let connectionProb = connectionProb / total
                    
                    // Choose method based on probabilities
                    if random.NextDouble() < connectionProb then
                        InsightGenerationMethod.ConnectionDiscovery
                    else
                        InsightGenerationMethod.ProblemRestructuring
                
                // Generate insight based on method
                let insightOption = 
                    match method with
                    | InsightGenerationMethod.ConnectionDiscovery ->
                        // Generate some sample ideas for connection discovery
                        let ideas = [
                            "Systems exhibit emergent properties that arise from interactions between components"
                            "Feedback loops can create both stability and instability in complex systems"
                            "Adaptation occurs through iterative cycles of variation, selection, and amplification"
                            "Information processing happens at multiple levels of organization simultaneously"
                            "Networks display characteristic patterns of connectivity and centrality"
                            "Self-organization emerges from local interactions following simple rules"
                            "Boundaries define system identity while enabling selective exchange with environment"
                            "Hierarchies of scale and function create nested levels of complexity"
                        ]
                        
                        ConnectionDiscovery.generateConnectionInsight ideas this.ConnectionDiscoveryLevel random
                    
                    | InsightGenerationMethod.ProblemRestructuring ->
                        // Generate a sample problem for restructuring
                        let problem = 
                            "How can we optimize the current system to improve efficiency and reduce errors? " +
                            "We need to identify bottlenecks and implement solutions that minimize resource usage " +
                            "while maintaining reliability. The system must continue operating during any changes, " +
                            "and all existing interfaces must be preserved."
                        
                        ProblemRestructuring.generateProblemRestructuringInsight problem this.ProblemRestructuringLevel random
                    
                    | _ ->
                        // Default to connection discovery for unknown methods
                        let ideas = [
                            "Systems exhibit emergent properties that arise from interactions between components"
                            "Feedback loops can create both stability and instability in complex systems"
                        ]
                        
                        ConnectionDiscovery.generateConnectionInsight ideas this.ConnectionDiscoveryLevel random
                
                // If insight was generated, add it to the list
                match insightOption with
                | Some insight ->
                    this.AddInsight(insight)
                    lastInsightTime <- DateTime.UtcNow
                    
                    logger.LogInformation("Generated insight: {Description} (Method: {Method}, Significance: {Significance:F2})",
                                         insight.Description, insight.Method, insight.Significance)
                    
                    return Some insight
                | None ->
                    logger.LogWarning("Failed to generate insight")
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error generating insight")
                return None
        }
    
    /// <summary>
    /// Generates an insight by a specific method.
    /// </summary>
    /// <param name="method">The insight generation method.</param>
    /// <returns>The generated insight.</returns>
    member this.GenerateInsightByMethodAsync(method: InsightGenerationMethod) =
        task {
            if not this.IsInitialized || not this.IsActive then
                logger.LogWarning("Cannot generate insight: insight generation not initialized or active")
                return None
            
            try
                logger.LogInformation("Generating insight using method: {Method}", method)
                
                // Generate insight based on method
                let insightOption = 
                    match method with
                    | InsightGenerationMethod.ConnectionDiscovery ->
                        // Generate some sample ideas for connection discovery
                        let ideas = [
                            "Systems exhibit emergent properties that arise from interactions between components"
                            "Feedback loops can create both stability and instability in complex systems"
                            "Adaptation occurs through iterative cycles of variation, selection, and amplification"
                            "Information processing happens at multiple levels of organization simultaneously"
                            "Networks display characteristic patterns of connectivity and centrality"
                            "Self-organization emerges from local interactions following simple rules"
                            "Boundaries define system identity while enabling selective exchange with environment"
                            "Hierarchies of scale and function create nested levels of complexity"
                        ]
                        
                        ConnectionDiscovery.generateConnectionInsight ideas this.ConnectionDiscoveryLevel random
                    
                    | InsightGenerationMethod.ProblemRestructuring ->
                        // Generate a sample problem for restructuring
                        let problem = 
                            "How can we optimize the current system to improve efficiency and reduce errors? " +
                            "We need to identify bottlenecks and implement solutions that minimize resource usage " +
                            "while maintaining reliability. The system must continue operating during any changes, " +
                            "and all existing interfaces must be preserved."
                        
                        ProblemRestructuring.generateProblemRestructuringInsight problem this.ProblemRestructuringLevel random
                    
                    | _ ->
                        // Default to connection discovery for unknown methods
                        let ideas = [
                            "Systems exhibit emergent properties that arise from interactions between components"
                            "Feedback loops can create both stability and instability in complex systems"
                        ]
                        
                        ConnectionDiscovery.generateConnectionInsight ideas this.ConnectionDiscoveryLevel random
                
                // If insight was generated, add it to the list
                match insightOption with
                | Some insight ->
                    this.AddInsight(insight)
                    lastInsightTime <- DateTime.UtcNow
                    
                    logger.LogInformation("Generated insight: {Description} (Method: {Method}, Significance: {Significance:F2})",
                                         insight.Description, insight.Method, insight.Significance)
                    
                    return Some insight
                | None ->
                    logger.LogWarning("Failed to generate insight using method: {Method}", method)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error generating insight by method")
                return None
        }
    
    /// <summary>
    /// Connects ideas for an insight.
    /// </summary>
    /// <param name="ideas">The ideas.</param>
    /// <returns>The insight.</returns>
    member this.ConnectIdeasForInsightAsync(ideas: string list) =
        task {
            if not this.IsInitialized || not this.IsActive then
                logger.LogWarning("Cannot connect ideas: insight generation not initialized or active")
                return None
            
            try
                logger.LogInformation("Connecting {Count} ideas for insight", ideas.Length)
                
                // Generate insight using connection discovery
                let insightOption = ConnectionDiscovery.generateConnectionInsight ideas this.ConnectionDiscoveryLevel random
                
                // If insight was generated, add it to the list
                match insightOption with
                | Some insight ->
                    this.AddInsight(insight)
                    lastInsightTime <- DateTime.UtcNow
                    
                    logger.LogInformation("Generated insight from connected ideas: {Description} (Significance: {Significance:F2})",
                                         insight.Description, insight.Significance)
                    
                    // Add concept connections
                    let concepts = ConnectionDiscovery.extractConcepts ideas
                    let connections = ConnectionDiscovery.findConceptConnections concepts ideas random
                    
                    // Add each connection
                    for (concept1, concept2) in connections do
                        let _ = this.AddConceptConnectionAsync(concept1, concept2) |> Async.AwaitTask |> Async.RunSynchronously
                        ()
                    
                    return Some insight
                | None ->
                    logger.LogWarning("Failed to generate insight from connected ideas")
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error connecting ideas for insight")
                return None
        }
    
    /// <summary>
    /// Restructures a problem for an insight.
    /// </summary>
    /// <param name="problem">The problem.</param>
    /// <returns>The insight.</returns>
    member this.RestructureProblemForInsightAsync(problem: string) =
        task {
            if not this.IsInitialized || not this.IsActive then
                logger.LogWarning("Cannot restructure problem: insight generation not initialized or active")
                return None
            
            try
                logger.LogInformation("Restructuring problem for insight: {Problem}", 
                                     if problem.Length > 100 then problem.Substring(0, 100) + "..." else problem)
                
                // Generate insight using problem restructuring
                let insightOption = ProblemRestructuring.generateProblemRestructuringInsight problem this.ProblemRestructuringLevel random
                
                // If insight was generated, add it to the list
                match insightOption with
                | Some insight ->
                    this.AddInsight(insight)
                    lastInsightTime <- DateTime.UtcNow
                    
                    logger.LogInformation("Generated insight from restructured problem: {Description} (Significance: {Significance:F2})",
                                         insight.Description, insight.Significance)
                    
                    return Some insight
                | None ->
                    logger.LogWarning("Failed to generate insight from restructured problem")
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error restructuring problem for insight")
                return None
        }
    
    interface IInsightGeneration with
        member this.InsightLevel = this.InsightLevel
        member this.ConnectionDiscoveryLevel = this.ConnectionDiscoveryLevel
        member this.ProblemRestructuringLevel = this.ProblemRestructuringLevel
        member this.IncubationLevel = this.IncubationLevel
        member this.Insights = this.Insights
        member this.ConceptConnections = this.ConceptConnections
        
        member this.InitializeAsync() = this.InitializeAsync()
        member this.ActivateAsync() = this.ActivateAsync()
        member this.DeactivateAsync() = this.DeactivateAsync()
        member this.UpdateAsync() = this.UpdateAsync()
        
        member this.GenerateInsightAsync() = this.GenerateInsightAsync()
        member this.GenerateInsightByMethodAsync(method) = this.GenerateInsightByMethodAsync(method)
        member this.ConnectIdeasForInsightAsync(ideas) = this.ConnectIdeasForInsightAsync(ideas)
        member this.RestructureProblemForInsightAsync(problem) = this.RestructureProblemForInsightAsync(problem)
        
        member this.GetRecentInsights(count) = this.GetRecentInsights(count)
        member this.GetInsightsByMethod(method, count) = this.GetInsightsByMethod(method, count)
        member this.GetInsightsByTag(tag, count) = this.GetInsightsByTag(tag, count)
        member this.GetMostSignificantInsights(count) = this.GetMostSignificantInsights(count)
        
        member this.AddConceptConnectionAsync(concept1, concept2) = this.AddConceptConnectionAsync(concept1, concept2)
        member this.GetConnectedConcepts(concept) = this.GetConnectedConcepts(concept)
        member this.CalculateConceptDistance(concept1, concept2) = this.CalculateConceptDistance(concept1, concept2)
