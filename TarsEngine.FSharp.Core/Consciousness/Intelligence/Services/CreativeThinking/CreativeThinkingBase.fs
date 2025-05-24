namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.CreativeThinking

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence
open TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

/// <summary>
/// Base implementation of the creative thinking capabilities.
/// </summary>
type CreativeThinkingBase(logger: ILogger<CreativeThinkingBase>) =
    // State variables
    let mutable isInitialized = false
    let mutable isActive = false
    let mutable creativityLevel = 0.5 // Starting with moderate creativity
    let mutable divergentThinkingLevel = 0.6 // Starting with moderate divergent thinking
    let mutable convergentThinkingLevel = 0.5 // Starting with moderate convergent thinking
    let mutable combinatorialCreativityLevel = 0.4 // Starting with moderate combinatorial creativity
    let mutable creativeIdeas = List.empty<CreativeIdea>
    let mutable creativeProcesses = List.empty<CreativeProcess>
    let random = System.Random()
    let mutable lastIdeaGenerationTime = DateTime.MinValue
    
    /// <summary>
    /// Gets the creativity level (0.0 to 1.0).
    /// </summary>
    member _.CreativityLevel = creativityLevel
    
    /// <summary>
    /// Gets the divergent thinking level (0.0 to 1.0).
    /// </summary>
    member _.DivergentThinkingLevel = divergentThinkingLevel
    
    /// <summary>
    /// Gets the convergent thinking level (0.0 to 1.0).
    /// </summary>
    member _.ConvergentThinkingLevel = convergentThinkingLevel
    
    /// <summary>
    /// Gets the combinatorial creativity level (0.0 to 1.0).
    /// </summary>
    member _.CombinatorialCreativityLevel = combinatorialCreativityLevel
    
    /// <summary>
    /// Gets the creative ideas.
    /// </summary>
    member _.CreativeIdeas = creativeIdeas
    
    /// <summary>
    /// Gets the creative processes.
    /// </summary>
    member _.CreativeProcesses = creativeProcesses
    
    /// <summary>
    /// Initializes the creative thinking.
    /// </summary>
    /// <returns>True if initialization was successful.</returns>
    member _.InitializeAsync() =
        task {
            try
                logger.LogInformation("Initializing creative thinking")
                
                // Initialize state
                isInitialized <- true
                
                logger.LogInformation("Creative thinking initialized successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error initializing creative thinking")
                return false
        }
    
    /// <summary>
    /// Activates the creative thinking.
    /// </summary>
    /// <returns>True if activation was successful.</returns>
    member _.ActivateAsync() =
        task {
            if not isInitialized then
                logger.LogWarning("Cannot activate creative thinking: not initialized")
                return false
            
            if isActive then
                logger.LogInformation("Creative thinking is already active")
                return true
            
            try
                logger.LogInformation("Activating creative thinking")
                
                // Activate state
                isActive <- true
                
                logger.LogInformation("Creative thinking activated successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error activating creative thinking")
                return false
        }
    
    /// <summary>
    /// Deactivates the creative thinking.
    /// </summary>
    /// <returns>True if deactivation was successful.</returns>
    member _.DeactivateAsync() =
        task {
            if not isActive then
                logger.LogInformation("Creative thinking is already inactive")
                return true
            
            try
                logger.LogInformation("Deactivating creative thinking")
                
                // Deactivate state
                isActive <- false
                
                logger.LogInformation("Creative thinking deactivated successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error deactivating creative thinking")
                return false
        }
    
    /// <summary>
    /// Updates the creative thinking.
    /// </summary>
    /// <returns>True if update was successful.</returns>
    member _.UpdateAsync() =
        task {
            if not isInitialized then
                logger.LogWarning("Cannot update creative thinking: not initialized")
                return false
            
            try
                // Gradually increase creativity levels over time (very slowly)
                if creativityLevel < 0.95 then
                    creativityLevel <- creativityLevel + 0.0001 * random.NextDouble()
                    creativityLevel <- Math.Min(creativityLevel, 1.0)
                
                if divergentThinkingLevel < 0.95 then
                    divergentThinkingLevel <- divergentThinkingLevel + 0.0001 * random.NextDouble()
                    divergentThinkingLevel <- Math.Min(divergentThinkingLevel, 1.0)
                
                if convergentThinkingLevel < 0.95 then
                    convergentThinkingLevel <- convergentThinkingLevel + 0.0001 * random.NextDouble()
                    convergentThinkingLevel <- Math.Min(convergentThinkingLevel, 1.0)
                
                if combinatorialCreativityLevel < 0.95 then
                    combinatorialCreativityLevel <- combinatorialCreativityLevel + 0.0001 * random.NextDouble()
                    combinatorialCreativityLevel <- Math.Min(combinatorialCreativityLevel, 1.0)
                
                return true
            with
            | ex ->
                logger.LogError(ex, "Error updating creative thinking")
                return false
        }
    
    /// <summary>
    /// Gets recent ideas.
    /// </summary>
    /// <param name="count">The number of ideas to get.</param>
    /// <returns>The recent ideas.</returns>
    member _.GetRecentIdeas(count: int) =
        creativeIdeas
        |> List.sortByDescending (fun idea -> idea.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets ideas by domain.
    /// </summary>
    /// <param name="domain">The domain.</param>
    /// <param name="count">The number of ideas to get.</param>
    /// <returns>The ideas in the domain.</returns>
    member _.GetIdeasByDomain(domain: string, count: int) =
        creativeIdeas
        |> List.filter (fun idea -> idea.Domain.Contains(domain, StringComparison.OrdinalIgnoreCase))
        |> List.sortByDescending (fun idea -> idea.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets ideas by tag.
    /// </summary>
    /// <param name="tag">The tag.</param>
    /// <param name="count">The number of ideas to get.</param>
    /// <returns>The ideas with the tag.</returns>
    member _.GetIdeasByTag(tag: string, count: int) =
        creativeIdeas
        |> List.filter (fun idea -> idea.Tags |> List.exists (fun t -> t.Contains(tag, StringComparison.OrdinalIgnoreCase)))
        |> List.sortByDescending (fun idea -> idea.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets the most original ideas.
    /// </summary>
    /// <param name="count">The number of ideas to get.</param>
    /// <returns>The most original ideas.</returns>
    member _.GetMostOriginalIdeas(count: int) =
        creativeIdeas
        |> List.sortByDescending (fun idea -> idea.Originality)
        |> List.truncate count
    
    /// <summary>
    /// Gets the most valuable ideas.
    /// </summary>
    /// <param name="count">The number of ideas to get.</param>
    /// <returns>The most valuable ideas.</returns>
    member _.GetMostValuableIdeas(count: int) =
        creativeIdeas
        |> List.sortByDescending (fun idea -> idea.Value)
        |> List.truncate count
    
    /// <summary>
    /// Gets the most effective creative processes.
    /// </summary>
    /// <param name="count">The number of processes to get.</param>
    /// <returns>The most effective creative processes.</returns>
    member _.GetMostEffectiveProcesses(count: int) =
        creativeProcesses
        |> List.sortByDescending (fun process -> process.Effectiveness)
        |> List.truncate count
    
    /// <summary>
    /// Adds a creative idea.
    /// </summary>
    /// <param name="idea">The idea to add.</param>
    member _.AddCreativeIdea(idea: CreativeIdea) =
        creativeIdeas <- idea :: creativeIdeas
        lastIdeaGenerationTime <- DateTime.UtcNow
    
    /// <summary>
    /// Adds a creative process.
    /// </summary>
    /// <param name="process">The process to add.</param>
    member _.AddCreativeProcess(process: CreativeProcess) =
        creativeProcesses <- process :: creativeProcesses
