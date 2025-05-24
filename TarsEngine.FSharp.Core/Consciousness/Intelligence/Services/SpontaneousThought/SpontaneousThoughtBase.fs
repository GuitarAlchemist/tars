namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.SpontaneousThought

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence
open TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

/// <summary>
/// Base implementation of the spontaneous thought capabilities.
/// </summary>
type SpontaneousThoughtBase(logger: ILogger<SpontaneousThoughtBase>) =
    // State variables
    let mutable isInitialized = false
    let mutable isActive = false
    let mutable spontaneityLevel = 0.5 // Starting with moderate spontaneity
    let mutable randomThoughtLevel = 0.4 // Starting with moderate random thought
    let mutable associativeJumpingLevel = 0.6 // Starting with moderate associative jumping
    let mutable mindWanderingLevel = 0.5 // Starting with moderate mind wandering
    let mutable thoughts = List.empty<ThoughtModel>
    let random = System.Random()
    let mutable lastThoughtTime = DateTime.MinValue
    
    /// <summary>
    /// Gets the spontaneity level (0.0 to 1.0).
    /// </summary>
    member _.SpontaneityLevel = spontaneityLevel
    
    /// <summary>
    /// Gets the random thought level (0.0 to 1.0).
    /// </summary>
    member _.RandomThoughtLevel = randomThoughtLevel
    
    /// <summary>
    /// Gets the associative jumping level (0.0 to 1.0).
    /// </summary>
    member _.AssociativeJumpingLevel = associativeJumpingLevel
    
    /// <summary>
    /// Gets the mind wandering level (0.0 to 1.0).
    /// </summary>
    member _.MindWanderingLevel = mindWanderingLevel
    
    /// <summary>
    /// Gets the thoughts.
    /// </summary>
    member _.Thoughts = thoughts
    
    /// <summary>
    /// Initializes the spontaneous thought.
    /// </summary>
    /// <returns>True if initialization was successful.</returns>
    member _.InitializeAsync() =
        task {
            try
                logger.LogInformation("Initializing spontaneous thought")
                
                // Initialize state
                isInitialized <- true
                
                logger.LogInformation("Spontaneous thought initialized successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error initializing spontaneous thought")
                return false
        }
    
    /// <summary>
    /// Activates the spontaneous thought.
    /// </summary>
    /// <returns>True if activation was successful.</returns>
    member _.ActivateAsync() =
        task {
            if not isInitialized then
                logger.LogWarning("Cannot activate spontaneous thought: not initialized")
                return false
            
            if isActive then
                logger.LogInformation("Spontaneous thought is already active")
                return true
            
            try
                logger.LogInformation("Activating spontaneous thought")
                
                // Activate state
                isActive <- true
                
                logger.LogInformation("Spontaneous thought activated successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error activating spontaneous thought")
                return false
        }
    
    /// <summary>
    /// Deactivates the spontaneous thought.
    /// </summary>
    /// <returns>True if deactivation was successful.</returns>
    member _.DeactivateAsync() =
        task {
            if not isActive then
                logger.LogInformation("Spontaneous thought is already inactive")
                return true
            
            try
                logger.LogInformation("Deactivating spontaneous thought")
                
                // Deactivate state
                isActive <- false
                
                logger.LogInformation("Spontaneous thought deactivated successfully")
                return true
            with
            | ex ->
                logger.LogError(ex, "Error deactivating spontaneous thought")
                return false
        }
    
    /// <summary>
    /// Updates the spontaneous thought.
    /// </summary>
    /// <returns>True if update was successful.</returns>
    member _.UpdateAsync() =
        task {
            if not isInitialized then
                logger.LogWarning("Cannot update spontaneous thought: not initialized")
                return false
            
            try
                // Gradually increase spontaneity levels over time (very slowly)
                if spontaneityLevel < 0.95 then
                    spontaneityLevel <- spontaneityLevel + 0.0001 * random.NextDouble()
                    spontaneityLevel <- Math.Min(spontaneityLevel, 1.0)
                
                if randomThoughtLevel < 0.95 then
                    randomThoughtLevel <- randomThoughtLevel + 0.0001 * random.NextDouble()
                    randomThoughtLevel <- Math.Min(randomThoughtLevel, 1.0)
                
                if associativeJumpingLevel < 0.95 then
                    associativeJumpingLevel <- associativeJumpingLevel + 0.0001 * random.NextDouble()
                    associativeJumpingLevel <- Math.Min(associativeJumpingLevel, 1.0)
                
                if mindWanderingLevel < 0.95 then
                    mindWanderingLevel <- mindWanderingLevel + 0.0001 * random.NextDouble()
                    mindWanderingLevel <- Math.Min(mindWanderingLevel, 1.0)
                
                return true
            with
            | ex ->
                logger.LogError(ex, "Error updating spontaneous thought")
                return false
        }
    
    /// <summary>
    /// Gets recent thoughts.
    /// </summary>
    /// <param name="count">The number of thoughts to get.</param>
    /// <returns>The recent thoughts.</returns>
    member _.GetRecentThoughts(count: int) =
        thoughts
        |> List.sortByDescending (fun thought -> thought.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets thoughts by method.
    /// </summary>
    /// <param name="method">The thought generation method.</param>
    /// <param name="count">The number of thoughts to get.</param>
    /// <returns>The thoughts generated by the specified method.</returns>
    member _.GetThoughtsByMethod(method: ThoughtGenerationMethod, count: int) =
        thoughts
        |> List.filter (fun thought -> thought.Method = method)
        |> List.sortByDescending (fun thought -> thought.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets thoughts by tag.
    /// </summary>
    /// <param name="tag">The tag.</param>
    /// <param name="count">The number of thoughts to get.</param>
    /// <returns>The thoughts with the tag.</returns>
    member _.GetThoughtsByTag(tag: string, count: int) =
        thoughts
        |> List.filter (fun thought -> thought.Tags |> List.exists (fun t -> t.Contains(tag, StringComparison.OrdinalIgnoreCase)))
        |> List.sortByDescending (fun thought -> thought.Timestamp)
        |> List.truncate count
    
    /// <summary>
    /// Gets the most significant thoughts.
    /// </summary>
    /// <param name="count">The number of thoughts to get.</param>
    /// <returns>The most significant thoughts.</returns>
    member _.GetMostSignificantThoughts(count: int) =
        thoughts
        |> List.sortByDescending (fun thought -> thought.Significance)
        |> List.truncate count
    
    /// <summary>
    /// Gets the most original thoughts.
    /// </summary>
    /// <param name="count">The number of thoughts to get.</param>
    /// <returns>The most original thoughts.</returns>
    member _.GetMostOriginalThoughts(count: int) =
        thoughts
        |> List.sortByDescending (fun thought -> thought.Originality)
        |> List.truncate count
    
    /// <summary>
    /// Gets the most coherent thoughts.
    /// </summary>
    /// <param name="count">The number of thoughts to get.</param>
    /// <returns>The most coherent thoughts.</returns>
    member _.GetMostCoherentThoughts(count: int) =
        thoughts
        |> List.sortByDescending (fun thought -> thought.Coherence)
        |> List.truncate count
    
    /// <summary>
    /// Adds a thought.
    /// </summary>
    /// <param name="thought">The thought to add.</param>
    member _.AddThought(thought: ThoughtModel) =
        thoughts <- thought :: thoughts
        lastThoughtTime <- DateTime.UtcNow
    
    /// <summary>
    /// Gets whether the spontaneous thought is initialized.
    /// </summary>
    member _.IsInitialized = isInitialized
    
    /// <summary>
    /// Gets whether the spontaneous thought is active.
    /// </summary>
    member _.IsActive = isActive
