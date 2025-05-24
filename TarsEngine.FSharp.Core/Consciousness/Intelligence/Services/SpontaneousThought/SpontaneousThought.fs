namespace TarsEngine.FSharp.Core.Consciousness.Intelligence.Services

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Intelligence
open TarsEngine.FSharp.Core.Consciousness.Intelligence.Services.SpontaneousThought

/// <summary>
/// Implementation of the spontaneous thought capabilities.
/// </summary>
type SpontaneousThought(logger: ILogger<SpontaneousThought>) =
    inherit SpontaneousThoughtBase(logger)
    
    let random = System.Random()
    let mutable lastThoughtTime = DateTime.MinValue
    
    /// <summary>
    /// Generates a spontaneous thought.
    /// </summary>
    /// <returns>The generated thought.</returns>
    member this.GenerateSpontaneousThoughtAsync() =
        task {
            if not this.IsInitialized || not this.IsActive then
                return None
            
            // Only generate thoughts periodically
            if (DateTime.UtcNow - lastThoughtTime).TotalSeconds < 30 then
                return None
            
            try
                logger.LogDebug("Generating spontaneous thought")
                
                // Choose a thought generation method based on current levels
                let method = ThoughtGeneration.chooseThoughtMethod 
                              this.RandomThoughtLevel 
                              this.AssociativeJumpingLevel 
                              this.MindWanderingLevel 
                              random
                
                // Generate thought based on method
                let thought = ThoughtGeneration.generateThoughtByMethod 
                               method 
                               this.Thoughts 
                               this.RandomThoughtLevel 
                               this.AssociativeJumpingLevel 
                               this.MindWanderingLevel 
                               random
                
                // Add to thoughts list
                this.AddThought(thought)
                
                lastThoughtTime <- DateTime.UtcNow
                
                logger.LogInformation("Generated spontaneous thought: {Content} (Method: {Method}, Significance: {Significance:F2})",
                                     thought.Content, thought.Method, thought.Significance)
                
                return Some thought
            with
            | ex ->
                logger.LogError(ex, "Error generating spontaneous thought")
                return None
        }
    
    /// <summary>
    /// Generates a thought by a specific method.
    /// </summary>
    /// <param name="method">The thought generation method.</param>
    /// <returns>The generated thought.</returns>
    member this.GenerateThoughtByMethodAsync(method: ThoughtGenerationMethod) =
        task {
            if not this.IsInitialized || not this.IsActive then
                logger.LogWarning("Cannot generate thought: spontaneous thought not initialized or active")
                return None
            
            try
                logger.LogInformation("Generating thought using method: {Method}", method)
                
                // Generate thought based on method
                let thought = ThoughtGeneration.generateThoughtByMethod 
                               method 
                               this.Thoughts 
                               this.RandomThoughtLevel 
                               this.AssociativeJumpingLevel 
                               this.MindWanderingLevel 
                               random
                
                // Add to thoughts list
                this.AddThought(thought)
                
                lastThoughtTime <- DateTime.UtcNow
                
                logger.LogInformation("Generated thought: {Content} (Method: {Method}, Significance: {Significance:F2})",
                                     thought.Content, thought.Method, thought.Significance)
                
                return Some thought
            with
            | ex ->
                logger.LogError(ex, "Error generating thought by method")
                return None
        }
    
    /// <summary>
    /// Marks a thought as leading to an insight.
    /// </summary>
    /// <param name="thoughtId">The thought ID.</param>
    /// <param name="insightId">The insight ID.</param>
    /// <returns>The updated thought.</returns>
    member this.MarkThoughtAsInsightfulAsync(thoughtId: string, insightId: string) =
        task {
            try
                logger.LogInformation("Marking thought with ID: {ThoughtId} as leading to insight: {InsightId}", 
                                     thoughtId, insightId)
                
                // Find the thought
                let thoughtOption = 
                    this.Thoughts 
                    |> List.tryFind (fun t -> t.Id = thoughtId)
                
                match thoughtOption with
                | Some thought ->
                    // Update the thought
                    let updatedThought = {
                        thought with
                            LedToInsight = true
                            InsightId = Some insightId
                    }
                    
                    // Replace the thought in the list
                    let updatedThoughts = 
                        this.Thoughts 
                        |> List.map (fun t -> if t.Id = thoughtId then updatedThought else t)
                    
                    // Update the thoughts list using reflection (since it's a mutable field in the base class)
                    let baseType = this.GetType().BaseType
                    let field = baseType.GetField("thoughts", System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
                    field.SetValue(this, updatedThoughts)
                    
                    logger.LogInformation("Marked thought with ID: {ThoughtId} as leading to insight", thoughtId)
                    
                    return Some updatedThought
                | None ->
                    logger.LogWarning("Thought with ID {ThoughtId} not found", thoughtId)
                    return None
            with
            | ex ->
                logger.LogError(ex, "Error marking thought as insightful")
                return None
        }
    
    interface ISpontaneousThought with
        member this.SpontaneityLevel = this.SpontaneityLevel
        member this.RandomThoughtLevel = this.RandomThoughtLevel
        member this.AssociativeJumpingLevel = this.AssociativeJumpingLevel
        member this.MindWanderingLevel = this.MindWanderingLevel
        member this.Thoughts = this.Thoughts
        
        member this.InitializeAsync() = this.InitializeAsync()
        member this.ActivateAsync() = this.ActivateAsync()
        member this.DeactivateAsync() = this.DeactivateAsync()
        member this.UpdateAsync() = this.UpdateAsync()
        
        member this.GenerateSpontaneousThoughtAsync() = this.GenerateSpontaneousThoughtAsync()
        member this.GenerateThoughtByMethodAsync(method) = this.GenerateThoughtByMethodAsync(method)
        
        member this.GetRecentThoughts(count) = this.GetRecentThoughts(count)
        member this.GetThoughtsByMethod(method, count) = this.GetThoughtsByMethod(method, count)
        member this.GetThoughtsByTag(tag, count) = this.GetThoughtsByTag(tag, count)
        member this.GetMostSignificantThoughts(count) = this.GetMostSignificantThoughts(count)
        member this.GetMostOriginalThoughts(count) = this.GetMostOriginalThoughts(count)
        member this.GetMostCoherentThoughts(count) = this.GetMostCoherentThoughts(count)
        
        member this.MarkThoughtAsInsightfulAsync(thoughtId, insightId) = this.MarkThoughtAsInsightfulAsync(thoughtId, insightId)
