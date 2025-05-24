namespace TarsEngine.FSharp.Core

open System
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Consciousness.Core
open TarsEngine.FSharp.Core.Consciousness.Services
open TarsEngine.FSharp.Core.Consciousness.DependencyInjection

/// <summary>
/// Main program for TarsEngine.FSharp.
/// </summary>
module Program =
    
    /// <summary>
    /// Configures the service provider.
    /// </summary>
    /// <returns>The configured service provider.</returns>
    let configureServices() =
        let services = ServiceCollection()
        
        // Add logging
        services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
        ) |> ignore
        
        // Add consciousness services
        services.AddTarsEngineFSharpConsciousness() |> ignore
        
        // Build the service provider
        services.BuildServiceProvider()
    
    /// <summary>
    /// Runs the consciousness demo.
    /// </summary>
    /// <param name="serviceProvider">The service provider.</param>
    let runConsciousnessDemo(serviceProvider: ServiceProvider) =
        // Get the consciousness service
        let consciousnessService = serviceProvider.GetRequiredService<IConsciousnessService>()
        
        // Get the current mental state
        let mentalState = consciousnessService.GetCurrentMentalState().Result
        
        // Print the current mental state
        printfn "Current mental state:"
        printfn "  Consciousness level: %A (%.2f)" mentalState.ConsciousnessLevel.LevelType mentalState.ConsciousnessLevel.Intensity
        printfn "  Mood: %s" mentalState.EmotionalState.Mood
        
        // Add an emotion
        let emotion = {
            Category = EmotionCategory.Joy
            Intensity = 0.8
            Description = "Feeling of joy"
            Trigger = Some "Success"
            Duration = None
            Data = Map.empty
        }
        
        consciousnessService.AddEmotion(emotion).Wait()
        
        // Set a thought process
        let thoughtProcess = {
            Type = ThoughtType.Analytical
            Content = "Analyzing data"
            Timestamp = DateTime.Now
            Duration = None
            AssociatedEmotions = []
            Data = Map.empty
        }
        
        consciousnessService.SetThoughtProcess(thoughtProcess).Wait()
        
        // Set attention focus
        consciousnessService.SetAttentionFocus("Task at hand").Wait()
        
        // Generate a report
        let report = consciousnessService.GenerateReport().Result
        
        // Print the report
        printfn "\nConsciousness report:"
        printfn "  Consciousness level: %A (%.2f)" report.CurrentMentalState.ConsciousnessLevel.LevelType report.CurrentMentalState.ConsciousnessLevel.Intensity
        printfn "  Mood: %s" report.CurrentMentalState.EmotionalState.Mood
        printfn "  Attention focus: %A" report.CurrentMentalState.AttentionFocus
        printfn "  Current thought: %A" (report.CurrentMentalState.CurrentThoughtProcess |> Option.map (fun tp -> tp.Type))
        printfn "  Recent emotions: %d" report.RecentEmotions.Length
    
    /// <summary>
    /// Main entry point.
    /// </summary>
    /// <param name="args">Command line arguments.</param>
    let main(args: string[]) =
        // Configure services
        use serviceProvider = configureServices()
        
        // Run the consciousness demo
        runConsciousnessDemo(serviceProvider)
        
        0 // Return success code
