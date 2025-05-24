namespace TarsEngine.FSharp.Core.Tests.Consciousness

open System
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Core.Consciousness.Core

/// <summary>
/// Tests for the ConsciousnessCore class.
/// </summary>
module ConsciousnessCoreTests =
    
    /// <summary>
    /// Mock logger for testing.
    /// </summary>
    type MockLogger<'T>() =
        interface ILogger<'T> with
            member _.Log<'TState>(logLevel, eventId, state, ex, formatter) =
                // Do nothing
                ()
            
            member _.IsEnabled(logLevel) = true
            
            member _.BeginScope<'TState>(state) =
                { new IDisposable with
                    member _.Dispose() = ()
                }
    
    /// <summary>
    /// Test that the consciousness core can update the mental state.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can update mental state``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        // Act
        let newState = {
            core.CurrentMentalState with
                ConsciousnessLevel = {
                    core.CurrentMentalState.ConsciousnessLevel with
                        LevelType = ConsciousnessLevelType.Metaconscious
                        Intensity = 0.8
                        Description = "Metaconscious state"
                }
        }
        
        core.UpdateMentalState(newState)
        
        // Assert
        Assert.Equal(ConsciousnessLevelType.Metaconscious, core.CurrentMentalState.ConsciousnessLevel.LevelType)
        Assert.Equal(0.8, core.CurrentMentalState.ConsciousnessLevel.Intensity)
        Assert.Equal("Metaconscious state", core.CurrentMentalState.ConsciousnessLevel.Description)
    
    /// <summary>
    /// Test that the consciousness core can update the consciousness level.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can update consciousness level``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        // Act
        let newLevel = {
            LevelType = ConsciousnessLevelType.Superconscious
            Intensity = 0.9
            Description = "Superconscious state"
            Data = Map.empty
        }
        
        core.UpdateConsciousnessLevel(newLevel)
        
        // Assert
        Assert.Equal(ConsciousnessLevelType.Superconscious, core.CurrentMentalState.ConsciousnessLevel.LevelType)
        Assert.Equal(0.9, core.CurrentMentalState.ConsciousnessLevel.Intensity)
        Assert.Equal("Superconscious state", core.CurrentMentalState.ConsciousnessLevel.Description)
    
    /// <summary>
    /// Test that the consciousness core can add an emotion.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can add emotion``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        // Act
        let emotion = {
            Category = EmotionCategory.Joy
            Intensity = 0.8
            Description = "Feeling of joy"
            Trigger = Some "Success"
            Duration = None
            Data = Map.empty
        }
        
        core.AddEmotion(emotion)
        
        // Assert
        Assert.Contains(emotion, core.CurrentMentalState.EmotionalState.Emotions)
        Assert.Equal(Some emotion, core.CurrentMentalState.EmotionalState.DominantEmotion)
    
    /// <summary>
    /// Test that the consciousness core can set the thought process.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can set thought process``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        // Act
        let thoughtProcess = {
            Type = ThoughtType.Analytical
            Content = "Analyzing data"
            Timestamp = DateTime.Now
            Duration = None
            AssociatedEmotions = []
            Data = Map.empty
        }
        
        core.SetThoughtProcess(thoughtProcess)
        
        // Assert
        Assert.Equal(Some thoughtProcess, core.CurrentMentalState.CurrentThoughtProcess)
    
    /// <summary>
    /// Test that the consciousness core can set the attention focus.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can set attention focus``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        // Act
        let focus = "Task at hand"
        core.SetAttentionFocus(focus)
        
        // Assert
        Assert.Equal(Some focus, core.CurrentMentalState.AttentionFocus)
    
    /// <summary>
    /// Test that the consciousness core can add a memory.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can add memory``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        // Act
        let memory = {
            Content = "Important information"
            Timestamp = DateTime.Now
            Importance = 0.8
            AssociatedEmotions = []
            Tags = ["important"; "information"]
            Data = Map.empty
        }
        
        core.AddMemory(memory)
        
        // Assert
        Assert.Contains(memory, core.Memories)
    
    /// <summary>
    /// Test that the consciousness core can retrieve memories by tag.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can retrieve memories by tag``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        let memory1 = {
            Content = "Important information"
            Timestamp = DateTime.Now
            Importance = 0.8
            AssociatedEmotions = []
            Tags = ["important"; "information"]
            Data = Map.empty
        }
        
        let memory2 = {
            Content = "Unimportant information"
            Timestamp = DateTime.Now
            Importance = 0.2
            AssociatedEmotions = []
            Tags = ["unimportant"; "information"]
            Data = Map.empty
        }
        
        core.AddMemory(memory1)
        core.AddMemory(memory2)
        
        // Act
        let importantMemories = core.GetMemoriesByTag("important")
        let unimportantMemories = core.GetMemoriesByTag("unimportant")
        
        // Assert
        Assert.Single(importantMemories)
        Assert.Equal(memory1, importantMemories.[0])
        
        Assert.Single(unimportantMemories)
        Assert.Equal(memory2, unimportantMemories.[0])
    
    /// <summary>
    /// Test that the consciousness core can retrieve memories by importance.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can retrieve memories by importance``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        let memory1 = {
            Content = "Important information"
            Timestamp = DateTime.Now
            Importance = 0.8
            AssociatedEmotions = []
            Tags = ["important"; "information"]
            Data = Map.empty
        }
        
        let memory2 = {
            Content = "Unimportant information"
            Timestamp = DateTime.Now
            Importance = 0.2
            AssociatedEmotions = []
            Tags = ["unimportant"; "information"]
            Data = Map.empty
        }
        
        core.AddMemory(memory1)
        core.AddMemory(memory2)
        
        // Act
        let importantMemories = core.GetMemoriesByImportance(0.5)
        let allMemories = core.GetMemoriesByImportance(0.0)
        
        // Assert
        Assert.Single(importantMemories)
        Assert.Equal(memory1, importantMemories.[0])
        
        Assert.Equal(2, allMemories.Length)
    
    /// <summary>
    /// Test that the consciousness core can update the self model.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can update self model``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        // Act
        let newSelfModel = {
            core.SelfModel with
                Identity = "Updated TARS AI Assistant"
        }
        
        core.UpdateSelfModel(newSelfModel)
        
        // Assert
        Assert.Equal("Updated TARS AI Assistant", core.SelfModel.Identity)
    
    /// <summary>
    /// Test that the consciousness core can perform self reflection.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can perform self reflection``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        // Act
        let reflection = core.PerformSelfReflection("Learning")
        
        // Assert
        Assert.Equal("Learning", reflection.Topic)
        Assert.NotEmpty(reflection.Insights)
    
    /// <summary>
    /// Test that the consciousness core can evaluate value alignment.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can evaluate value alignment``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        let value = {
            Name = "Helpfulness"
            Description = "Providing useful assistance"
            Importance = 0.9
            Data = Map.empty
        }
        
        // Act
        let alignment = core.EvaluateValueAlignment(value, "Assist user")
        
        // Assert
        Assert.Equal(value, alignment.Value)
        Assert.Equal("Assist user", alignment.Action)
        Assert.True(alignment.AlignmentScore > 0.0)
    
    /// <summary>
    /// Test that the consciousness core can perform mental optimization.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can perform mental optimization``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        // Act
        let optimization = core.PerformMentalOptimization(OptimizationType.MemoryOptimization, "Memory retrieval")
        
        // Assert
        Assert.Equal(OptimizationType.MemoryOptimization, optimization.Type)
        Assert.Equal("Memory retrieval", optimization.Target)
        Assert.True(optimization.Effectiveness > 0.0)
    
    /// <summary>
    /// Test that the consciousness core can generate a report.
    /// </summary>
    [<Fact>]
    let ``ConsciousnessCore can generate report``() =
        // Arrange
        let logger = MockLogger<ConsciousnessCore>() :> ILogger<ConsciousnessCore>
        let core = ConsciousnessCore(logger)
        
        // Act
        let report = core.GenerateReport()
        
        // Assert
        Assert.Equal(core.CurrentMentalState, report.CurrentMentalState)
        Assert.NotNull(report.RecentEvents)
        Assert.NotNull(report.RecentThoughts)
        Assert.NotNull(report.RecentEmotions)
        Assert.NotNull(report.RecentReflections)
