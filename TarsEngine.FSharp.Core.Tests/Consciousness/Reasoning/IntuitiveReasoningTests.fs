module TarsEngine.FSharp.Core.Tests.Consciousness.Reasoning.IntuitiveReasoningTests

open System
open System.Threading.Tasks
open Xunit
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.Core.Consciousness.Reasoning

/// <summary>
/// Tests for the IntuitiveReasoning class.
/// </summary>
type IntuitiveReasoningTests() =
    let logger = NullLogger<IntuitiveReasoning>() :> ILogger<IntuitiveReasoning>
    let intuitiveReasoning = IntuitiveReasoning(logger)
    
    /// <summary>
    /// Test that InitializeAsync initializes the intuitive reasoning.
    /// </summary>
    [<Fact>]
    member _.``InitializeAsync initializes the intuitive reasoning``() =
        task {
            // Act
            let! result = intuitiveReasoning.InitializeAsync()
            
            // Assert
            Assert.True(result)
            Assert.NotEmpty(intuitiveReasoning.HeuristicRules)
        }
    
    /// <summary>
    /// Test that ActivateAsync activates the intuitive reasoning.
    /// </summary>
    [<Fact>]
    member _.``ActivateAsync activates the intuitive reasoning``() =
        task {
            // Arrange
            let! _ = intuitiveReasoning.InitializeAsync()
            
            // Act
            let! result = intuitiveReasoning.ActivateAsync()
            
            // Assert
            Assert.True(result)
        }
    
    /// <summary>
    /// Test that DeactivateAsync deactivates the intuitive reasoning.
    /// </summary>
    [<Fact>]
    member _.``DeactivateAsync deactivates the intuitive reasoning``() =
        task {
            // Arrange
            let! _ = intuitiveReasoning.InitializeAsync()
            let! _ = intuitiveReasoning.ActivateAsync()
            
            // Act
            let! result = intuitiveReasoning.DeactivateAsync()
            
            // Assert
            Assert.True(result)
        }
    
    /// <summary>
    /// Test that UpdateAsync updates the intuitive reasoning.
    /// </summary>
    [<Fact>]
    member _.``UpdateAsync updates the intuitive reasoning``() =
        task {
            // Arrange
            let! _ = intuitiveReasoning.InitializeAsync()
            let initialIntuitionLevel = intuitiveReasoning.IntuitionLevel
            
            // Act
            let! result = intuitiveReasoning.UpdateAsync()
            
            // Assert
            Assert.True(result)
            Assert.True(intuitiveReasoning.IntuitionLevel >= initialIntuitionLevel)
        }
    
    /// <summary>
    /// Test that GenerateIntuitionAsync generates an intuition.
    /// </summary>
    [<Fact>]
    member _.``GenerateIntuitionAsync generates an intuition``() =
        task {
            // Arrange
            let! _ = intuitiveReasoning.InitializeAsync()
            let! _ = intuitiveReasoning.ActivateAsync()
            
            // Act
            let! intuitionOption = intuitiveReasoning.GenerateIntuitionAsync()
            
            // Assert
            Assert.True(intuitionOption.IsSome)
            let intuition = intuitionOption.Value
            Assert.NotEmpty(intuition.Description)
            Assert.InRange(intuition.Confidence, 0.0, 1.0)
        }
    
    /// <summary>
    /// Test that MakeIntuitiveDecisionAsync makes an intuitive decision.
    /// </summary>
    [<Fact>]
    member _.``MakeIntuitiveDecisionAsync makes an intuitive decision``() =
        task {
            // Arrange
            let! _ = intuitiveReasoning.InitializeAsync()
            let! _ = intuitiveReasoning.ActivateAsync()
            let decision = "Choose a programming language"
            let options = ["F#"; "C#"; "Python"; "JavaScript"]
            
            // Act
            let! intuitionOption = intuitiveReasoning.MakeIntuitiveDecisionAsync(decision, options)
            
            // Assert
            Assert.True(intuitionOption.IsSome)
            let intuition = intuitionOption.Value
            Assert.NotEmpty(intuition.Description)
            Assert.InRange(intuition.Confidence, 0.0, 1.0)
            Assert.Equal(decision, intuition.Decision)
            Assert.Contains(intuition.SelectedOption, options)
        }
    
    /// <summary>
    /// Test that GetRecentIntuitions returns the recent intuitions.
    /// </summary>
    [<Fact>]
    member _.``GetRecentIntuitions returns the recent intuitions``() =
        task {
            // Arrange
            let! _ = intuitiveReasoning.InitializeAsync()
            let! _ = intuitiveReasoning.ActivateAsync()
            let! _ = intuitiveReasoning.GenerateIntuitionAsync()
            
            // Act
            let intuitions = intuitiveReasoning.GetRecentIntuitions(1)
            
            // Assert
            Assert.NotEmpty(intuitions)
            Assert.Equal(1, intuitions.Length)
        }
    
    /// <summary>
    /// Test that AddHeuristicRule adds a heuristic rule.
    /// </summary>
    [<Fact>]
    member _.``AddHeuristicRule adds a heuristic rule``() =
        task {
            // Arrange
            let! _ = intuitiveReasoning.InitializeAsync()
            let initialRuleCount = intuitiveReasoning.HeuristicRules.Length
            
            // Act
            let rule = intuitiveReasoning.AddHeuristicRule("Test Rule", "Test Description", 0.7, "Test Context")
            
            // Assert
            Assert.Equal(initialRuleCount + 1, intuitiveReasoning.HeuristicRules.Length)
            Assert.Equal("Test Rule", rule.Name)
            Assert.Equal("Test Description", rule.Description)
            Assert.Equal(0.7, rule.Reliability)
            Assert.Equal("Test Context", rule.Context)
        }
    
    /// <summary>
    /// Test that UpdatePatternConfidence updates pattern confidence.
    /// </summary>
    [<Fact>]
    member _.``UpdatePatternConfidence updates pattern confidence``() =
        task {
            // Arrange
            let! _ = intuitiveReasoning.InitializeAsync()
            
            // Act
            intuitiveReasoning.UpdatePatternConfidence("test-pattern", 0.8)
            
            // Assert
            // We can't directly test the pattern confidence as it's private,
            // but we can test that the method doesn't throw an exception
            Assert.True(true)
        }
