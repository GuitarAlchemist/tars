module TarsEngine.FSharp.Core.Tests.Consciousness.Decision.DecisionServiceTests

open System
open System.Threading.Tasks
open Xunit
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.Core.Consciousness.Decision
open TarsEngine.FSharp.Core.Consciousness.Decision.Services

/// <summary>
/// Tests for the DecisionService class.
/// </summary>
type DecisionServiceTests() =
    let logger = NullLogger<DecisionService>() :> ILogger<DecisionService>
    let service = DecisionService(logger)
    
    /// <summary>
    /// Test that CreateDecision creates a decision with the correct properties.
    /// </summary>
    [<Fact>]
    member _.``CreateDecision creates a decision with the correct properties``() =
        task {
            // Arrange
            let description = "Test decision"
            let options = ["Option 1"; "Option 2"; "Option 3"]
            let domain = "Test domain"
            
            // Act
            let! decision = service.CreateDecision(description, options, domain)
            
            // Assert
            Assert.Equal(description, decision.Description)
            Assert.Equal(options, decision.Options)
            Assert.Equal(Some domain, decision.Domain)
            Assert.Equal(None, decision.SelectedOption)
            Assert.Equal(0.0, decision.Confidence)
            Assert.Equal(None, decision.Explanation)
            Assert.NotEqual(Guid.Empty, decision.Id)
        }
    
    /// <summary>
    /// Test that GetDecision returns the correct decision.
    /// </summary>
    [<Fact>]
    member _.``GetDecision returns the correct decision``() =
        task {
            // Arrange
            let description = "Test decision"
            let options = ["Option 1"; "Option 2"; "Option 3"]
            let! decision = service.CreateDecision(description, options)
            
            // Act
            let! retrievedDecision = service.GetDecision(decision.Id)
            
            // Assert
            Assert.Equal(Some decision, retrievedDecision)
        }
    
    /// <summary>
    /// Test that GetAllDecisions returns all decisions.
    /// </summary>
    [<Fact>]
    member _.``GetAllDecisions returns all decisions``() =
        task {
            // Arrange
            let! decision1 = service.CreateDecision("Decision 1", ["Option 1"; "Option 2"])
            let! decision2 = service.CreateDecision("Decision 2", ["Option A"; "Option B"])
            
            // Act
            let! decisions = service.GetAllDecisions()
            
            // Assert
            Assert.Contains(decision1, decisions)
            Assert.Contains(decision2, decisions)
        }
    
    /// <summary>
    /// Test that UpdateDecision updates the decision.
    /// </summary>
    [<Fact>]
    member _.``UpdateDecision updates the decision``() =
        task {
            // Arrange
            let! decision = service.CreateDecision("Original decision", ["Option 1"; "Option 2"])
            let updatedDecision = { decision with Description = "Updated decision" }
            
            // Act
            let! result = service.UpdateDecision(updatedDecision)
            
            // Assert
            Assert.Equal(updatedDecision, result)
            let! retrievedDecision = service.GetDecision(decision.Id)
            Assert.Equal(Some updatedDecision, retrievedDecision)
        }
    
    /// <summary>
    /// Test that DeleteDecision deletes the decision.
    /// </summary>
    [<Fact>]
    member _.``DeleteDecision deletes the decision``() =
        task {
            // Arrange
            let! decision = service.CreateDecision("Test decision", ["Option 1"; "Option 2"])
            
            // Act
            let! result = service.DeleteDecision(decision.Id)
            
            // Assert
            Assert.True(result)
            let! retrievedDecision = service.GetDecision(decision.Id)
            Assert.Equal(None, retrievedDecision)
        }
    
    /// <summary>
    /// Test that AddOption adds an option to a decision.
    /// </summary>
    [<Fact>]
    member _.``AddOption adds an option to a decision``() =
        task {
            // Arrange
            let! decision = service.CreateDecision("Test decision", ["Option 1"; "Option 2"])
            let newOption = "Option 3"
            
            // Act
            let! option = service.AddOption(decision.Id, newOption)
            
            // Assert
            Assert.Equal(decision.Id, option.DecisionId)
            Assert.Equal(newOption, option.Description)
            let! updatedDecision = service.GetDecision(decision.Id)
            Assert.Contains(newOption, updatedDecision.Value.Options)
        }
    
    /// <summary>
    /// Test that MakeIntuitiveDecision makes a decision with the correct properties.
    /// </summary>
    [<Fact>]
    member _.``MakeIntuitiveDecision makes a decision with the correct properties``() =
        task {
            // Arrange
            let description = "Test intuitive decision"
            let options = ["Option 1"; "Option 2"; "Option 3"]
            let domain = "Test domain"
            
            // Act
            let! decision = service.MakeIntuitiveDecision(description, options, domain)
            
            // Assert
            Assert.Equal(description, decision.Decision)
            Assert.Equal(options, decision.Options)
            Assert.Contains(decision.SelectedOption, options)
            Assert.InRange(decision.Confidence, 0.0, 1.0)
            Assert.NotEmpty(decision.Explanation)
        }
    
    /// <summary>
    /// Test that GenerateIntuition generates an intuition with the correct properties.
    /// </summary>
    [<Fact>]
    member _.``GenerateIntuition generates an intuition with the correct properties``() =
        task {
            // Arrange
            let situation = "Test situation"
            let domain = "Test domain"
            
            // Act
            let! intuition = service.GenerateIntuition(situation, domain)
            
            // Assert
            Assert.NotEmpty(intuition.Description)
            Assert.InRange(intuition.Confidence, 0.0, 1.0)
            Assert.Equal(DateTime.UtcNow.Date, intuition.Timestamp.Date)
        }
