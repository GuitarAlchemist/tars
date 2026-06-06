namespace TarsEngine.FSharp.Core.Tests.Consciousness.Decision.Services

open System
open System.Threading.Tasks
open Xunit
open Microsoft.Extensions.Logging
open Moq
open TarsEngine.FSharp.Core.Consciousness.Core
open TarsEngine.FSharp.Core.Consciousness.Decision
open TarsEngine.FSharp.Core.Consciousness.Decision.Services

/// <summary>
/// Tests for the DecisionServiceComplete class.
/// </summary>
module DecisionServiceTests =
    
    /// <summary>
    /// Creates a new instance of DecisionServiceComplete for testing.
    /// </summary>
    let createService() =
        let loggerMock = Mock<ILogger<DecisionServiceComplete>>()
        new DecisionServiceComplete(loggerMock.Object)
    
    [<Fact>]
    let ``CreateDecision should create a decision with the specified parameters`` () = task {
        // Arrange
        let service = createService()
        let name = "Test Decision"
        let description = "Test Description"
        let type' = DecisionType.Binary
        
        // Act
        let! decision = service.CreateDecision(name, description, type')
        
        // Assert
        Assert.Equal(name, decision.Name)
        Assert.Equal(description, decision.Description)
        Assert.Equal(type', decision.Type)
        Assert.Equal(DecisionStatus.Pending, decision.Status)
        Assert.Equal(DecisionPriority.Medium, decision.Priority)
        Assert.Empty(decision.Options)
        Assert.Empty(decision.Criteria)
        Assert.Empty(decision.Constraints)
        Assert.None(decision.SelectedOption)
        Assert.Equal(DateTime.UtcNow.Date, decision.CreationTime.Date)
        Assert.None(decision.CompletionTime)
        Assert.Empty(decision.AssociatedEmotions)
        Assert.None(decision.Context)
        Assert.None(decision.Justification)
    }
    
    [<Fact>]
    let ``GetDecision should return the decision with the specified ID`` () = task {
        // Arrange
        let service = createService()
        let! decision = service.CreateDecision("Test Decision", "Test Description", DecisionType.Binary)
        
        // Act
        let! result = service.GetDecision(decision.Id)
        
        // Assert
        Assert.True(result.IsSome)
        let retrievedDecision = result.Value
        Assert.Equal(decision.Id, retrievedDecision.Id)
        Assert.Equal(decision.Name, retrievedDecision.Name)
    }
    
    [<Fact>]
    let ``GetDecision should return None for a non-existent ID`` () = task {
        // Arrange
        let service = createService()
        
        // Act
        let! result = service.GetDecision(Guid.NewGuid())
        
        // Assert
        Assert.True(result.IsNone)
    }
    
    [<Fact>]
    let ``GetAllDecisions should return all decisions`` () = task {
        // Arrange
        let service = createService()
        let! decision1 = service.CreateDecision("Decision 1", "Description 1", DecisionType.Binary)
        let! decision2 = service.CreateDecision("Decision 2", "Description 2", DecisionType.MultipleChoice)
        
        // Act
        let! decisions = service.GetAllDecisions()
        
        // Assert
        Assert.Equal(2, decisions.Length)
        Assert.Contains(decisions, fun d -> d.Id = decision1.Id)
        Assert.Contains(decisions, fun d -> d.Id = decision2.Id)
    }
    
    [<Fact>]
    let ``UpdateDecision should update the decision with the specified parameters`` () = task {
        // Arrange
        let service = createService()
        let! decision = service.CreateDecision("Test Decision", "Test Description", DecisionType.Binary)
        let newDescription = "Updated Description"
        let newStatus = DecisionStatus.InProgress
        
        // Act
        let! updatedDecision = service.UpdateDecision(decision.Id, description = newDescription, status = newStatus)
        
        // Assert
        Assert.Equal(decision.Id, updatedDecision.Id)
        Assert.Equal(decision.Name, updatedDecision.Name)
        Assert.Equal(newDescription, updatedDecision.Description)
        Assert.Equal(newStatus, updatedDecision.Status)
    }
    
    [<Fact>]
    let ``DeleteDecision should delete the decision with the specified ID`` () = task {
        // Arrange
        let service = createService()
        let! decision = service.CreateDecision("Test Decision", "Test Description", DecisionType.Binary)
        
        // Act
        let! result = service.DeleteDecision(decision.Id)
        let! retrievedDecision = service.GetDecision(decision.Id)
        
        // Assert
        Assert.True(result)
        Assert.True(retrievedDecision.IsNone)
    }
    
    [<Fact>]
    let ``AddOption should add an option to the decision`` () = task {
        // Arrange
        let service = createService()
        let! decision = service.CreateDecision("Test Decision", "Test Description", DecisionType.Binary)
        let optionName = "Test Option"
        let optionDescription = "Test Option Description"
        
        // Act
        let! updatedDecision = service.AddOption(decision.Id, optionName, optionDescription)
        
        // Assert
        Assert.Equal(1, updatedDecision.Options.Length)
        let option = updatedDecision.Options.[0]
        Assert.Equal(optionName, option.Name)
        Assert.Equal(optionDescription, option.Description)
    }
    
    [<Fact>]
    let ``UpdateOption should update the option with the specified parameters`` () = task {
        // Arrange
        let service = createService()
        let! decision = service.CreateDecision("Test Decision", "Test Description", DecisionType.Binary)
        let! decisionWithOption = service.AddOption(decision.Id, "Test Option", "Test Option Description")
        let option = decisionWithOption.Options.[0]
        let newName = "Updated Option"
        let newDescription = "Updated Option Description"
        
        // Act
        let! updatedDecision = service.UpdateOption(decision.Id, option.Id, name = newName, description = newDescription)
        
        // Assert
        Assert.Equal(1, updatedDecision.Options.Length)
        let updatedOption = updatedDecision.Options.[0]
        Assert.Equal(option.Id, updatedOption.Id)
        Assert.Equal(newName, updatedOption.Name)
        Assert.Equal(newDescription, updatedOption.Description)
    }
    
    [<Fact>]
    let ``RemoveOption should remove the option from the decision`` () = task {
        // Arrange
        let service = createService()
        let! decision = service.CreateDecision("Test Decision", "Test Description", DecisionType.Binary)
        let! decisionWithOption = service.AddOption(decision.Id, "Test Option", "Test Option Description")
        let option = decisionWithOption.Options.[0]
        
        // Act
        let! updatedDecision = service.RemoveOption(decision.Id, option.Id)
        
        // Assert
        Assert.Empty(updatedDecision.Options)
    }
