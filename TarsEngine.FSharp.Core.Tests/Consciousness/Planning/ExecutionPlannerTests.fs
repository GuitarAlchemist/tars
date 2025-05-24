module TarsEngine.FSharp.Core.Tests.Consciousness.Planning.ExecutionPlannerTests

open System
open System.Threading.Tasks
open Xunit
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.Core.Consciousness.Planning
open TarsEngine.FSharp.Core.Consciousness.Planning.Services

/// <summary>
/// Tests for the ExecutionPlanner class.
/// </summary>
type ExecutionPlannerTests() =
    let logger = NullLogger<ExecutionPlanner>() :> ILogger<ExecutionPlanner>
    let planner = ExecutionPlanner(logger)
    
    /// <summary>
    /// Test that CreateExecutionPlan creates a plan with the correct properties.
    /// </summary>
    [<Fact>]
    member _.``CreateExecutionPlan creates a plan with the correct properties``() =
        task {
            // Arrange
            let name = "Test Plan"
            let description = "Test Description"
            let improvementId = "improvement-123"
            let metascriptId = "metascript-456"
            
            // Act
            let! plan = planner.CreateExecutionPlan(name, description, improvementId, metascriptId)
            
            // Assert
            Assert.Equal(name, plan.Name)
            Assert.Equal(description, plan.Description)
            Assert.Equal(improvementId, plan.ImprovementId)
            Assert.Equal(metascriptId, plan.MetascriptId)
            Assert.Equal(ExecutionPlanStatus.Created, plan.Status)
            Assert.Empty(plan.Steps)
            Assert.NotEqual(Guid.Empty.ToString(), plan.Id)
        }
    
    /// <summary>
    /// Test that GetExecutionPlan returns the correct plan.
    /// </summary>
    [<Fact>]
    member _.``GetExecutionPlan returns the correct plan``() =
        task {
            // Arrange
            let! plan = planner.CreateExecutionPlan("Test Plan", "Test Description", "improvement-123", "metascript-456")
            
            // Act
            let! retrievedPlan = planner.GetExecutionPlan(plan.Id)
            
            // Assert
            Assert.True(retrievedPlan.IsSome)
            Assert.Equal(plan.Id, retrievedPlan.Value.Id)
            Assert.Equal(plan.Name, retrievedPlan.Value.Name)
        }
    
    /// <summary>
    /// Test that GetAllExecutionPlans returns all plans.
    /// </summary>
    [<Fact>]
    member _.``GetAllExecutionPlans returns all plans``() =
        task {
            // Arrange
            let! plan1 = planner.CreateExecutionPlan("Plan 1", "Description 1", "improvement-1", "metascript-1")
            let! plan2 = planner.CreateExecutionPlan("Plan 2", "Description 2", "improvement-2", "metascript-2")
            
            // Act
            let! plans = planner.GetAllExecutionPlans()
            
            // Assert
            Assert.Contains(plans, fun p -> p.Id = plan1.Id)
            Assert.Contains(plans, fun p -> p.Id = plan2.Id)
        }
    
    /// <summary>
    /// Test that UpdateExecutionPlan updates the plan.
    /// </summary>
    [<Fact>]
    member _.``UpdateExecutionPlan updates the plan``() =
        task {
            // Arrange
            let! plan = planner.CreateExecutionPlan("Test Plan", "Test Description", "improvement-123", "metascript-456")
            let updatedPlan = { plan with Name = "Updated Plan"; Description = "Updated Description" }
            
            // Act
            let! result = planner.UpdateExecutionPlan(updatedPlan)
            
            // Assert
            Assert.Equal(updatedPlan.Id, result.Id)
            Assert.Equal(updatedPlan.Name, result.Name)
            Assert.Equal(updatedPlan.Description, result.Description)
            Assert.True(result.UpdatedAt.IsSome)
            
            // Verify the plan was updated in storage
            let! retrievedPlan = planner.GetExecutionPlan(plan.Id)
            Assert.True(retrievedPlan.IsSome)
            Assert.Equal(updatedPlan.Name, retrievedPlan.Value.Name)
            Assert.Equal(updatedPlan.Description, retrievedPlan.Value.Description)
        }
    
    /// <summary>
    /// Test that DeleteExecutionPlan deletes the plan.
    /// </summary>
    [<Fact>]
    member _.``DeleteExecutionPlan deletes the plan``() =
        task {
            // Arrange
            let! plan = planner.CreateExecutionPlan("Test Plan", "Test Description", "improvement-123", "metascript-456")
            
            // Act
            let! result = planner.DeleteExecutionPlan(plan.Id)
            
            // Assert
            Assert.True(result)
            
            // Verify the plan was deleted
            let! retrievedPlan = planner.GetExecutionPlan(plan.Id)
            Assert.False(retrievedPlan.IsSome)
        }
    
    /// <summary>
    /// Test that ValidatePlan validates the plan.
    /// </summary>
    [<Fact>]
    member _.``ValidatePlan validates the plan``() =
        task {
            // Arrange
            let! plan = planner.CreateExecutionPlan("Test Plan", "Test Description", "improvement-123", "metascript-456")
            
            // Act
            let! result = planner.ValidatePlan(plan.Id)
            
            // Assert
            Assert.True(result)
        }
    
    /// <summary>
    /// Test that MonitorPlan returns the correct status and progress.
    /// </summary>
    [<Fact>]
    member _.``MonitorPlan returns the correct status and progress``() =
        task {
            // Arrange
            let! plan = planner.CreateExecutionPlan("Test Plan", "Test Description", "improvement-123", "metascript-456")
            
            // Act
            let! result = planner.MonitorPlan(plan.Id)
            
            // Assert
            Assert.True(result.IsSome)
            let status, progress = result.Value
            Assert.Equal(ExecutionPlanStatus.Created, status)
            Assert.Equal(0.0, progress)
        }
    
    /// <summary>
    /// Test that AdaptPlan adapts the plan.
    /// </summary>
    [<Fact>]
    member _.``AdaptPlan adapts the plan``() =
        task {
            // Arrange
            let! plan = planner.CreateExecutionPlan("Test Plan", "Test Description", "improvement-123", "metascript-456")
            let adaptationReason = "Test adaptation"
            
            // Act
            let! result = planner.AdaptPlan(plan.Id, adaptationReason)
            
            // Assert
            Assert.True(result.IsSome)
            Assert.Equal(plan.Id, result.Value.Id)
            Assert.True(result.Value.UpdatedAt.IsSome)
            Assert.True(result.Value.Metadata.ContainsKey("AdaptationReason"))
            Assert.Equal(adaptationReason, result.Value.Metadata.["AdaptationReason"])
        }
    
    /// <summary>
    /// Test that GetExecutionContext returns the correct context.
    /// </summary>
    [<Fact>]
    member _.``GetExecutionContext returns None for a new plan``() =
        task {
            // Arrange
            let! plan = planner.CreateExecutionPlan("Test Plan", "Test Description", "improvement-123", "metascript-456")
            
            // Act
            let! result = planner.GetExecutionContext(plan.Id)
            
            // Assert
            Assert.False(result.IsSome)
        }
    
    /// <summary>
    /// Test that UpdateExecutionContext updates the context.
    /// </summary>
    [<Fact>]
    member _.``UpdateExecutionContext updates the context``() =
        task {
            // Arrange
            let! plan = planner.CreateExecutionPlan("Test Plan", "Test Description", "improvement-123", "metascript-456")
            let context = ExecutionContext.create plan.Id plan.ImprovementId plan.MetascriptId
            let updatedContext = { context with WorkingDirectory = "/test/dir" }
            
            // Act
            let! result = planner.UpdateExecutionContext(plan.Id, updatedContext)
            
            // Assert
            Assert.Equal(updatedContext.WorkingDirectory, result.WorkingDirectory)
            Assert.True(result.UpdatedAt.IsSome)
            
            // Verify the context was updated in storage
            let! retrievedContext = planner.GetExecutionContext(plan.Id)
            Assert.True(retrievedContext.IsSome)
            Assert.Equal(updatedContext.WorkingDirectory, retrievedContext.Value.WorkingDirectory)
        }
    
    /// <summary>
    /// Test that ExecutePlan executes the plan.
    /// </summary>
    [<Fact>]
    member _.``ExecutePlan executes an empty plan successfully``() =
        task {
            // Arrange
            let! plan = planner.CreateExecutionPlan("Test Plan", "Test Description", "improvement-123", "metascript-456")
            
            // Act
            let! result = planner.ExecutePlan(plan.Id)
            
            // Assert
            Assert.True(result.IsSome)
            Assert.Equal(plan.Id, result.Value.ExecutionPlanId)
            Assert.Equal(ExecutionPlanStatus.Completed, result.Value.Status)
            Assert.True(result.Value.IsSuccessful)
            Assert.True(result.Value.StartedAt.IsSome)
            Assert.True(result.Value.CompletedAt.IsSome)
            Assert.True(result.Value.DurationMs.IsSome)
            Assert.NotEmpty(result.Value.Output)
            Assert.Empty(result.Value.Error)
            
            // Verify the plan status was updated
            let! retrievedPlan = planner.GetExecutionPlan(plan.Id)
            Assert.True(retrievedPlan.IsSome)
            Assert.Equal(ExecutionPlanStatus.Completed, retrievedPlan.Value.Status)
        }
    
    /// <summary>
    /// Test that RollbackPlan rolls back the plan.
    /// </summary>
    [<Fact>]
    member _.``RollbackPlan rolls back an empty plan successfully``() =
        task {
            // Arrange
            let! plan = planner.CreateExecutionPlan("Test Plan", "Test Description", "improvement-123", "metascript-456")
            let! _ = planner.ExecutePlan(plan.Id)
            
            // Act
            let! result = planner.RollbackPlan(plan.Id)
            
            // Assert
            Assert.True(result)
            
            // Verify the plan status was updated
            let! retrievedPlan = planner.GetExecutionPlan(plan.Id)
            Assert.True(retrievedPlan.IsSome)
            Assert.Equal(ExecutionPlanStatus.Cancelled, retrievedPlan.Value.Status)
        }
