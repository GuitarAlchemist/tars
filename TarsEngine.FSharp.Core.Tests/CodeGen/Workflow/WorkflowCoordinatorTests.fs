namespace TarsEngine.FSharp.Core.Tests.CodeGen.Workflow

open System
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Core.CodeGen.Workflow

/// <summary>
/// Tests for the WorkflowCoordinator class.
/// </summary>
module WorkflowCoordinatorTests =
    
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
    /// Test that the workflow coordinator can create a workflow.
    /// </summary>
    [<Fact>]
    let ``WorkflowCoordinator can create a workflow``() =
        // Arrange
        let logger = MockLogger<WorkflowCoordinator>() :> ILogger<WorkflowCoordinator>
        let coordinator = WorkflowCoordinator(logger)
        
        // Act
        let workflow = coordinator.CreateWorkflow(
            "Test Workflow",
            "A test workflow",
            [
                "Step 1", "The first step", (fun () -> task { return "Step 1 result" :> obj })
                "Step 2", "The second step", (fun () -> task { return "Step 2 result" :> obj })
            ]
        )
        
        // Assert
        Assert.NotNull(workflow)
        Assert.Equal("Test Workflow", workflow.Name)
        Assert.Equal("A test workflow", workflow.Description)
        Assert.Equal(2, workflow.Steps.Length)
        Assert.Equal("Step 1", workflow.Steps.[0].Name)
        Assert.Equal("Step 2", workflow.Steps.[1].Name)
        Assert.Equal(WorkflowStatus.NotStarted, workflow.Status)
    
    /// <summary>
    /// Test that the workflow coordinator can execute a workflow.
    /// </summary>
    [<Fact>]
    let ``WorkflowCoordinator can execute a workflow``() = task {
        // Arrange
        let logger = MockLogger<WorkflowCoordinator>() :> ILogger<WorkflowCoordinator>
        let coordinator = WorkflowCoordinator(logger)
        
        let mutable step1Executed = false
        let mutable step2Executed = false
        
        let workflow = coordinator.CreateWorkflow(
            "Test Workflow",
            "A test workflow",
            [
                "Step 1", "The first step", (fun () -> 
                    task { 
                        step1Executed <- true
                        return "Step 1 result" :> obj 
                    }
                )
                "Step 2", "The second step", (fun () -> 
                    task { 
                        step2Executed <- true
                        return "Step 2 result" :> obj 
                    }
                )
            ]
        )
        
        // Act
        let! executedWorkflow = coordinator.ExecuteWorkflowAsync(workflow)
        
        // Assert
        Assert.NotNull(executedWorkflow)
        Assert.Equal(WorkflowStatus.Completed, executedWorkflow.Status)
        Assert.True(executedWorkflow.StartTime.IsSome)
        Assert.True(executedWorkflow.EndTime.IsSome)
        
        Assert.Equal(WorkflowStatus.Completed, executedWorkflow.Steps.[0].Status)
        Assert.Equal(WorkflowStatus.Completed, executedWorkflow.Steps.[1].Status)
        
        Assert.True(step1Executed)
        Assert.True(step2Executed)
        
        Assert.True(executedWorkflow.Steps.[0].Result.IsSome)
        Assert.True(executedWorkflow.Steps.[1].Result.IsSome)
        
        Assert.Equal("Step 1 result", executedWorkflow.Steps.[0].Result.Value)
        Assert.Equal("Step 2 result", executedWorkflow.Steps.[1].Result.Value)
    }
    
    /// <summary>
    /// Test that the workflow coordinator can handle a failed step.
    /// </summary>
    [<Fact>]
    let ``WorkflowCoordinator can handle a failed step``() = task {
        // Arrange
        let logger = MockLogger<WorkflowCoordinator>() :> ILogger<WorkflowCoordinator>
        let coordinator = WorkflowCoordinator(logger)
        
        let mutable step1Executed = false
        let mutable step2Executed = false
        
        let workflow = coordinator.CreateWorkflow(
            "Test Workflow",
            "A test workflow",
            [
                "Step 1", "The first step", (fun () -> 
                    task { 
                        step1Executed <- true
                        return "Step 1 result" :> obj 
                    }
                )
                "Step 2", "The second step", (fun () -> 
                    task { 
                        step2Executed <- true
                        failwith "Step 2 failed"
                        return "Step 2 result" :> obj 
                    }
                )
                "Step 3", "The third step", (fun () -> 
                    task { 
                        return "Step 3 result" :> obj 
                    }
                )
            ]
        )
        
        // Act
        let! executedWorkflow = coordinator.ExecuteWorkflowAsync(workflow)
        
        // Assert
        Assert.NotNull(executedWorkflow)
        Assert.Equal(WorkflowStatus.Failed, executedWorkflow.Status)
        Assert.True(executedWorkflow.StartTime.IsSome)
        Assert.True(executedWorkflow.EndTime.IsSome)
        
        Assert.Equal(WorkflowStatus.Completed, executedWorkflow.Steps.[0].Status)
        Assert.Equal(WorkflowStatus.Failed, executedWorkflow.Steps.[1].Status)
        Assert.Equal(WorkflowStatus.NotStarted, executedWorkflow.Steps.[2].Status)
        
        Assert.True(step1Executed)
        Assert.True(step2Executed)
        
        Assert.True(executedWorkflow.Steps.[0].Result.IsSome)
        Assert.False(executedWorkflow.Steps.[1].Result.IsSome)
        Assert.False(executedWorkflow.Steps.[2].Result.IsSome)
        
        Assert.True(executedWorkflow.Steps.[1].ErrorMessage.IsSome)
        Assert.Equal("Step 2 failed", executedWorkflow.Steps.[1].ErrorMessage.Value)
    }
    
    /// <summary>
    /// Test that the workflow coordinator can cancel a workflow.
    /// </summary>
    [<Fact>]
    let ``WorkflowCoordinator can cancel a workflow``() = task {
        // Arrange
        let logger = MockLogger<WorkflowCoordinator>() :> ILogger<WorkflowCoordinator>
        let coordinator = WorkflowCoordinator(logger)
        
        let mutable step1Executed = false
        let mutable step2Started = false
        let mutable step2Completed = false
        
        let workflow = coordinator.CreateWorkflow(
            "Test Workflow",
            "A test workflow",
            [
                "Step 1", "The first step", (fun () -> 
                    task { 
                        step1Executed <- true
                        return "Step 1 result" :> obj 
                    }
                )
                "Step 2", "The second step", (fun () -> 
                    task { 
                        step2Started <- true
                        
                        // Simulate a long-running operation
                        do! Task.Delay(1000)
                        
                        step2Completed <- true
                        return "Step 2 result" :> obj 
                    }
                )
            ]
        )
        
        // Act
        let executionTask = coordinator.ExecuteWorkflowAsync(workflow)
        
        // Wait for step 2 to start
        while not step2Started do
            do! Task.Delay(10)
        
        // Cancel the workflow
        let cancelled = coordinator.CancelWorkflow(workflow.Id)
        
        // Wait for the execution to complete
        let! executedWorkflow = executionTask
        
        // Assert
        Assert.True(cancelled)
        Assert.NotNull(executedWorkflow)
        Assert.Equal(WorkflowStatus.Cancelled, executedWorkflow.Status)
        Assert.True(executedWorkflow.StartTime.IsSome)
        Assert.True(executedWorkflow.EndTime.IsSome)
        
        Assert.Equal(WorkflowStatus.Completed, executedWorkflow.Steps.[0].Status)
        Assert.Equal(WorkflowStatus.Cancelled, executedWorkflow.Steps.[1].Status)
        
        Assert.True(step1Executed)
        Assert.True(step2Started)
        Assert.False(step2Completed)
        
        Assert.True(executedWorkflow.Steps.[0].Result.IsSome)
        Assert.False(executedWorkflow.Steps.[1].Result.IsSome)
    }
    
    /// <summary>
    /// Test that the workflow coordinator can get a workflow by ID.
    /// </summary>
    [<Fact>]
    let ``WorkflowCoordinator can get a workflow by ID``() =
        // Arrange
        let logger = MockLogger<WorkflowCoordinator>() :> ILogger<WorkflowCoordinator>
        let coordinator = WorkflowCoordinator(logger)
        
        let workflow = coordinator.CreateWorkflow(
            "Test Workflow",
            "A test workflow",
            [
                "Step 1", "The first step", (fun () -> task { return "Step 1 result" :> obj })
            ]
        )
        
        // Act
        let retrievedWorkflow = coordinator.GetWorkflow(workflow.Id)
        
        // Assert
        Assert.True(retrievedWorkflow.IsSome)
        Assert.Equal(workflow.Id, retrievedWorkflow.Value.Id)
        Assert.Equal(workflow.Name, retrievedWorkflow.Value.Name)
    
    /// <summary>
    /// Test that the workflow coordinator can get all workflows.
    /// </summary>
    [<Fact>]
    let ``WorkflowCoordinator can get all workflows``() =
        // Arrange
        let logger = MockLogger<WorkflowCoordinator>() :> ILogger<WorkflowCoordinator>
        let coordinator = WorkflowCoordinator(logger)
        
        let workflow1 = coordinator.CreateWorkflow(
            "Test Workflow 1",
            "A test workflow",
            [
                "Step 1", "The first step", (fun () -> task { return "Step 1 result" :> obj })
            ]
        )
        
        let workflow2 = coordinator.CreateWorkflow(
            "Test Workflow 2",
            "Another test workflow",
            [
                "Step 1", "The first step", (fun () -> task { return "Step 1 result" :> obj })
            ]
        )
        
        // Act
        let workflows = coordinator.GetAllWorkflows()
        
        // Assert
        Assert.Equal(2, workflows.Length)
        Assert.Contains(workflow1, workflows)
        Assert.Contains(workflow2, workflows)
    
    /// <summary>
    /// Test that the workflow coordinator can get the status of a workflow.
    /// </summary>
    [<Fact>]
    let ``WorkflowCoordinator can get the status of a workflow``() =
        // Arrange
        let logger = MockLogger<WorkflowCoordinator>() :> ILogger<WorkflowCoordinator>
        let coordinator = WorkflowCoordinator(logger)
        
        let workflow = coordinator.CreateWorkflow(
            "Test Workflow",
            "A test workflow",
            [
                "Step 1", "The first step", (fun () -> task { return "Step 1 result" :> obj })
            ]
        )
        
        // Act
        let status = coordinator.GetWorkflowStatus(workflow.Id)
        
        // Assert
        Assert.True(status.IsSome)
        Assert.Equal(WorkflowStatus.NotStarted, status.Value)
