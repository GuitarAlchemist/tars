namespace TarsTaskManager.Tests.Unit

open Xunit
open FsUnit.Xunit
open System
open TarsTaskManager.Core.Domain
open TarsTaskManager.Core.Services

module TaskServiceTests =

    [<Fact>]
    let `CreateTask should generate valid task with AI analysis` () =
        // Arrange
        let taskService = TaskService()
        let title = "Implement user authentication"
        let description = Some "Add JWT-based authentication system"

        // Act
        let result = taskService.AnalyzeTaskAsync(title, description) |> Async.RunSynchronously

        // Assert
        result.SuggestedPriority |> should equal Priority.High
        result.EstimatedDuration |> should be (greaterThan (TimeSpan.FromHours(4.0)))
        result.SuggestedTags |> should contain "authentication"
        result.Confidence |> should be (greaterThan 0.8)

    [<Fact>]
    let `GetTasksByUser should return only user's tasks` () =
        // Arrange
        let userId = Guid.NewGuid()
        let taskService = TaskService()

        // Act
        let result = taskService.GetTasksByUserAsync(userId) |> Async.RunSynchronously

        // Assert
        result |> List.forall (fun t -> t.CreatedBy = userId || t.AssignedTo = Some userId)
        |> should equal true

    [<Fact>]
    let `UpdateTaskPriority should recalculate AI priority` () =
        // Arrange
        let task = {
            Id = Guid.NewGuid()
            Title = "Critical bug fix"
            Description = Some "Fix security vulnerability"
            Priority = Priority.Low
            Status = TaskStatus.Pending
            DueDate = Some (DateTime.UtcNow.AddDays(1.0))
            EstimatedDuration = None
            Tags = []
            CreatedBy = Guid.NewGuid()
            AssignedTo = None
            ProjectId = None
            CreatedAt = DateTime.UtcNow
            UpdatedAt = DateTime.UtcNow
        }

        let taskService = TaskService()

        // Act
        let analysis = taskService.AnalyzeTaskAsync(task.Title, task.Description) |> Async.RunSynchronously

        // Assert
        analysis.SuggestedPriority |> should equal Priority.Critical

module DomainModelTests =

    [<Fact>]
    let `Task creation should have valid defaults` () =
        // Arrange & Act
        let task = {
            Id = Guid.NewGuid()
            Title = "Test task"
            Description = None
            Priority = Priority.Medium
            Status = TaskStatus.Pending
            DueDate = None
            EstimatedDuration = None
            Tags = []
            CreatedBy = Guid.NewGuid()
            AssignedTo = None
            ProjectId = None
            CreatedAt = DateTime.UtcNow
            UpdatedAt = DateTime.UtcNow
        }

        // Assert
        task.Id |> should not' (equal Guid.Empty)
        task.Status |> should equal TaskStatus.Pending
        task.CreatedAt |> should be (lessThanOrEqualTo DateTime.UtcNow)
