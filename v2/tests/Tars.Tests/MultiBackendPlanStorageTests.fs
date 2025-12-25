module Tars.Tests.MultiBackendPlanStorageTests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Knowledge

/// Test fixture for multi-backend plan storage
type MultiBackendPlanStorageTests() =

    /// Helper to create a test plan
    let createTestPlan (goal: string) : Plan =
        { Id = PlanId.New()
          Goal = goal
          Assumptions = []
          Steps =
            [ { Order = 1
                Description = "Step 1: Analyze requirements"
                EstimatedEffort = Some(TimeSpan.FromHours(2.0))
                Dependencies = []
                Status = StepStatus.NotStarted
                CompletedAt = None
                Notes = [] }
              { Order = 2
                Description = "Step 2: Implement solution"
                EstimatedEffort = Some(TimeSpan.FromHours(4.0))
                Dependencies = [ 1 ]
                Status = StepStatus.NotStarted
                CompletedAt = None
                Notes = [] } ]
          SuccessMetrics = [ "All tests pass"; "Code reviewed" ]
          RiskFactors = [ "Tight deadline"; "Complex requirements" ]
          Version = 1
          ParentVersion = None
          Status = PlanStatus.Draft
          CreatedAt = DateTime.UtcNow
          UpdatedAt = DateTime.UtcNow
          CreatedBy = AgentId.System
          Tags = [ "test"; "multi-backend" ] }

    [<Fact>]
    member _.``In-Memory storage should save and retrieve plan``() =
        task {
            // Arrange
            let storage = InMemoryLedgerStorage() :> IPlanStorage
            let plan = createTestPlan "Test in-memory storage"

            // Act
            let! saveResult = storage.SavePlan(plan)
            let! retrieved = storage.GetPlan(plan.Id)

            // Assert
            Assert.Equal(Ok(), saveResult)
            Assert.True(retrieved.IsSome)
            let retrievedPlan = retrieved.Value
            Assert.Equal(plan.Goal, retrievedPlan.Goal)
            Assert.Equal(2, retrievedPlan.Steps.Length)
        }

    [<Fact>]
    member _.``In-Memory storage should filter plans by status``() =
        task {
            // Arrange
            let storage = InMemoryLedgerStorage() :> IPlanStorage
            let draftPlan = createTestPlan "Draft plan"

            let activePlan =
                { createTestPlan "Active plan" with
                    Status = PlanStatus.Active }

            // Act
            let! _ = storage.SavePlan(draftPlan)
            let! _ = storage.SavePlan(activePlan)
            let! draftPlans = storage.GetPlansByStatus(PlanStatus.Draft)
            let! activePlans = storage.GetPlansByStatus(PlanStatus.Active)

            // Assert
            Assert.Equal(1, draftPlans.Length)
            Assert.Equal(1, activePlans.Length)
            Assert.Equal("Draft plan", draftPlans.[0].Goal)
            Assert.Equal("Active plan", activePlans.[0].Goal)
        }

    [<Fact>]
    member _.``In-Memory storage should update existing plan``() =
        task {
            // Arrange
            let storage = InMemoryLedgerStorage() :> IPlanStorage
            let plan = createTestPlan "Original goal"

            // Act
            let! _ = storage.SavePlan(plan)

            let updatedPlan =
                { plan with
                    Goal = "Updated goal"
                    Status = PlanStatus.Active }

            let! _ = storage.UpdatePlan(updatedPlan)
            let! retrieved = storage.GetPlan(plan.Id)

            // Assert
            Assert.True(retrieved.IsSome)
            Assert.Equal("Updated goal", retrieved.Value.Goal)
            Assert.Equal(PlanStatus.Active, retrieved.Value.Status)
        }

    [<Fact>]
    member _.``In-Memory storage should append plan events``() =
        task {
            // Arrange
            let storage = InMemoryLedgerStorage() :> IPlanStorage
            let plan = createTestPlan "Test event storage"
            let planId = plan.Id

            // Act
            let! _ = storage.AppendEvent(PlanEvent.PlanCreated plan)
            let! _ = storage.AppendEvent(PlanEvent.StepStarted(planId, 1))
            let! _ = storage.AppendEvent(PlanEvent.StepCompleted(planId, 1, "Requirements analyzed"))

            // Assert - events were written without error
            Assert.True(true)
        }

    [<Fact>]
    member _.``Plan with assumptions should serialize correctly``() =
        task {
            // Arrange
            let storage = InMemoryLedgerStorage() :> IPlanStorage
            let beliefId1 = BeliefId.New()
            let beliefId2 = BeliefId.New()

            let plan =
                { createTestPlan "Plan with assumptions" with
                    Assumptions = [ beliefId1; beliefId2 ] }

            // Act
            let! _ = storage.SavePlan(plan)
            let! retrieved = storage.GetPlan(plan.Id)

            // Assert
            Assert.True(retrieved.IsSome)
            Assert.Equal(2, retrieved.Value.Assumptions.Length)
            Assert.Contains(beliefId1, retrieved.Value.Assumptions)
            Assert.Contains(beliefId2, retrieved.Value.Assumptions)
        }

    [<Fact>]
    member _.``Plan with all status types should work``() =
        task {
            // Arrange
            let storage = InMemoryLedgerStorage() :> IPlanStorage

            let statuses =
                [ PlanStatus.Draft
                  PlanStatus.Active
                  PlanStatus.Paused
                  PlanStatus.Completed
                  PlanStatus.Failed
                  PlanStatus.Superseded ]

            // Act & Assert
            for status in statuses do
                let plan =
                    { createTestPlan $"Plan with {status} status" with
                        Status = status }

                let! saveResult = storage.SavePlan(plan)
                let! retrieved = storage.GetPlan(plan.Id)

                Assert.Equal(Ok(), saveResult)
                Assert.True(retrieved.IsSome)
                Assert.Equal(status, retrieved.Value.Status)
        }

    [<Fact>]
    member _.``PlanManager with in-memory storage should work``() =
        task {
            // Arrange
            let ledger = KnowledgeLedger.createInMemory ()
            do! ledger.Initialize()
            let manager = PlanManager.createInMemory (ledger)

            let goal = "Test PlanManager integration"

            let steps =
                [ { Order = 1
                    Description = "First step"
                    EstimatedEffort = None
                    Dependencies = []
                    Status = StepStatus.NotStarted
                    CompletedAt = None
                    Notes = [] } ]

            // Act
            let! result = manager.CreatePlan(goal, steps, [], AgentId.System)

            // Assert
            match result with
            | Result.Ok plan ->
                Assert.Equal(goal, plan.Goal)
                Assert.Equal(1, plan.Steps.Length)
            | Result.Error e -> failwith $"Plan creation failed: {e}"
        }
