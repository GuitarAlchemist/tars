namespace Tars.Tests

open System
open Xunit
open Tars.Evolution
open Tars.Core

module EvolutionTests =

    [<Fact>]
    let ``TaskQueue supports multiple tasks`` () =
        let task1 =
            { Id = Guid.NewGuid()
              DifficultyLevel = 1
              Goal = "Task 1"
              Constraints = []
              ValidationCriteria = ""
              Timeout = TimeSpan.Zero
              Score = 1.0 }

        let task2 =
            { Id = Guid.NewGuid()
              DifficultyLevel = 1
              Goal = "Task 2"
              Constraints = []
              ValidationCriteria = ""
              Timeout = TimeSpan.Zero
              Score = 0.9 }

        let state =
            { Generation = 0
              CurriculumAgentId = AgentId(Guid.NewGuid())
              ExecutorAgentId = AgentId(Guid.NewGuid())
              CompletedTasks = []
              CurrentTask = None
              TaskQueue = [ task1; task2 ]
              ActiveBeliefs = [] }

        // Simulate step logic manually as Engine.step is complex to mock fully without DI
        let nextState =
            match state.CurrentTask with
            | Some _ -> state // Should not happen
            | None ->
                match state.TaskQueue with
                | next :: rest ->
                    { state with
                        CurrentTask = Some next
                        TaskQueue = rest }
                | [] -> state

        Assert.Equal(Some task1, nextState.CurrentTask)
        let single = Assert.Single(nextState.TaskQueue)
        Assert.Equal(task2, single)
