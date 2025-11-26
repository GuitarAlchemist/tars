module Tars.Interface.Cli.Commands.Evolve

open System
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Kernel
open Tars.Evolution

let run (logger: ILogger) =
    task {
        logger.Information("Starting TARS v2 Evolution Engine...")

        let ctx = Kernel.init ()

        let curriculumId = Guid.NewGuid()
        let executorId = Guid.NewGuid()

        let curriculumAgent =
            Kernel.createAgent curriculumId "Curriculum" "llama3.2" "You are a teacher." []

        let executorAgent =
            Kernel.createAgent executorId "Executor" "llama3.2" "You are a student." []

        let ctx = Kernel.registerAgent curriculumAgent ctx
        let ctx = Kernel.registerAgent executorAgent ctx

        // 3. Initialize Evolution State
        let evoState: EvolutionState =
            { Generation = 0
              CurriculumAgentId = AgentId curriculumId
              ExecutorAgentId = AgentId executorId
              CompletedTasks = []
              CurrentTask = None }

        let evoCtx: Engine.EvolutionContext = { Kernel = ctx }

        let mutable currentState = evoState

        for i in 1..5 do
            printfn "--- Generation %d ---" currentState.Generation
            let! nextState = Engine.step evoCtx currentState
            currentState <- nextState

            match currentState.CurrentTask with
            | Some task -> printfn "Task Generated: %s" task.Goal
            | None -> printfn "Task Completed. History: %d" currentState.CompletedTasks.Length

            do! Task.Delay(1000)

        return 0
    }
