module Tars.Interface.Cli.Commands.Chat

open System
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Kernel
open Tars.Graph

let run (logger: ILogger) =
    task {
        logger.Information("Starting TARS v2 Chat...")

        let ctx = Kernel.init ()

        let agent =
            Kernel.createAgent (Guid.NewGuid()) "TARS" "llama3.2" "You are a helpful assistant." []

        let ctx = Kernel.registerAgent agent ctx
        let graphCtx: Graph.GraphContext = { Kernel = ctx; MaxSteps = 10 }

        let mutable currentAgent = agent
        let mutable running = true

        printfn "TARS v2 Chat initialized. Type 'exit' to quit."

        while running do
            printf "> "
            let input = Console.ReadLine()

            if isNull input || input = "exit" then
                running <- false
            else
                let msg =
                    { Id = Guid.NewGuid()
                      CorrelationId = CorrelationId(Guid.NewGuid())
                      Source = User
                      Target = Agent agent.Id
                      Content = input
                      Timestamp = DateTime.UtcNow
                      Metadata = Map.empty }

                currentAgent <- Kernel.receiveMessage msg currentAgent

                let mutable stepAgent = currentAgent
                let mutable stepCount = 0
                let mutable finished = false

                while not finished && stepCount < graphCtx.MaxSteps do
                    let! next = Graph.step stepAgent graphCtx
                    stepAgent <- next
                    stepCount <- stepCount + 1

                    match stepAgent.State with
                    | WaitingForUser _
                    | AgentState.Error _ -> finished <- true
                    | _ -> ()

                currentAgent <- stepAgent

                match currentAgent.State with
                | WaitingForUser prompt -> printfn $"TARS: %s{prompt}"
                | AgentState.Error err ->
                    printfn $"Agent Error: %s{err}"
                    logger.Error("Agent Error: {Error}", err)
                | _ -> ()

        return 0
    }
