module Tars.Interface.Cli.Program

open System
open Serilog
open Tars.Core
open Tars.Graph
open System.Threading.Tasks
open Tars.Kernel
open Tars.Security

type SimpleMessage(id: Guid, source: string, target: string option, content: obj) =
    interface IMessage with
        member _.Id = id
        member _.CorrelationId = Guid.NewGuid()
        member _.Source = source
        member _.Target = target
        member _.Content = content
        member _.Timestamp = DateTime.UtcNow

type DemoAgent(logger: ILogger) =
    interface IAgent with
        member _.Id = "demo-agent"
        member _.Name = "Demo Agent"
        member _.HandleAsync(msg: IMessage) =
            Task.Run(fun () ->
                match msg.Content with
                | :? string as text ->
                    logger.Information($"DemoAgent received: {text}")
                | _ ->
                    logger.Warning("DemoAgent received unknown message type")
            )

[<EntryPoint>]
let main argv =
    // Initialize Secrets
    CredentialVault.registerSecret "OPENWEBUI_EMAIL" "steph@email.me"
    CredentialVault.registerSecret "OPENWEBUI_PASSWORD" "letmein"

    Log.Logger <- LoggerConfiguration().WriteTo.Console().CreateLogger()
    let logger = Log.Logger

    match argv with
    | [| "ask"; prompt |] ->
        task {
            let apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
            if String.IsNullOrEmpty(apiKey) then
                 printfn "Error: OPENAI_API_KEY environment variable not set."
            else
                 let modelId = "gpt-4o" // Default to a capable model
                 let embeddingId = "text-embedding-3-small"
                 let provider = new Tars.Cortex.SemanticKernelProvider(apiKey, modelId, embeddingId) :> Tars.Kernel.ICognitiveProvider
                 
                 try
                     let! response = provider.AskAsync(prompt)
                     printfn "%s" response
                 with ex ->
                     printfn "Error: %s" ex.Message
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously
        |> ignore
        0

    | [| "test-grammar"; file |] ->
        try
            let content = System.IO.File.ReadAllText(file)
            let goals = Tars.Cortex.Grammar.Parser.parse content
            printfn "Parsed %d goals" goals.Length
            for g in goals do
                printfn "Goal: %s" g.Name
                for t in g.Tasks do
                    printfn "  - Task: %s" t.Name
            0
        with ex ->
            printfn "Error: %s" ex.Message
            1

    | [| "memory-add"; coll; id; text |] ->
        task {
            let apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
            if String.IsNullOrEmpty(apiKey) then
                 printfn "Error: OPENAI_API_KEY not set."
            else
                 let modelId = "gpt-4o"
                 let embeddingId = "text-embedding-3-small"
                 let provider = new Tars.Cortex.SemanticKernelProvider(apiKey, modelId, embeddingId) :> Tars.Kernel.ICognitiveProvider
                 let vectorStore = new Tars.Cortex.ChromaVectorStore("http://localhost:8000") :> Tars.Core.IVectorStore

                 try
                    let! embeddings = provider.GetEmbeddingsAsync([text])
                    let vector = embeddings.[0]
                    do! vectorStore.SaveAsync(coll, id, vector, Map [ "text", text ])
                    printfn "Stored %s in %s with id %s" text coll id
                 with ex ->
                    printfn "Error: %s" ex.Message
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously
        |> ignore
        0

    | [| "memory-search"; coll; text |] ->
        task {
            let apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
            if String.IsNullOrEmpty(apiKey) then
                 printfn "Error: OPENAI_API_KEY not set."
            else
                 let modelId = "gpt-4o"
                 let embeddingId = "text-embedding-3-small"
                 let provider = new Tars.Cortex.SemanticKernelProvider(apiKey, modelId, embeddingId) :> Tars.Kernel.ICognitiveProvider
                 let vectorStore = new Tars.Cortex.ChromaVectorStore("http://localhost:8000") :> Tars.Core.IVectorStore

                 try
                    let! embeddings = provider.GetEmbeddingsAsync([text])
                    let vector = embeddings.[0]
                    let! results = vectorStore.SearchAsync(coll, vector, 5)
                    printfn "Found %d results:" results.Length
                    for (id, dist, meta) in results do
                        printfn "  [%s] (dist: %f) %A" id dist meta
                 with ex ->
                    printfn "Error: %s" ex.Message
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously
        |> ignore
        0

    | [| "demo-ping" |] ->
        task {
            logger.Information("Starting TARS v2 Demo Ping...")
            Console.WriteLine("DEBUG: Program Started")

            // 1. Initialize EventBus
            // Cast to IEventBus to access interface members if needed, or use class directly if accessible.
            // EventBus constructor takes ILogger.
            let eventBus = new Tars.Kernel.EventBus(logger)
            let bus = eventBus :> Tars.Kernel.IEventBus

            // 2. Create and Register Agent
            let demoAgent = new DemoAgent(logger) :> Tars.Kernel.IAgent
            // Subscribe agent to its own ID
            logger.Information("Subscribing agent...")
            let _ = bus.Subscribe(demoAgent.Id, demoAgent.HandleAsync)
            logger.Information("Subscribed.")

            // 3. Publish Ping
            let msg = new SimpleMessage(Guid.NewGuid(), "CLI", Some demoAgent.Id, "PING")
            logger.Information("Publishing message...")
            do! bus.PublishAsync(msg)
            logger.Information("Published.")
            Console.WriteLine("DEBUG: Published (Console).")

            // 4. Wait a bit to ensure processing
            do! Task.Delay(3000)

            logger.Information("Ping sent.")
            Console.WriteLine("DEBUG: Ping sent (Console).")
            Console.Out.Flush()
            Log.CloseAndFlush()
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously
        |> ignore

        0
    | [| "chat" |] ->
        task {
            logger.Information("Starting TARS v2 Chat...")

            // 1. Initialize Kernel
            let ctx = Kernel.init ()

            // 2. Create Agent
            let agent =
                Kernel.createAgent (Guid.NewGuid()) "TARS" "llama3.2" "You are a helpful assistant." []

            let ctx = Kernel.registerAgent agent ctx

            // 3. Create Graph Context
            let graphCtx: Graph.GraphContext = { Kernel = ctx; MaxSteps = 10 }

            // 4. Run Chat Loop
            let mutable currentAgent = agent
            let mutable running = true

            printfn "TARS v2 Chat initialized. Type 'exit' to quit."

            while running do
                printf "> "
                let input = Console.ReadLine()

                if isNull input || input = "exit" then
                    running <- false
                else
                    // Add user message to memory
                    let msg =
                        { Id = Guid.NewGuid()
                          CorrelationId = CorrelationId(Guid.NewGuid())
                          Source = User
                          Target = Agent agent.Id
                          Content = input
                          Timestamp = DateTime.UtcNow
                          Metadata = Map.empty }

                    currentAgent <- Kernel.receiveMessage msg currentAgent

                    // Run Graph Step
                    // Loop until terminal state (WaitingForUser or Error)
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
                        | _ -> () // Continue stepping

                    currentAgent <- stepAgent

                    // Check state
                    match currentAgent.State with
                    | WaitingForUser prompt -> printfn $"TARS: %s{prompt}"
                    | AgentState.Error err ->
                        printfn $"Agent Error: %s{err}"
                        logger.Error("Agent Error: {Error}", err)
                    | _ -> () // Should not happen if step returns terminal
        }
        |> Async.AwaitTask
        |> Async.RunSynchronously
        |> ignore

        0
    | _ ->
        printfn "Usage: tars chat"
        1