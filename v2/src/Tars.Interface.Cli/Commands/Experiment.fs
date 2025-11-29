module Tars.Interface.Cli.Commands.Experiment

open System
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Kernel
open Tars.Connectors

let run (logger: ILogger) =
    task {
        logger.Information("Starting A/B Testing Experiment...")

        // 1. Setup Infrastructure
        let router = AgentRouter()
        // Use a dummy budget for now
        let governance =
            BudgetGovernor(
                { Budget.Infinite with
                    MaxTokens = Some 100000<token> }
            )

        let breaker = CircuitBreaker(5, TimeSpan.FromMinutes(1.0))
        let eventBus = new EventBus(logger, breaker, governance, router)
        let bus = eventBus :> IEventBus

        // Check available models
        let! (modelsResult: Result<Tars.Connectors.OpenWebUi.ModelInfo[], string>) =
            OpenWebUi.listModels "https://aialpha.bar-scouts.com/"

        let modelName =
            match modelsResult with
            | Result.Ok models ->
                let names = models |> Array.map (fun m -> m.id)
                let namesStr = String.Join(", ", names)
                logger.Information($"Available Models: {namesStr}")
                if names.Length > 0 then names[0] else "llama3.2"
            | Result.Error e ->
                logger.Error($"Failed to list models: {e}")
                "llama3.2"

        logger.Information($"Using Model: {modelName}")

        // 2. Create Agents
        let id1 = Guid.NewGuid()
        let id2 = Guid.NewGuid()

        // Agent A: Concise
        let agent1 =
            Kernel.createAgent
                id1
                "Assistant_V1"
                "1.0.0"
                modelName
                "You are a helpful assistant. You answer in extremely short, concise sentences. One line only."
                []

        // Agent B: Poetic
        let agent2 =
            Kernel.createAgent
                id2
                "Assistant_V2"
                "2.0.0"
                modelName
                "You are a helpful assistant. You answer in the form of a short haiku or poem."
                []

        // Register in Kernel (mocking kernel context since we are just using EventBus for this demo)

        let createAgentHandler (agent: Agent) =
            fun (msg: SemanticMessage<obj>) ->
                task {
                    logger.Information($"[{agent.Name}] Received: {msg.Content}")

                    let messages =
                        [ { OpenWebUi.OpenWebUiMessage.Role = "system"
                            OpenWebUi.OpenWebUiMessage.Content = agent.SystemPrompt }
                          { OpenWebUi.OpenWebUiMessage.Role = "user"
                            OpenWebUi.OpenWebUiMessage.Content = msg.Content.ToString() } ]

                    try
                        let! (result: Result<string, string>) =
                            OpenWebUi.generate "https://aialpha.bar-scouts.com/" agent.Model messages

                        match result with
                        | Result.Ok text -> logger.Information($"[{agent.Name}] Answered: {text.Trim()}")
                        | Result.Error err -> logger.Error($"[{agent.Name}] Failed to generate: {err}")
                    with ex ->
                        logger.Error($"[{agent.Name}] Failed to generate: {ex.Message}")
                }
                :> Task

        // Subscribe Agents
        let _ = bus.Subscribe(id1.ToString(), createAgentHandler agent1)
        let _ = bus.Subscribe(id2.ToString(), createAgentHandler agent2)

        // 3. Configure Router (Canary Strategy: 50/50)
        logger.Information("Configuring Router: Alias 'Assistant' -> Canary(V1, V2, 0.5)")
        router.SetRoute("Assistant", Canary(AgentId id1, AgentId id2, 0.5))

        // 4. Run Experiment
        let prompts =
            [ "What is the weather?"
              "Tell me a joke."
              "What is 2 + 2?"
              "Who are you?"
              "Explain quantum physics."
              "Write a hello world in F#." ]

        for prompt in prompts do
            logger.Information("---")
            logger.Information($"User sending: '{prompt}' to 'Assistant'")

            let msg =
                { Id = Guid.NewGuid()
                  CorrelationId = CorrelationId(Guid.NewGuid())
                  Sender = MessageEndpoint.User
                  Receiver = Some(MessageEndpoint.Alias "Assistant")
                  Performative = Performative.Request
                  Constraints = SemanticConstraints.Default
                  Ontology = None
                  Language = "text"
                  Content = prompt :> obj
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.empty }

            do! bus.PublishAsync(msg)
            do! Task.Delay(2000) // Wait for response

        return 0
    }
