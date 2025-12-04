module Tars.Interface.Cli.Commands.Demo

open System
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Kernel

type DemoAgent(id: Guid, logger: ILogger) =
    interface IAgent with
        member _.Id = id.ToString()
        member _.Name = "Demo Agent"

        member _.HandleAsync(msg: SemanticMessage<obj>) =
            task {
                do!
                    Task.Run(fun () ->
                        match msg.Content with
                        | :? string as text -> logger.Information($"DemoAgent received: {text}")
                        | _ -> logger.Warning("DemoAgent received unknown message type"))

                return Success()
            }

let ping (logger: ILogger) =
    task {
        logger.Information("Starting TARS v2 Demo Ping...")
        Console.WriteLine("DEBUG: Program Started")

        let eventBus = new EventBus(logger)
        let bus = eventBus :> IEventBus

        let agentId = Guid.NewGuid()
        let demoAgent = new DemoAgent(agentId, logger) :> IAgent
        logger.Information("Subscribing agent {Id}...", demoAgent.Id)
        let _ = bus.Subscribe(demoAgent.Id, fun msg -> demoAgent.HandleAsync(msg) :> Task)
        logger.Information("Subscribed.")

        let msg =
            { Id = Guid.NewGuid()
              CorrelationId = CorrelationId(Guid.NewGuid())
              Sender = MessageEndpoint.User
              Receiver = Some(MessageEndpoint.Agent(AgentId agentId))
              Performative = Performative.Request
              Intent = None
              Constraints = SemanticConstraints.Default
              Ontology = None
              Language = "text"
              Content = "PING" :> obj
              Timestamp = DateTime.UtcNow
              Metadata = Map.empty }

        logger.Information("Publishing message...")
        do! bus.PublishAsync(msg)
        logger.Information("Published.")
        Console.WriteLine("DEBUG: Published (Console).")

        do! Task.Delay(3000)

        logger.Information("Ping sent.")
        Console.WriteLine("DEBUG: Ping sent (Console).")
        Console.Out.Flush()
        Log.CloseAndFlush()
        return 0
    }
