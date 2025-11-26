module Tars.Interface.Cli.Commands.Demo

open System
open System.Threading.Tasks
open Serilog
open Tars.Kernel

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
                | :? string as text -> logger.Information($"DemoAgent received: {text}")
                | _ -> logger.Warning("DemoAgent received unknown message type"))

let ping (logger: ILogger) =
    task {
        logger.Information("Starting TARS v2 Demo Ping...")
        Console.WriteLine("DEBUG: Program Started")

        let eventBus = new EventBus(logger)
        let bus = eventBus :> IEventBus

        let demoAgent = new DemoAgent(logger) :> IAgent
        logger.Information("Subscribing agent...")
        let _ = bus.Subscribe(demoAgent.Id, demoAgent.HandleAsync)
        logger.Information("Subscribed.")

        let msg = new SimpleMessage(Guid.NewGuid(), "CLI", Some demoAgent.Id, "PING")
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
