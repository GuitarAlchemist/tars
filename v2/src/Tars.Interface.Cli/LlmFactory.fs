namespace Tars.Interface.Cli

open System
open Serilog
open Tars.Core
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService

module LlmFactory =
    let create (logger: ILogger) =
        let config = ConfigurationLoader.load ()
        let routingCfg = 
            { RoutingConfig.Default with
                OllamaBaseUri = 
                   config.Llm.BaseUrl 
                   |> Option.map (fun s -> try Uri(s) with _ -> Uri("http://localhost:11434")) 
                   |> Option.defaultValue (Uri "http://localhost:11434")
                LlamaCppBaseUri = config.Llm.LlamaCppUrl |> Option.map Uri
                LlamaSharpModelPath = config.Llm.LlamaSharpModelPath
                DefaultContextWindow = if config.Llm.ContextWindow > 0 then Some config.Llm.ContextWindow else None
                ReasoningModel = config.Llm.ReasoningModel
                CodingModel = config.Llm.CodingModel
                FastModel = config.Llm.FastModel }
        
        let serviceConfig = { LlmServiceConfig.Routing = routingCfg }
        // Simple client
        let client = new System.Net.Http.HttpClient()
        client.Timeout <- TimeSpan.FromMinutes(10.0)
        DefaultLlmService(client, serviceConfig) :> ILlmService
