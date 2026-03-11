namespace Tars.Interface.Cli

open System
open Serilog
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService

module LlmFactory =
    let create (logger: ILogger) =
        let config = ConfigurationLoader.load ()
        let routingCfg = RoutingConfig.fromTarsConfig config
        
        let serviceConfig = { LlmServiceConfig.Routing = routingCfg }
        // Simple client
        let client = new System.Net.Http.HttpClient()
        client.Timeout <- TimeSpan.FromMinutes(10.0)
        DefaultLlmService(client, serviceConfig) :> ILlmService
