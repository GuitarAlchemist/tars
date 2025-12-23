namespace Tars.Interface.Cli

open System
open Microsoft.Extensions.Configuration
open Tars.Core

module ConfigurationLoader =

    let load () : TarsConfig =
        try
            // Build configuration from sources
            let builder =
                ConfigurationBuilder()
                    .SetBasePath(System.AppContext.BaseDirectory)
                    .AddJsonFile("appsettings.json", true, true)
                    .AddEnvironmentVariables("TARS_")

            let root = builder.Build()

            // Helper to get value with default
            let get key (def: string) =
                let v = root.GetSection(key).Value
                if String.IsNullOrWhiteSpace(v) then def else v

            let getInt key (def: int) =
                let v = root.GetSection(key).Value

                match Int32.TryParse(v) with
                | true, res -> res
                | _ -> def

            let getFloat key (def: float) =
                let v = root.GetSection(key).Value

                match Double.TryParse(v) with
                | true, res -> res
                | _ -> def

            let getBool key (def: bool) =
                let v = root.GetSection(key).Value

                match Boolean.TryParse(v) with
                | true, res -> res
                | _ -> def

            let getOpt key (defOpt: string option) =
                let v = root.GetSection(key).Value
                if String.IsNullOrWhiteSpace(v) then defOpt else Some v

            // Construct LlmSettings
            let defLlm = ConfigurationDefaults.DefaultLlm

            let llm =
                { defLlm with
                    Provider = get "Llm:Provider" defLlm.Provider
                    Model = get "Llm:Model" defLlm.Model
                    EmbeddingModel = get "Llm:EmbeddingModel" defLlm.EmbeddingModel
                    BaseUrl = getOpt "Llm:BaseUrl" defLlm.BaseUrl
                    ApiKey = getOpt "Llm:ApiKey" defLlm.ApiKey
                    ContextWindow = getInt "Llm:ContextWindow" defLlm.ContextWindow
                    Temperature = getFloat "Llm:Temperature" defLlm.Temperature }

            // Construct MemorySettings
            let defConfig = ConfigurationDefaults.createDefault ()
            let defMem = defConfig.Memory

            let mem =
                { defMem with
                    DataDirectory = get "Memory:DataDirectory" defMem.DataDirectory
                    VectorStorePath = get "Memory:VectorStorePath" defMem.VectorStorePath
                    KnowledgeBasePath = get "Memory:KnowledgeBasePath" defMem.KnowledgeBasePath
                    EpisodeDbPath = get "Memory:EpisodeDbPath" defMem.EpisodeDbPath
                    GraphitiUrl = getOpt "Memory:GraphitiUrl" defMem.GraphitiUrl
                    PostgresConnectionString = getOpt "Memory:PostgresConnectionString" defMem.PostgresConnectionString }

            // Construct EvolutionSettings
            let defEvo = ConfigurationDefaults.DefaultEvolution

            let evo =
                { defEvo with
                    DefaultBudget = getFloat "Evolution:DefaultBudget" defEvo.DefaultBudget
                    MaxIterations = getInt "Evolution:MaxIterations" defEvo.MaxIterations
                    AutoSave = getBool "Evolution:AutoSave" defEvo.AutoSave
                    TraceEnabled = getBool "Evolution:TraceEnabled" defEvo.TraceEnabled }

            { Llm = llm
              Memory = mem
              Evolution = evo }
        with _ ->
            ConfigurationDefaults.createDefault ()
