namespace Tars.Interface.Cli

open System
open System.IO
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

            // Load secrets into CredentialVault if possible
            let secretsPath = Path.Combine(System.AppContext.BaseDirectory, "secrets.json")

            if File.Exists(secretsPath) then
                ignore (Tars.Security.CredentialVault.loadSecretsFromDisk (secretsPath))
            else
                // Check root dir too
                let rootSecrets = "secrets.json"

                if File.Exists(rootSecrets) then
                    ignore (Tars.Security.CredentialVault.loadSecretsFromDisk (rootSecrets))

            let root = builder.Build()

            // Helper to get value with default
            let get key (def: string) =
                let mapped =
                    match key with
                    | "Llm:Model" -> "DEFAULT_OLLAMA_MODEL"
                    | "Llm:BaseUrl" -> "OLLAMA_BASE_URL"
                    | "Llm:ApiKey" -> "OPENAI_API_KEY"
                    | "Llm:LlamaCppUrl" -> "LLAMA_CPP_URL"
                    | _ -> key

                match Tars.Security.CredentialVault.getSecret mapped with
                | Ok res -> res
                | _ ->
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
                let mapped =
                    match key with
                    | "Llm:Model" -> "DEFAULT_OLLAMA_MODEL"
                    | "Llm:BaseUrl" -> "OLLAMA_BASE_URL"
                    | "Llm:ApiKey" -> "OPENAI_API_KEY"
                    | "Llm:LlamaCppUrl" -> "LLAMA_CPP_URL"
                    | _ -> key

                match Tars.Security.CredentialVault.getSecret mapped with
                | Ok res -> Some res
                | _ ->
                    let v = root.GetSection(key).Value
                    if String.IsNullOrWhiteSpace(v) then defOpt else Some v

            let getList key (def: string list) =
                let section = root.GetSection(key)

                if not (section.Exists()) then
                    def
                else
                    let values =
                        section.GetChildren()
                        |> Seq.map (fun child -> child.Value)
                        |> Seq.filter (fun v -> not (String.IsNullOrWhiteSpace v))
                        |> Seq.toList

                    if List.isEmpty values then def else values

            // Construct LlmSettings
            let defLlm = ConfigurationDefaults.DefaultLlm

            let llm =
                { defLlm with
                    Provider = get "Llm:Provider" defLlm.Provider
                    Model = get "Llm:Model" defLlm.Model
                    LlamaSharpModelPath = getOpt "Llm:LlamaSharpModelPath" defLlm.LlamaSharpModelPath
                    EmbeddingModel = get "Llm:EmbeddingModel" defLlm.EmbeddingModel
                    BaseUrl = getOpt "Llm:BaseUrl" defLlm.BaseUrl
                    LlamaCppUrl = getOpt "Llm:LlamaCppUrl" defLlm.LlamaCppUrl
                    ApiKey = getOpt "Llm:ApiKey" defLlm.ApiKey
                    ContextWindow = getInt "Llm:ContextWindow" defLlm.ContextWindow
                    Temperature = getFloat "Llm:Temperature" defLlm.Temperature
                    ReasoningModel = getOpt "Llm:ReasoningModel" defLlm.ReasoningModel
                    CodingModel = getOpt "Llm:CodingModel" defLlm.CodingModel
                    FastModel = getOpt "Llm:FastModel" defLlm.FastModel }

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
                    PostgresConnectionString = getOpt "Memory:PostgresConnectionString" defMem.PostgresConnectionString
                    FusekiUrl = getOpt "Memory:FusekiUrl" defMem.FusekiUrl
                    FusekiAuth = getOpt "Memory:FusekiAuth" defMem.FusekiAuth }

            // Construct EvolutionSettings
            let defEvo = ConfigurationDefaults.DefaultEvolution

            let preLlmDef = ConfigurationDefaults.DefaultPreLlm

            let preLlm =
                { UseIntentClassifier = getBool "PreLlm:UseIntentClassifier" preLlmDef.UseIntentClassifier
                  DefaultPolicies = getList "PreLlm:DefaultPolicies" preLlmDef.DefaultPolicies }

            let evo =
                { defEvo with
                    DefaultBudget = getFloat "Evolution:DefaultBudget" defEvo.DefaultBudget
                    MaxIterations = getInt "Evolution:MaxIterations" defEvo.MaxIterations
                    AutoSave = getBool "Evolution:AutoSave" defEvo.AutoSave
                    TraceEnabled = getBool "Evolution:TraceEnabled" defEvo.TraceEnabled }

            { Llm = llm
              Memory = mem
              Evolution = evo
              PreLlm = preLlm
              VariantOverlays = Map.empty }
        with _ ->
            ConfigurationDefaults.createDefault ()
