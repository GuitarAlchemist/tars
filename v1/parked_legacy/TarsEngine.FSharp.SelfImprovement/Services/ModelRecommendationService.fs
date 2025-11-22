namespace TarsEngine.FSharp.SelfImprovement.Services

open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open Microsoft.Extensions.Logging
open YamlDotNet.RepresentationModel

type ModelRecommendation =
    { Key: string
      DisplayName: string
      Provider: string
      Model: string
      Capabilities: string list
      RequiresApiKey: bool
      Environment: Map<string, string>
      OptionalEnvironment: Map<string, string>
      Notes: string option }

module ModelRecommendationService =

    let private deserializer =
        lazy (
            DeserializerBuilder()
                .WithNamingConvention(UnderscoredNamingConvention.Instance)
                .IgnoreUnmatchedProperties()
                .Build()
        )

    let private placeholderPattern =
        Regex(@"\$\{([A-Za-z0-9_]+)(:-([^}]+))?\}", RegexOptions.Compiled)

    let internal resolveValue (value: string) =
        placeholderPattern.Replace(
            value,
            fun (m: Match) ->
                let envName = m.Groups.[1].Value
                let defaultGroup = m.Groups.[3]
                let envValue = Environment.GetEnvironmentVariable(envName)
                if String.IsNullOrEmpty(envValue) then
                    if defaultGroup.Success then defaultGroup.Value else ""
                else
                    envValue
        )

    let private toModelRecommendation (key: string) (entry: RecommendationEntry) =
        let capabilities =
            entry.Capabilities
            |> Option.ofObj
            |> Option.defaultValue [||]
            |> Array.map (fun c -> c.Trim())
            |> Array.filter (fun c -> not (String.IsNullOrWhiteSpace c))
            |> Array.toList

        let envRequired, envOptional =
            match entry.Env |> Option.ofObj with
            | None -> Map.empty, Map.empty
            | Some env ->
                let normalize value = value |> resolveValue

                let requiredPairs =
                    [ match env.ApiKey |> Option.ofObj with
                      | Some apiKey when not (String.IsNullOrWhiteSpace apiKey) -> yield "api_key", normalize apiKey
                      | _ -> ()
                      match env.Endpoint |> Option.ofObj with
                      | Some endpoint when not (String.IsNullOrWhiteSpace endpoint) -> yield "endpoint", normalize endpoint
                      | _ -> () ]

                let optionalPairs =
                    env.Optional
                    |> Option.ofObj
                    |> Option.map (fun dict ->
                        dict
                        |> Seq.map (fun kv -> kv.Key, normalize kv.Value)
                        |> Seq.toList)
                    |> Option.defaultValue []

                Map.ofSeq requiredPairs, Map.ofSeq optionalPairs

        { Key = key
          DisplayName = entry.DisplayName
          Provider = entry.Provider
          Model = resolveValue entry.Model
          Capabilities = capabilities
          RequiresApiKey = entry.RequiresApiKey
          Environment = envRequired
          OptionalEnvironment = envOptional
          Notes = entry.Notes |> Option.ofObj }

    let private toOption (value: string) =
        if String.IsNullOrWhiteSpace value then None else Some value

    let private candidatePaths (explicitPath: string option) =
        [ explicitPath
          Environment.GetEnvironmentVariable("TARS_LLM_RECOMMENDATIONS") |> toOption
          Some(Path.Combine(Environment.CurrentDirectory, "config", "llm-model-recommendations.yaml"))
          Some(Path.Combine(AppContext.BaseDirectory, "config", "llm-model-recommendations.yaml")) ]
        |> List.choose id
        |> List.map Path.GetFullPath
        |> List.distinct

    let internal loadFromText (logger: ILogger) (yaml: string) (source: string) : Map<string, ModelRecommendation> =
        try
            use reader = new StringReader(yaml)
            let document =
                deserializer.Value.Deserialize<RecommendationDocument>(reader)
                |> Option.ofObj

            match document with
            | None ->
                logger.LogWarning("Model recommendation document {Source} deserialized to null.", source)
                Map.empty
            | Some doc ->
                match doc.BestModels |> Option.ofObj with
                | None ->
                    logger.LogWarning("Model recommendation file {Source} contains no best_models section.", source)
                    Map.empty
                | Some entries ->
                    logger.LogInformation("Loaded {Count} LLM recommendations from {Source}.", entries.Count, source)
                    entries
                    |> Seq.map (fun kv -> kv.Key, toModelRecommendation kv.Key kv.Value)
                    |> Map.ofSeq
        with ex ->
            logger.LogError(ex, "Failed to parse model recommendation content from {Source}.", source)
            Map.empty

    let loadRecommendations (logger: ILogger) (path: string option) : Map<string, ModelRecommendation> =
        let paths = candidatePaths path

        let rec tryLoad remaining =
            match remaining with
            | [] ->
                logger.LogWarning("No model recommendation file found. Candidates: {Candidates}", paths)
                Map.empty
            | candidate :: rest ->
                if File.Exists candidate then
                    try
                        let yaml = File.ReadAllText(candidate)
                        let loaded = loadFromText logger yaml candidate
                        if Map.isEmpty loaded then
                            tryLoad rest
                        else
                            loaded
                    with ex ->
                        logger.LogError(ex, "Failed to parse model recommendation file {Path}.", candidate)
                        tryLoad rest
                else
                    tryLoad rest

        tryLoad paths
