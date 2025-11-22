namespace TarsEngine.FSharp.SelfImprovement

open System
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.SelfImprovement.Services

module ModelSelection =

    let private cacheLock = obj()
    let mutable private cachedRecommendations : Map<string, ModelRecommendation> option = None

    let private getRecommendations (logger: ILogger) =
        match cachedRecommendations with
        | Some recommendations -> recommendations
        | None ->
            lock cacheLock (fun () ->
                match cachedRecommendations with
                | Some recommendations -> recommendations
                | None ->
                    let loaded = ModelRecommendationService.loadRecommendations logger None
                    cachedRecommendations <- Some loaded
                    loaded)

    let private tryGetEnvironment (recommendation: ModelRecommendation) key =
        recommendation.Environment
        |> Map.tryFind key
        |> Option.filter (fun value -> not (String.IsNullOrWhiteSpace value))

    let private pickString (value: string option) (fallback: string) =
        value
        |> Option.filter (fun v -> not (String.IsNullOrWhiteSpace v))
        |> Option.defaultValue fallback

    type SelectedModel =
        { Provider: string
          ModelName: string
          Endpoint: string option
          RequiresApiKey: bool
          Notes: string option }

    let selectModel (logger: ILogger) (key: string) (defaultModel: string) (defaultEndpoint: string option) =
        let recommendations = getRecommendations logger
        match recommendations |> Map.tryFind key with
        | None ->
            { Provider = "custom"
              ModelName = defaultModel
              Endpoint = defaultEndpoint
              RequiresApiKey = false
              Notes = None }
        | Some recommendation ->
            let endpoint =
                match tryGetEnvironment recommendation "endpoint" with
                | Some value -> Some value
                | None -> defaultEndpoint

            { Provider = recommendation.Provider
              ModelName = pickString (Some recommendation.Model) defaultModel
              Endpoint = endpoint
              RequiresApiKey = recommendation.RequiresApiKey
              Notes = recommendation.Notes }
