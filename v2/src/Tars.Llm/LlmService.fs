namespace Tars.Llm

// High-level LLM service that abstracts over multiple backends.
// Provides a unified interface for chat completions and embeddings.

open System
open System.Net.Http
open System.Threading
open System.Threading.Tasks

/// <summary>
/// LLM service module providing the main service interface and implementation.
/// </summary>
module LlmService =

    open Tars.Llm
    open Tars.Llm.Routing

    open Tars.Core

    let inline private rerror<'a> (e: LlmError) : Result<'a, LlmError> = Result<'a, LlmError>.Error e

    let enrichRequest (routing: RoutingConfig) (req: LlmRequest) =
        { req with
            ContextWindow = req.ContextWindow |> Option.orElse routing.DefaultContextWindow
            Temperature = req.Temperature |> Option.orElse routing.DefaultTemperature }

    /// <summary>
    /// Default implementation of ILlmService and ILlmServiceFunctional.
    /// Routes requests to appropriate backends based on configuration.
    /// </summary>
    type DefaultLlmService(httpClient: HttpClient, cfg: LlmServiceConfig) =

        interface ILlmServiceFunctional with
            member _.CompleteAsync(req: LlmRequest) : AsyncResult<LlmResponse, LlmError> =
                asyncResult {
                    let req = enrichRequest cfg.Routing req
                    let routed = chooseBackend cfg.Routing req

                    // Ollama keeps its functional-path validation (generateValidated);
                    // every other backend delegates to the shared resolver and wraps
                    // exceptions into LlmError generically — same behaviour as before,
                    // dispatch defined once.
                    match routed.Backend with
                    | Ollama model ->
                        return! OllamaClientAsync.generateValidated httpClient routed.Endpoint model routed.ApiKey req
                    | _ ->
                        try
                            let! res =
                                (Backends.resolve cfg httpClient routed).Complete req
                                |> Async.AwaitTask
                                |> AsyncResult.ofAsync

                            return res
                        with ex ->
                            return! AsyncResult.ofResult (rerror<LlmResponse> (LlmError.fromException ex))
                }

            member _.EmbedAsync(text: string) : AsyncResult<float32[], LlmError> =
                asyncResult {
                    let model = cfg.Routing.DefaultEmbeddingModel

                    if
                        model.Contains("nomic")
                        || model.Contains("mxbai")
                        || model.Contains("llama")
                        || model.Contains("qwen")
                    then
                        return! OllamaClientAsync.getEmbeddingsAsync httpClient cfg.Routing.OllamaBaseUri model text
                    else
                        try
                            let! res =
                                OpenAiCompatibleClient.getEmbeddingsAsync
                                    httpClient
                                    cfg.Routing.OpenAIBaseUri
                                    model
                                    cfg.Routing.OpenAIKey
                                    text
                                |> Async.AwaitTask
                                |> AsyncResult.ofAsync

                            return res
                        with ex ->
                            return! AsyncResult.ofResult (rerror<float32[]> (LlmError.fromException ex))
                }

            member _.RouteAsync(req: LlmRequest) : AsyncResult<RoutedBackend, LlmError> =
                asyncResult {
                    let req = enrichRequest cfg.Routing req
                    return chooseBackend cfg.Routing req
                }

        interface ILlmService with
            member _.CompleteAsync(req: LlmRequest) : Task<LlmResponse> =
                task {
                    let req = enrichRequest cfg.Routing req
                    let routed = chooseBackend cfg.Routing req
                    return! (Backends.resolve cfg httpClient routed).Complete req
                }

            member _.CompleteStreamAsync(req: LlmRequest, onToken: string -> unit) : Task<LlmResponse> =
                task {
                    let req = enrichRequest cfg.Routing req
                    let routed = chooseBackend cfg.Routing req
                    return! (Backends.resolve cfg httpClient routed).Stream(req, onToken)
                }

            member _.EmbedAsync(text: string) : Task<float32[]> = Embedder.embed httpClient cfg.Routing text

            member _.RouteAsync(req: LlmRequest) : Task<RoutedBackend> =
                task {
                    let req = enrichRequest cfg.Routing req
                    return chooseBackend cfg.Routing req
                }

        interface ICancellableLlmService with
            member this.CompleteAsync(req, cancellationToken) =
                let work = (this :> ILlmService).CompleteAsync(req)

                if not cancellationToken.CanBeCanceled then
                    work
                else
                    task {
                        let delay = Task.Delay(Timeout.Infinite, cancellationToken)
                        let! completed = Task.WhenAny(work, delay)

                        if Object.ReferenceEquals(completed, work) then
                            return! work
                        else
                            return raise (OperationCanceledException(cancellationToken))
                    }

            member this.EmbedAsync(text, cancellationToken) =
                let work = (this :> ILlmService).EmbedAsync(text)

                if not cancellationToken.CanBeCanceled then
                    work
                else
                    task {
                        let delay = Task.Delay(Timeout.Infinite, cancellationToken)
                        let! completed = Task.WhenAny(work, delay)

                        if Object.ReferenceEquals(completed, work) then
                            return! work
                        else
                            return raise (OperationCanceledException(cancellationToken))
                    }

            member this.CompleteStreamAsync(req, onToken, cancellationToken) =
                let work = (this :> ILlmService).CompleteStreamAsync(req, onToken)

                if not cancellationToken.CanBeCanceled then
                    work
                else
                    task {
                        let delay = Task.Delay(Timeout.Infinite, cancellationToken)
                        let! completed = Task.WhenAny(work, delay)

                        if Object.ReferenceEquals(completed, work) then
                            return! work
                        else
                            return raise (OperationCanceledException(cancellationToken))
                    }

            member this.RouteAsync(req, cancellationToken) =
                let work = (this :> ILlmService).RouteAsync(req)

                if not cancellationToken.CanBeCanceled then
                    work
                else
                    task {
                        let delay = Task.Delay(Timeout.Infinite, cancellationToken)
                        let! completed = Task.WhenAny(work, delay)

                        if Object.ReferenceEquals(completed, work) then
                            return! work
                        else
                            return raise (OperationCanceledException(cancellationToken))
                    }
