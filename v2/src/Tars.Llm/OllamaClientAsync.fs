namespace Tars.Llm

// Ollama client with AsyncResult for functional error handling.
// This module provides the same functionality as OllamaClient but with type-safe error handling.

open System
open System.Net.Http
open Tars.Core

module OllamaClientAsync =
    open OllamaClient

    /// <summary>
    /// Generate chat completion with AsyncResult for clean error handling
    /// </summary>
    let generateAsync
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (apiKey: string option)
        (req: LlmRequest)
        : AsyncResult<LlmResponse, LlmError> =

        if String.IsNullOrWhiteSpace model then
            AsyncResult.ofResult (Result.Error(ModelNotFound model))
        elif req.Messages.Length = 0 then
            AsyncResult.ofResult (Result.Error(InvalidPrompt "No messages provided"))
        else
            async {
                try
                    let! response = sendChatAsync http baseUri model apiKey req |> Async.AwaitTask
                    return Result.Ok response
                with ex ->
                    return Result.Error(LlmError.fromException ex)
            }

    /// <summary>
    /// Get embeddings with AsyncResult
    /// </summary>
    let getEmbeddingsAsync
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (text: string)
        : AsyncResult<float32[], LlmError> =

        if String.IsNullOrWhiteSpace text then
            AsyncResult.ofResult (Result.Error(InvalidPrompt "Empty text for embedding"))
        else
            async {
                try
                    let! embeddings = OllamaClient.getEmbeddingsAsync http baseUri model text |> Async.AwaitTask
                    return Result.Ok embeddings
                with ex ->
                    return Result.Error(LlmError.fromException ex)
            }

    /// <summary>
    /// List available models with AsyncResult
    /// </summary>
    let listModelsAsync (http: HttpClient) (baseUri: Uri) : AsyncResult<string list, LlmError> =
        async {
            try
                let! models = OllamaClient.getTagsAsync http baseUri |> Async.AwaitTask
                return Result.Ok models
            with ex ->
                return Result.Error(LlmError.fromException ex)
        }

    /// <summary>
    /// Generate with retries on transient errors
    /// </summary>
    let generateWithRetry
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (apiKey: string option)
        (req: LlmRequest)
        (maxRetries: int)
        : AsyncResult<LlmResponse, LlmError> =

        let rec retry attempt =
            async {
                let! result = generateAsync http baseUri model apiKey req

                match result with
                | Result.Ok response -> return Result.Ok response
                | Result.Error(NetworkError _ as err) when attempt < maxRetries ->
                    do! Async.Sleep(100 * (pown 2 attempt))
                    return! retry (attempt + 1)
                | Result.Error(ApiTimeout _ as err) when attempt < maxRetries ->
                    do! Async.Sleep(100 * (pown 2 attempt))
                    return! retry (attempt + 1)
                | Result.Error err -> return Result.Error err
            }

        retry 0

    /// <summary>
    /// Generate with validation and better error messages
    /// </summary>
    let generateValidated
        (http: HttpClient)
        (baseUri: Uri)
        (model: string)
        (apiKey: string option)
        (req: LlmRequest)
        : AsyncResult<LlmResponse, LlmError> =

        // Validate model name
        if
            not (
                model.StartsWith("tars-")
                || model.StartsWith("llama")
                || model.StartsWith("mistral")
                || model.StartsWith("qwen")
                || model.Contains("magistral")
            )
        then
            AsyncResult.ofResult (Result.Error(ModelNotFound $"Unknown model: {model}"))
        // Validate messages
        elif req.Messages |> List.exists (fun m -> String.IsNullOrWhiteSpace m.Content) then
            AsyncResult.ofResult (Result.Error(InvalidPrompt "Message with empty content"))
        // Validate temperature
        elif req.Temperature |> Option.exists (fun t -> t < 0.0 || t > 2.0) then
            AsyncResult.ofResult (Result.Error(InvalidPrompt "Temperature out of range [0,2]"))
        else
            // Generate
            generateAsync http baseUri model apiKey req

/// <summary>
/// Example usage patterns for the AsyncResult-based client
/// </summary>
module OllamaClientAsyncExamples =
    open OllamaClientAsync

    /// Simple usage with pattern matching
    let exampleSimple http baseUri =
        async {
            let req: LlmRequest =
                { ModelHint = None
                  Model = None
                  SystemPrompt = None
                  MaxTokens = None
                  Temperature = Some 0.7
                  Stop = []
                  Messages = [ { Role = Role.User; Content = "Hello!" } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            let! result = generateAsync http baseUri "llama3.2" None req

            match result with
            | Result.Ok response -> printfn $"Success: %s{response.Text}"
            | Result.Error err -> printfn $"Error: %s{LlmError.toMessage err}"
        }

    /// With retries
    let exampleWithRetry http baseUri =
        async {
            let req: LlmRequest =
                { ModelHint = None
                  Model = None
                  SystemPrompt = None
                  MaxTokens = None
                  Temperature = Some 0.7
                  Stop = []
                  Messages = [ { Role = Role.User; Content = "Hello!" } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            let! result = generateWithRetry http baseUri "llama3.2" None req 3

            match result with
            | Result.Ok response -> printfn $"Got response after retries: %s{response.Text}"
            | Result.Error err -> printfn $"Failed after retries: %s{LlmError.toMessage err}"
        }

    /// Chaining multiple LLM calls
    let exampleChaining http baseUri =
        async {
            // First call
            let req1: LlmRequest =
                { ModelHint = None
                  Model = None
                  SystemPrompt = None
                  MaxTokens = None
                  Temperature = Some 0.1
                  Stop = []
                  Messages =
                    [ { Role = Role.User
                        Content = "What is 2+2?" } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            let! result1 = generateAsync http baseUri "llama3.2" None req1

            match result1 with
            | Result.Error err -> return Result.Error err
            | Result.Ok response1 ->
                // Second call using first response
                let req2: LlmRequest =
                    { ModelHint = None
                      Model = None
                      SystemPrompt = None
                      MaxTokens = None
                      Temperature = Some 0.1
                      Stop = []
                      Messages =
                        [ { Role = Role.User
                            Content = "What is 2+2?" }
                          { Role = Role.Assistant
                            Content = response1.Text }
                          { Role = Role.User
                            Content = "Now multiply that by 5" } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None
                      ContextWindow = None }

                let! result2 = generateAsync http baseUri "llama3.2" None req2

                match result2 with
                | Result.Error err -> return Result.Error err
                | Result.Ok response2 -> return Result.Ok response2.Text
        }
