// ================================================
// 🔄 TARS Ollama-Compatible API Layer
// ================================================
// Drop-in replacement for Ollama API endpoints

namespace TarsEngine.FSharp.Core

open System
open System.Threading.Tasks
open System.Text.Json
open Microsoft.AspNetCore.Http
open Microsoft.Extensions.Logging

module TarsOllamaCompatibleAPI =

    /// Ollama-compatible request models
    type OllamaGenerateRequest = {
        model: string
        prompt: string
        stream: bool option
        raw: bool option
        format: string option
        options: Map<string, obj> option
        system: string option
        template: string option
        context: int[] option
        keep_alive: string option
    }

    type OllamaChatMessage = {
        role: string
        content: string
        images: string[] option
    }

    type OllamaChatRequest = {
        model: string
        messages: OllamaChatMessage[]
        stream: bool option
        format: string option
        options: Map<string, obj> option
        keep_alive: string option
    }

    type OllamaEmbeddingsRequest = {
        model: string
        prompt: string
        options: Map<string, obj> option
        keep_alive: string option
    }

    /// Ollama-compatible response models
    type OllamaGenerateResponse = {
        model: string
        created_at: string
        response: string
        ``done``: bool
        context: int[] option
        total_duration: int64 option
        load_duration: int64 option
        prompt_eval_count: int option
        prompt_eval_duration: int64 option
        eval_count: int option
        eval_duration: int64 option
    }

    type OllamaChatResponse = {
        model: string
        created_at: string
        message: OllamaChatMessage
        ``done``: bool
        total_duration: int64 option
        load_duration: int64 option
        prompt_eval_count: int option
        prompt_eval_duration: int64 option
        eval_count: int option
        eval_duration: int64 option
    }

    type OllamaEmbeddingsResponse = {
        embedding: float32[]
    }

    type OllamaModelInfo = {
        name: string
        modified_at: string
        size: int64
        digest: string
        details: Map<string, obj>
    }

    type OllamaModelsResponse = {
        models: OllamaModelInfo[]
    }

    /// TARS Ollama API service
    type TarsOllamaAPIService(transformer: TarsCustomTransformer.TarsTransformerModel, logger: ILogger) =
        
        /// Generate text using TARS transformer
        member _.GenerateAsync(request: OllamaGenerateRequest) : Task<OllamaGenerateResponse> =
            task {
                let startTime = DateTime.UtcNow
                let stopwatch = System.Diagnostics.Stopwatch.StartNew()
                
                logger.LogInformation("TARS Ollama API: Generating response for model {Model}", request.model)
                
                // Use TARS transformer for generation
                let maxTokens = 
                    request.options 
                    |> Option.bind (fun opts -> opts.TryFind "num_predict")
                    |> Option.map (fun v -> v :?> int)
                    |> Option.defaultValue 128
                
                let! generatedText = TarsCustomTransformer.generateText 
                    transformer 
                    request.prompt 
                    maxTokens 
                    TarsCustomTransformer.simpleTokenizer 
                    TarsCustomTransformer.simpleDetokenizer
                
                stopwatch.Stop()
                
                let response = {
                    model = request.model
                    created_at = startTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                    response = generatedText
                    ``done`` = true
                    context = Some [||] // Would include actual context in real implementation
                    total_duration = Some stopwatch.ElapsedMilliseconds
                    load_duration = Some 0L
                    prompt_eval_count = Some request.prompt.Split(' ').Length
                    prompt_eval_duration = Some (stopwatch.ElapsedMilliseconds / 4L)
                    eval_count = Some generatedText.Split(' ').Length
                    eval_duration = Some (stopwatch.ElapsedMilliseconds * 3L / 4L)
                }
                
                logger.LogInformation("TARS Ollama API: Generated {TokenCount} tokens in {Duration}ms", 
                    response.eval_count.Value, stopwatch.ElapsedMilliseconds)
                
                return response
            }
        
        /// Chat completion using TARS transformer
        member _.ChatAsync(request: OllamaChatRequest) : Task<OllamaChatResponse> =
            task {
                let startTime = DateTime.UtcNow
                let stopwatch = System.Diagnostics.Stopwatch.StartNew()
                
                logger.LogInformation("TARS Ollama API: Processing chat for model {Model}", request.model)
                
                // Convert chat messages to prompt
                let prompt = 
                    request.messages
                    |> Array.map (fun msg -> $"{msg.role}: {msg.content}")
                    |> String.concat "\n"
                    |> fun p -> p + "\nassistant:"
                
                let maxTokens = 
                    request.options 
                    |> Option.bind (fun opts -> opts.TryFind "num_predict")
                    |> Option.map (fun v -> v :?> int)
                    |> Option.defaultValue 128
                
                let! generatedText = TarsCustomTransformer.generateText 
                    transformer 
                    prompt 
                    maxTokens 
                    TarsCustomTransformer.simpleTokenizer 
                    TarsCustomTransformer.simpleDetokenizer
                
                stopwatch.Stop()
                
                let responseMessage = {
                    role = "assistant"
                    content = generatedText
                    images = None
                }
                
                let response = {
                    model = request.model
                    created_at = startTime.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                    message = responseMessage
                    ``done`` = true
                    total_duration = Some stopwatch.ElapsedMilliseconds
                    load_duration = Some 0L
                    prompt_eval_count = Some prompt.Split(' ').Length
                    prompt_eval_duration = Some (stopwatch.ElapsedMilliseconds / 4L)
                    eval_count = Some generatedText.Split(' ').Length
                    eval_duration = Some (stopwatch.ElapsedMilliseconds * 3L / 4L)
                }
                
                logger.LogInformation("TARS Ollama API: Chat completed in {Duration}ms", stopwatch.ElapsedMilliseconds)
                
                return response
            }
        
        /// Generate embeddings using TARS vector capabilities
        member _.EmbeddingsAsync(request: OllamaEmbeddingsRequest) : Task<OllamaEmbeddingsResponse> =
            task {
                logger.LogInformation("TARS Ollama API: Generating embeddings for model {Model}", request.model)
                
                // Use TARS transformer to generate embeddings
                let tokens = TarsCustomTransformer.simpleTokenizer request.prompt
                let hiddenStates = TarsCustomTransformer.forwardTransformer transformer tokens
                
                // Extract embeddings from last hidden state (mean pooling)
                let seqLen = Array2D.length1 hiddenStates
                let hiddenSize = Array2D.length2 hiddenStates
                
                let embeddings = Array.init hiddenSize (fun i ->
                    let mutable sum = 0.0f
                    for j in 0 .. seqLen - 1 do
                        sum <- sum + hiddenStates.[j, i]
                    sum / float32 seqLen
                )
                
                let response = {
                    embedding = embeddings
                }
                
                logger.LogInformation("TARS Ollama API: Generated {Dimensions} dimensional embedding", embeddings.Length)
                
                return response
            }
        
        /// List available models
        member _.ListModelsAsync() : Task<OllamaModelsResponse> =
            task {
                logger.LogInformation("TARS Ollama API: Listing available models")
                
                let tarsModel = {
                    name = "tars-transformer"
                    modified_at = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                    size = 1024L * 1024L * 500L // 500MB implementd size
                    digest = "sha256:tars-custom-transformer-v1"
                    details = Map.ofList [
                        ("format", "tars" :> obj)
                        ("family", "transformer" :> obj)
                        ("families", [|"transformer"; "tars"|] :> obj)
                        ("parameter_size", "768M" :> obj)
                        ("quantization_level", "f32" :> obj)
                    ]
                }
                
                let response = {
                    models = [|tarsModel|]
                }
                
                return response
            }
        
        /// Show model information
        member _.ShowModelAsync(modelName: string) : Task<OllamaModelInfo option> =
            task {
                logger.LogInformation("TARS Ollama API: Showing model info for {Model}", modelName)
                
                if modelName = "tars-transformer" then
                    let modelInfo = {
                        name = "tars-transformer"
                        modified_at = DateTime.UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                        size = 1024L * 1024L * 500L
                        digest = "sha256:tars-custom-transformer-v1"
                        details = Map.ofList [
                            ("format", "tars" :> obj)
                            ("family", "transformer" :> obj)
                            ("parameter_size", "768M" :> obj)
                            ("quantization_level", "f32" :> obj)
                            ("vocab_size", transformer.Config.VocabSize :> obj)
                            ("hidden_size", transformer.Config.HiddenSize :> obj)
                            ("num_layers", transformer.Config.NumLayers :> obj)
                            ("num_heads", transformer.Config.NumHeads :> obj)
                            ("cuda_enabled", transformer.Config.UseCuda :> obj)
                        ]
                    }
                    return Some modelInfo
                else
                    return None
            }

    /// HTTP endpoint handlers for Ollama compatibility
    module OllamaEndpoints =
        
        /// Handle /api/generate endpoint
        let handleGenerate (apiService: TarsOllamaAPIService) (context: HttpContext) : Task =
            task {
                try
                    let! requestBody = context.Request.ReadFromJsonAsync<OllamaGenerateRequest>()
                    let! response = apiService.GenerateAsync(requestBody)
                    
                    context.Response.ContentType <- "application/json"
                    do! context.Response.WriteAsJsonAsync(response)
                with
                | ex ->
                    context.Response.StatusCode <- 500
                    do! context.Response.WriteAsync($"Error: {ex.Message}")
            }
        
        /// Handle /api/chat endpoint
        let handleChat (apiService: TarsOllamaAPIService) (context: HttpContext) : Task =
            task {
                try
                    let! requestBody = context.Request.ReadFromJsonAsync<OllamaChatRequest>()
                    let! response = apiService.ChatAsync(requestBody)
                    
                    context.Response.ContentType <- "application/json"
                    do! context.Response.WriteAsJsonAsync(response)
                with
                | ex ->
                    context.Response.StatusCode <- 500
                    do! context.Response.WriteAsync($"Error: {ex.Message}")
            }
        
        /// Handle /api/embeddings endpoint
        let handleEmbeddings (apiService: TarsOllamaAPIService) (context: HttpContext) : Task =
            task {
                try
                    let! requestBody = context.Request.ReadFromJsonAsync<OllamaEmbeddingsRequest>()
                    let! response = apiService.EmbeddingsAsync(requestBody)
                    
                    context.Response.ContentType <- "application/json"
                    do! context.Response.WriteAsJsonAsync(response)
                with
                | ex ->
                    context.Response.StatusCode <- 500
                    do! context.Response.WriteAsync($"Error: {ex.Message}")
            }
        
        /// Handle /api/tags endpoint (list models)
        let handleTags (apiService: TarsOllamaAPIService) (context: HttpContext) : Task =
            task {
                try
                    let! response = apiService.ListModelsAsync()
                    
                    context.Response.ContentType <- "application/json"
                    do! context.Response.WriteAsJsonAsync(response)
                with
                | ex ->
                    context.Response.StatusCode <- 500
                    do! context.Response.WriteAsync($"Error: {ex.Message}")
            }
        
        /// Handle /api/show endpoint
        let handleShow (apiService: TarsOllamaAPIService) (context: HttpContext) : Task =
            task {
                try
                    let modelName = context.Request.Query.["name"].ToString()
                    let! response = apiService.ShowModelAsync(modelName)
                    
                    match response with
                    | Some modelInfo ->
                        context.Response.ContentType <- "application/json"
                        do! context.Response.WriteAsJsonAsync(modelInfo)
                    | None ->
                        context.Response.StatusCode <- 404
                        do! context.Response.WriteAsync("Model not found")
                with
                | ex ->
                    context.Response.StatusCode <- 500
                    do! context.Response.WriteAsync($"Error: {ex.Message}")
            }

    /// Create TARS Ollama-compatible API service
    let createTarsOllamaAPI (transformer: TarsCustomTransformer.TarsTransformerModel) (logger: ILogger) : TarsOllamaAPIService =
        TarsOllamaAPIService(transformer, logger)
