namespace TarsEngine.FSharp.Cli.Services

open System
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Generic LLM request
type LlmRequest = {
    Model: string
    Prompt: string
    SystemPrompt: string option
    Temperature: float option
    MaxTokens: int option
    Context: string option
}

/// Generic LLM response
type LlmResponse = {
    Content: string
    Model: string
    TokensUsed: int option
    ResponseTime: TimeSpan
    Success: bool
    Error: string option
}

/// Generic LLM Service that can work with any model provider
type GenericLlmService(logger: ILogger<GenericLlmService>, httpClient: HttpClient) =
    
    let mutable ollamaBaseUrl = "http://localhost:11434"
    let mutable openAiBaseUrl = "https://api.openai.com/v1"
    let mutable anthropicBaseUrl = "https://api.anthropic.com/v1"
    
    /// Set Ollama base URL
    member this.SetOllamaUrl(url: string) =
        ollamaBaseUrl <- url
        
    /// Set OpenAI base URL  
    member this.SetOpenAiUrl(url: string) =
        openAiBaseUrl <- url
        
    /// Set Anthropic base URL
    member this.SetAnthropicUrl(url: string) =
        anthropicBaseUrl <- url
    
    /// Check if a model is available in Ollama
    member private this.CheckOllamaModel(modelName: string) =
        async {
            try
                let! response = httpClient.GetAsync($"{ollamaBaseUrl}/api/tags") |> Async.AwaitTask
                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                    return content.Contains(modelName)
                else
                    return false
            with
            | _ -> return false
        }
    
    /// Pull a model in Ollama
    member private this.PullOllamaModel(modelName: string) =
        async {
            try
                logger.LogInformation($"🔄 Pulling model {modelName}...")
                let pullPayload = {| name = modelName |}
                let pullJson = JsonSerializer.Serialize(pullPayload)
                let pullContent = new StringContent(pullJson, Encoding.UTF8, "application/json")
                
                let! pullResponse = httpClient.PostAsync($"{ollamaBaseUrl}/api/pull", pullContent) |> Async.AwaitTask
                if pullResponse.IsSuccessStatusCode then
                    logger.LogInformation($"✅ Successfully pulled model {modelName}")
                    return true
                else
                    logger.LogWarning($"❌ Failed to pull model {modelName}")
                    return false
            with
            | ex ->
                logger.LogError(ex, $"Error pulling model {modelName}")
                return false
        }
    
    /// Ensure Ollama model is available
    member private this.EnsureOllamaModel(modelName: string) =
        async {
            let! isAvailable = this.CheckOllamaModel(modelName)
            if isAvailable then
                return true
            else
                return! this.PullOllamaModel(modelName)
        }
    
    /// Send request to Ollama
    member private this.SendToOllama(request: LlmRequest) =
        async {
            try
                let startTime = DateTime.UtcNow
                
                // Ensure model is available
                let! modelReady = this.EnsureOllamaModel(request.Model)
                if not modelReady then
                    return {
                        Content = ""
                        Model = request.Model
                        TokensUsed = None
                        ResponseTime = DateTime.UtcNow - startTime
                        Success = false
                        Error = Some $"Model {request.Model} not available"
                    }
                else
                
                // Build messages
                let messages = 
                    match request.SystemPrompt with
                    | Some systemPrompt -> 
                        [| 
                            {| role = "system"; content = systemPrompt |}
                            {| role = "user"; content = request.Prompt |}
                        |]
                    | None -> 
                        [| {| role = "user"; content = request.Prompt |} |]
                
                // Create payload
                let payload = {|
                    model = request.Model
                    messages = messages
                    stream = false
                    options = {|
                        temperature = defaultArg request.Temperature 0.7
                        num_predict = defaultArg request.MaxTokens 1000
                    |}
                |}
                
                let json = JsonSerializer.Serialize(payload)
                let content = new StringContent(json, Encoding.UTF8, "application/json")
                
                let! response = httpClient.PostAsync($"{ollamaBaseUrl}/api/chat", content) |> Async.AwaitTask
                let! responseContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                
                if response.IsSuccessStatusCode then
                    try
                        let jsonDoc = JsonDocument.Parse(responseContent)
                        let messageContent = jsonDoc.RootElement.GetProperty("message").GetProperty("content").GetString()
                        
                        return {
                            Content = messageContent
                            Model = request.Model
                            TokensUsed = None
                            ResponseTime = DateTime.UtcNow - startTime
                            Success = true
                            Error = None
                        }
                    with
                    | ex ->
                        logger.LogError(ex, $"Error parsing Ollama response: {responseContent}")
                        return {
                            Content = ""
                            Model = request.Model
                            TokensUsed = None
                            ResponseTime = DateTime.UtcNow - startTime
                            Success = false
                            Error = Some $"Parse error: {ex.Message}"
                        }
                else
                    logger.LogError($"Ollama API error: {response.StatusCode} - {responseContent}")
                    return {
                        Content = ""
                        Model = request.Model
                        TokensUsed = None
                        ResponseTime = DateTime.UtcNow - startTime
                        Success = false
                        Error = Some $"API error: {response.StatusCode}"
                    }
            with
            | ex ->
                logger.LogError(ex, "Error calling Ollama API")
                return {
                    Content = ""
                    Model = request.Model
                    TokensUsed = None
                    ResponseTime = TimeSpan.Zero
                    Success = false
                    Error = Some ex.Message
                }
        }
    
    /// Auto-detect provider based on model name
    member private this.DetectProvider(modelName: string) =
        if modelName.StartsWith("gpt-") || modelName.StartsWith("text-") then
            "openai"
        elif modelName.StartsWith("claude-") then
            "anthropic"
        else
            "ollama"  // Default to Ollama for local models
    
    /// Send request to appropriate provider
    member this.SendRequest(request: LlmRequest) =
        async {
            let provider = this.DetectProvider(request.Model)
            logger.LogInformation($"🤖 Using {provider} provider for model {request.Model}")
            
            match provider with
            | "ollama" -> return! this.SendToOllama(request)
            | "openai" -> 
                logger.LogWarning("OpenAI provider not implemented yet, falling back to Ollama")
                return! this.SendToOllama(request)
            | "anthropic" -> 
                logger.LogWarning("Anthropic provider not implemented yet, falling back to Ollama")
                return! this.SendToOllama(request)
            | _ -> 
                return {
                    Content = ""
                    Model = request.Model
                    TokensUsed = None
                    ResponseTime = TimeSpan.Zero
                    Success = false
                    Error = Some $"Unknown provider: {provider}"
                }
        }
    
    /// List available models
    member this.ListAvailableModels() =
        async {
            try
                let! response = httpClient.GetAsync($"{ollamaBaseUrl}/api/tags") |> Async.AwaitTask
                if response.IsSuccessStatusCode then
                    let! content = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                    let jsonDoc = JsonDocument.Parse(content)
                    let models = 
                        jsonDoc.RootElement.GetProperty("models").EnumerateArray()
                        |> Seq.map (fun model -> model.GetProperty("name").GetString())
                        |> Seq.toArray
                    return Ok models
                else
                    return Error "Failed to connect to Ollama"
            with
            | ex ->
                logger.LogError(ex, "Error listing models")
                return Error ex.Message
        }
    
    /// Get recommended models for different use cases
    member this.GetRecommendedModels() =
        [|
            ("llama3.1", "General purpose, good balance of speed and quality")
            ("llama3.2", "Latest Llama model, improved reasoning")
            ("mistral", "Fast and efficient for most tasks")
            ("codellama", "Specialized for code generation")
            ("phi3", "Small but capable model, very fast")
            ("gemma2", "Google's efficient model")
        |]
