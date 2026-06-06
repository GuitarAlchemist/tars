namespace TarsEngine.FSharp.Cli.Services

open System
open System.Collections.Generic
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Advanced LLM providers
type LLMProvider =
    | OpenAI_GPT4
    | OpenAI_GPT4_Turbo
    | OpenAI_GPT4o
    | Anthropic_Claude3_Opus
    | Anthropic_Claude3_Sonnet
    | Anthropic_Claude3_Haiku
    | Google_Gemini_Pro
    | Google_Gemini_Ultra
    | Ollama_Mixtral
    | Ollama_Llama3_70B
    | Ollama_CodeLlama
    | Ollama_Qwen2_72B

/// LLM model configuration
type LLMModel = {
    Provider: LLMProvider
    ModelName: string
    DisplayName: string
    MaxTokens: int
    SupportsVision: bool
    SupportsCodeGeneration: bool
    SupportsReasoning: bool
    CostPerToken: float option
    ApiEndpoint: string
    RequiresApiKey: bool
}

/// LLM request configuration
type LLMRequest = {
    Model: LLMModel
    Messages: LLMMessage list
    Temperature: float
    MaxTokens: int option
    SystemPrompt: string option
    Tools: LLMTool list option
}

and LLMMessage = {
    Role: string // "system", "user", "assistant"
    Content: string
    ToolCalls: LLMToolCall list option
}

and LLMTool = {
    Type: string
    Function: LLMFunction
}

and LLMFunction = {
    Name: string
    Description: string
    Parameters: JsonElement
}

and LLMToolCall = {
    Id: string
    Type: string
    Function: LLMFunctionCall
}

and LLMFunctionCall = {
    Name: string
    Arguments: string
}

/// LLM response
type LLMResponse = {
    Content: string
    Model: string
    TokensUsed: int
    ResponseTime: TimeSpan
    Provider: LLMProvider
    FinishReason: string
    ToolCalls: LLMToolCall list option
    Confidence: float
}

/// Advanced LLM Service - Supports multiple state-of-the-art models
type AdvancedLLMService(logger: ILogger<AdvancedLLMService>, httpClient: HttpClient) =

    /// Available advanced LLM models
    member private this.GetAvailableModels() = [
        // OpenAI GPT-4 family (most advanced reasoning)
        {
            Provider = OpenAI_GPT4o
            ModelName = "gpt-4o"
            DisplayName = "GPT-4o (Latest)"
            MaxTokens = 128000
            SupportsVision = true
            SupportsCodeGeneration = true
            SupportsReasoning = true
            CostPerToken = Some 0.00003
            ApiEndpoint = "https://api.openai.com/v1/chat/completions"
            RequiresApiKey = true
        }
        {
            Provider = OpenAI_GPT4_Turbo
            ModelName = "gpt-4-turbo"
            DisplayName = "GPT-4 Turbo"
            MaxTokens = 128000
            SupportsVision = true
            SupportsCodeGeneration = true
            SupportsReasoning = true
            CostPerToken = Some 0.00001
            ApiEndpoint = "https://api.openai.com/v1/chat/completions"
            RequiresApiKey = true
        }
        
        // Anthropic Claude 3 family (excellent reasoning and safety)
        {
            Provider = Anthropic_Claude3_Opus
            ModelName = "claude-3-opus-20240229"
            DisplayName = "Claude 3 Opus (Most Capable)"
            MaxTokens = 200000
            SupportsVision = true
            SupportsCodeGeneration = true
            SupportsReasoning = true
            CostPerToken = Some 0.000015
            ApiEndpoint = "https://api.anthropic.com/v1/messages"
            RequiresApiKey = true
        }
        {
            Provider = Anthropic_Claude3_Sonnet
            ModelName = "claude-3-5-sonnet-20241022"
            DisplayName = "Claude 3.5 Sonnet (Balanced)"
            MaxTokens = 200000
            SupportsVision = true
            SupportsCodeGeneration = true
            SupportsReasoning = true
            CostPerToken = Some 0.000003
            ApiEndpoint = "https://api.anthropic.com/v1/messages"
            RequiresApiKey = true
        }
        
        // Google Gemini family (multimodal capabilities)
        {
            Provider = Google_Gemini_Pro
            ModelName = "gemini-1.5-pro"
            DisplayName = "Gemini 1.5 Pro"
            MaxTokens = 2000000
            SupportsVision = true
            SupportsCodeGeneration = true
            SupportsReasoning = true
            CostPerToken = Some 0.0000035
            ApiEndpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
            RequiresApiKey = true
        }
        
        // Local Ollama models (privacy-focused, no API costs)
        {
            Provider = Ollama_Qwen2_72B
            ModelName = "qwen2.5:7b"
            DisplayName = "Qwen2.5 7B (Advanced Reasoning)"
            MaxTokens = 32768
            SupportsVision = false
            SupportsCodeGeneration = true
            SupportsReasoning = true
            CostPerToken = None
            ApiEndpoint = "http://localhost:11434/v1/chat/completions"
            RequiresApiKey = false
        }
        {
            Provider = Ollama_Llama3_70B
            ModelName = "llama3.2:3b"
            DisplayName = "Llama 3.2 3B (Fast & Efficient)"
            MaxTokens = 8192
            SupportsVision = false
            SupportsCodeGeneration = true
            SupportsReasoning = true
            CostPerToken = None
            ApiEndpoint = "http://localhost:11434/v1/chat/completions"
            RequiresApiKey = false
        }
        {
            Provider = Ollama_CodeLlama
            ModelName = "codellama:7b"
            DisplayName = "Code Llama 7B (Code Specialist)"
            MaxTokens = 16384
            SupportsVision = false
            SupportsCodeGeneration = true
            SupportsReasoning = false
            CostPerToken = None
            ApiEndpoint = "http://localhost:11434/v1/chat/completions"
            RequiresApiKey = false
        }
        {
            Provider = Ollama_Mixtral
            ModelName = "phi3.5:3.8b"
            DisplayName = "Phi 3.5 3.8B (Microsoft Research)"
            MaxTokens = 32768
            SupportsVision = false
            SupportsCodeGeneration = true
            SupportsReasoning = true
            CostPerToken = None
            ApiEndpoint = "http://localhost:11434/v1/chat/completions"
            RequiresApiKey = false
        }
    ]

    /// Get the best model for a specific task
    member this.GetBestModelForTask(taskType: string, requiresLocal: bool) =
        let models = this.GetAvailableModels()
        
        match taskType.ToLower(), requiresLocal with
        | task, true when task.Contains("code") ->
            models |> List.find (fun m -> m.Provider = Ollama_CodeLlama)
        | task, true when task.Contains("reason") ->
            models |> List.find (fun m -> m.Provider = Ollama_Qwen2_72B)
        | _, true ->
            models |> List.find (fun m -> m.Provider = Ollama_Mixtral)
        | task, false when task.Contains("reason") || task.Contains("complex") ->
            models |> List.find (fun m -> m.Provider = Anthropic_Claude3_Opus)
        | task, false when task.Contains("code") ->
            models |> List.find (fun m -> m.Provider = OpenAI_GPT4o)
        | task, false when task.Contains("vision") || task.Contains("image") ->
            models |> List.find (fun m -> m.Provider = Google_Gemini_Pro)
        | _, false ->
            models |> List.find (fun m -> m.Provider = Anthropic_Claude3_Sonnet)

    /// Send request to OpenAI API
    member private this.SendOpenAIRequest(model: LLMModel, request: LLMRequest) =
        task {
            let apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
            if String.IsNullOrEmpty(apiKey) then
                return Error "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            else
                try
                    let startTime = DateTime.UtcNow
                    
                    let payload = {|
                        model = model.ModelName
                        messages = request.Messages |> List.map (fun m -> {| role = m.Role; content = m.Content |})
                        temperature = request.Temperature
                        max_tokens = request.MaxTokens |> Option.defaultValue model.MaxTokens
                    |}
                    
                    let json = JsonSerializer.Serialize(payload)
                    let content = new StringContent(json, Encoding.UTF8, "application/json")
                    
                    httpClient.DefaultRequestHeaders.Clear()
                    httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}")
                    
                    let! response = httpClient.PostAsync(model.ApiEndpoint, content)
                    let! responseContent = response.Content.ReadAsStringAsync()
                    
                    if response.IsSuccessStatusCode then
                        let responseJson = JsonDocument.Parse(responseContent)
                        let choices = responseJson.RootElement.GetProperty("choices")
                        let firstChoice = choices[0]
                        let message = firstChoice.GetProperty("message")
                        let responseText = message.GetProperty("content").GetString()
                        let usage = responseJson.RootElement.GetProperty("usage")
                        let totalTokens = usage.GetProperty("total_tokens").GetInt32()
                        
                        let result = {
                            Content = responseText
                            Model = model.ModelName
                            TokensUsed = totalTokens
                            ResponseTime = DateTime.UtcNow - startTime
                            Provider = model.Provider
                            FinishReason = firstChoice.GetProperty("finish_reason").GetString()
                            ToolCalls = None
                            Confidence = 0.9 // High confidence for GPT-4
                        }
                        
                        logger.LogInformation("OpenAI {Model} response: {Tokens} tokens in {Time}ms", 
                            model.DisplayName, totalTokens, result.ResponseTime.TotalMilliseconds)
                        
                        return Ok result
                    else
                        let error = $"OpenAI API error: {response.StatusCode} - {responseContent}"
                        logger.LogError("OpenAI API failed: {Error}", error)
                        return Error error
                        
                with ex ->
                    logger.LogError(ex, "Exception calling OpenAI API")
                    return Error ex.Message
        }

    /// Send request to Anthropic Claude API
    member private this.SendAnthropicRequest(model: LLMModel, request: LLMRequest) =
        task {
            let apiKey = Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY")
            if String.IsNullOrEmpty(apiKey) then
                return Error "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
            else
                try
                    let startTime = DateTime.UtcNow
                    
                    let payload = {|
                        model = model.ModelName
                        max_tokens = request.MaxTokens |> Option.defaultValue 4096
                        messages = request.Messages |> List.map (fun m -> {| role = m.Role; content = m.Content |})
                        temperature = request.Temperature
                    |}
                    
                    let json = JsonSerializer.Serialize(payload)
                    let content = new StringContent(json, Encoding.UTF8, "application/json")
                    
                    httpClient.DefaultRequestHeaders.Clear()
                    httpClient.DefaultRequestHeaders.Add("x-api-key", apiKey)
                    httpClient.DefaultRequestHeaders.Add("anthropic-version", "2023-06-01")
                    
                    let! response = httpClient.PostAsync(model.ApiEndpoint, content)
                    let! responseContent = response.Content.ReadAsStringAsync()
                    
                    if response.IsSuccessStatusCode then
                        let responseJson = JsonDocument.Parse(responseContent)
                        let content = responseJson.RootElement.GetProperty("content")
                        let firstContent = content[0]
                        let responseText = firstContent.GetProperty("text").GetString()
                        let usage = responseJson.RootElement.GetProperty("usage")
                        let totalTokens = usage.GetProperty("input_tokens").GetInt32() + usage.GetProperty("output_tokens").GetInt32()
                        
                        let result = {
                            Content = responseText
                            Model = model.ModelName
                            TokensUsed = totalTokens
                            ResponseTime = DateTime.UtcNow - startTime
                            Provider = model.Provider
                            FinishReason = responseJson.RootElement.GetProperty("stop_reason").GetString()
                            ToolCalls = None
                            Confidence = 0.95 // Very high confidence for Claude
                        }
                        
                        logger.LogInformation("Anthropic {Model} response: {Tokens} tokens in {Time}ms", 
                            model.DisplayName, totalTokens, result.ResponseTime.TotalMilliseconds)
                        
                        return Ok result
                    else
                        let error = $"Anthropic API error: {response.StatusCode} - {responseContent}"
                        logger.LogError("Anthropic API failed: {Error}", error)
                        return Error error
                        
                with ex ->
                    logger.LogError(ex, "Exception calling Anthropic API")
                    return Error ex.Message
        }

    /// Send request to Google Gemini API
    member private this.SendGeminiRequest(model: LLMModel, request: LLMRequest) =
        task {
            let apiKey = Environment.GetEnvironmentVariable("GOOGLE_API_KEY")
            if String.IsNullOrEmpty(apiKey) then
                return Error "Google API key not found. Set GOOGLE_API_KEY environment variable."
            else
                try
                    let startTime = DateTime.UtcNow

                    let payload = {|
                        contents = [|
                            {|
                                parts = [| {| text = request.Messages |> List.map (_.Content) |> String.concat "\n" |} |]
                            |}
                        |]
                        generationConfig = {|
                            temperature = request.Temperature
                            maxOutputTokens = request.MaxTokens |> Option.defaultValue 8192
                        |}
                    |}

                    let json = JsonSerializer.Serialize(payload)
                    let content = new StringContent(json, Encoding.UTF8, "application/json")

                    let endpoint = $"{model.ApiEndpoint}?key={apiKey}"

                    let! response = httpClient.PostAsync(endpoint, content)
                    let! responseContent = response.Content.ReadAsStringAsync()

                    if response.IsSuccessStatusCode then
                        let responseJson = JsonDocument.Parse(responseContent)
                        let candidates = responseJson.RootElement.GetProperty("candidates")
                        let firstCandidate = candidates[0]
                        let content = firstCandidate.GetProperty("content")
                        let parts = content.GetProperty("parts")
                        let responseText = parts[0].GetProperty("text").GetString()

                        let result = {
                            Content = responseText
                            Model = model.ModelName
                            TokensUsed = 0 // Gemini doesn't always return token count
                            ResponseTime = DateTime.UtcNow - startTime
                            Provider = model.Provider
                            FinishReason = firstCandidate.GetProperty("finishReason").GetString()
                            ToolCalls = None
                            Confidence = 0.85 // Good confidence for Gemini
                        }

                        logger.LogInformation("Google {Model} response in {Time}ms",
                            model.DisplayName, result.ResponseTime.TotalMilliseconds)

                        return Ok result
                    else
                        let error = $"Google API error: {response.StatusCode} - {responseContent}"
                        logger.LogError("Google API failed: {Error}", error)
                        return Error error

                with ex ->
                    logger.LogError(ex, "Exception calling Google API")
                    return Error ex.Message
        }

    /// Send request to Ollama (local models)
    member private this.SendOllamaRequest(model: LLMModel, request: LLMRequest) =
        task {
            try
                let startTime = DateTime.UtcNow

                let payload = {|
                    model = model.ModelName
                    messages = request.Messages |> List.map (fun m -> {| role = m.Role; content = m.Content |})
                    temperature = request.Temperature
                    stream = false
                |}

                let json = JsonSerializer.Serialize(payload)
                let content = new StringContent(json, Encoding.UTF8, "application/json")

                httpClient.DefaultRequestHeaders.Clear()

                let! response = httpClient.PostAsync(model.ApiEndpoint, content)
                let! responseContent = response.Content.ReadAsStringAsync()

                if response.IsSuccessStatusCode then
                    logger.LogInformation("Ollama response: {Response}", responseContent)
                    let responseJson = JsonDocument.Parse(responseContent)

                    // Parse OpenAI-compatible format: { "choices": [{"message": {"content": "..."}}] }
                    let responseText =
                        try
                            // Try OpenAI format: { "choices": [{"message": {"content": "..."}}] }
                            let choices = responseJson.RootElement.GetProperty("choices")
                            let firstChoice = choices[0]
                            let message = firstChoice.GetProperty("message")
                            message.GetProperty("content").GetString()
                        with
                        | :? KeyNotFoundException ->
                            try
                                // Try format 2: { "response": "..." }
                                responseJson.RootElement.GetProperty("response").GetString()
                            with
                            | :? KeyNotFoundException ->
                                try
                                    // Try format 3: { "content": "..." }
                                    responseJson.RootElement.GetProperty("content").GetString()
                                with
                                | :? KeyNotFoundException ->
                                    // Fallback: return the whole response as string
                                    responseContent

                    // Extract token count if available
                    let tokenCount =
                        try
                            let usage = responseJson.RootElement.GetProperty("usage")
                            usage.GetProperty("total_tokens").GetInt32()
                        with
                        | :? KeyNotFoundException -> 0

                    let result = {
                        Content = responseText
                        Model = model.ModelName
                        TokensUsed = tokenCount
                        ResponseTime = DateTime.UtcNow - startTime
                        Provider = model.Provider
                        FinishReason = "stop"
                        ToolCalls = None
                        Confidence = 0.8 // Good confidence for local models
                    }

                    logger.LogInformation("Ollama {Model} response in {Time}ms",
                        model.DisplayName, result.ResponseTime.TotalMilliseconds)

                    return Ok result
                else
                    let error = $"Ollama error: {response.StatusCode} - {responseContent}"
                    logger.LogError("Ollama failed: {Error}", error)
                    return Error error

            with ex ->
                logger.LogError(ex, "Exception calling Ollama")
                return Error ex.Message
        }

    /// Main query method - automatically selects best model and routes request
    member this.QueryAsync(query: string, ?taskType: string, ?requiresLocal: bool, ?preferredProvider: LLMProvider) =
        task {
            try
                let taskType = defaultArg taskType "general"
                let requiresLocal = defaultArg requiresLocal false

                let model =
                    match preferredProvider with
                    | Some provider ->
                        this.GetAvailableModels() |> List.find (fun m -> m.Provider = provider)
                    | None ->
                        this.GetBestModelForTask(taskType, requiresLocal)

                let request = {
                    Model = model
                    Messages = [
                        { Role = "user"; Content = query; ToolCalls = None }
                    ]
                    Temperature = 0.7
                    MaxTokens = None
                    SystemPrompt = None
                    Tools = None
                }

                logger.LogInformation("Routing query to {Model} for task: {Task}", model.DisplayName, taskType)

                match model.Provider with
                | OpenAI_GPT4 | OpenAI_GPT4_Turbo | OpenAI_GPT4o ->
                    return! this.SendOpenAIRequest(model, request)
                | Anthropic_Claude3_Opus | Anthropic_Claude3_Sonnet | Anthropic_Claude3_Haiku ->
                    return! this.SendAnthropicRequest(model, request)
                | Google_Gemini_Pro | Google_Gemini_Ultra ->
                    return! this.SendGeminiRequest(model, request)
                | Ollama_Mixtral | Ollama_Llama3_70B | Ollama_CodeLlama | Ollama_Qwen2_72B ->
                    return! this.SendOllamaRequest(model, request)

            with ex ->
                logger.LogError(ex, "Exception in QueryAsync")
                return Error ex.Message
        }

    /// Get available models with their capabilities
    member this.GetModelCapabilities() =
        let models = this.GetAvailableModels()
        models |> List.map (fun m ->
            $"{m.DisplayName} - Reasoning: {m.SupportsReasoning}, Code: {m.SupportsCodeGeneration}, Vision: {m.SupportsVision}, Local: {not m.RequiresApiKey}")

    /// Test connectivity to all available models
    member this.TestAllModelsAsync() =
        task {
            let models = this.GetAvailableModels()
            let results = ResizeArray<string * bool * string>()

            for model in models do
                try
                    let! result = this.QueryAsync("Hello, respond with just 'OK'", preferredProvider = model.Provider)
                    match result with
                    | Ok response ->
                        results.Add((model.DisplayName, true, $"✅ {response.ResponseTime.TotalMilliseconds:F0}ms"))
                    | Error error ->
                        results.Add((model.DisplayName, false, $"❌ {error}"))
                with ex ->
                    results.Add((model.DisplayName, false, $"❌ Exception: {ex.Message}"))

            return results |> Seq.toList
        }
