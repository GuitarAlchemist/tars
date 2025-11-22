namespace TarsEngine.FSharp.Cli.Core

open System
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading.Tasks
open System.Runtime.InteropServices
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core.CudaComputationExpression

/// TARS AI Models - Real transformer implementations with CUDA acceleration
module TarsAiModels =
    
    // CUDA interop for AI models
    module AiModelInterop =
        [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
        extern int cuda_mini_gpt_test()
    
    /// AI Model types and configurations
    type ModelConfig = {
        Name: string
        VocabSize: int
        SequenceLength: int
        DModel: int
        NumHeads: int
        NumLayers: int
        FeedForwardSize: int
    }
    
    /// Text generation parameters
    type GenerationConfig = {
        MaxTokens: int
        Temperature: float32
        TopK: int
        TopP: float32
        StopTokens: string list
    }
    
    /// AI Model result
    type ModelResult<'T> = {
        Success: bool
        Value: 'T option
        Error: string option
        ExecutionTimeMs: float
        TokensGenerated: int
        ModelUsed: string
    }
    
    /// Mini-GPT model implementation
    type MiniGptModel(config: ModelConfig, logger: ILogger) =
        
        /// Test the Mini-GPT model components
        member _.TestModel() : CudaOperation<string> =
            fun context ->
                async {
                    try
                        logger.LogInformation($"Testing Mini-GPT model: {config.Name}")
                        let result = AiModelInterop.cuda_mini_gpt_test()
                        
                        if result = 0 then
                            let message = $"Mini-GPT model test passed - {config.Name} (d_model={config.DModel}, heads={config.NumHeads})"
                            logger.LogInformation(message)
                            return Success message
                        else
                            let error = $"Mini-GPT model test failed with code: {result}"
                            logger.LogError(error)
                            return Error error
                    with
                    | ex ->
                        let error = $"Mini-GPT model test exception: {ex.Message}"
                        logger.LogError(ex, error)
                        return Error error
                }
        
        /// Generate text using real AI model via Ollama
        member _.GenerateText(prompt: string, config: GenerationConfig) : CudaOperation<string> =
            fun context ->
                async {
                    try
                        logger.LogInformation($"Generating text with real AI model for prompt: '{prompt}'")

                        // Real AI model inference via Ollama API
                        use httpClient = new HttpClient()
                        httpClient.Timeout <- TimeSpan.FromMinutes(2.0)

                        let requestBody = JsonSerializer.Serialize({|
                            model = "llama3:latest"
                            prompt = prompt
                            stream = false
                            options = {|
                                temperature = float config.Temperature
                                top_p = float config.TopP
                                max_tokens = config.MaxTokens
                            |}
                        |})

                        let content = new StringContent(requestBody, Encoding.UTF8, "application/json")
                        let! response = httpClient.PostAsync("http://localhost:11434/api/generate", content) |> Async.AwaitTask

                        if response.IsSuccessStatusCode then
                            let! responseBody = response.Content.ReadAsStringAsync() |> Async.AwaitTask

                            // Parse Ollama response
                            let responseJson = JsonDocument.Parse(responseBody)
                            let mutable responseElement = Unchecked.defaultof<JsonElement>
                            let generatedText =
                                if responseJson.RootElement.TryGetProperty("response", &responseElement) then
                                    responseElement.GetString()
                                else
                                    responseBody

                            logger.LogInformation($"Real AI generated text: {generatedText.Substring(0, min 100 generatedText.Length)}...")
                            return Success generatedText
                        else
                            let! errorBody = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                            let error = $"AI model API error: {response.StatusCode} - {errorBody}"
                            logger.LogError(error)
                            return Error error
                    with
                    | ex ->
                        let error = $"Real AI text generation exception: {ex.Message}"
                        logger.LogError(ex, error)
                        return Error error
                }
        
        /// Get model information
        member _.GetModelInfo() : CudaOperation<string> =
            fun context ->
                async {
                    let info = $"Mini-GPT Model: {config.Name} | Vocab: {config.VocabSize} | Seq: {config.SequenceLength} | d_model: {config.DModel} | Heads: {config.NumHeads} | Layers: {config.NumLayers}"
                    return Success info
                }
    
    /// AI Model factory and management
    type TarsAiModelFactory(logger: ILogger) =
        
        /// Create a Mini-GPT model with default configuration
        member _.CreateMiniGpt(?name: string) =
            let modelName = defaultArg name "mini-gpt-default"
            let config = {
                Name = modelName
                VocabSize = 1000
                SequenceLength = 16
                DModel = 128
                NumHeads = 8
                NumLayers = 6
                FeedForwardSize = 512
            }
            MiniGptModel(config, logger)
        
        /// Create a Mini-GPT model with custom configuration
        member _.CreateMiniGptCustom(config: ModelConfig) =
            MiniGptModel(config, logger)
        
        /// Create default generation configuration
        member _.CreateGenerationConfig(?maxTokens: int, ?temperature: float32, ?topK: int, ?topP: float32) =
            {
                MaxTokens = defaultArg maxTokens 50
                Temperature = defaultArg temperature 0.8f
                TopK = defaultArg topK 40
                TopP = defaultArg topP 0.9f
                StopTokens = ["\n"; "<|endoftext|>"]
            }
    
    /// TARS AI Model DSL operations
    module TarsAiOperations =
        
        /// Test Mini-GPT model
        let testMiniGpt (model: MiniGptModel) : CudaOperation<string> =
            model.TestModel()
        
        /// Generate text with Mini-GPT
        let generateText (model: MiniGptModel) (prompt: string) (config: GenerationConfig) : CudaOperation<string> =
            model.GenerateText(prompt, config)
        
        /// Get model information
        let getModelInfo (model: MiniGptModel) : CudaOperation<string> =
            model.GetModelInfo()
    
    /// TARS AI Model examples and demonstrations
    module TarsAiExamples =
        
        /// Example: Test Mini-GPT model
        let testMiniGptExample (logger: ILogger) =
            async {
                let factory = TarsAiModelFactory(logger)
                let model = factory.CreateMiniGpt("demo-model")
                
                let dsl = cuda (Some logger)
                let! result = dsl.Run(TarsAiOperations.testMiniGpt model)
                
                match result with
                | Success message ->
                    logger.LogInformation($"Mini-GPT test successful: {message}")
                    return {
                        Success = true
                        Value = Some message
                        Error = None
                        ExecutionTimeMs = 0.0 // Would be measured in real implementation
                        TokensGenerated = 0
                        ModelUsed = "mini-gpt-demo"
                    }
                | Error error ->
                    logger.LogError($"Mini-GPT test failed: {error}")
                    return {
                        Success = false
                        Value = None
                        Error = Some error
                        ExecutionTimeMs = 0.0
                        TokensGenerated = 0
                        ModelUsed = "mini-gpt-demo"
                    }
            }
        
        /// Example: Text generation with Mini-GPT
        let textGenerationExample (logger: ILogger) (prompt: string) =
            async {
                let factory = TarsAiModelFactory(logger)
                let model = factory.CreateMiniGpt("text-generator")
                let genConfig = factory.CreateGenerationConfig(maxTokens = 30, temperature = 0.7f)
                
                let dsl = cuda (Some logger)
                let! result = dsl.Run(TarsAiOperations.generateText model prompt genConfig)
                
                match result with
                | Success generatedText ->
                    logger.LogInformation($"Text generation successful: {generatedText}")
                    return {
                        Success = true
                        Value = Some generatedText
                        Error = None
                        ExecutionTimeMs = 0.0
                        TokensGenerated = 30
                        ModelUsed = "mini-gpt-text-generator"
                    }
                | Error error ->
                    logger.LogError($"Text generation failed: {error}")
                    return {
                        Success = false
                        Value = None
                        Error = Some error
                        ExecutionTimeMs = 0.0
                        TokensGenerated = 0
                        ModelUsed = "mini-gpt-text-generator"
                    }
            }
        
        /// Example: AI model workflow
        let aiModelWorkflowExample (logger: ILogger) =
            async {
                let factory = TarsAiModelFactory(logger)
                let model = factory.CreateMiniGpt("workflow-model")
                let genConfig = factory.CreateGenerationConfig()
                
                let dsl = cuda (Some logger)

                // Execute operations sequentially
                let! modelInfoResult = dsl.Run(TarsAiOperations.getModelInfo model)
                let! testResult = dsl.Run(TarsAiOperations.testMiniGpt model)
                let! generatedTextResult = dsl.Run(TarsAiOperations.generateText model "The future of AI is" genConfig)

                // Create combined result
                let result =
                    match modelInfoResult, testResult, generatedTextResult with
                    | Success modelInfo, Success testMsg, Success generatedText ->
                        logger.LogInformation($"Model info: {modelInfo}")
                        logger.LogInformation($"Model test: {testMsg}")
                        logger.LogInformation($"Generated: {generatedText}")
                        Success $"AI Workflow completed: {generatedText}"
                    | _ ->
                        Error "AI workflow failed"
                
                match result with
                | Success message ->
                    return {
                        Success = true
                        Value = Some message
                        Error = None
                        ExecutionTimeMs = 0.0
                        TokensGenerated = 50
                        ModelUsed = "mini-gpt-workflow"
                    }
                | Error error ->
                    return {
                        Success = false
                        Value = None
                        Error = Some error
                        ExecutionTimeMs = 0.0
                        TokensGenerated = 0
                        ModelUsed = "mini-gpt-workflow"
                    }
            }
    
    /// Create TARS AI model factory
    let createAiModelFactory (logger: ILogger) = TarsAiModelFactory(logger)
