namespace TarsEngine

open System
open System.Threading.Tasks
open TarsEngine.TarsTransformer
open TarsEngine.TarsTokenizer
open TarsEngine.TarsAiOptimization

/// TARS Complete AI Engine - Full AI system with transformer, tokenizer, and optimization
module TarsCompleteAiEngine =
    
    // ============================================================================
    // COMPLETE AI ENGINE TYPES
    // ============================================================================
    
    type CompleteAiConfig = {
        ModelName: string
        TransformerConfig: TransformerConfig
        TokenizerConfig: TokenizerConfig
        MaxNewTokens: int
        Temperature: float32
        TopK: int
        TopP: float32
        RepetitionPenalty: float32
        UseOptimization: bool
        OptimizationStrategy: OptimizationStrategy option
    }
    
    type AiRequest = {
        Prompt: string
        MaxTokens: int option
        Temperature: float32 option
        StopSequences: string[] option
        Stream: bool
    }
    
    type AiResponse = {
        GeneratedText: string
        TokensGenerated: int
        InferenceTimeMs: float
        TokensPerSecond: float
        ModelUsed: string
        CudaAccelerated: bool
        OptimizationUsed: bool
    }
    
    type EngineMetrics = {
        TotalInferences: int64
        AverageInferenceTimeMs: float
        AverageTokensPerSecond: float
        TotalTokensGenerated: int64
        CudaAccelerationRate: float
        OptimizationSuccessRate: float
    }
    
    // ============================================================================
    // TARS COMPLETE AI ENGINE
    // ============================================================================
    
    type TarsCompleteAiEngine() =
        let mutable config: CompleteAiConfig option = None
        let mutable transformer: TarsTransformerEngine option = None
        let mutable tokenizer: TarsTokenizer option = None
        let mutable isInitialized = false
        let mutable metrics = {
            TotalInferences = 0L
            AverageInferenceTimeMs = 0.0
            AverageTokensPerSecond = 0.0
            TotalTokensGenerated = 0L
            CudaAccelerationRate = 0.0
            OptimizationSuccessRate = 0.0
        }
        
        /// Initialize the complete AI engine
        member _.Initialize(aiConfig: CompleteAiConfig) = async {
            printfn ""
            printfn "🚀 Initializing TARS Complete AI Engine..."
            printfn "========================================"
            printfn ""
            
            config <- Some aiConfig
            
            // Initialize transformer
            printfn "🧠 Initializing Transformer Engine..."
            let transformerEngine = new TarsTransformerEngine()
            let! transformerInit = transformerEngine.Initialize()
            
            if transformerInit then
                let! modelLoaded = transformerEngine.LoadModel(aiConfig.TransformerConfig)
                if modelLoaded then
                    transformer <- Some transformerEngine
                    printfn "✅ Transformer engine ready"
                else
                    failwith "Failed to load transformer model"
            else
                failwith "Failed to initialize transformer engine"
            
            // Initialize tokenizer
            printfn ""
            printfn "🔤 Initializing Tokenizer..."
            let tokenizerEngine = new TarsTokenizer(aiConfig.TokenizerConfig)
            let! tokenizerInit = tokenizerEngine.Initialize()
            
            if tokenizerInit then
                tokenizer <- Some tokenizerEngine
                printfn "✅ Tokenizer ready"
            else
                failwith "Failed to initialize tokenizer"
            
            isInitialized <- true
            
            printfn ""
            printfn "🎉 TARS Complete AI Engine Initialized!"
            printfn "======================================"
            printfn $"📊 Model: {aiConfig.ModelName}"
            printfn $"🧠 Parameters: {this.GetParameterCount():N0}"
            printfn $"🔤 Vocab size: {aiConfig.TokenizerConfig.VocabSize:N0}"
            printfn $"📏 Max sequence: {aiConfig.TokenizerConfig.MaxSequenceLength}"
            printfn $"🚀 CUDA acceleration: {this.IsCudaAvailable()}"
            printfn $"🔧 Optimization: {aiConfig.UseOptimization}"
            printfn ""
            
            return true
        }
        
        /// Generate text from a prompt
        member _.GenerateText(request: AiRequest) = async {
            if not isInitialized then
                failwith "AI engine not initialized. Call Initialize() first."
            
            let startTime = DateTime.UtcNow
            
            match transformer, tokenizer, config with
            | Some t, Some tok, Some cfg ->
                
                printfn $"🤖 Generating text for prompt: \"{request.Prompt.[..Math.Min(50, request.Prompt.Length-1)]}...\""
                
                // Tokenize input
                let! tokenizationResult = tok.Tokenize(request.Prompt)
                printfn $"🔤 Tokenized: {tokenizationResult.Tokens.Length} tokens in {tokenizationResult.ProcessingTimeMs:F2}ms"
                
                // Determine generation parameters
                let maxTokens = request.MaxTokens |> Option.defaultValue cfg.MaxNewTokens
                let temperature = request.Temperature |> Option.defaultValue cfg.Temperature
                
                // Generate tokens
                let generatedTokenIds = ResizeArray<int>()
                let mutable currentTokenIds = tokenizationResult.TokenIds
                
                // Simple autoregressive generation
                for i in 1..maxTokens do
                    // Forward pass through transformer
                    let! logits = t.ForwardPass(currentTokenIds)
                    
                    // Get next token (simplified - just take most likely)
                    let lastLogits = logits.[logits.GetLength(0)-1, *]
                    let nextTokenId = this.SampleNextToken(lastLogits, temperature, cfg.TopK, cfg.TopP)
                    
                    generatedTokenIds.Add(nextTokenId)
                    
                    // Check for stop sequences
                    let! currentText = tok.Detokenize(generatedTokenIds.ToArray())
                    match request.StopSequences with
                    | Some stopSeqs ->
                        if stopSeqs |> Array.exists (fun stop -> currentText.Contains(stop)) then
                            break
                    | None -> ()
                    
                    // Update input for next iteration (simplified)
                    currentTokenIds <- Array.append currentTokenIds [| nextTokenId |]
                    if currentTokenIds.Length > cfg.TokenizerConfig.MaxSequenceLength then
                        currentTokenIds <- currentTokenIds.[1..]
                
                // Detokenize generated tokens
                let! generatedText = tok.Detokenize(generatedTokenIds.ToArray())
                
                let endTime = DateTime.UtcNow
                let totalTime = (endTime - startTime).TotalMilliseconds
                let tokensPerSecond = float generatedTokenIds.Count / (totalTime / 1000.0)
                
                // Update metrics
                this.UpdateMetrics(totalTime, generatedTokenIds.Count, this.IsCudaAvailable(), cfg.UseOptimization)
                
                printfn $"✅ Generated {generatedTokenIds.Count} tokens in {totalTime:F2}ms ({tokensPerSecond:F1} tokens/sec)"
                
                return {
                    GeneratedText = generatedText
                    TokensGenerated = generatedTokenIds.Count
                    InferenceTimeMs = totalTime
                    TokensPerSecond = tokensPerSecond
                    ModelUsed = cfg.ModelName
                    CudaAccelerated = this.IsCudaAvailable()
                    OptimizationUsed = cfg.UseOptimization
                }
                
            | _ -> failwith "Engine components not properly initialized"
        }
        
        /// Sample next token from logits
        member _.SampleNextToken(logits: float32[], temperature: float32, topK: int, topP: float32) : int =
            // Apply temperature
            let scaledLogits = logits |> Array.map (fun x -> x / temperature)
            
            // Apply softmax
            let maxLogit = scaledLogits |> Array.max
            let expLogits = scaledLogits |> Array.map (fun x -> exp(x - maxLogit))
            let sumExp = expLogits |> Array.sum
            let probabilities = expLogits |> Array.map (fun x -> x / sumExp)
            
            // Simple sampling - just take the most likely token (would implement proper sampling)
            let maxIndex = probabilities |> Array.mapi (fun i p -> (i, p)) |> Array.maxBy snd |> fst
            maxIndex
        
        /// Update performance metrics
        member _.UpdateMetrics(inferenceTimeMs: float, tokensGenerated: int, cudaUsed: bool, optimizationUsed: bool) =
            let newTotal = metrics.TotalInferences + 1L
            let newTokensTotal = metrics.TotalTokensGenerated + int64 tokensGenerated
            
            // Update running averages
            let newAvgTime = (metrics.AverageInferenceTimeMs * float (newTotal - 1L) + inferenceTimeMs) / float newTotal
            let newAvgTPS = float newTokensTotal / (newAvgTime * float newTotal / 1000.0)
            
            let newCudaRate = 
                if cudaUsed then
                    (metrics.CudaAccelerationRate * float (newTotal - 1L) + 1.0) / float newTotal
                else
                    (metrics.CudaAccelerationRate * float (newTotal - 1L)) / float newTotal
            
            let newOptimRate =
                if optimizationUsed then
                    (metrics.OptimizationSuccessRate * float (newTotal - 1L) + 1.0) / float newTotal
                else
                    (metrics.OptimizationSuccessRate * float (newTotal - 1L)) / float newTotal
            
            metrics <- {
                TotalInferences = newTotal
                AverageInferenceTimeMs = newAvgTime
                AverageTokensPerSecond = newAvgTPS
                TotalTokensGenerated = newTokensTotal
                CudaAccelerationRate = newCudaRate
                OptimizationSuccessRate = newOptimRate
            }
        
        /// Get current performance metrics
        member _.GetMetrics() = metrics
        
        /// Check if CUDA acceleration is available
        member _.IsCudaAvailable() =
            match transformer with
            | Some t -> 
                match t.GetModelInfo() with
                | Some info -> info.CudaAcceleration
                | None -> false
            | None -> false
        
        /// Get total parameter count
        member _.GetParameterCount() =
            match transformer with
            | Some t ->
                match t.GetModelInfo() with
                | Some info -> info.ParameterCount
                | None -> 0L
            | None -> 0L
        
        /// Get engine status
        member _.GetStatus() = {|
            IsInitialized = isInitialized
            ModelLoaded = transformer.IsSome
            TokenizerLoaded = tokenizer.IsSome
            CudaAvailable = this.IsCudaAvailable()
            ParameterCount = this.GetParameterCount()
            Metrics = metrics
            Config = config
        |}
        
        /// Cleanup resources
        member _.Cleanup() = async {
            printfn "🧹 Cleaning up TARS Complete AI Engine..."
            
            match transformer with
            | Some t -> 
                let! _ = t.Cleanup()
                ()
            | None -> ()
            
            isInitialized <- false
            printfn "✅ TARS Complete AI Engine cleanup complete"
            return true
        }
        
        interface IDisposable with
            member this.Dispose() =
                this.Cleanup() |> Async.RunSynchronously |> ignore
