namespace TarsEngine.FSharp.Cli.AI

open System
open TarsEngine.FSharp.Cli.AI.HuggingFaceTypes

/// Hugging Face Inference Types - Specific types for inference operations
module HuggingFaceInferenceTypes =
    
    /// Inference engine state
    type InferenceEngineState = {
        IsInitialized: bool
        CudaDeviceCount: int
        InitializedAt: DateTime option
        LastUsed: DateTime option
        TotalInferences: int64
    }
    
    /// Inference performance metrics
    type InferenceMetrics = {
        InferenceTime: TimeSpan
        TokensProcessed: int
        TokensPerSecond: float
        MemoryUsed: int64
        GpuUtilization: float
    }
    
    /// Inference result with metrics
    type InferenceResult<'T> = {
        Result: 'T
        Metrics: InferenceMetrics
        Success: bool
        ErrorMessage: string option
        CorrelationId: string
    }
    
    /// Text generation result
    type TextGenerationResult = {
        GeneratedText: string
        InputTokens: int
        OutputTokens: int
        FinishReason: string
        Logprobs: float[] option
    }
    
    /// Classification result
    type ClassificationResult = {
        Classifications: (string * float32)[]
        TopClass: string
        Confidence: float32
        AllScores: Map<string, float32>
    }
    
    /// Embeddings result
    type EmbeddingsResult = {
        Embeddings: float32[]
        Dimensions: int
        Magnitude: float32
        Normalized: bool
    }
    
    /// Question answering result
    type QuestionAnsweringResult = {
        Answer: string
        Confidence: float32
        StartIndex: int
        EndIndex: int
        Context: string
    }
    
    /// Inference request context
    type InferenceContext = {
        RequestId: string
        ModelId: string
        StartTime: DateTime
        Parameters: Map<string, obj>
        Metadata: Map<string, string>
    }
    
    /// Engine capabilities
    type EngineCapability = {
        Name: string
        Description: string
        Supported: bool
        RequiresCuda: bool
        EstimatedPerformance: string
    }
    
    /// Create default inference engine state
    let createDefaultEngineState() = {
        IsInitialized = false
        CudaDeviceCount = 0
        InitializedAt = None
        LastUsed = None
        TotalInferences = 0L
    }
    
    /// Create inference metrics
    let createInferenceMetrics (inferenceTime: TimeSpan) (tokensProcessed: int) = {
        InferenceTime = inferenceTime
        TokensProcessed = tokensProcessed
        TokensPerSecond = if inferenceTime.TotalSeconds > 0.0 then float tokensProcessed / inferenceTime.TotalSeconds else 0.0
        MemoryUsed = 0L // Would be calculated from actual usage
        GpuUtilization = 0.0 // Would be measured from CUDA
    }
    
    /// Create inference context
    let createInferenceContext (requestId: string) (modelId: string) = {
        RequestId = requestId
        ModelId = modelId
        StartTime = DateTime.UtcNow
        Parameters = Map.empty
        Metadata = Map.empty
    }
    
    /// Get supported capabilities
    let getSupportedCapabilities() = [
        {
            Name = "Text Generation"
            Description = "Generate human-like text with CUDA acceleration"
            Supported = true
            RequiresCuda = true
            EstimatedPerformance = "~200ms for 100 tokens"
        }
        {
            Name = "Text Classification"
            Description = "Classify text sentiment, topics, and more"
            Supported = true
            RequiresCuda = true
            EstimatedPerformance = "~150ms per classification"
        }
        {
            Name = "Sentence Embeddings"
            Description = "Convert text to high-dimensional vectors"
            Supported = true
            RequiresCuda = true
            EstimatedPerformance = "~100ms for 384-dim embeddings"
        }
        {
            Name = "Question Answering"
            Description = "Extract answers from context with CUDA"
            Supported = true
            RequiresCuda = true
            EstimatedPerformance = "~250ms per question"
        }
        {
            Name = "Summarization"
            Description = "Generate concise summaries of long text"
            Supported = true
            RequiresCuda = true
            EstimatedPerformance = "~500ms for article summary"
        }
        {
            Name = "Translation"
            Description = "Translate between multiple languages"
            Supported = true
            RequiresCuda = true
            EstimatedPerformance = "~300ms per sentence"
        }
        {
            Name = "Token Classification"
            Description = "Named entity recognition and POS tagging"
            Supported = true
            RequiresCuda = true
            EstimatedPerformance = "~200ms per sentence"
        }
        {
            Name = "Zero-shot Classification"
            Description = "Classify without training data"
            Supported = true
            RequiresCuda = true
            EstimatedPerformance = "~180ms per classification"
        }
        {
            Name = "Batch Processing"
            Description = "Process multiple inputs efficiently"
            Supported = true
            RequiresCuda = true
            EstimatedPerformance = "~50% faster than sequential"
        }
        {
            Name = "Custom Models"
            Description = "Load and run your own fine-tuned models"
            Supported = true
            RequiresCuda = false
            EstimatedPerformance = "Varies by model size"
        }
    ]
