namespace TarsEngine.FSharp.Cli.AI

open System
open System.Collections.Generic

/// Hugging Face Integration Types - Types for HF model integration
module HuggingFaceTypes =
    
    /// Hugging Face model information
    type HuggingFaceModelInfo = {
        ModelId: string
        ModelName: string
        Author: string
        Task: string
        Architecture: string
        Framework: string // "pytorch", "onnx", "tensorflow"
        Downloads: int64
        Likes: int
        Tags: string[]
        Description: string
        CreatedAt: DateTime
        UpdatedAt: DateTime
        ModelSize: int64 // Size in bytes
        IsPrivate: bool
        License: string option
    }
    
    /// Model download status
    type ModelDownloadStatus =
        | NotDownloaded
        | Downloading of progress: float
        | Downloaded of localPath: string
        | Failed of error: string
    
    /// Tokenizer types
    type TokenizerType =
        | BertTokenizer
        | GPTTokenizer
        | T5Tokenizer
        | SentencePieceTokenizer
        | CustomTokenizer of name: string
    
    /// Tokenization result
    type TokenizationResult = {
        InputIds: int[]
        AttentionMask: int[]
        TokenTypeIds: int[] option
        SpecialTokensMask: int[] option
        OffsetMapping: (int * int)[] option
        Tokens: string[]
    }
    
    /// NLP task types
    type NLPTask =
        | TextGeneration of maxLength: int * temperature: float * topP: float
        | TextClassification of labels: string[]
        | QuestionAnswering of context: string
        | Summarization of maxLength: int * minLength: int
        | Translation of sourceLanguage: string * targetLanguage: string
        | SentenceEmbeddings
        | TokenClassification of labels: string[]
        | FillMask of maskToken: string
        | FeatureExtraction
        | ZeroShotClassification of candidateLabels: string[]
    
    /// Inference request for Hugging Face models
    type HuggingFaceInferenceRequest = {
        RequestId: string
        ModelId: string
        Task: NLPTask
        InputText: string
        BatchInputs: string[] option
        Parameters: Map<string, obj>
        UseCache: bool
        ReturnTokens: bool
        ReturnAttentions: bool
        ReturnHiddenStates: bool
        CorrelationId: string
    }
    
    /// Inference response from Hugging Face models
    type HuggingFaceInferenceResponse = {
        RequestId: string
        ModelId: string
        Task: string
        GeneratedText: string option
        Classifications: (string * float)[] option
        Embeddings: float32[] option
        Tokens: string[] option
        Attentions: float32[][][] option
        HiddenStates: float32[][][] option
        Answer: string option
        Score: float option
        StartIndex: int option
        EndIndex: int option
        InferenceTime: TimeSpan
        TokensProcessed: int
        Success: bool
        ErrorMessage: string option
        CorrelationId: string
    }
    
    /// Model cache entry
    type ModelCacheEntry = {
        ModelInfo: HuggingFaceModelInfo
        LocalPath: string
        CachedAt: DateTime
        LastAccessed: DateTime
        AccessCount: int64
        FileSize: int64
        IsLoaded: bool
        LoadedAt: DateTime option
    }
    
    /// Hugging Face API configuration
    type HuggingFaceConfig = {
        ApiToken: string option
        CacheDirectory: string
        MaxCacheSize: int64 // Max cache size in bytes
        AutoCleanup: bool
        CleanupThreshold: float // Cleanup when cache exceeds this percentage
        DefaultTimeout: TimeSpan
        MaxConcurrentDownloads: int
        PreferredFormat: string // "onnx", "pytorch", "tensorflow"
        EnableTelemetry: bool
    }
    
    /// Model repository information
    type ModelRepository = {
        Owner: string
        Name: string
        Branch: string
        Revision: string
        Files: string[]
        TotalSize: int64
        LastModified: DateTime
    }
    
    /// Download progress information
    type DownloadProgress = {
        ModelId: string
        FileName: string
        BytesDownloaded: int64
        TotalBytes: int64
        ProgressPercentage: float
        DownloadSpeed: float // Bytes per second
        EstimatedTimeRemaining: TimeSpan option
        StartedAt: DateTime
        Status: string
    }
