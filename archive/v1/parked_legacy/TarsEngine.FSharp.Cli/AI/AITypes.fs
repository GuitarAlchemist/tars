namespace TarsEngine.FSharp.Cli.AI

open System

/// AI Types - Core type definitions for AI inference
module AITypes =
    
    /// TARS tensor data type
    type TarsTensor = {
        Data: float32[]
        Shape: int[]
        Device: string // "cpu" or "cuda"
        DevicePtr: nativeint option
        RequiresGrad: bool
        GradientData: float32[] option
    }
    
    /// Neural network layer types
    type LayerType =
        | Linear of inputSize: int * outputSize: int
        | Embedding of vocabSize: int * embedDim: int
        | LayerNorm of size: int * eps: float
        | MultiHeadAttention of numHeads: int * headDim: int * seqLen: int
        | FeedForward of hiddenSize: int * intermediateSize: int
        | Activation of funcType: string // "relu", "gelu", "softmax"
        | Dropout of rate: float
        | Custom of name: string * parameters: Map<string, obj>
    
    /// Neural network layer definition
    type NeuralLayer = {
        LayerId: string
        LayerType: LayerType
        Weights: TarsTensor option
        Bias: TarsTensor option
        Parameters: Map<string, TarsTensor>
        IsTrainable: bool
        DeviceId: int
    }
    
    /// Neural network model architecture
    type TarsModel = {
        ModelId: string
        ModelName: string
        Architecture: string // "transformer", "cnn", "rnn", "custom"
        Layers: NeuralLayer[]
        ModelSize: int64 // Total parameters
        MemoryRequirement: int64 // GPU memory needed
        MaxSequenceLength: int
        VocabularySize: int
        HiddenSize: int
        NumLayers: int
        NumAttentionHeads: int
        IntermediateSize: int
        IsLoaded: bool
        DeviceId: int
        CreatedAt: DateTime
        LastUsed: DateTime
    }
    
    /// Inference request
    type InferenceRequest = {
        RequestId: string
        ModelId: string
        InputTensors: TarsTensor[]
        MaxOutputLength: int option
        Temperature: float option
        TopP: float option
        TopK: int option
        DoSample: bool
        ReturnAttentions: bool
        ReturnHiddenStates: bool
        CorrelationId: string
    }
    
    /// Inference response
    type InferenceResponse = {
        RequestId: string
        ModelId: string
        OutputTensors: TarsTensor[]
        Logits: TarsTensor option
        Attentions: TarsTensor[] option
        HiddenStates: TarsTensor[] option
        InferenceTime: TimeSpan
        TokensGenerated: int
        TokensPerSecond: float
        MemoryUsed: int64
        GpuUtilization: float
        Success: bool
        ErrorMessage: string option
        CorrelationId: string
    }
    
    /// Training configuration
    type TrainingConfig = {
        LearningRate: float
        BatchSize: int
        MaxEpochs: int
        OptimizerType: string // "adam", "sgd", "adamw"
        LossFunction: string // "cross_entropy", "mse", "huber"
        GradientClipping: float option
        WeightDecay: float
        WarmupSteps: int
        SaveCheckpoints: bool
        CheckpointInterval: int
        ValidationSplit: float
        EarlyStopping: bool
        Patience: int
    }
    
    /// Model performance metrics
    type ModelMetrics = {
        ModelId: string
        TotalInferences: int64
        SuccessfulInferences: int64
        FailedInferences: int64
        AverageInferenceTime: TimeSpan
        AverageTokensPerSecond: float
        PeakMemoryUsage: int64
        AverageMemoryUsage: int64
        GpuUtilization: float
        ThroughputOpsPerSecond: float
        AccuracyMetrics: Map<string, float>
        LastUpdate: DateTime
    }
