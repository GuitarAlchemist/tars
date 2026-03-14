namespace TarsEngine.FSharp.Cli.Acceleration

open System

/// CUDA Types - Core type definitions for CUDA operations
module CudaTypes =
    
    /// CUDA error codes
    type CudaError =
        | Success = 0
        | InvalidDevice = 1
        | OutOfMemory = 2
        | KernelLaunch = 3
        | InvalidValue = 4
        | NotInitialized = 5
        | RuntimeError = 6
    
    /// CUDA operation types
    type CudaOperationType =
        | VectorSimilarity of dimensions: int
        | MatrixMultiplication of m: int * n: int * k: int
        | TensorOperation of shape: int array
        | ReasoningKernel of complexity: int
        | DataProcessing of size: int
        | CustomKernel of name: string
        // AI-specific operations
        | NeuralNetworkInference of modelName: string * inputShape: int array
        | TransformerAttention of seqLen: int * headDim: int * numHeads: int
        | LayerNormalization of size: int * eps: float
        | ActivationFunction of funcType: string * size: int
        | EmbeddingLookup of vocabSize: int * embedDim: int * seqLen: int
        | ModelTraining of batchSize: int * modelSize: int
        | GradientComputation of paramCount: int
        | OptimizerStep of optimizerType: string * paramCount: int
    
    /// CUDA device information
    type CudaDeviceInfo = {
        DeviceId: int
        Name: string
        TotalMemory: int64
        ComputeCapability: float
        MultiprocessorCount: int
        MaxThreadsPerBlock: int
        IsAvailable: bool
    }
    
    /// CUDA operation context
    type CudaOperationContext = {
        OperationId: string
        OperationType: CudaOperationType
        DeviceId: int
        StreamId: int64
        Priority: int
        EstimatedTime: TimeSpan
        MemoryRequired: int64
        CreatedAt: DateTime
    }
    
    /// CUDA operation result
    type CudaOperationResult = {
        OperationId: string
        Success: bool
        ExecutionTime: TimeSpan
        MemoryUsed: int64
        ThroughputGFlops: float
        ErrorMessage: string option
        ResultData: obj option
    }
    
    /// CUDA performance metrics
    type CudaPerformanceMetrics = {
        TotalOperations: int64
        SuccessfulOperations: int64
        FailedOperations: int64
        AverageExecutionTime: TimeSpan
        TotalGpuTime: TimeSpan
        MemoryUtilization: float
        ThroughputGFlops: float
        LastUpdate: DateTime
    }
