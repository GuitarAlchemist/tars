namespace TarsEngine.FSharp.Cli.Acceleration

open System
open TarsEngine.FSharp.Cli.Acceleration.CudaTypes

/// CUDA Operation Factory - Creates CUDA operation contexts
module CudaOperationFactory =
    
    /// Create vector similarity operation
    let createVectorSimilarity (dimensions: int) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = VectorSimilarity dimensions
            DeviceId = 0
            StreamId = 0L
            Priority = 1
            EstimatedTime = TimeSpan.FromMilliseconds(float dimensions / 1000.0)
            MemoryRequired = int64 dimensions * 4L * 2L // Two vectors
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create matrix multiplication operation
    let createMatrixMultiplication (m: int) (n: int) (k: int) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = MatrixMultiplication (m, n, k)
            DeviceId = 0
            StreamId = 0L
            Priority = 2
            EstimatedTime = TimeSpan.FromMilliseconds(float (m * n * k) / 100000.0)
            MemoryRequired = int64 (m * n + n * k + m * k) * 4L
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create tensor operation
    let createTensorOperation (shape: int array) : CudaOperationContext =
        let totalElements = Array.fold (*) 1 shape
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = TensorOperation shape
            DeviceId = 0
            StreamId = 0L
            Priority = 1
            EstimatedTime = TimeSpan.FromMilliseconds(float totalElements / 10000.0)
            MemoryRequired = int64 totalElements * 4L
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create reasoning kernel operation
    let createReasoningKernel (complexity: int) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = ReasoningKernel complexity
            DeviceId = 0
            StreamId = 0L
            Priority = 3
            EstimatedTime = TimeSpan.FromMilliseconds(float complexity * 0.1)
            MemoryRequired = int64 complexity * 1024L
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create data processing operation
    let createDataProcessing (size: int) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = DataProcessing size
            DeviceId = 0
            StreamId = 0L
            Priority = 1
            EstimatedTime = TimeSpan.FromMilliseconds(float size / 1000.0)
            MemoryRequired = int64 size * 4L
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create custom kernel operation
    let createCustomKernel (name: string) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = CustomKernel name
            DeviceId = 0
            StreamId = 0L
            Priority = 2
            EstimatedTime = TimeSpan.FromMilliseconds(10.0)
            MemoryRequired = 1024L * 1024L // 1MB default
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create neural network inference operation
    let createNeuralNetworkInference (modelName: string) (inputShape: int array) : CudaOperationContext =
        let totalElements = Array.fold (*) 1 inputShape
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = NeuralNetworkInference (modelName, inputShape)
            DeviceId = 0
            StreamId = 0L
            Priority = 3
            EstimatedTime = TimeSpan.FromMilliseconds(float totalElements / 1000.0)
            MemoryRequired = int64 totalElements * 4L * 2L // Input + output
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create transformer attention operation
    let createTransformerAttention (seqLen: int) (headDim: int) (numHeads: int) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = TransformerAttention (seqLen, headDim, numHeads)
            DeviceId = 0
            StreamId = 0L
            Priority = 3
            EstimatedTime = TimeSpan.FromMilliseconds(float (seqLen * seqLen * numHeads) / 10000.0)
            MemoryRequired = int64 (seqLen * seqLen * numHeads * headDim) * 4L
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create layer normalization operation
    let createLayerNormalization (size: int) (eps: float) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = LayerNormalization (size, eps)
            DeviceId = 0
            StreamId = 0L
            Priority = 1
            EstimatedTime = TimeSpan.FromMilliseconds(float size / 10000.0)
            MemoryRequired = int64 size * 4L * 2L // Input + output
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create activation function operation
    let createActivationFunction (funcType: string) (size: int) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = ActivationFunction (funcType, size)
            DeviceId = 0
            StreamId = 0L
            Priority = 1
            EstimatedTime = TimeSpan.FromMilliseconds(float size / 50000.0)
            MemoryRequired = int64 size * 4L * 2L // Input + output
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create embedding lookup operation
    let createEmbeddingLookup (vocabSize: int) (embedDim: int) (seqLen: int) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = EmbeddingLookup (vocabSize, embedDim, seqLen)
            DeviceId = 0
            StreamId = 0L
            Priority = 2
            EstimatedTime = TimeSpan.FromMilliseconds(float (seqLen * embedDim) / 10000.0)
            MemoryRequired = int64 (vocabSize * embedDim + seqLen * embedDim) * 4L
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create model training operation
    let createModelTraining (batchSize: int) (modelSize: int) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = ModelTraining (batchSize, modelSize)
            DeviceId = 0
            StreamId = 0L
            Priority = 4
            EstimatedTime = TimeSpan.FromMilliseconds(float (batchSize * modelSize) / 1000.0)
            MemoryRequired = int64 (batchSize * modelSize) * 4L * 3L // Forward + backward + gradients
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create gradient computation operation
    let createGradientComputation (paramCount: int) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = GradientComputation paramCount
            DeviceId = 0
            StreamId = 0L
            Priority = 3
            EstimatedTime = TimeSpan.FromMilliseconds(float paramCount / 10000.0)
            MemoryRequired = int64 paramCount * 4L * 2L // Parameters + gradients
            CreatedAt = DateTime.UtcNow
        }
    
    /// Create optimizer step operation
    let createOptimizerStep (optimizerType: string) (paramCount: int) : CudaOperationContext =
        {
            OperationId = Guid.NewGuid().ToString("N").[..15]
            OperationType = OptimizerStep (optimizerType, paramCount)
            DeviceId = 0
            StreamId = 0L
            Priority = 2
            EstimatedTime = TimeSpan.FromMilliseconds(float paramCount / 20000.0)
            MemoryRequired = int64 paramCount * 4L * 3L // Parameters + gradients + momentum
            CreatedAt = DateTime.UtcNow
        }
