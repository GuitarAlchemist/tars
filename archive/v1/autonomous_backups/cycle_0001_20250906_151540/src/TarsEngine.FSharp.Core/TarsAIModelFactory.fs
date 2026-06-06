namespace TarsEngine

open TarsEngine.TarsAIInferenceEngine

/// Factory for creating realistic AI model configurations for TARS Hyperlight inference
module TarsAIModelFactory =
    
    /// Factory for creating AI model configurations
    type TarsAIModelFactory() =
        
        /// Create a small text generation model (GPT-2 style)
        static member CreateSmallTextModel() = {
            ModelId = "tars-gpt2-small"
            ModelName = "TARS GPT-2 Small"
            ModelType = TextGeneration("small", 124_000_000L) // 124M parameters
            ModelPath = "/models/tars-gpt2-small.onnx"
            WasmBinary = "tars_gpt2_small.wasm"
            InputFormat = "text"
            OutputFormat = "text"
            MaxBatchSize = 4
            MaxSequenceLength = 1024
            MemoryRequirementMB = 512  // Realistic for 124M model
            ExpectedLatencyMs = 80.0   // Realistic for small model
            ThroughputRPS = 25.0       // Conservative estimate
        }
        
        /// Create a medium text generation model (GPT-2 medium style)
        static member CreateMediumTextModel() = {
            ModelId = "tars-gpt2-medium"
            ModelName = "TARS GPT-2 Medium"
            ModelType = TextGeneration("medium", 355_000_000L) // 355M parameters
            ModelPath = "/models/tars-gpt2-medium.onnx"
            WasmBinary = "tars_gpt2_medium.wasm"
            InputFormat = "text"
            OutputFormat = "text"
            MaxBatchSize = 2
            MaxSequenceLength = 1024
            MemoryRequirementMB = 1024 // Realistic for 355M model
            ExpectedLatencyMs = 150.0  // Realistic for medium model
            ThroughputRPS = 12.0       // Conservative estimate
        }
        
        /// Create a text embedding model (Sentence-BERT style)
        static member CreateTextEmbeddingModel() = {
            ModelId = "tars-sentence-bert"
            ModelName = "TARS Sentence BERT"
            ModelType = TextEmbedding(384) // 384-dimensional embeddings
            ModelPath = "/models/tars-sentence-bert.onnx"
            WasmBinary = "tars_sentence_bert.wasm"
            InputFormat = "text"
            OutputFormat = "vector"
            MaxBatchSize = 16
            MaxSequenceLength = 512
            MemoryRequirementMB = 256  // Smaller than generative models
            ExpectedLatencyMs = 25.0   // Fast embedding generation
            ThroughputRPS = 80.0       // High throughput for embeddings
        }
        
        /// Create a sentiment analysis model
        static member CreateSentimentModel() = {
            ModelId = "tars-sentiment-analysis"
            ModelName = "TARS Sentiment Analyzer"
            ModelType = SentimentAnalysis(50000) // 50K vocabulary
            ModelPath = "/models/tars-sentiment.onnx"
            WasmBinary = "tars_sentiment.wasm"
            InputFormat = "text"
            OutputFormat = "classification"
            MaxBatchSize = 32
            MaxSequenceLength = 256
            MemoryRequirementMB = 128  // Small classification model
            ExpectedLatencyMs = 15.0   // Very fast classification
            ThroughputRPS = 150.0      // High throughput
        }
        
        /// Create an image classification model (ResNet style)
        static member CreateImageClassificationModel() = {
            ModelId = "tars-resnet50"
            ModelName = "TARS ResNet-50"
            ModelType = ImageClassification(224, 1000) // 224x224 input, 1000 classes
            ModelPath = "/models/tars-resnet50.onnx"
            WasmBinary = "tars_resnet50.wasm"
            InputFormat = "image"
            OutputFormat = "classification"
            MaxBatchSize = 8
            MaxSequenceLength = 1 // Not applicable for images
            MemoryRequirementMB = 384  // ResNet-50 size
            ExpectedLatencyMs = 120.0  // Image processing latency
            ThroughputRPS = 20.0       // Moderate throughput
        }
        
        /// Create a code generation model (CodeT5 style)
        static member CreateCodeGenerationModel() = {
            ModelId = "tars-codet5-small"
            ModelName = "TARS CodeT5 Small"
            ModelType = CodeGeneration("python", 2048) // Python code, 2K context
            ModelPath = "/models/tars-codet5-small.onnx"
            WasmBinary = "tars_codet5_small.wasm"
            InputFormat = "text"
            OutputFormat = "code"
            MaxBatchSize = 2
            MaxSequenceLength = 2048
            MemoryRequirementMB = 768  // Code models are larger
            ExpectedLatencyMs = 200.0  // Code generation is slower
            ThroughputRPS = 8.0        // Lower throughput for code
        }
        
        /// Create a TARS autonomous reasoning model
        static member CreateTarsReasoningModel() = {
            ModelId = "tars-reasoning-v1"
            ModelName = "TARS Autonomous Reasoning Model"
            ModelType = ReasoningModel("autonomous_decision_making")
            ModelPath = "/models/tars-reasoning-v1.onnx"
            WasmBinary = "tars_reasoning_v1.wasm"
            InputFormat = "structured"
            OutputFormat = "decision"
            MaxBatchSize = 1 // Reasoning is typically sequential
            MaxSequenceLength = 4096
            MemoryRequirementMB = 1536 // Large reasoning model
            ExpectedLatencyMs = 300.0  // Complex reasoning takes time
            ThroughputRPS = 5.0        // Low throughput for complex reasoning
        }
        
        /// Create a lightweight edge model for IoT devices
        static member CreateEdgeModel() = {
            ModelId = "tars-edge-tiny"
            ModelName = "TARS Edge Tiny Model"
            ModelType = TextGeneration("tiny", 10_000_000L) // 10M parameters
            ModelPath = "/models/tars-edge-tiny.onnx"
            WasmBinary = "tars_edge_tiny.wasm"
            InputFormat = "text"
            OutputFormat = "text"
            MaxBatchSize = 1
            MaxSequenceLength = 256
            MemoryRequirementMB = 64   // Very small for edge devices
            ExpectedLatencyMs = 30.0   // Fast for edge
            ThroughputRPS = 40.0       // Good throughput for tiny model
        }
        
        /// Create a multimodal model (text + image)
        static member CreateMultimodalModel() = {
            ModelId = "tars-multimodal-v1"
            ModelName = "TARS Multimodal Vision-Language"
            ModelType = TextGeneration("multimodal", 400_000_000L) // 400M parameters
            ModelPath = "/models/tars-multimodal-v1.onnx"
            WasmBinary = "tars_multimodal_v1.wasm"
            InputFormat = "multimodal"
            OutputFormat = "text"
            MaxBatchSize = 1
            MaxSequenceLength = 1024
            MemoryRequirementMB = 1280 // Large multimodal model
            ExpectedLatencyMs = 400.0  // Complex multimodal processing
            ThroughputRPS = 3.0        // Low throughput for complexity
        }
        
        /// Get all available model configurations
        static member GetAllModels() = [
            TarsAIModelFactory.CreateSmallTextModel()
            TarsAIModelFactory.CreateMediumTextModel()
            TarsAIModelFactory.CreateTextEmbeddingModel()
            TarsAIModelFactory.CreateSentimentModel()
            TarsAIModelFactory.CreateImageClassificationModel()
            TarsAIModelFactory.CreateCodeGenerationModel()
            TarsAIModelFactory.CreateTarsReasoningModel()
            TarsAIModelFactory.CreateEdgeModel()
            TarsAIModelFactory.CreateMultimodalModel()
        ]
        
        /// Get models suitable for edge deployment
        static member GetEdgeModels() = [
            TarsAIModelFactory.CreateEdgeModel()
            TarsAIModelFactory.CreateTextEmbeddingModel()
            TarsAIModelFactory.CreateSentimentModel()
        ]
        
        /// Get models suitable for server deployment
        static member GetServerModels() = [
            TarsAIModelFactory.CreateSmallTextModel()
            TarsAIModelFactory.CreateMediumTextModel()
            TarsAIModelFactory.CreateImageClassificationModel()
            TarsAIModelFactory.CreateCodeGenerationModel()
            TarsAIModelFactory.CreateTarsReasoningModel()
            TarsAIModelFactory.CreateMultimodalModel()
        ]
        
        /// Get models by memory requirement (for resource-constrained environments)
        static member GetModelsByMemoryLimit(maxMemoryMB: int) =
            TarsAIModelFactory.GetAllModels()
            |> List.filter (fun model -> model.MemoryRequirementMB <= maxMemoryMB)
            |> List.sortBy (fun model -> model.MemoryRequirementMB)
        
        /// Get models by latency requirement (for real-time applications)
        static member GetModelsByLatencyLimit(maxLatencyMs: float) =
            TarsAIModelFactory.GetAllModels()
            |> List.filter (fun model -> model.ExpectedLatencyMs <= maxLatencyMs)
            |> List.sortBy (fun model -> model.ExpectedLatencyMs)
        
        /// Get models by throughput requirement (for high-volume applications)
        static member GetModelsByThroughputRequirement(minThroughputRPS: float) =
            TarsAIModelFactory.GetAllModels()
            |> List.filter (fun model -> model.ThroughputRPS >= minThroughputRPS)
            |> List.sortByDescending (fun model -> model.ThroughputRPS)
    
    /// Model deployment recommendations based on use case
    type ModelDeploymentRecommendations() =
        
        /// Recommend models for real-time chat applications
        static member ForRealTimeChat() = [
            TarsAIModelFactory.CreateSmallTextModel()  // Good balance of quality and speed
            TarsAIModelFactory.CreateEdgeModel()       // For ultra-low latency
        ]
        
        /// Recommend models for batch processing
        static member ForBatchProcessing() = [
            TarsAIModelFactory.CreateMediumTextModel()     // Higher quality for batch
            TarsAIModelFactory.CreateMultimodalModel()     // Complex processing acceptable
        ]
        
        /// Recommend models for edge IoT devices
        static member ForEdgeIoT() = [
            TarsAIModelFactory.CreateEdgeModel()           // Minimal resources
            TarsAIModelFactory.CreateSentimentModel()      // Fast classification
        ]
        
        /// Recommend models for enterprise applications
        static member ForEnterprise() = [
            TarsAIModelFactory.CreateTarsReasoningModel()  // Business logic
            TarsAIModelFactory.CreateCodeGenerationModel() // Development assistance
            TarsAIModelFactory.CreateTextEmbeddingModel()  // Search and similarity
        ]
        
        /// Recommend models for research and development
        static member ForResearch() = [
            TarsAIModelFactory.CreateMultimodalModel()     // Cutting-edge capabilities
            TarsAIModelFactory.CreateTarsReasoningModel()  // Advanced reasoning
            TarsAIModelFactory.CreateMediumTextModel()     // Good baseline
        ]
