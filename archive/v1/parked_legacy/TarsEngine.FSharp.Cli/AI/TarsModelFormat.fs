namespace TarsEngine.FSharp.Cli.AI

open System
open System.IO
open System.Text.Json
open System.IO.Compression
open System.Threading
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
// Forward reference - types will be defined in this module

/// TARS Model Format - Custom serialization format for TARS neural networks
module TarsModelFormat =

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
    
    /// Model metadata structure
    type ModelMetadata = {
        ModelId: string
        ModelName: string
        Architecture: string
        Version: string
        CreatedAt: DateTime
        CreatedBy: string
        Description: string
        Tags: string[]
        ModelSize: int64
        MemoryRequirement: int64
        MaxSequenceLength: int
        VocabularySize: int
        HiddenSize: int
        NumLayers: int
        NumAttentionHeads: int
        IntermediateSize: int
        ActivationFunction: string
        NormalizationType: string
        PositionalEncoding: string
        Quantization: string option
        CompressionRatio: float option
        TrainingConfig: TrainingConfigMetadata option
        PerformanceBenchmarks: PerformanceBenchmark[]
        HardwareRequirements: HardwareRequirement[]
        LicenseInfo: string option
        ChecksumSHA256: string
    }
    
    /// Training configuration metadata
    and TrainingConfigMetadata = {
        DatasetName: string
        DatasetSize: int64
        TrainingSteps: int64
        LearningRate: float
        BatchSize: int
        OptimizerType: string
        LossFunction: string
        ValidationAccuracy: float option
        TrainingTime: TimeSpan
        HyperParameters: Map<string, obj>
    }
    
    /// Performance benchmark data
    and PerformanceBenchmark = {
        BenchmarkName: string
        Hardware: string
        InferenceTimeMs: float
        TokensPerSecond: float
        MemoryUsageMB: int64
        GpuUtilization: float
        ThroughputOpsPerSec: float
        AccuracyScore: float option
        BenchmarkDate: DateTime
    }
    
    /// Hardware requirement specification
    and HardwareRequirement = {
        RequirementType: string // "minimum", "recommended", "optimal"
        GpuMemoryMB: int64
        SystemMemoryMB: int64
        ComputeCapability: float option
        CudaCores: int option
        TensorCores: bool
        StorageSpaceMB: int64
        NetworkBandwidthMbps: int option
    }
    
    /// Layer weight data structure
    type LayerWeightData = {
        LayerId: string
        LayerType: string
        WeightShape: int[]
        WeightData: byte[] // Compressed weight data
        BiasShape: int[] option
        BiasData: byte[] option
        ParameterData: Map<string, byte[]>
        Quantization: string option
        CompressionType: string
        OriginalSize: int64
        CompressedSize: int64
    }
    
    /// Complete TARS model file structure
    type TarsModelFile = {
        Metadata: ModelMetadata
        Layers: LayerWeightData[]
        ConfigData: byte[]
        CustomData: Map<string, byte[]>
        FileVersion: string
        CreatedAt: DateTime
        TotalSizeBytes: int64
    }
    
    /// TARS model serializer/deserializer
    type TarsModelSerializer(logger: ITarsLogger) =
        
        let fileVersion = "1.0.0"
        let magicBytes = [| 0x54uy; 0x41uy; 0x52uy; 0x53uy |] // "TARS"
        
        /// Serialize TARS model to file
        member this.SerializeModelAsync(model: TarsModel, outputPath: string, cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    logger.LogInformation(correlationId, $"💾 Serializing TARS model to: {outputPath}")
                    
                    // Create model metadata
                    let metadata = {
                        ModelId = model.ModelId
                        ModelName = model.ModelName
                        Architecture = model.Architecture
                        Version = "1.0.0"
                        CreatedAt = DateTime.UtcNow
                        CreatedBy = "TARS AI Engine"
                        Description = $"TARS {model.Architecture} model with {model.ModelSize:N0} parameters"
                        Tags = [| model.Architecture; "tars"; "neural-network" |]
                        ModelSize = model.ModelSize
                        MemoryRequirement = model.MemoryRequirement
                        MaxSequenceLength = model.MaxSequenceLength
                        VocabularySize = model.VocabularySize
                        HiddenSize = model.HiddenSize
                        NumLayers = model.NumLayers
                        NumAttentionHeads = model.NumAttentionHeads
                        IntermediateSize = model.IntermediateSize
                        ActivationFunction = "gelu"
                        NormalizationType = "layer_norm"
                        PositionalEncoding = "learned"
                        Quantization = None
                        CompressionRatio = None
                        TrainingConfig = None
                        PerformanceBenchmarks = [||]
                        HardwareRequirements = [|
                            {
                                RequirementType = "minimum"
                                GpuMemoryMB = model.MemoryRequirement / 1024L / 1024L
                                SystemMemoryMB = 8192L
                                ComputeCapability = Some 6.0
                                CudaCores = Some 1024
                                TensorCores = false
                                StorageSpaceMB = model.MemoryRequirement / 1024L / 1024L * 2L
                                NetworkBandwidthMbps = None
                            }
                        |]
                        LicenseInfo = Some "TARS Open Source License"
                        ChecksumSHA256 = ""
                    }
                    
                    // Serialize layer weights
                    let layerData = 
                        model.Layers
                        |> Array.map (fun layer -> this.SerializeLayerWeights(layer, correlationId))
                    
                    // Create model file structure
                    let modelFile = {
                        Metadata = metadata
                        Layers = layerData
                        ConfigData = [||]
                        CustomData = Map.empty
                        FileVersion = fileVersion
                        CreatedAt = DateTime.UtcNow
                        TotalSizeBytes = 0L
                    }
                    
                    // Write to compressed file
                    let! writeResult = this.WriteModelFile(modelFile, outputPath, correlationId)
                    
                    match writeResult with
                    | Success (fileSize, _) ->
                        logger.LogInformation(correlationId, $"✅ Model serialized successfully: {fileSize:N0} bytes")
                        return Success (fileSize, Map [("outputPath", box outputPath); ("fileSize", box fileSize)])
                    
                    | Failure (error, _) ->
                        return Failure (error, correlationId)
                
                with
                | ex ->
                    let error = ExecutionError ($"Model serialization failed: {ex.Message}", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Deserialize TARS model from file
        member this.DeserializeModelAsync(inputPath: string, cancellationToken: CancellationToken) =
            task {
                try
                    let correlationId = generateCorrelationId()
                    logger.LogInformation(correlationId, $"📖 Deserializing TARS model from: {inputPath}")
                    
                    if not (File.Exists(inputPath)) then
                        let error = ValidationError ($"Model file not found: {inputPath}", "inputPath")
                        return Failure (error, correlationId)
                    else
                        // Read and parse model file
                        let! readResult = this.ReadModelFile(inputPath, correlationId)
                        
                        match readResult with
                        | Success (modelFile, _) ->
                            // Convert to TarsModel
                            let! modelResult = this.ConvertToTarsModel(modelFile, correlationId)
                            
                            match modelResult with
                            | Success (model, metadata) ->
                                logger.LogInformation(correlationId, $"✅ Model deserialized: {model.ModelName} ({model.ModelSize:N0} parameters)")
                                return Success (model, metadata)
                            
                            | Failure (error, _) ->
                                return Failure (error, correlationId)
                        
                        | Failure (error, _) ->
                            return Failure (error, correlationId)
                
                with
                | ex ->
                    let error = ExecutionError ($"Model deserialization failed: {ex.Message}", Some ex)
                    logger.LogError(generateCorrelationId(), error, ex)
                    return Failure (error, generateCorrelationId())
            }
        
        /// Serialize layer weights with compression
        member private this.SerializeLayerWeights(layer: NeuralLayer, correlationId: string) : LayerWeightData =
            try
                let weightData = 
                    match layer.Weights with
                    | Some weights -> this.CompressFloatArray(weights.Data)
                    | None -> [||]
                
                let biasData = 
                    match layer.Bias with
                    | Some bias -> Some (this.CompressFloatArray(bias.Data))
                    | None -> None
                
                let weightShape = 
                    match layer.Weights with
                    | Some weights -> weights.Shape
                    | None -> [||]
                
                let biasShape = 
                    match layer.Bias with
                    | Some bias -> Some bias.Shape
                    | None -> None
                
                {
                    LayerId = layer.LayerId
                    LayerType = this.GetLayerTypeName(layer.LayerType)
                    WeightShape = weightShape
                    WeightData = weightData
                    BiasShape = biasShape
                    BiasData = biasData
                    ParameterData = Map.empty
                    Quantization = None
                    CompressionType = "gzip"
                    OriginalSize = int64 (weightData.Length + (biasData |> Option.map Array.length |> Option.defaultValue 0))
                    CompressedSize = int64 weightData.Length
                }
            
            with
            | ex ->
                logger.LogError(correlationId, TarsError.create "LayerSerializationError" $"Failed to serialize layer {layer.LayerId}" (Some ex), ex)
                {
                    LayerId = layer.LayerId
                    LayerType = "unknown"
                    WeightShape = [||]
                    WeightData = [||]
                    BiasShape = None
                    BiasData = None
                    ParameterData = Map.empty
                    Quantization = None
                    CompressionType = "none"
                    OriginalSize = 0L
                    CompressedSize = 0L
                }
        
        /// Get layer type name for serialization
        member private this.GetLayerTypeName(layerType: LayerType) : string =
            match layerType with
            | Linear (_, _) -> "linear"
            | Embedding (_, _) -> "embedding"
            | LayerNorm (_, _) -> "layer_norm"
            | MultiHeadAttention (_, _, _) -> "multi_head_attention"
            | FeedForward (_, _) -> "feed_forward"
            | Activation funcType -> $"activation_{funcType}"
            | Dropout _ -> "dropout"
            | Custom (name, _) -> $"custom_{name}"
        
        /// Compress float array using gzip
        member private this.CompressFloatArray(data: float32[]) : byte[] =
            try
                use memoryStream = new MemoryStream()
                use gzipStream = new GZipStream(memoryStream, CompressionMode.Compress)
                use binaryWriter = new BinaryWriter(gzipStream)

                for value in data do
                    binaryWriter.Write(value)

                gzipStream.Close()
                memoryStream.ToArray()

            with
            | _ -> [||]

        /// Decompress float array from gzip
        member private this.DecompressFloatArray(compressedData: byte[], expectedLength: int) : float32[] =
            try
                use memoryStream = new MemoryStream(compressedData)
                use gzipStream = new GZipStream(memoryStream, CompressionMode.Decompress)
                use binaryReader = new BinaryReader(gzipStream)

                Array.init expectedLength (fun _ -> binaryReader.ReadSingle())

            with
            | _ -> Array.create expectedLength 0.0f

        /// Write model file to disk
        member private this.WriteModelFile(modelFile: TarsModelFile, outputPath: string, correlationId: string) =
            task {
                try
                    use fileStream = new FileStream(outputPath, FileMode.Create, FileAccess.Write)
                    use binaryWriter = new BinaryWriter(fileStream)

                    // Write magic bytes
                    binaryWriter.Write(magicBytes)

                    // Write file version
                    binaryWriter.Write(fileVersion)

                    // Serialize and write metadata
                    let metadataJson = JsonSerializer.Serialize(modelFile.Metadata, JsonSerializerOptions(WriteIndented = false))
                    let metadataBytes = System.Text.Encoding.UTF8.GetBytes(metadataJson)
                    binaryWriter.Write(metadataBytes.Length)
                    binaryWriter.Write(metadataBytes)

                    // Write layer count
                    binaryWriter.Write(modelFile.Layers.Length)

                    // Write each layer
                    for layer in modelFile.Layers do
                        let layerJson = JsonSerializer.Serialize(layer, JsonSerializerOptions(WriteIndented = false))
                        let layerBytes = System.Text.Encoding.UTF8.GetBytes(layerJson)
                        binaryWriter.Write(layerBytes.Length)
                        binaryWriter.Write(layerBytes)

                    // Write config data
                    binaryWriter.Write(modelFile.ConfigData.Length)
                    binaryWriter.Write(modelFile.ConfigData)

                    // Write custom data count
                    binaryWriter.Write(modelFile.CustomData.Count)
                    for kvp in modelFile.CustomData do
                        let keyBytes = System.Text.Encoding.UTF8.GetBytes(kvp.Key)
                        binaryWriter.Write(keyBytes.Length)
                        binaryWriter.Write(keyBytes)
                        binaryWriter.Write(kvp.Value.Length)
                        binaryWriter.Write(kvp.Value)

                    let fileSize = fileStream.Length
                    logger.LogInformation(correlationId, $"📝 Model file written: {fileSize:N0} bytes")

                    return Success (fileSize, Map [("fileSize", box fileSize)])

                with
                | ex ->
                    let error = ExecutionError ($"Failed to write model file: {ex.Message}", Some ex)
                    return Failure (error, correlationId)
            }

        /// Read model file from disk
        member private this.ReadModelFile(inputPath: string, correlationId: string) =
            task {
                try
                    use fileStream = new FileStream(inputPath, FileMode.Open, FileAccess.Read)
                    use binaryReader = new BinaryReader(fileStream)

                    // Verify magic bytes
                    let readMagicBytes = binaryReader.ReadBytes(4)
                    if readMagicBytes <> magicBytes then
                        let error = ValidationError ("Invalid TARS model file format", "fileFormat")
                        return Failure (error, correlationId)

                    // Read file version
                    let readVersion = binaryReader.ReadString()
                    if readVersion <> fileVersion then
                        logger.LogWarning(correlationId, $"Version mismatch: expected {fileVersion}, got {readVersion}")

                    // Read metadata
                    let metadataLength = binaryReader.ReadInt32()
                    let metadataBytes = binaryReader.ReadBytes(metadataLength)
                    let metadataJson = System.Text.Encoding.UTF8.GetString(metadataBytes)
                    let metadata = JsonSerializer.Deserialize<ModelMetadata>(metadataJson)

                    // Read layers
                    let layerCount = binaryReader.ReadInt32()
                    let layers = Array.create layerCount Unchecked.defaultof<LayerWeightData>

                    for i in 0 .. layerCount - 1 do
                        let layerLength = binaryReader.ReadInt32()
                        let layerBytes = binaryReader.ReadBytes(layerLength)
                        let layerJson = System.Text.Encoding.UTF8.GetString(layerBytes)
                        layers.[i] <- JsonSerializer.Deserialize<LayerWeightData>(layerJson)

                    // Read config data
                    let configLength = binaryReader.ReadInt32()
                    let configData = binaryReader.ReadBytes(configLength)

                    // Read custom data
                    let customDataCount = binaryReader.ReadInt32()
                    let mutable customData = Map.empty

                    for _ in 0 .. customDataCount - 1 do
                        let keyLength = binaryReader.ReadInt32()
                        let keyBytes = binaryReader.ReadBytes(keyLength)
                        let key = System.Text.Encoding.UTF8.GetString(keyBytes)
                        let valueLength = binaryReader.ReadInt32()
                        let value = binaryReader.ReadBytes(valueLength)
                        customData <- customData.Add(key, value)

                    let modelFile = {
                        Metadata = metadata
                        Layers = layers
                        ConfigData = configData
                        CustomData = customData
                        FileVersion = readVersion
                        CreatedAt = DateTime.UtcNow
                        TotalSizeBytes = fileStream.Length
                    }

                    logger.LogInformation(correlationId, $"📖 Model file read: {fileStream.Length:N0} bytes")
                    return Success (modelFile, Map [("fileSize", box fileStream.Length)])

                with
                | ex ->
                    let error = ExecutionError ($"Failed to read model file: {ex.Message}", Some ex)
                    return Failure (error, correlationId)
            }

        /// Convert TarsModelFile to TarsModel
        member private this.ConvertToTarsModel(modelFile: TarsModelFile, correlationId: string) =
            task {
                try
                    logger.LogInformation(correlationId, $"🔄 Converting model file to TarsModel: {modelFile.Metadata.ModelName}")

                    // Convert layers
                    let layers =
                        modelFile.Layers
                        |> Array.map (fun layerData -> this.ConvertToNeuralLayer(layerData, correlationId))

                    let model = {
                        ModelId = modelFile.Metadata.ModelId
                        ModelName = modelFile.Metadata.ModelName
                        Architecture = modelFile.Metadata.Architecture
                        Layers = layers
                        ModelSize = modelFile.Metadata.ModelSize
                        MemoryRequirement = modelFile.Metadata.MemoryRequirement
                        MaxSequenceLength = modelFile.Metadata.MaxSequenceLength
                        VocabularySize = modelFile.Metadata.VocabularySize
                        HiddenSize = modelFile.Metadata.HiddenSize
                        NumLayers = modelFile.Metadata.NumLayers
                        NumAttentionHeads = modelFile.Metadata.NumAttentionHeads
                        IntermediateSize = modelFile.Metadata.IntermediateSize
                        IsLoaded = true
                        DeviceId = 0
                        CreatedAt = modelFile.Metadata.CreatedAt
                        LastUsed = DateTime.UtcNow
                    }

                    return Success (model, Map [("layerCount", box layers.Length)])

                with
                | ex ->
                    let error = ExecutionError ($"Failed to convert model file: {ex.Message}", Some ex)
                    return Failure (error, correlationId)
            }

        /// Convert LayerWeightData to NeuralLayer
        member private this.ConvertToNeuralLayer(layerData: LayerWeightData, correlationId: string) : NeuralLayer =
            try
                let layerType = this.ParseLayerType(layerData.LayerType, layerData.WeightShape)

                let weights =
                    if layerData.WeightData.Length > 0 then
                        let expectedLength = Array.fold (*) 1 layerData.WeightShape
                        let weightArray = this.DecompressFloatArray(layerData.WeightData, expectedLength)
                        Some {
                            Data = weightArray
                            Shape = layerData.WeightShape
                            Device = "cuda"
                            DevicePtr = None
                            RequiresGrad = false
                            GradientData = None
                        }
                    else
                        None

                let bias =
                    match layerData.BiasData, layerData.BiasShape with
                    | Some biasData, Some biasShape when biasData.Length > 0 ->
                        let expectedLength = Array.fold (*) 1 biasShape
                        let biasArray = this.DecompressFloatArray(biasData, expectedLength)
                        Some {
                            Data = biasArray
                            Shape = biasShape
                            Device = "cuda"
                            DevicePtr = None
                            RequiresGrad = false
                            GradientData = None
                        }
                    | _ -> None

                {
                    LayerId = layerData.LayerId
                    LayerType = layerType
                    Weights = weights
                    Bias = bias
                    Parameters = Map.empty
                    IsTrainable = true
                    DeviceId = 0
                }

            with
            | ex ->
                logger.LogError(correlationId, TarsError.create "LayerConversionError" $"Failed to convert layer {layerData.LayerId}" (Some ex), ex)
                {
                    LayerId = layerData.LayerId
                    LayerType = Custom ("unknown", Map.empty)
                    Weights = None
                    Bias = None
                    Parameters = Map.empty
                    IsTrainable = false
                    DeviceId = 0
                }

        /// Parse layer type from string
        member private this.ParseLayerType(layerTypeName: string, weightShape: int[]) : LayerType =
            match layerTypeName.ToLower() with
            | "linear" when weightShape.Length >= 2 -> Linear (weightShape.[0], weightShape.[1])
            | "embedding" when weightShape.Length >= 2 -> Embedding (weightShape.[0], weightShape.[1])
            | "layer_norm" when weightShape.Length >= 1 -> LayerNorm (weightShape.[0], 1e-5)
            | "multi_head_attention" -> MultiHeadAttention (12, 64, 512) // Default values
            | "feed_forward" -> FeedForward (768, 3072) // Default values
            | name when name.StartsWith("activation_") -> Activation (name.Substring(11))
            | "dropout" -> Dropout 0.1
            | name when name.StartsWith("custom_") -> Custom (name.Substring(7), Map.empty)
            | _ -> Custom (layerTypeName, Map.empty)

    /// Create TARS model serializer
    let createModelSerializer (logger: ITarsLogger) =
        new TarsModelSerializer(logger)
