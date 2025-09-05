namespace TarsEngine.FSharp.Cli.Tests.AI

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open Xunit
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.AI.TarsModelFormat

/// Comprehensive tests for TARS Model Format
module TarsModelFormatTests =
    
    /// Mock logger for testing
    type MockLogger() =
        interface ITarsLogger with
            member _.LogInformation(correlationId: string, message: string) = 
                Console.WriteLine($"[INFO] {correlationId}: {message}")
            member _.LogInformation(correlationId: string, message: string, args: obj[]) = 
                Console.WriteLine($"[INFO] {correlationId}: {String.Format(message, args)}")
            member _.LogWarning(correlationId: string, message: string) = 
                Console.WriteLine($"[WARN] {correlationId}: {message}")
            member _.LogError(correlationId: string, error: TarsError, ex: Exception) = 
                Console.WriteLine($"[ERROR] {correlationId}: {error} - {ex.Message}")
    
    /// Create test tensor
    let createTestTensor (shape: int[]) (value: float32) =
        let size = Array.fold (*) 1 shape
        {
            Data = Array.create size value
            Shape = shape
            Device = "cuda"
            DevicePtr = None
            RequiresGrad = false
            GradientData = None
        }
    
    /// Create comprehensive test model
    let createComprehensiveTestModel() =
        {
            ModelId = "comprehensive-test-model"
            ModelName = "Comprehensive Test Transformer"
            Architecture = "transformer"
            Layers = [|
                // Embedding layer
                {
                    LayerId = "embedding"
                    LayerType = Embedding (5000, 512)
                    Weights = Some (createTestTensor [|5000; 512|] 0.02f)
                    Bias = None
                    Parameters = Map.empty
                    IsTrainable = true
                    DeviceId = 0
                }
                
                // Multiple attention layers
                for i in 0..5 do
                    {
                        LayerId = $"attention_{i}"
                        LayerType = MultiHeadAttention (8, 64, 256)
                        Weights = Some (createTestTensor [|512; 512; 3|] 0.01f)
                        Bias = Some (createTestTensor [|512; 3|] 0.0f)
                        Parameters = Map.empty
                        IsTrainable = true
                        DeviceId = 0
                    }
                
                // Feed-forward layers
                for i in 0..5 do
                    {
                        LayerId = $"feedforward_{i}"
                        LayerType = FeedForward (512, 2048)
                        Weights = Some (createTestTensor [|512; 2048; 2|] 0.01f)
                        Bias = Some (createTestTensor [|2048; 2|] 0.0f)
                        Parameters = Map.empty
                        IsTrainable = true
                        DeviceId = 0
                    }
                
                // Layer normalization
                for i in 0..11 do
                    {
                        LayerId = $"layernorm_{i}"
                        LayerType = LayerNorm (512, 1e-5)
                        Weights = Some (createTestTensor [|512|] 1.0f)
                        Bias = Some (createTestTensor [|512|] 0.0f)
                        Parameters = Map.empty
                        IsTrainable = true
                        DeviceId = 0
                    }
                
                // Activation layers
                {
                    LayerId = "gelu_activation"
                    LayerType = Activation "gelu"
                    Weights = None
                    Bias = None
                    Parameters = Map.empty
                    IsTrainable = false
                    DeviceId = 0
                }
                
                // Dropout layer
                {
                    LayerId = "dropout"
                    LayerType = Dropout 0.1
                    Weights = None
                    Bias = None
                    Parameters = Map.empty
                    IsTrainable = false
                    DeviceId = 0
                }
                
                // Custom layer
                {
                    LayerId = "custom_tars_layer"
                    LayerType = Custom ("tars_reasoning", Map [("complexity", box 100); ("mode", box "inference")])
                    Weights = Some (createTestTensor [|512; 256|] 0.05f)
                    Bias = Some (createTestTensor [|256|] 0.0f)
                    Parameters = Map [
                        ("reasoning_weights", createTestTensor [|256; 128|] 0.03f)
                        ("attention_bias", createTestTensor [|128|] 0.0f)
                    ]
                    IsTrainable = true
                    DeviceId = 0
                }
                
                // Output projection
                {
                    LayerId = "output_projection"
                    LayerType = Linear (512, 5000)
                    Weights = Some (createTestTensor [|512; 5000|] 0.02f)
                    Bias = Some (createTestTensor [|5000|] 0.0f)
                    Parameters = Map.empty
                    IsTrainable = true
                    DeviceId = 0
                }
            |]
            ModelSize = 125_000_000L
            MemoryRequirement = 500L * 1024L * 1024L // 500MB
            MaxSequenceLength = 256
            VocabularySize = 5000
            HiddenSize = 512
            NumLayers = 6
            NumAttentionHeads = 8
            IntermediateSize = 2048
            IsLoaded = false
            DeviceId = 0
            CreatedAt = DateTime.UtcNow
            LastUsed = DateTime.UtcNow
        }

    [<Fact>]
    let ``Model serializer should serialize and deserialize models correctly`` () =
        task {
            // Arrange
            let logger = MockLogger() :> ITarsLogger
            let serializer = createModelSerializer logger
            let testModel = createComprehensiveTestModel()
            let testPath = "test_serialization.tars"
            
            try
                // Act - Serialize
                let! serializeResult = serializer.SerializeModelAsync(testModel, testPath, CancellationToken.None)
                
                // Assert serialization
                match serializeResult with
                | Success (fileSize, _) ->
                    Assert.True(fileSize > 0L)
                    Assert.True(File.Exists(testPath))
                    
                    // Act - Deserialize
                    let! deserializeResult = serializer.DeserializeModelAsync(testPath, CancellationToken.None)
                    
                    // Assert deserialization
                    match deserializeResult with
                    | Success (deserializedModel, _) ->
                        // Verify model metadata
                        Assert.Equal(testModel.ModelId, deserializedModel.ModelId)
                        Assert.Equal(testModel.ModelName, deserializedModel.ModelName)
                        Assert.Equal(testModel.Architecture, deserializedModel.Architecture)
                        Assert.Equal(testModel.ModelSize, deserializedModel.ModelSize)
                        Assert.Equal(testModel.MemoryRequirement, deserializedModel.MemoryRequirement)
                        Assert.Equal(testModel.MaxSequenceLength, deserializedModel.MaxSequenceLength)
                        Assert.Equal(testModel.VocabularySize, deserializedModel.VocabularySize)
                        Assert.Equal(testModel.HiddenSize, deserializedModel.HiddenSize)
                        Assert.Equal(testModel.NumLayers, deserializedModel.NumLayers)
                        Assert.Equal(testModel.NumAttentionHeads, deserializedModel.NumAttentionHeads)
                        Assert.Equal(testModel.IntermediateSize, deserializedModel.IntermediateSize)
                        
                        // Verify layers
                        Assert.Equal(testModel.Layers.Length, deserializedModel.Layers.Length)
                        
                        for i in 0 .. testModel.Layers.Length - 1 do
                            let originalLayer = testModel.Layers.[i]
                            let deserializedLayer = deserializedModel.Layers.[i]
                            
                            Assert.Equal(originalLayer.LayerId, deserializedLayer.LayerId)
                            Assert.Equal(originalLayer.IsTrainable, deserializedLayer.IsTrainable)
                            Assert.Equal(originalLayer.DeviceId, deserializedLayer.DeviceId)
                            
                            // Verify weights if present
                            match originalLayer.Weights, deserializedLayer.Weights with
                            | Some originalWeights, Some deserializedWeights ->
                                Assert.Equal(originalWeights.Shape, deserializedWeights.Shape)
                                Assert.Equal(originalWeights.Device, deserializedWeights.Device)
                                Assert.Equal(originalWeights.RequiresGrad, deserializedWeights.RequiresGrad)
                                // Note: Data might be compressed/decompressed, so exact equality may not hold
                                Assert.Equal(originalWeights.Data.Length, deserializedWeights.Data.Length)
                            | None, None -> () // Both None, which is correct
                            | _ -> Assert.True(false, $"Weight mismatch in layer {originalLayer.LayerId}")
                            
                            // Verify bias if present
                            match originalLayer.Bias, deserializedLayer.Bias with
                            | Some originalBias, Some deserializedBias ->
                                Assert.Equal(originalBias.Shape, deserializedBias.Shape)
                                Assert.Equal(originalBias.Data.Length, deserializedBias.Data.Length)
                            | None, None -> () // Both None, which is correct
                            | _ -> Assert.True(false, $"Bias mismatch in layer {originalLayer.LayerId}")
                        
                    | Failure (error, _) ->
                        Assert.True(false, $"Deserialization failed: {error}")
                        
                | Failure (error, _) ->
                    Assert.True(false, $"Serialization failed: {error}")
                    
            finally
                // Cleanup
                if File.Exists(testPath) then
                    File.Delete(testPath)
        }

    [<Fact>]
    let ``Model serializer should handle different layer types correctly`` () =
        task {
            // Arrange
            let logger = MockLogger() :> ITarsLogger
            let serializer = createModelSerializer logger
            
            // Create model with all layer types
            let testModel = {
                createComprehensiveTestModel() with
                    ModelId = "layer-types-test"
                    ModelName = "Layer Types Test Model"
            }
            
            let testPath = "test_layer_types.tars"
            
            try
                // Act
                let! serializeResult = serializer.SerializeModelAsync(testModel, testPath, CancellationToken.None)
                
                match serializeResult with
                | Success (_, _) ->
                    let! deserializeResult = serializer.DeserializeModelAsync(testPath, CancellationToken.None)
                    
                    match deserializeResult with
                    | Success (deserializedModel, _) ->
                        // Verify specific layer types
                        let embeddingLayer = deserializedModel.Layers |> Array.find (fun l -> l.LayerId = "embedding")
                        match embeddingLayer.LayerType with
                        | Embedding (vocabSize, embedDim) ->
                            Assert.Equal(5000, vocabSize)
                            Assert.Equal(512, embedDim)
                        | _ -> Assert.True(false, "Embedding layer type not preserved")
                        
                        let attentionLayer = deserializedModel.Layers |> Array.find (fun l -> l.LayerId = "attention_0")
                        match attentionLayer.LayerType with
                        | MultiHeadAttention (numHeads, headDim, seqLen) ->
                            Assert.Equal(8, numHeads)
                            Assert.Equal(64, headDim)
                            Assert.Equal(256, seqLen)
                        | _ -> Assert.True(false, "Attention layer type not preserved")
                        
                        let customLayer = deserializedModel.Layers |> Array.find (fun l -> l.LayerId = "custom_tars_layer")
                        match customLayer.LayerType with
                        | Custom (name, parameters) ->
                            Assert.Equal("tars_reasoning", name)
                            Assert.True(parameters.ContainsKey("complexity"))
                            Assert.True(parameters.ContainsKey("mode"))
                        | _ -> Assert.True(false, "Custom layer type not preserved")
                        
                    | Failure (error, _) ->
                        Assert.True(false, $"Deserialization failed: {error}")
                        
                | Failure (error, _) ->
                    Assert.True(false, $"Serialization failed: {error}")
                    
            finally
                if File.Exists(testPath) then
                    File.Delete(testPath)
        }

    [<Fact>]
    let ``Model serializer should handle file compression correctly`` () =
        task {
            // Arrange
            let logger = MockLogger() :> ITarsLogger
            let serializer = createModelSerializer logger
            let testModel = createComprehensiveTestModel()
            let testPath = "test_compression.tars"
            
            try
                // Act
                let! result = serializer.SerializeModelAsync(testModel, testPath, CancellationToken.None)
                
                // Assert
                match result with
                | Success (fileSize, _) ->
                    Assert.True(fileSize > 0L)
                    
                    // File should be smaller than uncompressed data due to compression
                    let uncompressedSize = testModel.Layers 
                                         |> Array.sumBy (fun layer ->
                                             let weightSize = layer.Weights |> Option.map (fun w -> int64 w.Data.Length * 4L) |> Option.defaultValue 0L
                                             let biasSize = layer.Bias |> Option.map (fun b -> int64 b.Data.Length * 4L) |> Option.defaultValue 0L
                                             weightSize + biasSize)
                    
                    // Compressed file should be significantly smaller (assuming some compression)
                    Assert.True(fileSize < uncompressedSize)
                    
                | Failure (error, _) ->
                    Assert.True(false, $"Serialization failed: {error}")
                    
            finally
                if File.Exists(testPath) then
                    File.Delete(testPath)
        }

    [<Fact>]
    let ``Model serializer should handle invalid file paths`` () =
        task {
            // Arrange
            let logger = MockLogger() :> ITarsLogger
            let serializer = createModelSerializer logger
            let testModel = createComprehensiveTestModel()
            let invalidPath = "/invalid/path/that/does/not/exist/model.tars"
            
            // Act
            let! result = serializer.SerializeModelAsync(testModel, invalidPath, CancellationToken.None)
            
            // Assert
            match result with
            | Success _ ->
                Assert.True(false, "Should have failed with invalid path")
            | Failure (error, _) ->
                Assert.IsType<ExecutionError>(error) |> ignore
        }

    [<Fact>]
    let ``Model serializer should handle nonexistent files for deserialization`` () =
        task {
            // Arrange
            let logger = MockLogger() :> ITarsLogger
            let serializer = createModelSerializer logger
            let nonexistentPath = "nonexistent_model.tars"
            
            // Act
            let! result = serializer.DeserializeModelAsync(nonexistentPath, CancellationToken.None)
            
            // Assert
            match result with
            | Success _ ->
                Assert.True(false, "Should have failed with nonexistent file")
            | Failure (error, _) ->
                Assert.IsType<ValidationError>(error) |> ignore
        }

    [<Fact>]
    let ``Model format should preserve tensor data integrity`` () =
        task {
            // Arrange
            let logger = MockLogger() :> ITarsLogger
            let serializer = createModelSerializer logger
            
            // Create model with specific tensor values for verification
            let specificValues = [| 1.5f; -2.3f; 0.0f; 42.7f; -0.001f |]
            let testTensor = {
                Data = specificValues
                Shape = [|5|]
                Device = "cuda"
                DevicePtr = None
                RequiresGrad = true
                GradientData = Some [| 0.1f; 0.2f; 0.3f; 0.4f; 0.5f |]
            }
            
            let testModel = {
                createComprehensiveTestModel() with
                    ModelId = "tensor-integrity-test"
                    Layers = [|
                        {
                            LayerId = "test_layer"
                            LayerType = Linear (5, 1)
                            Weights = Some testTensor
                            Bias = None
                            Parameters = Map.empty
                            IsTrainable = true
                            DeviceId = 0
                        }
                    |]
            }
            
            let testPath = "test_tensor_integrity.tars"
            
            try
                // Act
                let! serializeResult = serializer.SerializeModelAsync(testModel, testPath, CancellationToken.None)
                
                match serializeResult with
                | Success (_, _) ->
                    let! deserializeResult = serializer.DeserializeModelAsync(testPath, CancellationToken.None)
                    
                    match deserializeResult with
                    | Success (deserializedModel, _) ->
                        let deserializedLayer = deserializedModel.Layers.[0]
                        match deserializedLayer.Weights with
                        | Some deserializedTensor ->
                            Assert.Equal(testTensor.Shape, deserializedTensor.Shape)
                            Assert.Equal(testTensor.Device, deserializedTensor.Device)
                            Assert.Equal(testTensor.RequiresGrad, deserializedTensor.RequiresGrad)
                            Assert.Equal(testTensor.Data.Length, deserializedTensor.Data.Length)
                            
                            // Note: Due to compression/decompression, exact floating-point equality might not hold
                            // So we check that values are approximately equal
                            for i in 0 .. testTensor.Data.Length - 1 do
                                let original = testTensor.Data.[i]
                                let deserialized = deserializedTensor.Data.[i]
                                let tolerance = 0.001f
                                Assert.True(abs(original - deserialized) < tolerance, 
                                           $"Value mismatch at index {i}: expected {original}, got {deserialized}")
                                           
                        | None ->
                            Assert.True(false, "Weights were not preserved")
                            
                    | Failure (error, _) ->
                        Assert.True(false, $"Deserialization failed: {error}")
                        
                | Failure (error, _) ->
                    Assert.True(false, $"Serialization failed: {error}")
                    
            finally
                if File.Exists(testPath) then
                    File.Delete(testPath)
        }
