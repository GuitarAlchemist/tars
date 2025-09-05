namespace TarsEngine.FSharp.Cli.AI

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.AI.AITypes
open TarsEngine.FSharp.Cli.AI.LayerExecutors

/// Forward Pass Executor - Handles neural network forward pass execution
module ForwardPassExecutor =
    
    /// Execute a single neural network layer
    let executeLayer 
        (logger: ITarsLogger) 
        (layer: NeuralLayer) 
        (inputTensor: TarsTensor) 
        (correlationId: string) =
        task {
            try
                match layer.LayerType with
                | Linear (inputSize, outputSize) ->
                    return! LayerExecutors.executeLinearLayer logger layer inputTensor inputSize outputSize correlationId
                
                | Embedding (vocabSize, embedDim) ->
                    return! LayerExecutors.executeEmbeddingLayer logger layer inputTensor vocabSize embedDim correlationId
                
                | LayerNorm (size, eps) ->
                    return! LayerExecutors.executeLayerNormalization logger layer inputTensor size eps correlationId
                
                | MultiHeadAttention (numHeads, headDim, seqLen) ->
                    return! LayerExecutors.executeMultiHeadAttention logger layer inputTensor numHeads headDim seqLen correlationId
                
                | FeedForward (hiddenSize, intermediateSize) ->
                    return! LayerExecutors.executeFeedForward logger layer inputTensor hiddenSize intermediateSize correlationId
                
                | Activation funcType ->
                    return! LayerExecutors.executeActivation logger inputTensor funcType correlationId
                
                | Dropout rate ->
                    // During inference, dropout is disabled
                    return Success (inputTensor, Map [("dropout", box "disabled")])
                
                | Custom (name, parameters) ->
                    return! LayerExecutors.executeCustomLayer logger layer inputTensor name parameters correlationId
            
            with
            | ex ->
                let error = ExecutionError ($"Layer execution failed ({layer.LayerId}): {ex.Message}", Some ex)
                logger.LogError(correlationId, error, ex)
                return Failure (error, correlationId)
        }
    
    /// Execute forward pass through neural network layers
    let executeForwardPass 
        (logger: ITarsLogger) 
        (model: TarsModel) 
        (inputTensor: TarsTensor) 
        (correlationId: string) =
        task {
            try
                logger.LogInformation(correlationId, 
                    $"⚡ Executing forward pass through {model.Layers.Length} layers")
                
                let mutable currentTensor = inputTensor
                let mutable layerIndex = 0
                
                // Process each layer sequentially
                for layer in model.Layers do
                    let! layerResult = executeLayer logger layer currentTensor correlationId
                    
                    match layerResult with
                    | Success (outputTensor, _) ->
                        currentTensor <- outputTensor
                        layerIndex <- layerIndex + 1
                        
                        if layerIndex % 5 = 0 then
                            logger.LogInformation(correlationId, 
                                $"📊 Processed {layerIndex}/{model.Layers.Length} layers")
                    
                    | Failure (error, _) ->
                        return Failure (error, correlationId)
                
                logger.LogInformation(correlationId, 
                    $"✅ Forward pass completed through all {model.Layers.Length} layers")
                return Success (currentTensor, Map [
                    ("layersProcessed", box model.Layers.Length)
                ])
            
            with
            | ex ->
                let error = ExecutionError ($"Forward pass failed: {ex.Message}", Some ex)
                logger.LogError(correlationId, error, ex)
                return Failure (error, correlationId)
        }
