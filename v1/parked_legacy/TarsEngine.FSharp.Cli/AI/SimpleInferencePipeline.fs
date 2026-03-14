namespace TarsEngine.FSharp.Cli.AI

open System
open System.Threading
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.AI.AITypes
open TarsEngine.FSharp.Cli.AI.ForwardPassExecutor

/// Simple Inference Pipeline - Simplified neural network inference execution
module SimpleInferencePipeline =
    
    /// Execute the complete inference pipeline
    let executeInferencePipeline 
        (logger: ITarsLogger) 
        (model: TarsModel) 
        (request: InferenceRequest) 
        (cancellationToken: CancellationToken) =
        task {
            try
                let startTime = DateTime.UtcNow
                logger.LogInformation(request.CorrelationId, 
                    $"🔄 Executing inference pipeline for {model.Architecture} model")
                
                // Validate input tensors
                if request.InputTensors.Length = 0 then
                    let error = ValidationError ("No input tensors provided", "inputTensors")
                    return Failure (error, request.CorrelationId)
                
                let inputTensor = request.InputTensors.[0]
                
                // Execute forward pass through all layers
                let! forwardResult = ForwardPassExecutor.executeForwardPass logger model inputTensor request.CorrelationId
                
                match forwardResult with
                | Success (outputTensor, metadata) ->
                    let inferenceTime = DateTime.UtcNow - startTime
                    let tokensGenerated = outputTensor.Shape.[0] // Sequence length
                    let tokensPerSecond = float tokensGenerated / inferenceTime.TotalSeconds
                    
                    let response = {
                        RequestId = request.RequestId
                        ModelId = model.ModelId
                        OutputTensors = [| outputTensor |]
                        Logits = Some outputTensor
                        Attentions = if request.ReturnAttentions then Some [||] else None
                        HiddenStates = if request.ReturnHiddenStates then Some [||] else None
                        InferenceTime = inferenceTime
                        TokensGenerated = tokensGenerated
                        TokensPerSecond = tokensPerSecond
                        MemoryUsed = model.MemoryRequirement
                        GpuUtilization = 
                            metadata.TryFind("gpuUtilization") 
                            |> Option.map unbox<float> 
                            |> Option.defaultValue 0.0
                        Success = true
                        ErrorMessage = None
                        CorrelationId = request.CorrelationId
                    }
                    
                    logger.LogInformation(request.CorrelationId, 
                        $"✅ Forward pass completed: {tokensGenerated} tokens, {tokensPerSecond:F2} tokens/sec")
                    return Success (response, Map [
                        ("tokensGenerated", box tokensGenerated)
                        ("tokensPerSecond", box tokensPerSecond)
                    ])
                
                | Failure (error, _) ->
                    return Failure (error, request.CorrelationId)
            
            with
            | ex ->
                let error = ExecutionError ($"Inference pipeline failed: {ex.Message}", Some ex)
                logger.LogError(request.CorrelationId, error, ex)
                return Failure (error, request.CorrelationId)
        }
