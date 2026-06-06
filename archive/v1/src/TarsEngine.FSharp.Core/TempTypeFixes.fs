// Temporary type fixes for compilation
namespace TarsEngine.FSharp.Core

open System
open System.Threading.Tasks

module TempTypeFixes =

    // Simplified ReasoningAgent for compilation
    type TempReasoningAgent = {
        Id: int
        Strategy: string
        QualityScore: float
        Payoff: float
        BestResponse: string
    }

    // Simplified InferenceModelConfig
    type TempInferenceModelConfig = {
        ModelName: string
        VocabularySize: int
        EmbeddingDimension: int
        HiddenSize: int
        NumLayers: int
        NumAttentionHeads: int
        MaxSequenceLength: int
        UseMultiSpaceEmbeddings: bool
        GeometricSpaces: obj list
    }

    // Placeholder types to prevent compilation errors
    type TaskCode<'TOverall, 'TResult> = TaskCode of Task<'TResult>
    type TaskBuilderBase() =
        member _.Bind(task: Task<'T>, f: 'T -> TaskCode<'U, 'V>) : TaskCode<'U, 'V> =
            TaskCode (Task.FromResult(Unchecked.defaultof<'V>))
        member _.Return(value: 'T) : TaskCode<'U, 'T> =
            TaskCode (Task.FromResult(value))
        member _.Zero() : TaskCode<'T, unit> =
            TaskCode (Task.FromResult(()))

    // Missing types for test projects
    type TarsError =
        | ValidationError of string
        | ProcessingError of string
        | NetworkError of string
        | UnknownError of string

    type ITarsLogger =
        abstract member LogInformation: correlationId: string * message: string -> unit
        abstract member LogInformation: correlationId: string * message: string * args: obj[] -> unit
        abstract member LogWarning: correlationId: string * message: string -> unit
        abstract member LogError: correlationId: string * error: TarsError * ex: Exception -> unit

    // Result types for tests
    type TarsResult<'T> =
        | Success of 'T
        | Error of TarsError

    // Tensor type for AI tests
    type TarsTensor = {
        Data: float32[]
        Shape: int[]
        Device: string
        DevicePtr: nativeint option
        RequiresGrad: bool
        GradientData: float32[] option
    }

    // Model types for AI tests
    type TarsModel = {
        ModelId: string
        ModelName: string
        Architecture: string
        Layers: int
        ModelSize: int64
        MemoryRequirement: int64
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

    // Request types for tests
    type TarsRequest = {
        RequestId: string
        ModelId: string
        Data: obj
    }

    // Mock functions for tests
    let createCudaEngine (logger: ITarsLogger) =
        { new obj() with
            member _.ToString() = "MockCudaEngine" }

    let createAIInferenceEngine (logger: ITarsLogger) =
        { new obj() with
            member _.ToString() = "MockAIInferenceEngine" }

    let createTestCudaEngine (logger: ITarsLogger) = createCudaEngine logger
    let createTestAIEngine (logger: ITarsLogger) = createAIInferenceEngine logger

    let createTestTensor () = {
        Data = [| 1.0f; 2.0f; 3.0f |]
        Shape = [| 3 |]
        Device = "cpu"
        DevicePtr = None
        RequiresGrad = false
        GradientData = None
    }

    let createTestModel () = {
        ModelId = "test-model-123"
        ModelName = "Test Model"
        Architecture = "transformer"
        Layers = 12
        ModelSize = 1024L * 1024L * 100L // 100MB
        MemoryRequirement = 1024L * 1024L * 200L // 200MB
        MaxSequenceLength = 2048
        VocabularySize = 50000
        HiddenSize = 768
        NumLayers = 12
        NumAttentionHeads = 12
        IntermediateSize = 3072
        IsLoaded = false
        DeviceId = 0
        CreatedAt = DateTime.UtcNow
        LastUsed = DateTime.UtcNow
    }

    let createModelSerializer () =
        { new obj() with
            member _.ToString() = "MockModelSerializer" }
