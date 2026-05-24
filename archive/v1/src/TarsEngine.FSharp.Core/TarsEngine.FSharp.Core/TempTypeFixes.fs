// Temporary type fixes for compilation
namespace TarsEngine.FSharp.Core

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
