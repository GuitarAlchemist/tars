namespace Tars.Cortex

open System
open Tars.Core
open Tars.Llm

/// Minimal entity extraction module for Phase 2
module EntityExtractor =

    /// Result of entity extraction
    type ExtractionResult = {
        Entities: TarsEntity list
        Facts: TarsFact list
        Confidence: float
        ExtractedAt: DateTime
    }
    
    /// Error during extraction
    type ExtractionError =
        | LlmError of message: string
        | ParseError of message: string
        | EmptyContent
    
    /// Create an empty extraction result
    let empty () : ExtractionResult = {
        Entities = []
        Facts = []
        Confidence = 0.0
        ExtractedAt = DateTime.UtcNow
    }
    
    /// Merge multiple extraction results
    let mergeResults (results: ExtractionResult list) : ExtractionResult =
        let allEntities = results |> List.collect (fun r -> r.Entities)
        let allFacts = results |> List.collect (fun r -> r.Facts)
        let avgConfidence = 
            if results.IsEmpty then 0.0
            else results |> List.averageBy (fun r -> r.Confidence)
        {
            Entities = allEntities
            Facts = allFacts
            Confidence = avgConfidence
            ExtractedAt = DateTime.UtcNow
        }
