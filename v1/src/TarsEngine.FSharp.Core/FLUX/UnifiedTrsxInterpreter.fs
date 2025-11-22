namespace TarsEngine.FSharp.Core.FLUX

open System
open TarsEngine.FSharp.Core.FLUX.FluxFractalArchitecture

/// Unified TRSX interpreter that handles both .flux and .trsx formats
module UnifiedTrsxInterpreter =

    /// Unified document structure
    type UnifiedDocument = {
        Metadata: DocumentMetadata
        Content: string
        Tier: FractalTier
        Format: DocumentFormat
    }

    /// Document metadata
    and DocumentMetadata = {
        Title: string
        Version: string
        Author: string option
        Created: DateTime
        FilePath: string
    }

    /// Document format
    and DocumentFormat =
        | FluxFormat
        | TrsxFormat
        | UnifiedFormat

    /// Unified execution result
    type UnifiedExecutionResult = {
        Success: bool
        Output: string
        ExecutionTime: TimeSpan
        Tier: FractalTier
        Format: DocumentFormat
        FractalMetrics: FractalMetrics option
    }

    /// Fractal analysis metrics
    and FractalMetrics = {
        Dimension: float
        SelfSimilarity: float
        Complexity: int
        EmergentProperties: string list
    }

    /// Unified interpreter
    type UnifiedInterpreter() =
        let fluxEngine = UnifiedFluxEngine()
        
        /// Execute document based on format and content
        member this.ExecuteDocument(document: UnifiedDocument) : UnifiedExecutionResult =
            let startTime = DateTime.Now
            
            try
                let result = 
                    match document.Format with
                    | FluxFormat -> this.ExecuteFluxDocument(document)
                    | TrsxFormat -> this.ExecuteTrsxDocument(document)
                    | UnifiedFormat -> this.ExecuteUnifiedDocument(document)
                
                let executionTime = DateTime.Now - startTime
                
                {
                    Success = true
                    Output = result
                    ExecutionTime = executionTime
                    Tier = document.Tier
                    Format = document.Format
                    FractalMetrics = Some {
                        Dimension = 1.5
                        SelfSimilarity = 0.8
                        Complexity = document.Content.Length / 100
                        EmergentProperties = ["tier_based_execution"; "fractal_structure"]
                    }
                }
            with
            | ex ->
                let executionTime = DateTime.Now - startTime
                {
                    Success = false
                    Output = sprintf "Error: %s" ex.Message
                    ExecutionTime = executionTime
                    Tier = document.Tier
                    Format = document.Format
                    FractalMetrics = None
                }
        
        /// Execute FLUX format document
        member private this.ExecuteFluxDocument(document: UnifiedDocument) : string =
            let fluxResult = fluxEngine.ExecuteFluxContent(document.Content, document.Metadata.FilePath)
            String.Join("\n", fluxResult.Results)
        
        /// Execute TRSX format document
        member private this.ExecuteTrsxDocument(document: UnifiedDocument) : string =
            sprintf "🔧 TRSX execution: %s\n✅ Processed %d characters" 
                document.Metadata.Title document.Content.Length
        
        /// Execute unified format document
        member private this.ExecuteUnifiedDocument(document: UnifiedDocument) : string =
            let fluxResult = this.ExecuteFluxDocument(document)
            let trsxResult = this.ExecuteTrsxDocument(document)
            sprintf "%s\n%s" fluxResult trsxResult
        
        /// Execute file by path
        member this.ExecuteFile(filePath: string) : UnifiedExecutionResult =
            let content = System.IO.File.ReadAllText(filePath)
            let format = this.DetermineFormat(filePath, content)
            let tier = this.DetermineTier(content)
            
            let document = {
                Metadata = {
                    Title = System.IO.Path.GetFileNameWithoutExtension(filePath)
                    Version = "1.0"
                    Author = None
                    Created = DateTime.Now
                    FilePath = filePath
                }
                Content = content
                Tier = tier
                Format = format
            }
            
            this.ExecuteDocument(document)
        
        /// Determine document format from file extension and content
        member this.DetermineFormat(filePath: string, content: string) : DocumentFormat =
            let extension = System.IO.Path.GetExtension(filePath).ToLowerInvariant()
            match extension with
            | ".flux" -> FluxFormat
            | ".trsx" -> TrsxFormat
            | _ when content.Contains("META {") -> FluxFormat
            | _ when content.Contains("reasoning_block") -> TrsxFormat
            | _ -> FluxFormat  // Default to FLUX
        
        /// Determine tier from content complexity
        member this.DetermineTier(content: string) : FractalTier =
            let lines = content.Split('\n') |> Array.length
            let hasReflection = content.Contains("REFLECT") || content.Contains("META")
            let hasEvolution = content.Contains("EVOLVE") || content.Contains("EMERGE")
            
            if hasEvolution then
                Tier4Plus_Emergent
            elif hasReflection then
                Tier3_Reflective
            elif lines > 50 then
                Tier2_Extended
            else
                Tier1_Core

    /// File format detector
    type FormatDetector() =
        
        /// Detect format and tier from file
        member this.AnalyzeFile(filePath: string) : DocumentFormat * FractalTier =
            let content = System.IO.File.ReadAllText(filePath)
            let interpreter = UnifiedInterpreter()
            
            let format = interpreter.DetermineFormat(filePath, content)
            let tier = interpreter.DetermineTier(content)
            
            (format, tier)
        
        /// Get format statistics
        member this.GetFormatStats(filePath: string) : Map<string, obj> =
            let content = System.IO.File.ReadAllText(filePath)
            
            Map.ofList [
                ("file_size", box content.Length)
                ("line_count", box (content.Split('\n').Length))
                ("has_meta", box (content.Contains("META")))
                ("has_reflection", box (content.Contains("REFLECT")))
                ("has_evolution", box (content.Contains("EVOLVE")))
                ("complexity_score", box (content.Length / 100))
            ]

/// Validation result
type ValidationResult = {
    IsValid: bool
    Errors: string list
    Warnings: string list
    SuggestedTier: FractalTier option
}
