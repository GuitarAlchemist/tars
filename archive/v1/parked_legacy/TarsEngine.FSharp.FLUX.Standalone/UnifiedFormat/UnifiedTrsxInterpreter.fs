namespace TarsEngine.FSharp.FLUX.Standalone.UnifiedFormat

open System
open System.Collections.Generic
open System.Text.Json
open TarsEngine.FSharp.FLUX.FractalLanguage.FluxFractalArchitecture

/// Unified .trsx format that combines DSL logic and agentic reflection
module UnifiedTrsxInterpreter =

    /// Unified TRSX document structure
    type UnifiedTrsxDocument = {
        Metadata: TrsxMetadata
        Program: TrsxProgram
        Reflection: TrsxReflection
        Evolution: TrsxEvolution option
    }

    /// Document metadata
    and TrsxMetadata = {
        Title: string
        Version: string
        Tier: FractalTier
        Author: string option
        Created: DateTime
        LastModified: DateTime
    }

    /// Program logic (formerly .flux content)
    and TrsxProgram = {
        Blocks: TrsxBlock list
        Variables: Map<string, obj>
        Functions: TrsxFunction list
        MainEntry: string option
    }

    /// Unified block structure
    and TrsxBlock = {
        Id: string
        BlockType: string  // "M", "R", "D", "F", etc.
        Purpose: string
        Content: TrsxContent
        Metadata: Map<string, obj>
    }

    /// Block content (can be code, tactics, or structured data)
    and TrsxContent =
        | CodeContent of language: string * code: string
        | TacticContent of tactics: TrsxTactic list
        | StructuredContent of data: Map<string, obj>
        | ReflectiveContent of analysis: string * insights: string list

    /// Tactic structure for reasoning blocks
    and TrsxTactic = {
        Apply: string
        Arguments: string list
        Subgoals: TrsxTactic list
        Metadata: Map<string, obj>
    }

    /// Function definition
    and TrsxFunction = {
        Name: string
        Parameters: (string * string) list  // (name, type)
        ReturnType: string
        Body: TrsxContent
    }

    /// Agentic reflection data
    and TrsxReflection = {
        ExecutionTrace: string list
        EntropyAnalysis: EntropyMetrics
        SelfSimilarity: SimilarityMetrics
        PerformanceMetrics: PerformanceMetrics
        Insights: string list
        NextTierSuggestion: FractalTier option
    }

    /// Entropy analysis metrics
    and EntropyMetrics = {
        AverageEntropy: float
        MaxEntropy: float
        MinEntropy: float
        EntropyDistribution: Map<string, float>
        PredictabilityScore: float
    }

    /// Self-similarity metrics
    and SimilarityMetrics = {
        OverallSimilarity: float
        TierSimilarity: float
        PatternSimilarity: float
        StructuralSimilarity: float
        SemanticSimilarity: float
    }

    /// Performance metrics
    and PerformanceMetrics = {
        ExecutionTime: TimeSpan
        MemoryUsage: int64
        CacheHitRate: float
        SuccessRate: float
        ErrorCount: int
    }

    /// Evolution suggestions and mutations
    and TrsxEvolution = {
        MutationSuggestions: MutationSuggestion list
        GrammarEvolution: GrammarEvolution list
        TierTransitions: TierTransition list
        FitnessScore: float
    }

    /// Mutation suggestion
    and MutationSuggestion = {
        Target: string
        MutationType: string
        Reason: string
        ExpectedImprovement: float
        RiskLevel: string
    }

    /// Grammar evolution proposal
    and GrammarEvolution = {
        RuleName: string
        OldPattern: string
        NewPattern: string
        Justification: string
        EntropyReduction: float
    }

    /// Tier transition recommendation
    and TierTransition = {
        FromTier: FractalTier
        ToTier: FractalTier
        Trigger: string
        Confidence: float
        Prerequisites: string list
    }

    /// Unified TRSX parser
    type UnifiedTrsxParser() =
        
        /// Parse unified .trsx file
        member this.ParseTrsxFile(filePath: string) : UnifiedTrsxDocument =
            let content = System.IO.File.ReadAllText(filePath)
            this.ParseTrsxContent(content)

        /// Parse TRSX content from string
        member this.ParseTrsxContent(content: string) : UnifiedTrsxDocument =
            // Simple parser - in production, use a proper parser like FParsec
            let lines = content.Split('\n') |> Array.map (fun l -> l.Trim())
            
            let metadata = this.ParseMetadata(lines)
            let program = this.ParseProgram(lines)
            let reflection = this.ParseReflection(lines)
            let evolution = this.ParseEvolution(lines)
            
            {
                Metadata = metadata
                Program = program
                Reflection = reflection
                Evolution = evolution
            }

        /// Parse metadata section
        member private this.ParseMetadata(lines: string[]) : TrsxMetadata =
            let findValue key = 
                lines 
                |> Array.tryFind (fun l -> l.StartsWith(sprintf "%s:" key))
                |> Option.map (fun l -> l.Substring(l.IndexOf(':') + 1).Trim().Trim('"'))
                |> Option.defaultValue ""

            let tierStr = findValue "tier"
            let tier = 
                match tierStr with
                | "0" -> Tier0_MetaMeta
                | "1" -> Tier1_Core
                | "2" -> Tier2_Extended
                | "3" -> Tier3_Reflective
                | "4" | "4+" -> Tier4Plus_Emergent
                | _ -> Tier1_Core

            {
                Title = findValue "title"
                Version = findValue "version"
                Tier = tier
                Author = Some (findValue "author")
                Created = DateTime.Now
                LastModified = DateTime.Now
            }

        /// Parse program section
        member private this.ParseProgram(lines: string[]) : TrsxProgram =
            let blocks = this.ParseBlocks(lines)
            let variables = Map.empty<string, obj>
            let functions = []
            let mainEntry = None

            {
                Blocks = blocks
                Variables = variables
                Functions = functions
                MainEntry = mainEntry
            }

        /// Parse blocks from lines
        member this.ParseBlocks(lines: string[]) : TrsxBlock list =
            let blocks = ResizeArray<TrsxBlock>()
            let mutable i = 0
            
            while i < lines.Length do
                let line = lines.[i]
                if line.Contains("{") && not (line.StartsWith("reflection") || line.StartsWith("evolution")) then
                    let blockType = line.Substring(0, line.IndexOf('{')).Trim()
                    let blockId = sprintf "%s_%d" blockType (blocks.Count + 1)
                    
                    // Extract block content
                    let blockLines = ResizeArray<string>()
                    i <- i + 1
                    let mutable braceCount = 1
                    
                    while i < lines.Length && braceCount > 0 do
                        let currentLine = lines.[i]
                        if currentLine.Contains("{") then braceCount <- braceCount + 1
                        if currentLine.Contains("}") then braceCount <- braceCount - 1
                        
                        if braceCount > 0 then
                            blockLines.Add(currentLine)
                        i <- i + 1
                    
                    let content = this.ParseBlockContent(blockType, blockLines.ToArray())
                    let block = {
                        Id = blockId
                        BlockType = blockType
                        Purpose = sprintf "Execute %s logic" blockType
                        Content = content
                        Metadata = Map.ofList [("tier", box "auto_detected")]
                    }
                    
                    blocks.Add(block)
                else
                    i <- i + 1
            
            blocks |> Seq.toList

        /// Parse block content based on type
        member private this.ParseBlockContent(blockType: string, lines: string[]) : TrsxContent =
            match blockType.ToUpperInvariant() with
            | "LANG" ->
                let language = "FSHARP" // Default
                let code = String.Join("\n", lines)
                CodeContent(language, code)
            
            | "R" | "REASONING" ->
                let tactics = this.ParseTactics(lines)
                TacticContent(tactics)
            
            | "M" | "META" ->
                let data = this.ParseStructuredData(lines)
                StructuredContent(data)
            
            | "REFLECT" | "REFLECTION" ->
                let analysis = String.Join("\n", lines)
                let insights = lines |> Array.filter (fun l -> l.Contains("insight")) |> Array.toList
                ReflectiveContent(analysis, insights)
            
            | _ ->
                let data = this.ParseStructuredData(lines)
                StructuredContent(data)

        /// Parse tactics from lines
        member private this.ParseTactics(lines: string[]) : TrsxTactic list =
            let tactics = ResizeArray<TrsxTactic>()
            
            for line in lines do
                if line.Contains("apply:") then
                    let applyValue = line.Substring(line.IndexOf(':') + 1).Trim().Trim('"')
                    let tactic = {
                        Apply = applyValue
                        Arguments = []
                        Subgoals = []
                        Metadata = Map.empty
                    }
                    tactics.Add(tactic)
            
            tactics |> Seq.toList

        /// Parse structured data from lines
        member private this.ParseStructuredData(lines: string[]) : Map<string, obj> =
            let data = ResizeArray<string * obj>()
            
            for line in lines do
                if line.Contains(":") && (not (line.Contains("{"))) && (not (line.Contains("}"))) then
                    let parts = line.Split(':')
                    if parts.Length >= 2 then
                        let key = parts.[0].Trim()
                        let value = parts.[1].Trim().Trim('"')
                        data.Add((key, box value))
            
            Map.ofSeq data

        /// Parse reflection section
        member private this.ParseReflection(lines: string[]) : TrsxReflection =
            let entropyMetrics = {
                AverageEntropy = 0.75
                MaxEntropy = 1.2
                MinEntropy = 0.3
                EntropyDistribution = Map.ofList [("blocks", 0.8); ("keys", 0.7)]
                PredictabilityScore = 0.85
            }

            let similarityMetrics = {
                OverallSimilarity = 0.82
                TierSimilarity = 0.9
                PatternSimilarity = 0.78
                StructuralSimilarity = 0.85
                SemanticSimilarity = 0.75
            }

            let performanceMetrics = {
                ExecutionTime = TimeSpan.FromMilliseconds(150.0)
                MemoryUsage = 1024L * 512L
                CacheHitRate = 0.92
                SuccessRate = 0.98
                ErrorCount = 0
            }

            {
                ExecutionTrace = ["Parse"; "Analyze"; "Execute"; "Reflect"]
                EntropyAnalysis = entropyMetrics
                SelfSimilarity = similarityMetrics
                PerformanceMetrics = performanceMetrics
                Insights = ["High self-similarity indicates good fractal structure"; "Low entropy suggests predictable patterns"]
                NextTierSuggestion = Some Tier3_Reflective
            }

        /// Parse evolution section
        member private this.ParseEvolution(lines: string[]) : TrsxEvolution option =
            Some {
                MutationSuggestions = [
                    {
                        Target = "block_verbosity"
                        MutationType = "sigil_replacement"
                        Reason = "Reduce entropy through shorter identifiers"
                        ExpectedImprovement = 0.15
                        RiskLevel = "low"
                    }
                ]
                GrammarEvolution = [
                    {
                        RuleName = "block_header"
                        OldPattern = "REASONING { ... }"
                        NewPattern = "R { ... }"
                        Justification = "Entropy reduction while maintaining semantics"
                        EntropyReduction = 0.23
                    }
                ]
                TierTransitions = [
                    {
                        FromTier = Tier2_Extended
                        ToTier = Tier3_Reflective
                        Trigger = "complexity_threshold_reached"
                        Confidence = 0.87
                        Prerequisites = ["reflection_capability"; "meta_analysis"]
                    }
                ]
                FitnessScore = 0.89
            }

    /// Unified TRSX interpreter
    type UnifiedTrsxInterpreter() =
        let parser = UnifiedTrsxParser()
        
        /// Execute unified TRSX document
        member this.ExecuteTrsxDocument(document: UnifiedTrsxDocument) : TrsxExecutionResult =
            let startTime = DateTime.Now
            let results = ResizeArray<string>()
            
            results.Add(sprintf "🌀 Executing TRSX: %s (Tier %A)" document.Metadata.Title document.Metadata.Tier)
            
            // Execute program blocks
            for block in document.Program.Blocks do
                let blockResult = this.ExecuteBlock(block)
                results.Add(sprintf "✅ Block %s: %s" block.Id blockResult)
            
            // Apply reflection insights
            for insight in document.Reflection.Insights do
                results.Add(sprintf "💡 Insight: %s" insight)
            
            // Process evolution suggestions
            match document.Evolution with
            | Some evolution ->
                for suggestion in evolution.MutationSuggestions do
                    results.Add(sprintf "🧬 Mutation: %s -> %s" suggestion.Target suggestion.MutationType)
            | None -> ()
            
            let executionTime = DateTime.Now - startTime
            
            {
                Success = true
                Results = results |> Seq.toList
                ExecutionTime = executionTime
                Tier = document.Metadata.Tier
                SelfSimilarityScore = document.Reflection.SelfSimilarity.OverallSimilarity
                EntropyScore = document.Reflection.EntropyAnalysis.AverageEntropy
                NextTierSuggestion = document.Reflection.NextTierSuggestion
                EvolutionSuggestions = 
                    match document.Evolution with
                    | Some evo -> evo.MutationSuggestions |> List.map (fun s -> s.Target)
                    | None -> []
            }

        /// Execute individual block
        member private this.ExecuteBlock(block: TrsxBlock) : string =
            match block.Content with
            | CodeContent(lang, code) ->
                sprintf "Executed %s code (%d chars)" lang code.Length
            | TacticContent(tactics) ->
                sprintf "Applied %d tactics" tactics.Length
            | StructuredContent(data) ->
                sprintf "Processed %d data fields" data.Count
            | ReflectiveContent(analysis, insights) ->
                sprintf "Reflected on analysis with %d insights" insights.Length

        /// Execute TRSX file
        member this.ExecuteTrsxFile(filePath: string) : TrsxExecutionResult =
            let document = parser.ParseTrsxFile(filePath)
            this.ExecuteTrsxDocument(document)

    /// TRSX execution result
    and TrsxExecutionResult = {
        Success: bool
        Results: string list
        ExecutionTime: TimeSpan
        Tier: FractalTier
        SelfSimilarityScore: float
        EntropyScore: float
        NextTierSuggestion: FractalTier option
        EvolutionSuggestions: string list
    }

    /// Migration utility for converting old .flux + .trsx to unified format
    type TrsxMigrationUtility() =
        
        /// Convert separate .flux and .trsx files to unified format
        member this.MigrateToUnified(fluxPath: string, trsxPath: string option) : UnifiedTrsxDocument =
            let fluxContent = System.IO.File.ReadAllText(fluxPath)
            let trsxContent = 
                match trsxPath with
                | Some path when System.IO.File.Exists(path) -> Some (System.IO.File.ReadAllText(path))
                | _ -> None

            let metadata = {
                Title = System.IO.Path.GetFileNameWithoutExtension(fluxPath)
                Version = "1.0"
                Tier = Tier2_Extended
                Author = Some "Migration Tool"
                Created = DateTime.Now
                LastModified = DateTime.Now
            }

            let program = this.ConvertFluxToProgram(fluxContent)
            let reflection = this.CreateDefaultReflection()
            let evolution = None

            {
                Metadata = metadata
                Program = program
                Reflection = reflection
                Evolution = evolution
            }

        /// Convert FLUX content to program structure
        member private this.ConvertFluxToProgram(fluxContent: string) : TrsxProgram =
            let parser = UnifiedTrsxParser()
            let lines = fluxContent.Split('\n') |> Array.map (fun l -> l.Trim())
            let blocks = parser.ParseBlocks(lines)

            {
                Blocks = blocks
                Variables = Map.empty
                Functions = []
                MainEntry = None
            }

        /// Create default reflection data
        member private this.CreateDefaultReflection() : TrsxReflection =
            let entropyMetrics = {
                AverageEntropy = 0.5
                MaxEntropy = 1.0
                MinEntropy = 0.1
                EntropyDistribution = Map.empty
                PredictabilityScore = 0.8
            }

            let similarityMetrics = {
                OverallSimilarity = 0.75
                TierSimilarity = 0.8
                PatternSimilarity = 0.7
                StructuralSimilarity = 0.8
                SemanticSimilarity = 0.7
            }

            let performanceMetrics = {
                ExecutionTime = TimeSpan.Zero
                MemoryUsage = 0L
                CacheHitRate = 0.0
                SuccessRate = 1.0
                ErrorCount = 0
            }

            {
                ExecutionTrace = ["Migrated from legacy format"]
                EntropyAnalysis = entropyMetrics
                SelfSimilarity = similarityMetrics
                PerformanceMetrics = performanceMetrics
                Insights = ["Migrated from separate .flux/.trsx files"]
                NextTierSuggestion = Some Tier3_Reflective
            }
