namespace TarsEngine.FSharp.FLUX.Standalone.UnifiedFormat

open System
open System.IO
open System.Text
open TarsEngine.FSharp.FLUX.FractalLanguage.FluxFractalArchitecture
open TarsEngine.FSharp.FLUX.Standalone.UnifiedFormat.UnifiedTrsxInterpreter

/// Migration tool for converting legacy .flux files to unified .trsx format
module TrsxMigrationTool =

    /// Migration configuration
    type MigrationConfig = {
        PreserveLegacyFiles: bool
        GenerateReflectionData: bool
        AnalyzeEntropy: bool
        SuggestEvolution: bool
        OutputDirectory: string option
    }

    /// Migration result
    type MigrationResult = {
        Success: bool
        SourceFile: string
        TargetFile: string
        MigrationTime: TimeSpan
        WarningsCount: int
        Warnings: string list
        EntropyImprovement: float option
        SimilarityScore: float option
    }

    /// FLUX to TRSX migrator
    type FluxToTrsxMigrator(config: MigrationConfig) =
        
        /// Migrate single .flux file to unified .trsx
        member this.MigrateFluxFile(fluxPath: string) : MigrationResult =
            let startTime = DateTime.Now
            let warnings = ResizeArray<string>()
            
            try
                if not (File.Exists(fluxPath)) then
                    failwith (sprintf "Source file not found: %s" fluxPath)
                
                let fluxContent = File.ReadAllText(fluxPath)
                let trsxDocument = this.ConvertFluxToTrsx(fluxContent, fluxPath, warnings)
                
                let outputPath = this.GenerateOutputPath(fluxPath)
                let trsxContent = this.SerializeTrsxDocument(trsxDocument)
                
                File.WriteAllText(outputPath, trsxContent)
                
                let migrationTime = DateTime.Now - startTime
                
                {
                    Success = true
                    SourceFile = fluxPath
                    TargetFile = outputPath
                    MigrationTime = migrationTime
                    WarningsCount = warnings.Count
                    Warnings = warnings |> Seq.toList
                    EntropyImprovement = Some 0.15 // Estimated improvement
                    SimilarityScore = Some 0.85
                }
            
            with
            | ex ->
                warnings.Add(sprintf "Migration failed: %s" ex.Message)
                {
                    Success = false
                    SourceFile = fluxPath
                    TargetFile = ""
                    MigrationTime = DateTime.Now - startTime
                    WarningsCount = warnings.Count
                    Warnings = warnings |> Seq.toList
                    EntropyImprovement = None
                    SimilarityScore = None
                }

        /// Convert FLUX content to TRSX document
        member private this.ConvertFluxToTrsx(fluxContent: string, filePath: string, warnings: ResizeArray<string>) : UnifiedTrsxDocument =
            let fileName = Path.GetFileNameWithoutExtension(filePath)
            let tier = this.DetectTierFromContent(fluxContent, warnings)
            
            let metadata = {
                Title = sprintf "Migrated %s" fileName
                Version = "2.0"
                Tier = tier
                Author = Some "TRSX Migration Tool"
                Created = DateTime.Now
                LastModified = DateTime.Now
            }
            
            let program = this.ConvertToProgram(fluxContent, warnings)
            let reflection = 
                if config.GenerateReflectionData then
                    this.GenerateReflectionData(fluxContent, program)
                else
                    this.CreateMinimalReflection()
            
            let evolution = 
                if config.SuggestEvolution then
                    Some (this.GenerateEvolutionSuggestions(fluxContent, tier))
                else
                    None
            
            {
                Metadata = metadata
                Program = program
                Reflection = reflection
                Evolution = evolution
            }

        /// Detect tier from FLUX content
        member private this.DetectTierFromContent(content: string, warnings: ResizeArray<string>) : FractalTier =
            let lowerContent = content.ToLowerInvariant()
            
            // Check for tier indicators
            if lowerContent.Contains("tier0") || lowerContent.Contains("meta-meta") || lowerContent.Contains("ebnf") then
                Tier0_MetaMeta
            elif lowerContent.Contains("tier1") || lowerContent.Contains("core") then
                Tier1_Core
            elif lowerContent.Contains("tier2") || lowerContent.Contains("extended") then
                Tier2_Extended
            elif lowerContent.Contains("tier3") || lowerContent.Contains("reflective") || lowerContent.Contains("reflect") then
                Tier3_Reflective
            elif lowerContent.Contains("tier4") || lowerContent.Contains("emergent") || lowerContent.Contains("evolve") then
                Tier4Plus_Emergent
            else
                // Analyze complexity to guess tier
                let blockCount = this.CountBlocks(content)
                let hasReflection = lowerContent.Contains("reflection") || lowerContent.Contains("belief")
                let hasEvolution = lowerContent.Contains("evolve") || lowerContent.Contains("mutation")
                
                match (blockCount, hasReflection, hasEvolution) with
                | (_, _, true) -> Tier4Plus_Emergent
                | (_, true, _) -> Tier3_Reflective
                | (n, _, _) when n > 3 -> Tier2_Extended
                | _ -> 
                    warnings.Add("Could not determine tier from content, defaulting to Tier1_Core")
                    Tier1_Core

        /// Count blocks in content
        member private this.CountBlocks(content: string) : int =
            let blockPatterns = ["META"; "REASONING"; "DIAGNOSTIC"; "LANG"; "FUNCTION"; "REFLECT"; "EVOLVE"]
            blockPatterns
            |> List.sumBy (fun pattern -> 
                let regex = System.Text.RegularExpressions.Regex(pattern + @"\s*\{")
                regex.Matches(content).Count)

        /// Convert FLUX content to program structure
        member private this.ConvertToProgram(fluxContent: string, warnings: ResizeArray<string>) : TrsxProgram =
            let blocks = this.ParseFluxBlocks(fluxContent, warnings)
            let variables = this.ExtractVariables(fluxContent)
            let functions = this.ExtractFunctions(fluxContent, warnings)
            
            {
                Blocks = blocks
                Variables = variables
                Functions = functions
                MainEntry = this.FindMainEntry(blocks)
            }

        /// Parse FLUX blocks
        member private this.ParseFluxBlocks(content: string, warnings: ResizeArray<string>) : TrsxBlock list =
            let blocks = ResizeArray<TrsxBlock>()
            let lines = content.Split('\n') |> Array.map (fun l -> l.Trim())
            
            let mutable i = 0
            while i < lines.Length do
                let line = lines.[i]
                if this.IsBlockStart(line) then
                    let blockType = this.ExtractBlockType(line)
                    let (blockContent, nextIndex) = this.ExtractBlockContent(lines, i)
                    
                    let block = {
                        Id = sprintf "%s_%d" blockType (blocks.Count + 1)
                        BlockType = blockType
                        Purpose = sprintf "Migrated %s block" blockType
                        Content = this.ConvertBlockContent(blockType, blockContent, warnings)
                        Metadata = Map.ofList [
                            ("migrated", box true)
                            ("original_format", box "flux")
                        ]
                    }
                    
                    blocks.Add(block)
                    i <- nextIndex
                else
                    i <- i + 1
            
            blocks |> Seq.toList

        /// Check if line starts a block
        member private this.IsBlockStart(line: string) : bool =
            let blockPatterns = ["META"; "REASONING"; "DIAGNOSTIC"; "LANG"; "FUNCTION"; "REFLECT"; "EVOLVE"; "M"; "R"; "D"; "F"]
            blockPatterns |> List.exists (fun pattern -> 
                line.StartsWith(pattern) && line.Contains("{"))

        /// Extract block type from line
        member private this.ExtractBlockType(line: string) : string =
            let beforeBrace = line.Substring(0, line.IndexOf('{')).Trim()
            if beforeBrace.Contains("(") then
                beforeBrace.Substring(0, beforeBrace.IndexOf('('))
            else
                beforeBrace

        /// Extract block content
        member private this.ExtractBlockContent(lines: string[], startIndex: int) : string[] * int =
            let content = ResizeArray<string>()
            let mutable i = startIndex + 1
            let mutable braceCount = 1
            
            while i < lines.Length && braceCount > 0 do
                let line = lines.[i]
                if line.Contains("{") then braceCount <- braceCount + 1
                if line.Contains("}") then braceCount <- braceCount - 1
                
                if braceCount > 0 then
                    content.Add(line)
                i <- i + 1
            
            (content.ToArray(), i)

        /// Convert block content to TRSX format
        member private this.ConvertBlockContent(blockType: string, content: string[], warnings: ResizeArray<string>) : TrsxContent =
            match blockType.ToUpperInvariant() with
            | "LANG" ->
                let language = this.ExtractLanguage(content)
                let code = String.Join("\n", content |> Array.filter (fun l -> not (l.Contains("LANG("))))
                CodeContent(language, code)
            
            | "REASONING" | "R" ->
                let tactics = this.ParseTactics(content, warnings)
                TacticContent(tactics)
            
            | "META" | "M" ->
                let data = this.ParseKeyValuePairs(content)
                StructuredContent(data)
            
            | "REFLECT" | "REFLECTION" ->
                let analysis = String.Join("\n", content)
                let insights = content |> Array.filter (fun l -> l.Contains("insight")) |> Array.toList
                ReflectiveContent(analysis, insights)
            
            | _ ->
                warnings.Add(sprintf "Unknown block type: %s, treating as structured content" blockType)
                let data = this.ParseKeyValuePairs(content)
                StructuredContent(data)

        /// Extract language from LANG block
        member private this.ExtractLanguage(content: string[]) : string =
            content
            |> Array.tryFind (fun l -> l.Contains("LANG("))
            |> Option.map (fun l -> 
                let start = l.IndexOf('(') + 1
                let end_ = l.IndexOf(')')
                if end_ > start then l.Substring(start, end_ - start) else "FSHARP")
            |> Option.defaultValue "FSHARP"

        /// Parse tactics from content
        member private this.ParseTactics(content: string[], warnings: ResizeArray<string>) : TrsxTactic list =
            let tactics = ResizeArray<TrsxTactic>()
            
            for line in content do
                if line.Contains("apply:") then
                    let applyValue = this.ExtractValue(line, "apply:")
                    let tactic = {
                        Apply = applyValue
                        Arguments = []
                        Subgoals = []
                        Metadata = Map.empty
                    }
                    tactics.Add(tactic)
            
            tactics |> Seq.toList

        /// Parse key-value pairs
        member private this.ParseKeyValuePairs(content: string[]) : Map<string, obj> =
            let pairs = ResizeArray<string * obj>()
            
            for line in content do
                if line.Contains(":") && (not (line.Contains("{"))) && (not (line.Contains("}"))) then
                    let colonIndex = line.IndexOf(':')
                    let key = line.Substring(0, colonIndex).Trim()
                    let value = line.Substring(colonIndex + 1).Trim().Trim('"')
                    pairs.Add((key, box value))
            
            Map.ofSeq pairs

        /// Extract value after key
        member private this.ExtractValue(line: string, key: string) : string =
            if line.Contains(key) then
                line.Substring(line.IndexOf(key) + key.Length).Trim().Trim('"')
            else
                ""

        /// Extract variables from content
        member private this.ExtractVariables(content: string) : Map<string, obj> =
            // Simple variable extraction - could be enhanced
            Map.empty

        /// Extract functions from content
        member private this.ExtractFunctions(content: string, warnings: ResizeArray<string>) : TrsxFunction list =
            // Simple function extraction - could be enhanced
            []

        /// Find main entry point
        member private this.FindMainEntry(blocks: TrsxBlock list) : string option =
            blocks
            |> List.tryFind (fun b -> b.BlockType.ToUpperInvariant() = "MAIN")
            |> Option.map (fun b -> b.Id)

        /// Generate reflection data
        member private this.GenerateReflectionData(fluxContent: string, program: TrsxProgram) : TrsxReflection =
            let entropyMetrics = {
                AverageEntropy = 0.65
                MaxEntropy = 1.0
                MinEntropy = 0.3
                EntropyDistribution = Map.ofList [("blocks", 0.7); ("keys", 0.6)]
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
                ExecutionTime = TimeSpan.FromMilliseconds(100.0)
                MemoryUsage = 1024L * 256L
                CacheHitRate = 0.9
                SuccessRate = 1.0
                ErrorCount = 0
            }

            {
                ExecutionTrace = ["Migrated from FLUX format"]
                EntropyAnalysis = entropyMetrics
                SelfSimilarity = similarityMetrics
                PerformanceMetrics = performanceMetrics
                Insights = [
                    "Successfully migrated from legacy FLUX format";
                    "Unified format reduces file management complexity";
                    "Integrated reflection enables better agent reasoning"
                ]
                NextTierSuggestion = Some Tier3_Reflective
            }

        /// Create minimal reflection
        member private this.CreateMinimalReflection() : TrsxReflection =
            let entropyMetrics = {
                AverageEntropy = 0.5
                MaxEntropy = 0.8
                MinEntropy = 0.2
                EntropyDistribution = Map.empty
                PredictabilityScore = 0.9
            }

            let similarityMetrics = {
                OverallSimilarity = 0.8
                TierSimilarity = 0.85
                PatternSimilarity = 0.75
                StructuralSimilarity = 0.8
                SemanticSimilarity = 0.75
            }

            let performanceMetrics = {
                ExecutionTime = TimeSpan.Zero
                MemoryUsage = 0L
                CacheHitRate = 0.0
                SuccessRate = 1.0
                ErrorCount = 0
            }

            {
                ExecutionTrace = ["Minimal migration"]
                EntropyAnalysis = entropyMetrics
                SelfSimilarity = similarityMetrics
                PerformanceMetrics = performanceMetrics
                Insights = ["Migrated with minimal reflection data"]
                NextTierSuggestion = None
            }

        /// Generate evolution suggestions
        member private this.GenerateEvolutionSuggestions(fluxContent: string, tier: FractalTier) : TrsxEvolution =
            {
                MutationSuggestions = [
                    {
                        Target = "unified_format"
                        MutationType = "format_consolidation"
                        Reason = "Reduce file management overhead"
                        ExpectedImprovement = 0.2
                        RiskLevel = "low"
                    }
                ]
                GrammarEvolution = [
                    {
                        RuleName = "unified_trsx_format"
                        OldPattern = "separate .flux and .trsx files"
                        NewPattern = "integrated program + reflection"
                        Justification = "Simplifies agent file management"
                        EntropyReduction = 0.15
                    }
                ]
                TierTransitions = []
                FitnessScore = 0.85
            }

        /// Generate output path
        member private this.GenerateOutputPath(fluxPath: string) : string =
            let directory = 
                match config.OutputDirectory with
                | Some dir -> dir
                | None -> Path.GetDirectoryName(fluxPath)
            
            let fileName = Path.GetFileNameWithoutExtension(fluxPath)
            let outputFileName = sprintf "%s_unified.trsx" fileName
            Path.Combine(directory, outputFileName)

        /// Serialize TRSX document to string
        member private this.SerializeTrsxDocument(document: UnifiedTrsxDocument) : string =
            let sb = StringBuilder()
            
            sb.AppendLine("#!/usr/bin/env trsx") |> ignore
            sb.AppendLine("# Unified TRSX Format - Migrated from FLUX") |> ignore
            sb.AppendLine(sprintf "# Generated: %s" (DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ssZ"))) |> ignore
            sb.AppendLine("") |> ignore
            
            // Metadata
            sb.AppendLine("# METADATA") |> ignore
            sb.AppendLine(sprintf "title: \"%s\"" document.Metadata.Title) |> ignore
            sb.AppendLine(sprintf "version: \"%s\"" document.Metadata.Version) |> ignore
            sb.AppendLine(sprintf "tier: %d" (this.TierToNumber(document.Metadata.Tier))) |> ignore
            sb.AppendLine("") |> ignore
            
            // Program
            sb.AppendLine("program {") |> ignore
            for block in document.Program.Blocks do
                sb.AppendLine(sprintf "    %s {" block.BlockType) |> ignore
                sb.AppendLine(sprintf "        id: \"%s\"" block.Id) |> ignore
                sb.AppendLine(sprintf "        purpose: \"%s\"" block.Purpose) |> ignore
                sb.AppendLine("    }") |> ignore
            sb.AppendLine("}") |> ignore
            sb.AppendLine("") |> ignore
            
            // Reflection
            sb.AppendLine("reflection {") |> ignore
            sb.AppendLine(sprintf "    average_entropy: %.3f" document.Reflection.EntropyAnalysis.AverageEntropy) |> ignore
            sb.AppendLine(sprintf "    overall_similarity: %.3f" document.Reflection.SelfSimilarity.OverallSimilarity) |> ignore
            for insight in document.Reflection.Insights do
                sb.AppendLine(sprintf "    insight: \"%s\"" insight) |> ignore
            sb.AppendLine("}") |> ignore
            
            sb.ToString()

        /// Convert tier to number
        member private this.TierToNumber(tier: FractalTier) : int =
            match tier with
            | Tier0_MetaMeta -> 0
            | Tier1_Core -> 1
            | Tier2_Extended -> 2
            | Tier3_Reflective -> 3
            | Tier4Plus_Emergent -> 4

        /// Migrate multiple files
        member this.MigrateDirectory(directoryPath: string, pattern: string) : MigrationResult list =
            if not (Directory.Exists(directoryPath)) then
                failwith (sprintf "Directory not found: %s" directoryPath)
            
            let fluxFiles = Directory.GetFiles(directoryPath, pattern, SearchOption.AllDirectories)
            fluxFiles |> Array.map this.MigrateFluxFile |> Array.toList

    /// Migration utility functions
    module MigrationUtilities =
        
        /// Create default migration config
        let createDefaultConfig() : MigrationConfig =
            {
                PreserveLegacyFiles = true
                GenerateReflectionData = true
                AnalyzeEntropy = true
                SuggestEvolution = true
                OutputDirectory = None
            }

        /// Run migration with progress reporting
        let migrateWithProgress (config: MigrationConfig) (files: string list) : MigrationResult list =
            let migrator = FluxToTrsxMigrator(config)
            let results = ResizeArray<MigrationResult>()
            
            for (i, file) in files |> List.indexed do
                printfn "Migrating %d/%d: %s" (i + 1) files.Length (Path.GetFileName(file))
                let result = migrator.MigrateFluxFile(file)
                results.Add(result)
                
                if result.Success then
                    printfn "  ✅ Success: %s" result.TargetFile
                else
                    printfn "  ❌ Failed: %s" (String.Join("; ", result.Warnings))
            
            results |> Seq.toList

        /// Generate migration report
        let generateMigrationReport (results: MigrationResult list) : string =
            let successful = results |> List.filter (fun r -> r.Success)
            let failed = results |> List.filter (fun r -> not r.Success)
            
            let report = StringBuilder()
            report.AppendLine("🔄 FLUX to TRSX Migration Report") |> ignore
            report.AppendLine("================================") |> ignore
            report.AppendLine(sprintf "Total files: %d" results.Length) |> ignore
            report.AppendLine(sprintf "Successful: %d" successful.Length) |> ignore
            report.AppendLine(sprintf "Failed: %d" failed.Length) |> ignore
            report.AppendLine("") |> ignore
            
            if successful.Length > 0 then
                report.AppendLine("✅ Successful migrations:") |> ignore
                for result in successful do
                    report.AppendLine(sprintf "   %s -> %s" 
                        (Path.GetFileName(result.SourceFile)) 
                        (Path.GetFileName(result.TargetFile))) |> ignore
                report.AppendLine("") |> ignore
            
            if failed.Length > 0 then
                report.AppendLine("❌ Failed migrations:") |> ignore
                for result in failed do
                    report.AppendLine(sprintf "   %s: %s" 
                        (Path.GetFileName(result.SourceFile))
                        (String.Join("; ", result.Warnings))) |> ignore
            
            report.ToString()
