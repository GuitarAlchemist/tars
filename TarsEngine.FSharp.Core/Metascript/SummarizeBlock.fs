namespace TarsEngine.FSharp.Core.Metascript

open System
open System.Collections.Generic
open TarsEngine.FSharp.Core.AI

/// <summary>
/// SUMMARIZE block configuration from metascript DSL
/// </summary>
type SummarizeBlockConfig = {
    Source: string
    Sources: string list option
    Levels: SummarizationLevel list
    Output: string
    MoeConsensus: bool
    AutoCorrect: bool
    PreserveFacts: bool
    TargetAudience: string
    ExpertWeights: Map<ExpertType, float>
    MaxIterations: int
    QualityThreshold: float
    ConditionalLevels: Map<string, SummarizationLevel list> option
    MergeStrategy: string option
    ConflictResolution: string option
    IterativeProcess: bool
    ImprovementThreshold: float
    ConvergenceCriteria: string
    OutputFormat: SummarizeOutputFormat
}

and SummarizeOutputFormat = {
    Structure: string
    IncludeConfidence: bool
    ShowCompressionRatio: bool
    IncludeExpertOpinions: bool
    IncludeCorrections: bool
}

/// <summary>
/// Parsed SUMMARIZE block from metascript
/// </summary>
type SummarizeBlock = {
    Config: SummarizeBlockConfig
    Variables: Map<string, obj>
    Conditions: Map<string, string> option
}

/// <summary>
/// SUMMARIZE block execution result
/// </summary>
type SummarizeBlockResult = {
    Success: bool
    Output: MultiLevelSummary option
    OutputVariable: string
    ProcessingTime: TimeSpan
    Errors: string list
    Warnings: string list
    Metadata: Map<string, obj>
}

/// <summary>
/// SUMMARIZE DSL Block Parser and Executor
/// Handles the new SUMMARIZE block in TARS metascripts
/// </summary>
type SummarizeBlockParser() =
    
    let summarizer = ResponseSummarizer()
    
    /// Parse expert type from string
    member private this.ParseExpertType(expertStr: string) =
        match expertStr.ToLower().Replace("_", "").Replace("-", "") with
        | "clarity" | "clarityexpert" -> Some ClarityExpert
        | "accuracy" | "accuracyexpert" -> Some AccuracyExpert
        | "brevity" | "brevityexpert" -> Some BrevityExpert
        | "structure" | "structureexpert" -> Some StructureExpert
        | "domain" | "domainexpert" -> Some DomainExpert
        | _ -> None
    
    /// Parse summarization level from string or number
    member private this.ParseSummarizationLevel(levelStr: string) =
        match levelStr.ToLower() with
        | "1" | "executive" -> Some SummarizationLevel.Executive
        | "2" | "tactical" -> Some SummarizationLevel.Tactical
        | "3" | "operational" -> Some SummarizationLevel.Operational
        | "4" | "comprehensive" -> Some SummarizationLevel.Comprehensive
        | "5" | "detailed" -> Some SummarizationLevel.Detailed
        | _ -> None
    
    /// Parse levels list from various formats
    member private this.ParseLevels(levelsInput: obj) =
        match levelsInput with
        | :? string as str ->
            str.Split([|','; ';'; ' '|], StringSplitOptions.RemoveEmptyEntries)
            |> Array.choose this.ParseSummarizationLevel
            |> Array.toList
        | :? (int list) as intList ->
            intList 
            |> List.choose (fun i -> this.ParseSummarizationLevel(i.ToString()))
        | :? (string list) as strList ->
            strList 
            |> List.choose this.ParseSummarizationLevel
        | _ -> [SummarizationLevel.Executive; SummarizationLevel.Tactical; SummarizationLevel.Operational]
    
    /// Parse expert weights from configuration
    member private this.ParseExpertWeights(expertsConfig: Map<string, obj>) =
        expertsConfig
        |> Map.toSeq
        |> Seq.choose (fun (expertStr, weightObj) ->
            match this.ParseExpertType(expertStr), weightObj with
            | Some expert, (:? float as weight) -> Some (expert, weight)
            | Some expert, (:? int as weight) -> Some (expert, float weight)
            | Some expert, (:? string as weightStr) ->
                match Double.TryParse(weightStr) with
                | true, weight -> Some (expert, weight)
                | false, _ -> None
            | _ -> None)
        |> Map.ofSeq
    
    /// Parse conditional levels
    member private this.ParseConditionalLevels(conditionsConfig: Map<string, obj>) =
        conditionsConfig
        |> Map.map (fun condition levelsObj -> this.ParseLevels(levelsObj))
    
    /// Parse output format configuration
    member private this.ParseOutputFormat(formatConfig: Map<string, obj>) =
        {
            Structure = formatConfig.TryFind("structure") |> Option.map string |> Option.defaultValue "hierarchical"
            IncludeConfidence = formatConfig.TryFind("include_confidence") |> Option.map (fun v -> v :?> bool) |> Option.defaultValue true
            ShowCompressionRatio = formatConfig.TryFind("show_compression_ratio") |> Option.map (fun v -> v :?> bool) |> Option.defaultValue true
            IncludeExpertOpinions = formatConfig.TryFind("include_expert_opinions") |> Option.map (fun v -> v :?> bool) |> Option.defaultValue false
            IncludeCorrections = formatConfig.TryFind("include_corrections") |> Option.map (fun v -> v :?> bool) |> Option.defaultValue false
        }
    
    /// Parse SUMMARIZE block from metascript configuration
    member this.ParseSummarizeBlock(blockConfig: Map<string, obj>, variables: Map<string, obj>) =
        try
            let source = blockConfig.TryFind("source") |> Option.map string |> Option.defaultValue ""
            let sources = blockConfig.TryFind("sources") |> Option.map (fun v -> v :?> string list)
            let output = blockConfig.TryFind("output") |> Option.map string |> Option.defaultValue "summary_result"
            
            let levels = 
                blockConfig.TryFind("levels") 
                |> Option.map this.ParseLevels
                |> Option.defaultValue [SummarizationLevel.Executive; SummarizationLevel.Tactical]
            
            let configuration = blockConfig.TryFind("configuration") |> Option.map (fun v -> v :?> Map<string, obj>) |> Option.defaultValue Map.empty
            let expertsConfig = blockConfig.TryFind("experts") |> Option.map (fun v -> v :?> Map<string, obj>) |> Option.defaultValue Map.empty
            let correctionsConfig = blockConfig.TryFind("corrections") |> Option.map (fun v -> v :?> Map<string, obj>) |> Option.defaultValue Map.empty
            let outputFormatConfig = blockConfig.TryFind("output_format") |> Option.map (fun v -> v :?> Map<string, obj>) |> Option.defaultValue Map.empty
            
            let conditionalLevels = 
                blockConfig.TryFind("conditional_levels") 
                |> Option.map (fun v -> this.ParseConditionalLevels(v :?> Map<string, obj>))
            
            let config = {
                Source = source
                Sources = sources
                Levels = levels
                Output = output
                MoeConsensus = configuration.TryFind("moe_consensus") |> Option.map (fun v -> v :?> bool) |> Option.defaultValue true
                AutoCorrect = correctionsConfig.TryFind("auto_correct") |> Option.map (fun v -> v :?> bool) |> Option.defaultValue true
                PreserveFacts = configuration.TryFind("preserve_facts") |> Option.map (fun v -> v :?> bool) |> Option.defaultValue true
                TargetAudience = configuration.TryFind("target_audience") |> Option.map string |> Option.defaultValue "general"
                ExpertWeights = this.ParseExpertWeights(expertsConfig)
                MaxIterations = configuration.TryFind("max_iterations") |> Option.map (fun v -> v :?> int) |> Option.defaultValue 3
                QualityThreshold = configuration.TryFind("quality_threshold") |> Option.map (fun v -> v :?> float) |> Option.defaultValue 0.8
                ConditionalLevels = conditionalLevels
                MergeStrategy = blockConfig.TryFind("merge_strategy") |> Option.map string
                ConflictResolution = blockConfig.TryFind("conflict_resolution") |> Option.map string
                IterativeProcess = blockConfig.TryFind("iterative_process") |> Option.map (fun v -> v :?> bool) |> Option.defaultValue false
                ImprovementThreshold = configuration.TryFind("improvement_threshold") |> Option.map (fun v -> v :?> float) |> Option.defaultValue 0.1
                ConvergenceCriteria = configuration.TryFind("convergence_criteria") |> Option.map string |> Option.defaultValue "quality_threshold"
                OutputFormat = this.ParseOutputFormat(outputFormatConfig)
            }
            
            let conditions = blockConfig.TryFind("conditions") |> Option.map (fun v -> v :?> Map<string, string>)
            
            Some {
                Config = config
                Variables = variables
                Conditions = conditions
            }
        with
        | ex -> 
            None
    
    /// Resolve source text from variables
    member private this.ResolveSourceText(source: string, variables: Map<string, obj>) =
        match variables.TryFind(source) with
        | Some value -> string value
        | None -> source // Treat as literal text if not found in variables
    
    /// Resolve multiple source texts
    member private this.ResolveSourceTexts(sources: string list, variables: Map<string, obj>) =
        sources |> List.map (fun source -> this.ResolveSourceText(source, variables))
    
    /// Evaluate conditional levels
    member private this.EvaluateConditionalLevels(text: string, conditionalLevels: Map<string, SummarizationLevel list>) =
        let textLength = text.Length
        
        // Simple condition evaluation (in real implementation, this would be more sophisticated)
        let matchedCondition = 
            conditionalLevels
            |> Map.tryPick (fun condition levels ->
                match condition with
                | c when c.Contains("length_gt_1000") && textLength > 1000 -> Some levels
                | c when c.Contains("length_gt_500") && textLength > 500 -> Some levels
                | "else" -> Some levels
                | _ -> None)
        
        matchedCondition |> Option.defaultValue [SummarizationLevel.Executive]
    
    /// Merge multiple summaries using specified strategy
    member private this.MergeSummaries(summaries: MultiLevelSummary list, strategy: string) =
        match strategy.ToLower() with
        | "consensus_synthesis" ->
            // Simple consensus - combine summaries at each level
            let firstSummary = summaries.Head
            let mergedSummaries = 
                firstSummary.Summaries
                |> Map.map (fun level _ ->
                    let levelSummaries = summaries |> List.choose (fun s -> s.Summaries.TryFind(level))
                    if levelSummaries.Length > 1 then
                        let combinedText = levelSummaries |> List.map (fun ls -> ls.Summary) |> String.concat " "
                        let avgConfidence = levelSummaries |> List.map (fun ls -> ls.ConfidenceScore) |> List.average
                        { levelSummaries.Head with Summary = combinedText; ConfidenceScore = avgConfidence }
                    else
                        levelSummaries.Head)
            
            { firstSummary with Summaries = mergedSummaries }
        
        | "comprehensive_synthesis" ->
            // Take the most comprehensive summary
            summaries |> List.maxBy (fun s -> s.OverallQuality)
        
        | _ ->
            // Default: take first summary
            summaries.Head
    
    /// Execute SUMMARIZE block
    member this.ExecuteSummarizeBlock(block: SummarizeBlock) =
        let startTime = DateTime.UtcNow
        let errors = ResizeArray<string>()
        let warnings = ResizeArray<string>()
        
        try
            // Resolve source text(s)
            let sourceTexts = 
                match block.Config.Sources with
                | Some sources -> this.ResolveSourceTexts(sources, block.Variables)
                | None -> [this.ResolveSourceText(block.Config.Source, block.Variables)]
            
            if sourceTexts |> List.exists String.IsNullOrWhiteSpace then
                errors.Add("Source text is empty or not found")
                {
                    Success = false
                    Output = None
                    OutputVariable = block.Config.Output
                    ProcessingTime = DateTime.UtcNow - startTime
                    Errors = errors |> Seq.toList
                    Warnings = warnings |> Seq.toList
                    Metadata = Map.empty
                }
            else
                // Determine levels to use
                let levelsToUse = 
                    match block.Config.ConditionalLevels with
                    | Some conditionalLevels -> 
                        let combinedText = String.Join(" ", sourceTexts)
                        this.EvaluateConditionalLevels(combinedText, conditionalLevels)
                    | None -> block.Config.Levels
                
                // Create summarization configuration
                let summarizerConfig = {
                    Levels = levelsToUse
                    MoeConsensus = block.Config.MoeConsensus
                    AutoCorrect = block.Config.AutoCorrect
                    PreserveFacts = block.Config.PreserveFacts
                    TargetAudience = block.Config.TargetAudience
                    ExpertWeights = 
                        if block.Config.ExpertWeights.IsEmpty then
                            Map.ofList [
                                (ClarityExpert, 0.8)
                                (AccuracyExpert, 0.9)
                                (BrevityExpert, 0.7)
                                (StructureExpert, 0.8)
                                (DomainExpert, 0.8)
                            ]
                        else
                            block.Config.ExpertWeights
                    MaxIterations = block.Config.MaxIterations
                    QualityThreshold = block.Config.QualityThreshold
                }
                
                // Process summaries
                let summaries = 
                    sourceTexts 
                    |> List.map (fun text -> summarizer.SummarizeMultiLevel(text, summarizerConfig))
                
                // Merge if multiple sources
                let finalSummary = 
                    match summaries with
                    | [single] -> single
                    | multiple -> 
                        let strategy = block.Config.MergeStrategy |> Option.defaultValue "consensus_synthesis"
                        this.MergeSummaries(multiple, strategy)
                
                let processingTime = DateTime.UtcNow - startTime
                
                {
                    Success = true
                    Output = Some finalSummary
                    OutputVariable = block.Config.Output
                    ProcessingTime = processingTime
                    Errors = errors |> Seq.toList
                    Warnings = warnings |> Seq.toList
                    Metadata = Map.ofList [
                        ("levels_processed", box levelsToUse.Length)
                        ("sources_processed", box sourceTexts.Length)
                        ("moe_consensus", box block.Config.MoeConsensus)
                        ("target_audience", box block.Config.TargetAudience)
                        ("overall_quality", box finalSummary.OverallQuality)
                    ]
                }
        with
        | ex ->
            errors.Add($"Execution error: {ex.Message}")
            {
                Success = false
                Output = None
                OutputVariable = block.Config.Output
                ProcessingTime = DateTime.UtcNow - startTime
                Errors = errors |> Seq.toList
                Warnings = warnings |> Seq.toList
                Metadata = Map.ofList [("error", box ex.Message)]
            }
    
    /// Format output according to configuration
    member this.FormatOutput(result: SummarizationResult, format: SummarizeOutputFormat) =
        let parts = ResizeArray<string>()
        
        parts.Add($"Summary: {result.Summary}")
        
        if format.ShowCompressionRatio then
            parts.Add($"Compression: {result.CompressionRatio:P1}")
        
        if format.IncludeConfidence then
            parts.Add($"Confidence: {result.ConfidenceScore:P1}")
        
        if format.IncludeCorrections && result.CorrectionsMade.Length > 0 then
            parts.Add($"Corrections: {String.Join(", ", result.CorrectionsMade)}")
        
        if format.IncludeExpertOpinions && result.ExpertConsensus.IsSome then
            let opinions = result.ExpertConsensus.Value
            let expertSummary = opinions |> List.map (fun o -> $"{o.ExpertType}: {o.ConfidenceScore:P1}") |> String.concat(", ")
            parts.Add($"Expert Consensus: {expertSummary}")
        
        String.Join("\n", parts)
    
    /// Get supported DSL syntax help
    member this.GetSyntaxHelp() =
        """
SUMMARIZE Block Syntax:

Basic Usage:
SUMMARIZE:
  source: "response_variable"
  levels: [1, 2, 3]
  output: "summary_result"

Advanced Configuration:
SUMMARIZE:
  source: "llm_response"
  levels: ["executive", "tactical", "operational"]
  output: "multi_level_summary"
  
  CONFIGURATION:
    moe_consensus: true
    auto_correct: true
    preserve_facts: true
    target_audience: "technical"
  
  EXPERTS:
    clarity_expert: 0.8
    accuracy_expert: 0.9
    brevity_expert: 0.7
  
  OUTPUT_FORMAT:
    structure: "hierarchical"
    include_confidence: true
    show_compression_ratio: true

Multi-Source:
SUMMARIZE:
  sources: ["response_1", "response_2", "response_3"]
  merge_strategy: "consensus_synthesis"
  levels: [1, 2]

Conditional Levels:
SUMMARIZE:
  source: "variable_response"
  CONDITIONAL_LEVELS:
    if_length_gt_1000: [1, 2, 3]
    if_length_gt_500: [1, 2]
    else: [1]
        """
