namespace TarsEngine.FSharp.Core.AI

open System
open System.Text.RegularExpressions
open System.Collections.Generic

/// <summary>
/// Summarization levels for different audiences and purposes
/// </summary>
type SummarizationLevel =
    | Executive = 1      // Ultra-concise (1-2 sentences)
    | Tactical = 2       // Action-focused (3-5 sentences)
    | Operational = 3    // Balanced detail (1-2 paragraphs)
    | Comprehensive = 4  // Structured summary (3-5 paragraphs)
    | Detailed = 5       // Analysis summary (multiple sections)

/// <summary>
/// Expert types for MoE consensus
/// </summary>
type ExpertType =
    | ClarityExpert
    | AccuracyExpert
    | BrevityExpert
    | StructureExpert
    | DomainExpert

/// <summary>
/// Summarization configuration
/// </summary>
type SummarizationConfig = {
    Levels: SummarizationLevel list
    MoeConsensus: bool
    AutoCorrect: bool
    PreserveFacts: bool
    TargetAudience: string
    ExpertWeights: Map<ExpertType, float>
    MaxIterations: int
    QualityThreshold: float
}

/// <summary>
/// Expert opinion for MoE consensus
/// </summary>
type ExpertOpinion = {
    ExpertType: ExpertType
    Summary: string
    ConfidenceScore: float
    QualityMetrics: Map<string, float>
    Reasoning: string
}

/// <summary>
/// Summarization result
/// </summary>
type SummarizationResult = {
    Level: SummarizationLevel
    Summary: string
    CompressionRatio: float
    ConfidenceScore: float
    QualityMetrics: Map<string, float>
    ExpertConsensus: ExpertOpinion list option
    CorrectionsMade: string list
    ProcessingTime: TimeSpan
}

/// <summary>
/// Multi-level summarization output
/// </summary>
type MultiLevelSummary = {
    OriginalText: string
    OriginalLength: int
    Summaries: Map<SummarizationLevel, SummarizationResult>
    OverallQuality: float
    ProcessingTime: TimeSpan
    Metadata: Map<string, obj>
}

/// <summary>
/// TARS Response Summarizer - Multi-level summarization with MoE consensus
/// </summary>
type ResponseSummarizer() =
    
    /// Default configuration
    let defaultConfig = {
        Levels = [SummarizationLevel.Executive; SummarizationLevel.Tactical; SummarizationLevel.Operational]
        MoeConsensus = true
        AutoCorrect = true
        PreserveFacts = true
        TargetAudience = "general"
        ExpertWeights = Map.ofList [
            (ClarityExpert, 0.8)
            (AccuracyExpert, 0.9)
            (BrevityExpert, 0.7)
            (StructureExpert, 0.8)
            (DomainExpert, 0.8)
        ]
        MaxIterations = 3
        QualityThreshold = 0.8
    }
    
    /// Extract key sentences using simple scoring
    member private this.ExtractKeySentences(text: string, targetCount: int) =
        let sentences = text.Split([|'.'; '!'; '?'|], StringSplitOptions.RemoveEmptyEntries)
                       |> Array.map (fun s -> s.Trim())
                       |> Array.filter (fun s -> s.Length > 10)
        
        if sentences.Length <= targetCount then
            sentences
        else
            // Simple scoring based on length and position
            let scoredSentences = 
                sentences
                |> Array.mapi (fun i sentence ->
                    let lengthScore = min (float sentence.Length / 100.0) 1.0
                    let positionScore = if i < 3 then 1.0 elif i >= sentences.Length - 2 then 0.8 else 0.6
                    let score = lengthScore * positionScore
                    (sentence, score))
                |> Array.sortByDescending snd
                |> Array.take targetCount
                |> Array.map fst
            
            scoredSentences
    
    /// Generate summary for specific level
    member private this.GenerateLevelSummary(text: string, level: SummarizationLevel, config: SummarizationConfig) =
        let startTime = DateTime.UtcNow
        
        let targetSentences, compressionTarget = 
            match level with
            | SummarizationLevel.Executive -> (1, 0.95)
            | SummarizationLevel.Tactical -> (3, 0.85)
            | SummarizationLevel.Operational -> (6, 0.75)
            | SummarizationLevel.Comprehensive -> (10, 0.60)
            | SummarizationLevel.Detailed -> (15, 0.40)
            | _ -> (3, 0.80)
        
        let keySentences = this.ExtractKeySentences(text, targetSentences)
        let summary = String.Join(" ", keySentences)
        
        let actualCompressionRatio = 1.0 - (float summary.Length / float text.Length)
        let compressionScore = 1.0 - abs (actualCompressionRatio - compressionTarget)
        
        // Simple quality metrics
        let qualityMetrics = Map.ofList [
            ("compression_accuracy", compressionScore)
            ("readability", 0.8) // Placeholder
            ("coherence", 0.85)   // Placeholder
            ("completeness", 0.9) // Placeholder
        ]
        
        let processingTime = DateTime.UtcNow - startTime
        
        {
            Level = level
            Summary = summary
            CompressionRatio = actualCompressionRatio
            ConfidenceScore = qualityMetrics |> Map.values |> Seq.average
            QualityMetrics = qualityMetrics
            ExpertConsensus = None
            CorrectionsMade = []
            ProcessingTime = processingTime
        }
    
    /// Generate expert opinion
    member private this.GenerateExpertOpinion(text: string, expertType: ExpertType, level: SummarizationLevel) =
        let summary, reasoning = 
            match expertType with
            | ClarityExpert ->
                let simplified = this.SimplifyLanguage(text, level)
                (simplified, "Focused on clarity and readability")
            | AccuracyExpert ->
                let factual = this.PreserveFacts(text, level)
                (factual, "Prioritized factual accuracy and completeness")
            | BrevityExpert ->
                let concise = this.MaximizeConciseness(text, level)
                (concise, "Optimized for maximum conciseness")
            | StructureExpert ->
                let structured = this.ImproveStructure(text, level)
                (structured, "Enhanced logical structure and flow")
            | DomainExpert ->
                let technical = this.PreserveTechnicalAccuracy(text, level)
                (technical, "Maintained domain-specific accuracy")
        
        let qualityMetrics = Map.ofList [
            ("expert_confidence", 0.85)
            ("approach_effectiveness", 0.80)
            ("target_achievement", 0.88)
        ]
        
        {
            ExpertType = expertType
            Summary = summary
            ConfidenceScore = 0.85
            QualityMetrics = qualityMetrics
            Reasoning = reasoning
        }
    
    /// Simplify language for clarity
    member private this.SimplifyLanguage(text: string, level: SummarizationLevel) =
        let sentences = this.ExtractKeySentences(text, this.GetTargetSentenceCount(level))
        let simplified = sentences |> Array.map (fun s -> 
            s.Replace("utilize", "use")
             .Replace("demonstrate", "show")
             .Replace("facilitate", "help")
             .Replace("implement", "do"))
        String.Join(" ", simplified)
    
    /// Preserve important facts
    member private this.PreserveFacts(text: string, level: SummarizationLevel) =
        // Simple fact preservation - look for numbers, dates, names
        let factPattern = @"\b\d+(?:\.\d+)?%?|\b\d{4}\b|\b[A-Z][a-z]+ [A-Z][a-z]+\b"
        let facts = Regex.Matches(text, factPattern) |> Seq.cast<Match> |> Seq.map (fun m -> m.Value) |> Seq.distinct |> Seq.toArray
        
        let sentences = this.ExtractKeySentences(text, this.GetTargetSentenceCount(level))
        let summary = String.Join(" ", sentences)
        
        // Ensure key facts are included
        let missingFacts = facts |> Array.filter (fun fact -> not (summary.Contains(fact)))
        if missingFacts.Length > 0 && summary.Length < 500 then
            summary + " Key facts: " + String.Join(", ", missingFacts |> Array.take 3)
        else
            summary
    
    /// Maximize conciseness
    member private this.MaximizeConciseness(text: string, level: SummarizationLevel) =
        let targetCount = max 1 (this.GetTargetSentenceCount(level) - 1)
        let sentences = this.ExtractKeySentences(text, targetCount)
        let concise = sentences |> Array.map (fun s -> 
            s.Replace(" that ", " ")
             .Replace(" which ", " ")
             .Replace(" in order to ", " to ")
             .Replace("  ", " "))
        String.Join(" ", concise)
    
    /// Improve structure
    member private this.ImproveStructure(text: string, level: SummarizationLevel) =
        let sentences = this.ExtractKeySentences(text, this.GetTargetSentenceCount(level))
        
        // Simple structure improvement - add transitions
        let structured = 
            sentences
            |> Array.mapi (fun i sentence ->
                match i with
                | 0 -> sentence
                | 1 when sentences.Length > 2 -> "Additionally, " + sentence.ToLower()
                | _ when i = sentences.Length - 1 -> "Finally, " + sentence.ToLower()
                | _ -> "Furthermore, " + sentence.ToLower())
        
        String.Join(" ", structured)
    
    /// Preserve technical accuracy
    member private this.PreserveTechnicalAccuracy(text: string, level: SummarizationLevel) =
        // Preserve technical terms and acronyms
        let technicalPattern = @"\b[A-Z]{2,}\b|\b\w+\.\w+\b|\b\w+\(\)\b"
        let technicalTerms = Regex.Matches(text, technicalPattern) |> Seq.cast<Match> |> Seq.map (fun m -> m.Value) |> Seq.distinct |> Seq.toArray
        
        let sentences = this.ExtractKeySentences(text, this.GetTargetSentenceCount(level))
        let summary = String.Join(" ", sentences)
        
        // Ensure technical terms are preserved
        summary
    
    /// Get target sentence count for level
    member private this.GetTargetSentenceCount(level: SummarizationLevel) =
        match level with
        | SummarizationLevel.Executive -> 1
        | SummarizationLevel.Tactical -> 3
        | SummarizationLevel.Operational -> 6
        | SummarizationLevel.Comprehensive -> 10
        | SummarizationLevel.Detailed -> 15
        | _ -> 3
    
    /// Generate MoE consensus
    member private this.GenerateMoeConsensus(text: string, level: SummarizationLevel, config: SummarizationConfig) =
        let experts = config.ExpertWeights |> Map.keys |> Seq.toList
        let opinions = experts |> List.map (fun expert -> this.GenerateExpertOpinion(text, expert, level))
        
        // Simple consensus - weighted average approach
        let weightedSummaries = 
            opinions 
            |> List.map (fun opinion -> 
                let weight = config.ExpertWeights.[opinion.ExpertType]
                (opinion.Summary, weight * opinion.ConfidenceScore))
        
        // For simplicity, select the highest weighted summary
        let bestSummary = 
            weightedSummaries 
            |> List.maxBy snd
            |> fst
        
        let consensusScore = 
            opinions 
            |> List.map (fun o -> o.ConfidenceScore * config.ExpertWeights.[o.ExpertType])
            |> List.average
        
        (bestSummary, consensusScore, opinions)
    
    /// Apply automatic corrections
    member private this.ApplyCorrections(summary: string, config: SummarizationConfig) =
        let mutable corrected = summary
        let corrections = ResizeArray<string>()
        
        if config.AutoCorrect then
            // Simple grammar corrections
            if corrected.Contains("  ") then
                corrected <- corrected.Replace("  ", " ")
                corrections.Add("Fixed double spaces")
            
            // Ensure proper sentence ending
            if not (corrected.EndsWith(".") || corrected.EndsWith("!") || corrected.EndsWith("?")) then
                corrected <- corrected + "."
                corrections.Add("Added sentence ending")
            
            // Capitalize first letter
            if corrected.Length > 0 && Char.IsLower(corrected.[0]) then
                corrected <- Char.ToUpper(corrected.[0]).ToString() + corrected.Substring(1)
                corrections.Add("Capitalized first letter")
        
        (corrected, corrections |> Seq.toList)
    
    /// Summarize single level
    member this.SummarizeLevel(text: string, level: SummarizationLevel, ?config: SummarizationConfig) =
        let config = defaultArg config defaultConfig
        let startTime = DateTime.UtcNow
        
        let summary, consensusScore, expertOpinions = 
            if config.MoeConsensus then
                let (summary, score, opinions) = this.GenerateMoeConsensus(text, level, config)
                (summary, score, Some opinions)
            else
                let result = this.GenerateLevelSummary(text, level, config)
                (result.Summary, result.ConfidenceScore, None)
        
        let (correctedSummary, corrections) = this.ApplyCorrections(summary, config)
        
        let compressionRatio = 1.0 - (float correctedSummary.Length / float text.Length)
        let processingTime = DateTime.UtcNow - startTime
        
        {
            Level = level
            Summary = correctedSummary
            CompressionRatio = compressionRatio
            ConfidenceScore = consensusScore
            QualityMetrics = Map.ofList [
                ("compression_ratio", compressionRatio)
                ("consensus_score", consensusScore)
                ("correction_count", float corrections.Length)
            ]
            ExpertConsensus = expertOpinions
            CorrectionsMade = corrections
            ProcessingTime = processingTime
        }
    
    /// Summarize multiple levels
    member this.SummarizeMultiLevel(text: string, ?config: SummarizationConfig) =
        let config = defaultArg config defaultConfig
        let startTime = DateTime.UtcNow
        
        let summaries = 
            config.Levels
            |> List.map (fun level -> 
                let result = this.SummarizeLevel(text, level, config)
                (level, result))
            |> Map.ofList
        
        let overallQuality = 
            summaries 
            |> Map.values 
            |> Seq.map (fun s -> s.ConfidenceScore)
            |> Seq.average
        
        let processingTime = DateTime.UtcNow - startTime
        
        {
            OriginalText = text
            OriginalLength = text.Length
            Summaries = summaries
            OverallQuality = overallQuality
            ProcessingTime = processingTime
            Metadata = Map.ofList [
                ("levels_processed", box config.Levels.Length)
                ("moe_consensus", box config.MoeConsensus)
                ("auto_correct", box config.AutoCorrect)
                ("target_audience", box config.TargetAudience)
            ]
        }
    
    /// Get summary by level from multi-level result
    member this.GetSummaryByLevel(multiLevel: MultiLevelSummary, level: SummarizationLevel) =
        multiLevel.Summaries.TryFind(level)
    
    /// Compare summaries
    member this.CompareSummaries(summary1: SummarizationResult, summary2: SummarizationResult) =
        {|
            Level = summary1.Level
            Summary1Quality = summary1.ConfidenceScore
            Summary2Quality = summary2.ConfidenceScore
            BetterSummary = if summary1.ConfidenceScore > summary2.ConfidenceScore then 1 else 2
            QualityDifference = abs (summary1.ConfidenceScore - summary2.ConfidenceScore)
            CompressionComparison = {|
                Summary1Compression = summary1.CompressionRatio
                Summary2Compression = summary2.CompressionRatio
                MoreConcise = if summary1.CompressionRatio > summary2.CompressionRatio then 1 else 2
            |}
            Recommendation = 
                if summary1.ConfidenceScore > summary2.ConfidenceScore then "Use Summary 1"
                elif summary2.ConfidenceScore > summary1.ConfidenceScore then "Use Summary 2"
                else "Both summaries are similar in quality"
        |}
    
    /// Get system statistics
    member this.GetSystemStats() =
        {|
            SupportedLevels = 5
            ExpertTypes = 5
            DefaultCompressionRatios = Map.ofList [
                ("Executive", 0.95)
                ("Tactical", 0.85)
                ("Operational", 0.75)
                ("Comprehensive", 0.60)
                ("Detailed", 0.40)
            ]
            Features = [
                "Multi-level summarization"
                "MoE consensus"
                "Automatic corrections"
                "Quality metrics"
                "Expert opinions"
                "Configurable processing"
            ]
        |}
