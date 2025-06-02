namespace TarsEngine.FSharp.Core.AI

open System
open System.IO
open System.Text.Json
open System.Collections.Generic
open System.Text.RegularExpressions

/// <summary>
/// Prompt improvement strategies
/// </summary>
type PromptImprovementStrategy =
    | ClarityEnhancement
    | ContextEnrichment
    | ExampleAddition
    | ConstraintSpecification
    | FormatStandardization
    | PerformanceOptimization
    | ErrorReduction
    | UserExperienceImprovement

/// <summary>
/// Prompt analysis results
/// </summary>
type PromptAnalysis = {
    OriginalPrompt: string
    Issues: string list
    Suggestions: string list
    ConfidenceScore: float
    EstimatedImprovement: float
}

/// <summary>
/// Improved prompt with metadata
/// </summary>
type ImprovedPrompt = {
    Original: string
    Improved: string
    Strategy: PromptImprovementStrategy
    Reasoning: string
    ExpectedBenefit: string
    ConfidenceScore: float
    Version: string
    CreatedAt: DateTime
}

/// <summary>
/// Prompt performance metrics
/// </summary>
type PromptPerformance = {
    PromptId: string
    UsageCount: int
    SuccessRate: float
    AverageResponseTime: float
    UserSatisfactionScore: float
    ErrorCount: int
    LastUsed: DateTime
}

/// <summary>
/// TARS Prompt Optimizer - Universal prompt improvement system
/// Can be used across all TARS operations for prompt optimization
/// </summary>
type PromptOptimizer() =
    
    let performanceData = Dictionary<string, PromptPerformance>()
    let promptHistory = Dictionary<string, ImprovedPrompt list>()
    
    /// Analyze prompt for potential improvements
    member this.AnalyzePrompt(prompt: string) : PromptAnalysis =
        let issues = ResizeArray<string>()
        let suggestions = ResizeArray<string>()
        
        // Check for clarity issues
        if prompt.Length < 20 then
            issues.Add("Prompt is too short and may lack context")
            suggestions.Add("Add more specific instructions and context")
        
        if prompt.Length > 2000 then
            issues.Add("Prompt is very long and may be inefficient")
            suggestions.Add("Consider breaking into smaller, focused prompts")
        
        // Check for vague language
        let vaguePhrases = ["please help", "do something", "figure out", "handle this", "deal with"]
        let hasVagueLanguage = vaguePhrases |> List.exists (fun phrase -> prompt.ToLower().Contains(phrase))
        if hasVagueLanguage then
            issues.Add("Contains vague language that may lead to unclear results")
            suggestions.Add("Use specific action verbs and clear instructions")
        
        // Check for missing examples
        if not (prompt.Contains("example") || prompt.Contains("for instance") || prompt.Contains("such as")) then
            if prompt.Length > 100 then
                suggestions.Add("Consider adding examples to clarify expectations")
        
        // Check for missing format specification
        if not (prompt.Contains("format") || prompt.Contains("structure") || prompt.Contains("organize")) then
            if prompt.Contains("list") || prompt.Contains("report") || prompt.Contains("summary") then
                suggestions.Add("Specify the desired output format or structure")
        
        // Check for missing constraints
        if not (prompt.Contains("limit") || prompt.Contains("maximum") || prompt.Contains("minimum") || prompt.Contains("between")) then
            suggestions.Add("Consider adding constraints for length, scope, or detail level")
        
        // Calculate confidence score based on analysis
        let issueCount = issues.Count
        let confidenceScore = 
            match issueCount with
            | 0 -> 0.9
            | 1 -> 0.7
            | 2 -> 0.5
            | _ -> 0.3
        
        let estimatedImprovement = float issueCount * 0.15 + 0.1
        
        {
            OriginalPrompt = prompt
            Issues = issues |> Seq.toList
            Suggestions = suggestions |> Seq.toList
            ConfidenceScore = confidenceScore
            EstimatedImprovement = min estimatedImprovement 0.8
        }
    
    /// Improve prompt using specific strategy
    member this.ImprovePrompt(prompt: string, strategy: PromptImprovementStrategy) : ImprovedPrompt =
        let improved, reasoning, benefit = 
            match strategy with
            | ClarityEnhancement ->
                let enhanced = this.EnhanceClarity(prompt)
                (enhanced, "Enhanced clarity by adding specific instructions and removing ambiguous language", "Improved task understanding and execution accuracy")
            
            | ContextEnrichment ->
                let enriched = this.EnrichContext(prompt)
                (enriched, "Added contextual information to help AI understand the task better", "Better context awareness and more relevant responses")
            
            | ExampleAddition ->
                let withExamples = this.AddExamples(prompt)
                (withExamples, "Added examples to demonstrate expected output format and quality", "Clearer expectations and more consistent results")
            
            | ConstraintSpecification ->
                let withConstraints = this.AddConstraints(prompt)
                (withConstraints, "Added specific constraints and boundaries for the task", "More focused and controlled outputs")
            
            | FormatStandardization ->
                let standardized = this.StandardizeFormat(prompt)
                (standardized, "Standardized the prompt format for consistency and clarity", "More predictable and structured responses")
            
            | PerformanceOptimization ->
                let optimized = this.OptimizeForPerformance(prompt)
                (optimized, "Optimized prompt for faster processing and reduced token usage", "Improved response time and cost efficiency")
            
            | ErrorReduction ->
                let errorProof = this.AddErrorHandling(prompt)
                (errorProof, "Added error handling instructions and edge case considerations", "Reduced error rates and more robust responses")
            
            | UserExperienceImprovement ->
                let uxImproved = this.ImproveUserExperience(prompt)
                (uxImproved, "Enhanced user experience with clearer communication and helpful guidance", "Better user satisfaction and engagement")
        
        {
            Original = prompt
            Improved = improved
            Strategy = strategy
            Reasoning = reasoning
            ExpectedBenefit = benefit
            ConfidenceScore = 0.75
            Version = "1.0.0"
            CreatedAt = DateTime.UtcNow
        }
    
    /// Enhance prompt clarity
    member private this.EnhanceClarity(prompt: string) =
        let enhanced = prompt
                      |> fun p -> if not (p.Contains("specific")) then p + "\n\nBe specific and detailed in your response." else p
                      |> fun p -> if not (p.Contains("clear")) then p + " Provide clear and unambiguous information." else p
                      |> fun p -> Regex.Replace(p, @"\b(please help|figure out|deal with)\b", "analyze and provide", RegexOptions.IgnoreCase)
        enhanced
    
    /// Enrich prompt context
    member private this.EnrichContext(prompt: string) =
        let contextAddition = "\n\nContext: Consider the following when responding:\n- The user's technical background and expertise level\n- The specific domain and use case\n- Any relevant constraints or requirements\n- The intended audience for the response"
        prompt + contextAddition
    
    /// Add examples to prompt
    member private this.AddExamples(prompt: string) =
        if prompt.Contains("example") then prompt
        else prompt + "\n\nExample format:\n[Provide a brief example of the expected output format or structure]"
    
    /// Add constraints to prompt
    member private this.AddConstraints(prompt: string) =
        let constraints = "\n\nConstraints:\n- Keep response focused and relevant\n- Limit to essential information\n- Maintain professional tone\n- Provide actionable guidance where applicable"
        prompt + constraints
    
    /// Standardize prompt format
    member private this.StandardizeFormat(prompt: string) =
        let standardized = $"Task: {prompt.Trim()}\n\nInstructions:\n- Follow the specified requirements exactly\n- Provide complete and accurate information\n- Use clear and professional language\n- Structure your response logically"
        standardized
    
    /// Optimize for performance
    member private this.OptimizeForPerformance(prompt: string) =
        let optimized = prompt
                       |> fun p -> p.Replace("comprehensive and detailed", "concise")
                       |> fun p -> p.Replace("extensive analysis", "focused analysis")
                       |> fun p -> p.Replace("please provide a complete", "provide")
                       |> fun p -> if p.Length > 500 then p.Substring(0, 400) + "... [optimized for efficiency]" else p
        optimized
    
    /// Add error handling
    member private this.AddErrorHandling(prompt: string) =
        let errorHandling = "\n\nError Handling:\nIf you cannot complete the request:\n1. Explain what information is missing\n2. Suggest alternative approaches\n3. Provide partial results if possible\n4. Ask clarifying questions if needed"
        prompt + errorHandling
    
    /// Improve user experience
    member private this.ImproveUserExperience(prompt: string) =
        let uxImprovement = "\n\nUser Experience Guidelines:\n- Make your response easy to understand\n- Provide actionable next steps\n- Include relevant examples or references\n- Offer additional help or clarification if needed"
        prompt + uxImprovement
    
    /// Get best improvement strategy for a prompt
    member this.GetBestStrategy(prompt: string) : PromptImprovementStrategy =
        let analysis = this.AnalyzePrompt(prompt)
        
        // Determine best strategy based on issues found
        if analysis.Issues |> List.exists (fun issue -> issue.Contains("vague") || issue.Contains("unclear")) then
            ClarityEnhancement
        elif analysis.Issues |> List.exists (fun issue -> issue.Contains("context")) then
            ContextEnrichment
        elif analysis.Issues |> List.exists (fun issue -> issue.Contains("long")) then
            PerformanceOptimization
        elif analysis.Suggestions |> List.exists (fun suggestion -> suggestion.Contains("example")) then
            ExampleAddition
        elif analysis.Suggestions |> List.exists (fun suggestion -> suggestion.Contains("format")) then
            FormatStandardization
        else
            UserExperienceImprovement
    
    /// Auto-improve prompt using best strategy
    member this.AutoImprove(prompt: string) : ImprovedPrompt =
        let bestStrategy = this.GetBestStrategy(prompt)
        this.ImprovePrompt(prompt, bestStrategy)
    
    /// Compare two prompts and suggest which is better
    member this.ComparePrompts(prompt1: string, prompt2: string) =
        let analysis1 = this.AnalyzePrompt(prompt1)
        let analysis2 = this.AnalyzePrompt(prompt2)
        
        let score1 = analysis1.ConfidenceScore
        let score2 = analysis2.ConfidenceScore
        
        {|
            Prompt1 = {| Text = prompt1; Score = score1; Issues = analysis1.Issues |}
            Prompt2 = {| Text = prompt2; Score = score2; Issues = analysis2.Issues |}
            Recommendation = 
                if score1 > score2 then "Use Prompt 1"
                elif score2 > score1 then "Use Prompt 2"
                else "Both prompts are similar in quality"
            ScoreDifference = abs (score1 - score2)
            BetterPrompt = if score1 > score2 then 1 elif score2 > score1 then 2 else 0
        |}
    
    /// Record prompt performance
    member this.RecordPerformance(promptId: string, responseTime: float, success: bool, userSatisfaction: float option) =
        let existing = 
            if performanceData.ContainsKey(promptId) then
                performanceData.[promptId]
            else
                {
                    PromptId = promptId
                    UsageCount = 0
                    SuccessRate = 0.0
                    AverageResponseTime = 0.0
                    UserSatisfactionScore = 0.0
                    ErrorCount = 0
                    LastUsed = DateTime.UtcNow
                }
        
        let newUsageCount = existing.UsageCount + 1
        let newSuccessCount = if success then (int (existing.SuccessRate * float existing.UsageCount)) + 1 
                             else (int (existing.SuccessRate * float existing.UsageCount))
        let newErrorCount = if not success then existing.ErrorCount + 1 else existing.ErrorCount
        
        let newAverageResponseTime = 
            (existing.AverageResponseTime * float existing.UsageCount + responseTime) / float newUsageCount
        
        let newSatisfactionScore = 
            match userSatisfaction with
            | Some score -> 
                (existing.UserSatisfactionScore * float existing.UsageCount + score) / float newUsageCount
            | None -> existing.UserSatisfactionScore
        
        let updated = {
            existing with
                UsageCount = newUsageCount
                SuccessRate = float newSuccessCount / float newUsageCount
                AverageResponseTime = newAverageResponseTime
                UserSatisfactionScore = newSatisfactionScore
                ErrorCount = newErrorCount
                LastUsed = DateTime.UtcNow
        }
        
        performanceData.[promptId] <- updated
    
    /// Get performance data for a prompt
    member this.GetPerformance(promptId: string) =
        if performanceData.ContainsKey(promptId) then
            Some performanceData.[promptId]
        else
            None
    
    /// Get all performance data
    member this.GetAllPerformance() =
        performanceData.Values |> Seq.toArray
    
    /// Save prompt improvement to history
    member this.SaveToHistory(promptId: string, improvement: ImprovedPrompt) =
        if promptHistory.ContainsKey(promptId) then
            promptHistory.[promptId] <- improvement :: promptHistory.[promptId]
        else
            promptHistory.[promptId] <- [improvement]
    
    /// Get improvement history for a prompt
    member this.GetHistory(promptId: string) =
        if promptHistory.ContainsKey(promptId) then
            promptHistory.[promptId]
        else
            []
    
    /// Get system-wide improvement statistics
    member this.GetImprovementStats() =
        let allPerformance = performanceData.Values |> Seq.toArray
        let allHistory = promptHistory.Values |> Seq.concat |> Seq.toArray
        
        {|
            TotalPromptsTracked = performanceData.Count
            TotalImprovements = allHistory.Length
            AverageSuccessRate = 
                if allPerformance.Length > 0 then
                    allPerformance |> Array.averageBy (fun p -> p.SuccessRate)
                else 0.0
            AverageResponseTime = 
                if allPerformance.Length > 0 then
                    allPerformance |> Array.averageBy (fun p -> p.AverageResponseTime)
                else 0.0
            AverageUserSatisfaction = 
                if allPerformance.Length > 0 then
                    allPerformance |> Array.averageBy (fun p -> p.UserSatisfactionScore)
                else 0.0
            MostUsedStrategy = 
                if allHistory.Length > 0 then
                    allHistory 
                    |> Array.groupBy (fun h -> h.Strategy)
                    |> Array.maxBy (fun (_, group) -> group.Length)
                    |> fst
                    |> string
                else "None"
            PromptsNeedingImprovement = 
                allPerformance 
                |> Array.filter (fun p -> p.SuccessRate < 0.8 || p.UserSatisfactionScore < 4.0)
                |> Array.length
        |}
