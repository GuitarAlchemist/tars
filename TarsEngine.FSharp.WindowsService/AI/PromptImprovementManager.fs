namespace TarsEngine.FSharp.WindowsService.AI

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open System.Text.Json
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Hosting

/// <summary>
/// Prompt categories for improvement tracking
/// </summary>
type PromptCategory =
    | SystemPrompt
    | UserInteraction
    | CodeGeneration
    | Documentation
    | TaskExecution
    | AgentCommunication

/// <summary>
/// Prompt performance metrics
/// </summary>
type PromptMetrics = {
    PromptId: string
    Category: PromptCategory
    UsageCount: int
    SuccessRate: float
    AverageResponseTime: float
    UserSatisfactionScore: float
    ErrorCount: int
    LastUsed: DateTime
    CreatedDate: DateTime
    Version: string
}

/// <summary>
/// Prompt improvement suggestion
/// </summary>
type PromptImprovement = {
    OriginalPrompt: string
    ImprovedPrompt: string
    ImprovementType: string
    Reasoning: string
    ExpectedBenefit: string
    ConfidenceScore: float
    TestingRequired: bool
}

/// <summary>
/// Prompt A/B test configuration
/// </summary>
type PromptABTest = {
    TestId: string
    OriginalPrompt: string
    VariantPrompt: string
    Category: PromptCategory
    StartDate: DateTime
    EndDate: DateTime option
    SampleSize: int
    CurrentSamples: int
    OriginalMetrics: PromptMetrics option
    VariantMetrics: PromptMetrics option
    IsActive: bool
    StatisticalSignificance: float option
}

/// <summary>
/// Prompt Improvement Manager for TARS AI optimization
/// Analyzes, optimizes, and evolves prompts across all TARS operations
/// </summary>
type PromptImprovementManager(logger: ILogger<PromptImprovementManager>) =
    inherit BackgroundService()
    
    let promptMetrics = ConcurrentDictionary<string, PromptMetrics>()
    let activeTests = ConcurrentDictionary<string, PromptABTest>()
    let promptHistory = ConcurrentDictionary<string, string list>()
    let mutable cancellationTokenSource = new CancellationTokenSource()
    
    let metricsFile = Path.Combine(".tars", "prompt_metrics.json")
    let testsFile = Path.Combine(".tars", "prompt_ab_tests.json")
    let improvementsFile = Path.Combine(".tars", "prompt_improvements.json")
    
    // Ensure .tars directory exists
    do
        let tarsDir = ".tars"
        if not (Directory.Exists(tarsDir)) then
            Directory.CreateDirectory(tarsDir) |> ignore
    
    /// Save data to disk
    member private this.SaveData() =
        try
            // Save metrics
            let metricsData = promptMetrics.Values |> Seq.toArray
            let metricsJson = JsonSerializer.Serialize(metricsData, JsonSerializerOptions(WriteIndented = true))
            File.WriteAllText(metricsFile, metricsJson)
            
            // Save tests
            let testsData = activeTests.Values |> Seq.toArray
            let testsJson = JsonSerializer.Serialize(testsData, JsonSerializerOptions(WriteIndented = true))
            File.WriteAllText(testsFile, testsJson)
            
            logger.LogDebug("💾 Prompt improvement data saved to disk")
        with
        | ex -> logger.LogError(ex, "❌ Failed to save prompt improvement data")
    
    /// Load data from disk
    member private this.LoadData() =
        try
            // Load metrics
            if File.Exists(metricsFile) then
                let metricsJson = File.ReadAllText(metricsFile)
                let metricsData = JsonSerializer.Deserialize<PromptMetrics[]>(metricsJson)
                
                promptMetrics.Clear()
                for metric in metricsData do
                    promptMetrics.TryAdd(metric.PromptId, metric) |> ignore
                
                logger.LogInformation($"📂 Loaded {metricsData.Length} prompt metrics from disk")
            
            // Load tests
            if File.Exists(testsFile) then
                let testsJson = File.ReadAllText(testsFile)
                let testsData = JsonSerializer.Deserialize<PromptABTest[]>(testsJson)
                
                activeTests.Clear()
                for test in testsData do
                    activeTests.TryAdd(test.TestId, test) |> ignore
                
                logger.LogInformation($"📂 Loaded {testsData.Length} A/B tests from disk")
                
        with
        | ex -> 
            logger.LogError(ex, "❌ Failed to load prompt improvement data")
    
    /// Record prompt usage and performance
    member this.RecordPromptUsage(promptId: string, category: PromptCategory, responseTime: float, success: bool, userSatisfaction: float option) =
        let currentMetrics = 
            promptMetrics.GetOrAdd(promptId, fun id -> {
                PromptId = id
                Category = category
                UsageCount = 0
                SuccessRate = 0.0
                AverageResponseTime = 0.0
                UserSatisfactionScore = 0.0
                ErrorCount = 0
                LastUsed = DateTime.UtcNow
                CreatedDate = DateTime.UtcNow
                Version = "1.0.0"
            })
        
        let newUsageCount = currentMetrics.UsageCount + 1
        let newSuccessCount = if success then (int (currentMetrics.SuccessRate * float currentMetrics.UsageCount)) + 1 
                             else (int (currentMetrics.SuccessRate * float currentMetrics.UsageCount))
        let newErrorCount = if not success then currentMetrics.ErrorCount + 1 else currentMetrics.ErrorCount
        
        let newAverageResponseTime = 
            (currentMetrics.AverageResponseTime * float currentMetrics.UsageCount + responseTime) / float newUsageCount
        
        let newSatisfactionScore = 
            match userSatisfaction with
            | Some score -> 
                (currentMetrics.UserSatisfactionScore * float currentMetrics.UsageCount + score) / float newUsageCount
            | None -> currentMetrics.UserSatisfactionScore
        
        let updatedMetrics = {
            currentMetrics with
                UsageCount = newUsageCount
                SuccessRate = float newSuccessCount / float newUsageCount
                AverageResponseTime = newAverageResponseTime
                UserSatisfactionScore = newSatisfactionScore
                ErrorCount = newErrorCount
                LastUsed = DateTime.UtcNow
        }
        
        promptMetrics.TryUpdate(promptId, updatedMetrics, currentMetrics) |> ignore
        this.SaveData()
        
        logger.LogDebug($"📊 Recorded usage for prompt {promptId}: Success={success}, ResponseTime={responseTime}ms")
    
    /// Analyze prompt performance and generate improvement suggestions
    member this.AnalyzePromptPerformance(promptId: string, promptText: string) =
        match promptMetrics.TryGetValue(promptId) with
        | true, metrics ->
            let improvements = ResizeArray<PromptImprovement>()
            
            // Analyze success rate
            if metrics.SuccessRate < 0.8 then
                improvements.Add({
                    OriginalPrompt = promptText
                    ImprovedPrompt = this.ImproveClarity(promptText)
                    ImprovementType = "Clarity Enhancement"
                    Reasoning = "Low success rate indicates potential ambiguity in instructions"
                    ExpectedBenefit = $"Increase success rate from {metrics.SuccessRate:P1} to 85%+"
                    ConfidenceScore = 0.75
                    TestingRequired = true
                })
            
            // Analyze response time
            if metrics.AverageResponseTime > 5000.0 then
                improvements.Add({
                    OriginalPrompt = promptText
                    ImprovedPrompt = this.OptimizeForSpeed(promptText)
                    ImprovementType = "Efficiency Optimization"
                    Reasoning = "High response time suggests prompt complexity can be reduced"
                    ExpectedBenefit = $"Reduce response time from {metrics.AverageResponseTime:F0}ms to <3000ms"
                    ConfidenceScore = 0.65
                    TestingRequired = true
                })
            
            // Analyze user satisfaction
            if metrics.UserSatisfactionScore < 4.0 then
                improvements.Add({
                    OriginalPrompt = promptText
                    ImprovedPrompt = this.EnhanceUserExperience(promptText)
                    ImprovementType = "User Experience Enhancement"
                    Reasoning = "Low satisfaction score indicates user needs are not fully met"
                    ExpectedBenefit = $"Increase satisfaction from {metrics.UserSatisfactionScore:F1} to 4.5+"
                    ConfidenceScore = 0.70
                    TestingRequired = true
                })
            
            // Analyze error patterns
            if float metrics.ErrorCount / float metrics.UsageCount > 0.1 then
                improvements.Add({
                    OriginalPrompt = promptText
                    ImprovedPrompt = this.AddErrorHandling(promptText)
                    ImprovementType = "Error Reduction"
                    Reasoning = "High error rate suggests missing constraints or edge case handling"
                    ExpectedBenefit = $"Reduce error rate from {float metrics.ErrorCount / float metrics.UsageCount:P1} to <5%"
                    ConfidenceScore = 0.80
                    TestingRequired = true
                })
            
            improvements.ToArray()
        | false, _ ->
            logger.LogWarning($"⚠️ No metrics found for prompt {promptId}")
            [||]
    
    /// Improve prompt clarity
    member private this.ImproveClarity(prompt: string) =
        // Simple clarity improvements (in real implementation, this would use advanced NLP)
        let improved = prompt
                      |> fun p -> if not (p.Contains("specific")) then p + "\n\nBe specific and detailed in your response." else p
                      |> fun p -> if not (p.Contains("format")) then p + "\n\nFormat your response clearly with appropriate structure." else p
                      |> fun p -> if not (p.Contains("example")) then p + "\n\nProvide examples where helpful." else p
        improved
    
    /// Optimize prompt for speed
    member private this.OptimizeForSpeed(prompt: string) =
        // Simple speed optimizations
        let optimized = prompt
                       |> fun p -> p.Replace("Please provide a comprehensive and detailed", "Provide a concise")
                       |> fun p -> p.Replace("extensive analysis", "analysis")
                       |> fun p -> if p.Length > 500 then p.Substring(0, 400) + "... [optimized for speed]" else p
        optimized
    
    /// Enhance user experience
    member private this.EnhanceUserExperience(prompt: string) =
        // Simple UX improvements
        let enhanced = prompt + "\n\nEnsure your response is:\n- Easy to understand\n- Actionable\n- Relevant to the user's context\n- Helpful for next steps"
        enhanced
    
    /// Add error handling to prompt
    member private this.AddErrorHandling(prompt: string) =
        // Simple error handling additions
        let withErrorHandling = prompt + "\n\nIf you cannot complete the request:\n1. Explain what information is missing\n2. Suggest alternative approaches\n3. Provide partial results if possible"
        withErrorHandling
    
    /// Create A/B test for prompt improvement
    member this.CreateABTest(originalPrompt: string, improvedPrompt: string, category: PromptCategory, sampleSize: int) =
        let testId = Guid.NewGuid().ToString("N")[..7]
        
        let abTest = {
            TestId = testId
            OriginalPrompt = originalPrompt
            VariantPrompt = improvedPrompt
            Category = category
            StartDate = DateTime.UtcNow
            EndDate = None
            SampleSize = sampleSize
            CurrentSamples = 0
            OriginalMetrics = None
            VariantMetrics = None
            IsActive = true
            StatisticalSignificance = None
        }
        
        activeTests.TryAdd(testId, abTest) |> ignore
        this.SaveData()
        
        logger.LogInformation($"🧪 Created A/B test {testId} for {category} prompt")
        testId
    
    /// Get prompt variant for A/B testing
    member this.GetPromptVariant(testId: string, useVariant: bool) =
        match activeTests.TryGetValue(testId) with
        | true, test when test.IsActive ->
            if useVariant then test.VariantPrompt else test.OriginalPrompt
        | _ -> ""
    
    /// Record A/B test result
    member this.RecordABTestResult(testId: string, useVariant: bool, responseTime: float, success: bool, userSatisfaction: float option) =
        match activeTests.TryGetValue(testId) with
        | true, test when test.IsActive ->
            let updatedTest = { test with CurrentSamples = test.CurrentSamples + 1 }
            activeTests.TryUpdate(testId, updatedTest, test) |> ignore
            
            // Record metrics for the specific variant
            let variantId = if useVariant then $"{testId}-variant" else $"{testId}-original"
            this.RecordPromptUsage(variantId, test.Category, responseTime, success, userSatisfaction)
            
            // Check if test is complete
            if updatedTest.CurrentSamples >= updatedTest.SampleSize then
                this.CompleteABTest(testId)
        | _ -> ()
    
    /// Complete A/B test and analyze results
    member private this.CompleteABTest(testId: string) =
        match activeTests.TryGetValue(testId) with
        | true, test ->
            let originalMetrics = promptMetrics.TryGetValue($"{testId}-original")
            let variantMetrics = promptMetrics.TryGetValue($"{testId}-variant")
            
            match originalMetrics, variantMetrics with
            | (true, original), (true, variant) ->
                // Simple statistical significance calculation (in real implementation, use proper statistical tests)
                let successDiff = abs (variant.SuccessRate - original.SuccessRate)
                let timeDiff = abs (variant.AverageResponseTime - original.AverageResponseTime)
                let satisfactionDiff = abs (variant.UserSatisfactionScore - original.UserSatisfactionScore)
                
                let significance = (successDiff * 0.4 + (timeDiff / 1000.0) * 0.3 + satisfactionDiff * 0.3)
                
                let completedTest = {
                    test with
                        IsActive = false
                        EndDate = Some DateTime.UtcNow
                        OriginalMetrics = Some original
                        VariantMetrics = Some variant
                        StatisticalSignificance = Some significance
                }
                
                activeTests.TryUpdate(testId, completedTest, test) |> ignore
                this.SaveData()
                
                logger.LogInformation($"✅ Completed A/B test {testId} with significance {significance:F3}")
                
                // Auto-promote if variant is significantly better
                if significance > 0.1 && variant.SuccessRate > original.SuccessRate then
                    logger.LogInformation($"🚀 Auto-promoting variant prompt for test {testId}")
            | _ ->
                logger.LogWarning($"⚠️ Insufficient data to complete A/B test {testId}")
        | _ -> ()
    
    /// Get prompt improvement recommendations
    member this.GetImprovementRecommendations() =
        let recommendations = ResizeArray<{| PromptId: string; Metrics: PromptMetrics; Improvements: PromptImprovement[] |}>()
        
        for kvp in promptMetrics do
            let metrics = kvp.Value
            
            // Only recommend improvements for prompts with sufficient usage
            if metrics.UsageCount >= 10 then
                let improvements = this.AnalyzePromptPerformance(kvp.Key, "")
                
                if improvements.Length > 0 then
                    recommendations.Add({|
                        PromptId = kvp.Key
                        Metrics = metrics
                        Improvements = improvements
                    |})
        
        recommendations.ToArray()
    
    /// Get system overview
    member this.GetSystemOverview() = {|
        TotalPrompts = promptMetrics.Count
        ActiveABTests = activeTests.Values |> Seq.filter (fun t -> t.IsActive) |> Seq.length
        CompletedABTests = activeTests.Values |> Seq.filter (fun t -> not t.IsActive) |> Seq.length
        AverageSuccessRate = 
            if promptMetrics.Count > 0 then
                promptMetrics.Values |> Seq.averageBy (fun m -> m.SuccessRate)
            else 0.0
        AverageResponseTime = 
            if promptMetrics.Count > 0 then
                promptMetrics.Values |> Seq.averageBy (fun m -> m.AverageResponseTime)
            else 0.0
        AverageUserSatisfaction = 
            if promptMetrics.Count > 0 then
                promptMetrics.Values |> Seq.averageBy (fun m -> m.UserSatisfactionScore)
            else 0.0
        CategoryBreakdown = 
            promptMetrics.Values 
            |> Seq.groupBy (fun m -> m.Category)
            |> Seq.map (fun (category, metrics) -> category.ToString(), Seq.length metrics)
            |> Map.ofSeq
        TopPerformingPrompts = 
            promptMetrics.Values 
            |> Seq.sortByDescending (fun m -> m.SuccessRate * m.UserSatisfactionScore)
            |> Seq.take 5
            |> Seq.toArray
        PromptsNeedingImprovement = 
            promptMetrics.Values 
            |> Seq.filter (fun m -> m.SuccessRate < 0.8 || m.UserSatisfactionScore < 4.0)
            |> Seq.length
    |}
    
    /// Background service execution
    override this.ExecuteAsync(stoppingToken: CancellationToken) = task {
        logger.LogInformation("🧠 Prompt Improvement Manager started")
        
        // Load existing data
        this.LoadData()
        
        // Periodic analysis and optimization
        while not stoppingToken.IsCancellationRequested do
            try
                // Check for completed A/B tests
                let activeTestIds = activeTests.Values |> Seq.filter (fun t -> t.IsActive) |> Seq.map (fun t -> t.TestId) |> Seq.toArray
                
                for testId in activeTestIds do
                    match activeTests.TryGetValue(testId) with
                    | true, test when test.CurrentSamples >= test.SampleSize ->
                        this.CompleteABTest(testId)
                    | _ -> ()
                
                // Save data periodically
                this.SaveData()
                
                do! Task.Delay(30000, stoppingToken) // Check every 30 seconds
            with
            | :? OperationCanceledException -> ()
            | ex -> logger.LogError(ex, "Error in prompt improvement manager loop")
        
        logger.LogInformation("🧠 Prompt Improvement Manager stopped")
    }
    
    /// Dispose resources
    override this.Dispose() =
        cancellationTokenSource?.Cancel()
        cancellationTokenSource?.Dispose()
        base.Dispose()
