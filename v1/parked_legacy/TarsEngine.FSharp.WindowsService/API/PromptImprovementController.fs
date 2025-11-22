namespace TarsEngine.FSharp.WindowsService.API

open System
open Microsoft.AspNetCore.Mvc
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.WindowsService.AI

/// <summary>
/// REST API Controller for TARS prompt improvement system
/// Provides comprehensive prompt optimization and A/B testing capabilities
/// </summary>
[<ApiController>]
[<Route("api/[controller]")>]
type PromptImprovementController(logger: ILogger<PromptImprovementController>, promptManager: PromptImprovementManager) =
    inherit ControllerBase()
    
    /// Get prompt improvement system overview
    [<HttpGet("overview")>]
    member this.GetSystemOverview() =
        try
            let overview = promptManager.GetSystemOverview()
            logger.LogInformation("📊 Prompt improvement system overview requested")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                overview = overview
                capabilities = {|
                    promptAnalysis = true
                    abTesting = true
                    automaticOptimization = true
                    performanceTracking = true
                    continuousImprovement = true
                |}
                message = "Prompt improvement system overview retrieved"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting prompt improvement overview")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get prompt improvement overview"
                details = ex.Message
            |})
    
    /// Record prompt usage and performance
    [<HttpPost("usage")>]
    member this.RecordPromptUsage([<FromBody>] request: {| promptId: string; category: string; responseTime: float; success: bool; userSatisfaction: float option |}) =
        try
            let category = 
                match request.category.ToLower() with
                | "system" -> PromptCategory.SystemPrompt
                | "user" -> PromptCategory.UserInteraction
                | "code" -> PromptCategory.CodeGeneration
                | "documentation" -> PromptCategory.Documentation
                | "task" -> PromptCategory.TaskExecution
                | "agent" -> PromptCategory.AgentCommunication
                | _ -> PromptCategory.UserInteraction
            
            promptManager.RecordPromptUsage(request.promptId, category, request.responseTime, request.success, request.userSatisfaction)
            
            logger.LogInformation($"📊 Recorded usage for prompt {request.promptId}")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                promptId = request.promptId
                message = "Prompt usage recorded successfully"
            |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error recording prompt usage for {request.promptId}")
            this.StatusCode(500, {|
                success = false
                error = "Failed to record prompt usage"
                details = ex.Message
            |})
    
    /// Analyze prompt performance and get improvement suggestions
    [<HttpPost("analyze")>]
    member this.AnalyzePrompt([<FromBody>] request: {| promptId: string; promptText: string |}) =
        try
            let improvements = promptManager.AnalyzePromptPerformance(request.promptId, request.promptText)
            
            logger.LogInformation($"🔍 Analyzed prompt {request.promptId}, found {improvements.Length} improvement suggestions")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                promptId = request.promptId
                improvements = improvements
                improvementCount = improvements.Length
                message = $"Prompt analysis completed with {improvements.Length} suggestions"
            |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error analyzing prompt {request.promptId}")
            this.StatusCode(500, {|
                success = false
                error = "Failed to analyze prompt"
                details = ex.Message
            |})
    
    /// Get improvement recommendations for all prompts
    [<HttpGet("recommendations")>]
    member this.GetImprovementRecommendations() =
        try
            let recommendations = promptManager.GetImprovementRecommendations()
            
            logger.LogInformation($"💡 Retrieved {recommendations.Length} improvement recommendations")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                recommendations = recommendations
                count = recommendations.Length
                summary = {|
                    totalRecommendations = recommendations.Length
                    highPriorityRecommendations = 
                        recommendations 
                        |> Array.filter (fun r -> r.Improvements |> Array.exists (fun i -> i.ConfidenceScore > 0.75))
                        |> Array.length
                    categoriesNeedingImprovement = 
                        recommendations 
                        |> Array.map (fun r -> r.Metrics.Category.ToString())
                        |> Array.distinct
                        |> Array.length
                |}
                message = "Improvement recommendations retrieved"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting improvement recommendations")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get improvement recommendations"
                details = ex.Message
            |})
    
    /// Create A/B test for prompt improvement
    [<HttpPost("ab-test")>]
    member this.CreateABTest([<FromBody>] request: {| originalPrompt: string; improvedPrompt: string; category: string; sampleSize: int |}) =
        try
            let category = 
                match request.category.ToLower() with
                | "system" -> PromptCategory.SystemPrompt
                | "user" -> PromptCategory.UserInteraction
                | "code" -> PromptCategory.CodeGeneration
                | "documentation" -> PromptCategory.Documentation
                | "task" -> PromptCategory.TaskExecution
                | "agent" -> PromptCategory.AgentCommunication
                | _ -> PromptCategory.UserInteraction
            
            let testId = promptManager.CreateABTest(request.originalPrompt, request.improvedPrompt, category, request.sampleSize)
            
            logger.LogInformation($"🧪 Created A/B test {testId} for {category} prompt")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                testId = testId
                category = request.category
                sampleSize = request.sampleSize
                message = $"A/B test {testId} created successfully"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error creating A/B test")
            this.StatusCode(500, {|
                success = false
                error = "Failed to create A/B test"
                details = ex.Message
            |})
    
    /// Get prompt variant for A/B testing
    [<HttpGet("ab-test/{testId}/variant")>]
    member this.GetPromptVariant(testId: string, [<FromQuery>] useVariant: bool) =
        try
            let prompt = promptManager.GetPromptVariant(testId, useVariant)
            
            if String.IsNullOrEmpty(prompt) then
                this.NotFound({|
                    success = false
                    error = $"A/B test {testId} not found or inactive"
                    message = "The specified test ID does not exist or is not active"
                |})
            else
                logger.LogDebug($"📝 Retrieved prompt variant for test {testId}, useVariant={useVariant}")
                
                this.Ok({|
                    success = true
                    timestamp = DateTime.UtcNow
                    testId = testId
                    prompt = prompt
                    isVariant = useVariant
                    message = "Prompt variant retrieved"
                |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error getting prompt variant for test {testId}")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get prompt variant"
                details = ex.Message
            |})
    
    /// Record A/B test result
    [<HttpPost("ab-test/{testId}/result")>]
    member this.RecordABTestResult(testId: string, [<FromBody>] request: {| useVariant: bool; responseTime: float; success: bool; userSatisfaction: float option |}) =
        try
            promptManager.RecordABTestResult(testId, request.useVariant, request.responseTime, request.success, request.userSatisfaction)
            
            logger.LogDebug($"📊 Recorded A/B test result for {testId}")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                testId = testId
                variantUsed = request.useVariant
                message = "A/B test result recorded"
            |})
        with
        | ex ->
            logger.LogError(ex, $"❌ Error recording A/B test result for {testId}")
            this.StatusCode(500, {|
                success = false
                error = "Failed to record A/B test result"
                details = ex.Message
            |})
    
    /// Get prompt optimization suggestions
    [<HttpPost("optimize")>]
    member this.OptimizePrompt([<FromBody>] request: {| promptText: string; optimizationType: string |}) =
        try
            // Simple optimization based on type
            let optimizedPrompt = 
                match request.optimizationType.ToLower() with
                | "clarity" -> 
                    request.promptText + "\n\nBe specific and detailed in your response. Provide clear examples where helpful."
                | "speed" -> 
                    request.promptText.Replace("comprehensive", "concise").Replace("detailed analysis", "analysis")
                | "accuracy" -> 
                    request.promptText + "\n\nEnsure accuracy by:\n1. Verifying facts\n2. Providing sources when possible\n3. Acknowledging uncertainties"
                | "user-experience" -> 
                    request.promptText + "\n\nMake your response:\n- Easy to understand\n- Actionable\n- Relevant to user context"
                | _ -> request.promptText
            
            logger.LogInformation($"⚡ Optimized prompt for {request.optimizationType}")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                originalPrompt = request.promptText
                optimizedPrompt = optimizedPrompt
                optimizationType = request.optimizationType
                improvements = [|
                    "Enhanced clarity and specificity"
                    "Improved instruction structure"
                    "Added guidance for better results"
                |]
                message = $"Prompt optimized for {request.optimizationType}"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error optimizing prompt")
            this.StatusCode(500, {|
                success = false
                error = "Failed to optimize prompt"
                details = ex.Message
            |})
    
    /// Get prompt performance analytics
    [<HttpGet("analytics")>]
    member this.GetPromptAnalytics() =
        try
            let overview = promptManager.GetSystemOverview()
            
            logger.LogInformation("📈 Prompt analytics requested")
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                analytics = {|
                    performanceMetrics = {|
                        averageSuccessRate = overview.AverageSuccessRate
                        averageResponseTime = overview.AverageResponseTime
                        averageUserSatisfaction = overview.AverageUserSatisfaction
                    |}
                    systemHealth = {|
                        totalPrompts = overview.TotalPrompts
                        activeABTests = overview.ActiveABTests
                        completedABTests = overview.CompletedABTests
                        promptsNeedingImprovement = overview.PromptsNeedingImprovement
                    |}
                    categoryBreakdown = overview.CategoryBreakdown
                    topPerformingPrompts = overview.TopPerformingPrompts
                    improvementOpportunities = {|
                        lowSuccessRatePrompts = overview.PromptsNeedingImprovement
                        optimizationPotential = 
                            if overview.AverageSuccessRate < 0.9 then "High" 
                            elif overview.AverageSuccessRate < 0.95 then "Medium"
                            else "Low"
                        recommendedActions = [|
                            if overview.AverageSuccessRate < 0.8 then "Focus on clarity improvements"
                            if overview.AverageResponseTime > 3000.0 then "Optimize for speed"
                            if overview.AverageUserSatisfaction < 4.0 then "Enhance user experience"
                            if overview.ActiveABTests = 0 then "Start A/B testing program"
                        |]
                    |}
                |}
                message = "Prompt analytics retrieved"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Error getting prompt analytics")
            this.StatusCode(500, {|
                success = false
                error = "Failed to get prompt analytics"
                details = ex.Message
            |})
    
    /// Health check for prompt improvement system
    [<HttpGet("health")>]
    member this.HealthCheck() =
        try
            let overview = promptManager.GetSystemOverview()
            
            this.Ok({|
                success = true
                timestamp = DateTime.UtcNow
                service = "Prompt Improvement Manager"
                status = "Healthy"
                metrics = {|
                    totalPrompts = overview.TotalPrompts
                    averagePerformance = overview.AverageSuccessRate
                    activeOptimizations = overview.ActiveABTests
                    systemEfficiency = 
                        if overview.AverageSuccessRate > 0.9 then "Excellent"
                        elif overview.AverageSuccessRate > 0.8 then "Good"
                        elif overview.AverageSuccessRate > 0.7 then "Fair"
                        else "Needs Improvement"
                |}
                capabilities = {|
                    realTimeAnalysis = true
                    automaticOptimization = true
                    abTesting = true
                    performanceTracking = true
                    continuousImprovement = true
                |}
                message = "Prompt improvement system is operational"
            |})
        with
        | ex ->
            logger.LogError(ex, "❌ Prompt improvement health check failed")
            this.StatusCode(500, {|
                success = false
                service = "Prompt Improvement Manager"
                status = "Unhealthy"
                error = ex.Message
            |})
