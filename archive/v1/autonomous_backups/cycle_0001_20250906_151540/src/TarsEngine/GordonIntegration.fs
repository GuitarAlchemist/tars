namespace TarsEngine

open System
open System.Threading.Tasks
open System.Net.Http
open System.Text.Json
open Microsoft.Extensions.Logging

/// Gordon AI Assistant Integration for TARS
/// Provides intelligent analysis and recommendations for infrastructure operations
module GordonIntegration =
    
    type GordonAnalysisType =
        | ContainerHealth
        | DatabaseStatus
        | NetworkConnectivity
        | ResourceUsage
        | ConsolidationStrategy
        | SecurityAssessment
        | PerformanceOptimization
    
    type GordonRecommendation = {
        Priority: int
        Category: string
        Action: string
        Reasoning: string
        RiskLevel: string
        EstimatedImpact: string
        Prerequisites: string list
    }
    
    type GordonAnalysisResult = {
        AnalysisType: GordonAnalysisType
        Timestamp: DateTime
        Summary: string
        Recommendations: GordonRecommendation list
        HealthScore: int // 0-100
        CriticalIssues: string list
        Warnings: string list
        NextSteps: string list
    }
    
    type IGordonService =
        abstract member AnalyzeInfrastructure: GordonAnalysisType -> Task<Result<GordonAnalysisResult, string>>
        abstract member GetRecommendations: string -> Task<Result<GordonRecommendation list, string>>
        abstract member ValidateConsolidationPlan: string -> Task<Result<bool * string list, string>>
        abstract member MonitorOperation: string -> Task<Result<string, string>>
    
    type GordonService(httpClient: HttpClient, logger: ILogger<GordonService>) =
        
        let gordonEndpoint = "http://localhost:8997" // Gordon's Docker container port
        
        let createAnalysisPrompt analysisType =
            match analysisType with
            | ContainerHealth -> 
                "Analyze current Docker container health, identify failing services, resource conflicts, and provide remediation steps."
            | DatabaseStatus -> 
                "Assess database container status including MongoDB, ChromaDB, Redis. Check for connection issues, data integrity, and performance."
            | NetworkConnectivity -> 
                "Evaluate network connectivity between TARS services, identify port conflicts, and assess network security."
            | ResourceUsage -> 
                "Analyze CPU, memory, and disk usage across all TARS containers. Identify resource bottlenecks and optimization opportunities."
            | ConsolidationStrategy -> 
                "Develop a comprehensive strategy for consolidating multiple Docker compose files into a unified TARS stack."
            | SecurityAssessment -> 
                "Perform security analysis of TARS infrastructure, identify vulnerabilities, and recommend security improvements."
            | PerformanceOptimization -> 
                "Analyze TARS performance metrics and provide optimization recommendations for better throughput and response times."
        
        let parseGordonResponse (response: string) (analysisType: GordonAnalysisType) =
            try
                // Parse Gordon's natural language response into structured data
                let lines = response.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
                
                let summary = 
                    lines 
                    |> Array.tryFind (fun line -> line.Contains("Summary:") || line.Contains("Analysis:"))
                    |> Option.defaultValue "Gordon analysis completed"
                
                let recommendations = 
                    lines
                    |> Array.filter (fun line -> line.Contains("Recommend:") || line.Contains("Action:"))
                    |> Array.mapi (fun i line -> {
                        Priority = i + 1
                        Category = analysisType.ToString()
                        Action = line.Replace("Recommend:", "").Replace("Action:", "").Trim()
                        Reasoning = "Gordon AI analysis"
                        RiskLevel = if line.Contains("critical") || line.Contains("urgent") then "High" else "Medium"
                        EstimatedImpact = "Moderate"
                        Prerequisites = []
                    })
                    |> Array.toList
                
                let criticalIssues = 
                    lines
                    |> Array.filter (fun line -> line.Contains("critical") || line.Contains("error") || line.Contains("failed"))
                    |> Array.toList
                
                let warnings = 
                    lines
                    |> Array.filter (fun line -> line.Contains("warning") || line.Contains("caution") || line.Contains("potential"))
                    |> Array.toList
                
                let healthScore = 
                    if criticalIssues.Length > 0 then 30
                    elif warnings.Length > 2 then 60
                    elif warnings.Length > 0 then 80
                    else 95
                
                {
                    AnalysisType = analysisType
                    Timestamp = DateTime.UtcNow
                    Summary = summary
                    Recommendations = recommendations
                    HealthScore = healthScore
                    CriticalIssues = criticalIssues
                    Warnings = warnings
                    NextSteps = [
                        "Review Gordon's recommendations"
                        "Implement high-priority actions"
                        "Monitor system health"
                        "Schedule follow-up analysis"
                    ]
                }
            with ex ->
                logger.LogError(ex, "Failed to parse Gordon response")
                {
                    AnalysisType = analysisType
                    Timestamp = DateTime.UtcNow
                    Summary = "Gordon analysis parsing failed"
                    Recommendations = []
                    HealthScore = 50
                    CriticalIssues = [ex.Message]
                    Warnings = ["Could not parse Gordon's response"]
                    NextSteps = ["Retry analysis"; "Check Gordon service health"]
                }
        
        let queryGordon (prompt: string) = 
            task {
                try
                    logger.LogInformation("Querying Gordon with prompt: {Prompt}", prompt)
                    
                    let requestBody = {|
                        message = prompt
                        useAdvancedReasoning = true
                        enableMemory = true
                        maxTokens = 2000
                    |}
                    
                    let json = JsonSerializer.Serialize(requestBody)
                    let content = new StringContent(json, System.Text.Encoding.UTF8, "application/json")
                    
                    let! response = httpClient.PostAsync($"{gordonEndpoint}/api/chat", content)
                    
                    if response.IsSuccessStatusCode then
                        let! responseContent = response.Content.ReadAsStringAsync()
                        let result = JsonSerializer.Deserialize<JsonElement>(responseContent)
                        
                        if result.TryGetProperty("success", out let success) && success.GetBoolean() then
                            if result.TryGetProperty("result", out let resultProp) then
                                return Ok (resultProp.GetString())
                            else
                                return Error "No result in Gordon response"
                        else
                            if result.TryGetProperty("error", out let errorProp) then
                                return Error (errorProp.GetString())
                            else
                                return Error "Gordon request failed"
                    else
                        return Error $"Gordon service returned {response.StatusCode}"
                        
                with ex ->
                    logger.LogError(ex, "Error querying Gordon")
                    return Error $"Gordon query failed: {ex.Message}"
            }
        
        interface IGordonService with
            member _.AnalyzeInfrastructure(analysisType: GordonAnalysisType) =
                task {
                    let prompt = createAnalysisPrompt analysisType
                    let! gordonResponse = queryGordon prompt
                    
                    match gordonResponse with
                    | Ok response ->
                        let analysis = parseGordonResponse response analysisType
                        return Ok analysis
                    | Error error ->
                        return Error error
                }
            
            member _.GetRecommendations(context: string) =
                task {
                    let prompt = $"Provide specific recommendations for: {context}. Include priority, actions, and reasoning."
                    let! gordonResponse = queryGordon prompt
                    
                    match gordonResponse with
                    | Ok response ->
                        let analysis = parseGordonResponse response ConsolidationStrategy
                        return Ok analysis.Recommendations
                    | Error error ->
                        return Error error
                }
            
            member _.ValidateConsolidationPlan(plan: string) =
                task {
                    let prompt = $"Validate this consolidation plan and identify potential issues: {plan}"
                    let! gordonResponse = queryGordon prompt
                    
                    match gordonResponse with
                    | Ok response ->
                        let isValid = not (response.Contains("error") || response.Contains("critical") || response.Contains("fail"))
                        let issues = 
                            response.Split([|'\n'|], StringSplitOptions.RemoveEmptyEntries)
                            |> Array.filter (fun line -> line.Contains("issue") || line.Contains("problem") || line.Contains("risk"))
                            |> Array.toList
                        return Ok (isValid, issues)
                    | Error error ->
                        return Error error
                }
            
            member _.MonitorOperation(operation: string) =
                task {
                    let prompt = $"Monitor and provide status update for operation: {operation}"
                    let! gordonResponse = queryGordon prompt
                    return gordonResponse
                }
    
    /// Gordon-assisted consolidation orchestrator
    type GordonConsolidationOrchestrator(gordonService: IGordonService, logger: ILogger<GordonConsolidationOrchestrator>) =
        
        member _.ExecuteConsolidation() =
            task {
                logger.LogInformation("Starting Gordon-assisted consolidation")
                
                // Step 1: Initial infrastructure analysis
                let! healthAnalysis = gordonService.AnalyzeInfrastructure(ContainerHealth)
                match healthAnalysis with
                | Ok analysis ->
                    logger.LogInformation("Gordon health analysis complete. Score: {Score}", analysis.HealthScore)
                    
                    if analysis.HealthScore < 50 then
                        logger.LogWarning("Gordon detected critical issues. Consolidation may be risky.")
                        for issue in analysis.CriticalIssues do
                            logger.LogError("Critical: {Issue}", issue)
                    
                    // Step 2: Get consolidation strategy
                    let! strategyResult = gordonService.GetRecommendations("Docker container consolidation for TARS stack")
                    match strategyResult with
                    | Ok recommendations ->
                        logger.LogInformation("Gordon provided {Count} recommendations", recommendations.Length)
                        
                        // Step 3: Execute high-priority recommendations
                        for rec in recommendations |> List.filter (fun r -> r.Priority <= 3) do
                            logger.LogInformation("Executing: {Action}", rec.Action)
                            let! monitorResult = gordonService.MonitorOperation(rec.Action)
                            match monitorResult with
                            | Ok status -> logger.LogInformation("Status: {Status}", status)
                            | Error error -> logger.LogError("Failed: {Error}", error)
                        
                        return Ok "Gordon-assisted consolidation completed"
                    | Error error ->
                        logger.LogError("Failed to get Gordon recommendations: {Error}", error)
                        return Error error
                | Error error ->
                    logger.LogError("Gordon health analysis failed: {Error}", error)
                    return Error error
            }
        
        member _.GetConsolidationStatus() =
            task {
                let! analysis = gordonService.AnalyzeInfrastructure(ContainerHealth)
                match analysis with
                | Ok result ->
                    return Ok {|
                        HealthScore = result.HealthScore
                        Summary = result.Summary
                        CriticalIssues = result.CriticalIssues.Length
                        Warnings = result.Warnings.Length
                        LastAnalysis = result.Timestamp
                    |}
                | Error error ->
                    return Error error
            }
