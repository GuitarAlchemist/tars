namespace TarsEngine.FSharp.Core

open System
open System.IO
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// LLM analysis request
type LLMAnalysisRequest = {
    ComponentName: string
    SourceCode: string
    FilePath: string
    Timestamp: DateTime
    AnalysisType: string
    Context: Map<string, obj>
}

/// LLM analysis response
type LLMAnalysisResponse = {
    ComponentName: string
    AnalysisType: string
    GeneratedInsights: string[]
    DesignRecommendations: string[]
    ArchitecturalAnalysis: string
    QualityAssessment: string
    ImprovementSuggestions: string[]
    ExecutionTime: float
    Timestamp: DateTime
    LLMModel: string
    TokensUsed: int
}

/// Real-time LLM analyzer that generates fresh insights
type RealTimeLLMAnalyzer(logger: ILogger<RealTimeLLMAnalyzer>) =
    
    let httpClient = new HttpClient()
    
    /// Generate dynamic prompt based on component analysis
    member private this.GenerateDynamicPrompt(request: LLMAnalysisRequest) =
        let timestamp = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss.fff")
        let codeLines = request.SourceCode.Split('\n').Length
        let codeSize = request.SourceCode.Length
        
        sprintf """
REAL-TIME COMPONENT ANALYSIS REQUEST
===================================
Timestamp: %s
Component: %s
File: %s
Code Size: %d characters, %d lines

ANALYSIS REQUIREMENTS:
1. Provide FRESH analysis - no templates or canned responses
2. Focus on actual code structure and patterns
3. Generate specific, actionable recommendations
4. Analyze architectural decisions made in this component

SOURCE CODE TO ANALYZE:
```fsharp
%s
```

PROVIDE ANALYSIS IN THIS FORMAT:
1. ARCHITECTURAL ANALYSIS: Describe the actual architectural patterns used
2. DESIGN QUALITY: Assess the design decisions made
3. SPECIFIC IMPROVEMENTS: List concrete improvements with code examples
4. INTEGRATION OPPORTUNITIES: How this component could work with others
5. REAL-TIME INSIGHTS: Fresh observations about this specific code

IMPORTANT: Base your analysis ONLY on the actual code provided. Generate fresh insights every time.
""" timestamp request.ComponentName request.FilePath codeSize codeLines request.SourceCode
    
    /// Call local LLM (Ollama) for analysis
    member private this.CallLocalLLM(prompt: string, model: string) =
        async {
            try
                let requestBody = {|
                    model = model
                    prompt = prompt
                    stream = false
                    options = {|
                        temperature = 0.7
                        top_p = 0.9
                        max_tokens = 2000
                    |}
                |}
                
                let json = JsonSerializer.Serialize(requestBody)
                let content = new StringContent(json, Encoding.UTF8, "application/json")
                
                let! response = httpClient.PostAsync("http://localhost:11434/api/generate", content) |> Async.AwaitTask
                
                if response.IsSuccessStatusCode then
                    let! responseText = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                    let responseJson = JsonDocument.Parse(responseText)
                    let generatedText = responseJson.RootElement.GetProperty("response").GetString()
                    return Some generatedText
                else
                    logger.LogWarning(sprintf "LLM call failed with status: %s" (response.StatusCode.ToString()))
                    return None
            with
            | ex ->
                logger.LogError(ex, "Failed to call local LLM")
                return None
        }
    
    /// Parse LLM response into structured analysis
    member private this.ParseLLMResponse(response: string, request: LLMAnalysisRequest, executionTime: float) =
        try
            let lines = response.Split('\n') |> Array.map (fun line -> line.Trim()) |> Array.filter (fun line -> not (String.IsNullOrEmpty(line)))
            
            let mutable architecturalAnalysis = ""
            let mutable qualityAssessment = ""
            let insights = ResizeArray<string>()
            let recommendations = ResizeArray<string>()
            let improvements = ResizeArray<string>()
            
            let mutable currentSection = ""
            
            for line in lines do
                if line.Contains("ARCHITECTURAL ANALYSIS") then
                    currentSection <- "ARCHITECTURE"
                elif line.Contains("DESIGN QUALITY") then
                    currentSection <- "QUALITY"
                elif line.Contains("SPECIFIC IMPROVEMENTS") then
                    currentSection <- "IMPROVEMENTS"
                elif line.Contains("INTEGRATION OPPORTUNITIES") then
                    currentSection <- "RECOMMENDATIONS"
                elif line.Contains("REAL-TIME INSIGHTS") then
                    currentSection <- "INSIGHTS"
                else
                    match currentSection with
                    | "ARCHITECTURE" -> architecturalAnalysis <- architecturalAnalysis + line + " "
                    | "QUALITY" -> qualityAssessment <- qualityAssessment + line + " "
                    | "IMPROVEMENTS" -> if line.Length > 10 then improvements.Add(line)
                    | "RECOMMENDATIONS" -> if line.Length > 10 then recommendations.Add(line)
                    | "INSIGHTS" -> if line.Length > 10 then insights.Add(line)
                    | _ -> ()
            
            // If parsing failed, extract insights from the raw response
            if insights.Count = 0 then
                let sentences = response.Split([|'.'; '!'; '?'|], StringSplitOptions.RemoveEmptyEntries)
                for sentence in sentences do
                    let trimmed = sentence.Trim()
                    if trimmed.Length > 20 then
                        insights.Add(trimmed)
            
            {
                ComponentName = request.ComponentName
                AnalysisType = request.AnalysisType
                GeneratedInsights = insights.ToArray()
                DesignRecommendations = recommendations.ToArray()
                ArchitecturalAnalysis = if String.IsNullOrEmpty(architecturalAnalysis) then response.Substring(0, Math.Min(500, response.Length)) else architecturalAnalysis.Trim()
                QualityAssessment = if String.IsNullOrEmpty(qualityAssessment) then "Quality assessment generated from LLM analysis" else qualityAssessment.Trim()
                ImprovementSuggestions = improvements.ToArray()
                ExecutionTime = executionTime
                Timestamp = DateTime.UtcNow
                LLMModel = "ollama-local"
                TokensUsed = response.Length / 4 // Rough estimate
            }
        with
        | ex ->
            logger.LogError(ex, "Failed to parse LLM response")
            {
                ComponentName = request.ComponentName
                AnalysisType = request.AnalysisType
                GeneratedInsights = [| sprintf "LLM analysis completed at %s" (DateTime.UtcNow.ToString()) |]
                DesignRecommendations = [| "Consider reviewing component architecture" |]
                ArchitecturalAnalysis = response.Substring(0, Math.Min(200, response.Length))
                QualityAssessment = "Analysis generated but parsing failed"
                ImprovementSuggestions = [| "Review LLM response parsing" |]
                ExecutionTime = executionTime
                Timestamp = DateTime.UtcNow
                LLMModel = "ollama-local"
                TokensUsed = response.Length / 4
            }
    
    /// Analyze component using real-time LLM
    member this.AnalyzeComponentWithLLM(componentName: string, filePath: string, analysisType: string) =
        async {
            try
                let startTime = DateTime.UtcNow
                logger.LogInformation(sprintf "Starting real-time LLM analysis of component: %s" componentName)
                
                // Read source code
                let sourceCode = 
                    if File.Exists(filePath) then
                        File.ReadAllText(filePath)
                    else
                        sprintf "// File not found: %s" filePath
                
                let request = {
                    ComponentName = componentName
                    SourceCode = sourceCode
                    FilePath = filePath
                    Timestamp = startTime
                    AnalysisType = analysisType
                    Context = Map.ofList [
                        ("fileSize", sourceCode.Length :> obj)
                        ("lineCount", sourceCode.Split('\n').Length :> obj)
                        ("analysisTime", startTime :> obj)
                    ]
                }
                
                // Generate dynamic prompt
                let prompt = this.GenerateDynamicPrompt(request)
                
                // Call LLM with different models based on analysis type
                let model = 
                    match analysisType with
                    | "ARCHITECTURE" -> "codellama:7b"
                    | "QUALITY" -> "mistral:7b"
                    | "PERFORMANCE" -> "codellama:7b"
                    | _ -> "llama2:7b"
                
                let! llmResponse = this.CallLocalLLM(prompt, model)
                
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                match llmResponse with
                | Some response ->
                    let analysis = this.ParseLLMResponse(response, request, executionTime)
                    logger.LogInformation(sprintf "LLM analysis completed for %s in %.1fms" componentName executionTime)
                    return Some analysis
                | None ->
                    logger.LogWarning(sprintf "LLM analysis failed for component: %s" componentName)
                    return None
            with
            | ex ->
                logger.LogError(ex, sprintf "Real-time LLM analysis failed for component: %s" componentName)
                return None
        }
    
    /// Analyze multiple components concurrently
    member this.AnalyzeMultipleComponents(components: (string * string)[], analysisType: string) =
        async {
            try
                logger.LogInformation(sprintf "Starting concurrent LLM analysis of %d components" components.Length)
                
                let! analyses = 
                    components
                    |> Array.map (fun (name, path) -> this.AnalyzeComponentWithLLM(name, path, analysisType))
                    |> Async.Parallel
                
                let successful = analyses |> Array.choose id
                let failed = analyses.Length - successful.Length
                
                logger.LogInformation(sprintf "LLM analysis completed: %d successful, %d failed" successful.Length failed)
                
                return {|
                    TotalComponents = components.Length
                    SuccessfulAnalyses = successful.Length
                    FailedAnalyses = failed
                    Analyses = successful
                    TotalExecutionTime = successful |> Array.sumBy (fun a -> a.ExecutionTime)
                    AverageExecutionTime = if successful.Length > 0 then successful |> Array.averageBy (fun a -> a.ExecutionTime) else 0.0
                    Timestamp = DateTime.UtcNow
                |}
            with
            | ex ->
                logger.LogError(ex, "Failed to analyze multiple components")
                return {|
                    TotalComponents = components.Length
                    SuccessfulAnalyses = 0
                    FailedAnalyses = components.Length
                    Analyses = [||]
                    TotalExecutionTime = 0.0
                    AverageExecutionTime = 0.0
                    Timestamp = DateTime.UtcNow
                |}
        }
    
    /// Generate fresh design insights for component
    member this.GenerateFreshDesignInsights(componentAnalysis: ComponentAnalysis) =
        async {
            try
                logger.LogInformation(sprintf "Generating fresh design insights for: %s" componentAnalysis.Name)
                
                // Generate multiple analysis types for comprehensive insights
                let analysisTypes = ["ARCHITECTURE"; "QUALITY"; "PERFORMANCE"]
                
                let! analyses = 
                    analysisTypes
                    |> List.map (fun analysisType -> this.AnalyzeComponentWithLLM(componentAnalysis.Name, componentAnalysis.FilePath, analysisType))
                    |> Async.Parallel
                
                let validAnalyses = analyses |> Array.choose id
                
                if validAnalyses.Length > 0 then
                    let combinedInsights = validAnalyses |> Array.collect (fun a -> a.GeneratedInsights)
                    let combinedRecommendations = validAnalyses |> Array.collect (fun a -> a.DesignRecommendations)
                    let combinedImprovements = validAnalyses |> Array.collect (fun a -> a.ImprovementSuggestions)
                    
                    return {|
                        ComponentName = componentAnalysis.Name
                        FreshInsights = combinedInsights
                        DesignRecommendations = combinedRecommendations
                        ImprovementSuggestions = combinedImprovements
                        AnalysisCount = validAnalyses.Length
                        TotalExecutionTime = validAnalyses |> Array.sumBy (fun a -> a.ExecutionTime)
                        GeneratedAt = DateTime.UtcNow
                        LLMModelsUsed = validAnalyses |> Array.map (fun a -> a.LLMModel) |> Array.distinct
                    |}
                else
                    return {|
                        ComponentName = componentAnalysis.Name
                        FreshInsights = [| sprintf "Fresh analysis generated at %s - LLM unavailable" (DateTime.UtcNow.ToString()) |]
                        DesignRecommendations = [| "Consider manual code review" |]
                        ImprovementSuggestions = [| "Ensure LLM service is available" |]
                        AnalysisCount = 0
                        TotalExecutionTime = 0.0
                        GeneratedAt = DateTime.UtcNow
                        LLMModelsUsed = [||]
                    |}
            with
            | ex ->
                logger.LogError(ex, sprintf "Failed to generate fresh insights for: %s" componentAnalysis.Name)
                return {|
                    ComponentName = componentAnalysis.Name
                    FreshInsights = [| sprintf "Analysis failed at %s: %s" (DateTime.UtcNow.ToString()) ex.Message |]
                    DesignRecommendations = [| "Review component manually" |]
                    ImprovementSuggestions = [| "Fix LLM integration issues" |]
                    AnalysisCount = 0
                    TotalExecutionTime = 0.0
                    GeneratedAt = DateTime.UtcNow
                    LLMModelsUsed = [||]
                |}
        }
