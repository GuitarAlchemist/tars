namespace TarsEngine.FSharp.SelfImprovement

open System
open System.Net.Http
open System.Text
open System.Text.Json
open Microsoft.Extensions.Logging
open ImprovementTypes
open MetaReasoningCritic

/// AI-powered code analyzer using Ollama
module OllamaCodeAnalyzer =
    
    type OllamaRequest = {
        model: string
        prompt: string
        stream: bool
        options: Map<string, obj> option
    }
    
    type OllamaResponse = {
        response: string
        completed: bool
    }

    type OllamaIssue = {
        line: int
        severity: string
        issueType: string
        description: string
        suggestion: string
    }

    type OllamaAnalysis = {
        issues: OllamaIssue array
        overall_quality: float
        summary: string
    }

    type OllamaModelInfo = {
        name: string
    }

    type OllamaModelsResponse = {
        models: OllamaModelInfo array
    }
    
    let private tryDeserialize<'T> (payload: string) =
        let value = JsonSerializer.Deserialize<'T>(payload)
        if obj.ReferenceEquals(value, null) then None else Some value
    
    /// Ollama code analyzer service
    type OllamaCodeAnalyzer(httpClient: HttpClient, logger: ILogger) =
        
        let baseUrl =
            match Environment.GetEnvironmentVariable("OLLAMA_BASE_URL") with
            | null
            | "" -> "http://localhost:11434"
            | value -> value
        
        /// Analyze code using Ollama AI
        member this.AnalyzeCodeAsync(filePath: string, content: string, model: string) =
            async {
                try
                    logger.LogInformation("Analyzing code with Ollama: {FilePath}", filePath)

                    let adaptiveDirectives =
                        MetaReasoningCritic.loadPromptAdjustments MetaReasoningCritic.defaultPromptPath
                        |> Map.toList
                        |> List.map snd
                        |> List.distinct

                    let adaptiveBlock =
                        if adaptiveDirectives.IsEmpty then
                            []
                        else
                            [ "Adaptive guardrails from recent critic training:" ]
                            @ (adaptiveDirectives |> List.map (fun directive -> "- " + directive))
                            @ [ "" ]

                    let prompt =
                        [
                            "You are an expert code analyzer. Analyze the following code and provide improvement suggestions.";
                            "";
                            $"File: {filePath}";
                            "Code:";
                            "`";
                            content;
                            "`";
                            "";
                        ]
                        @ adaptiveBlock @
                        [
                            "Please provide your analysis in the following JSON format:";
                            "{";
                            "  \"issues\": [";
                            "    {";
                            "      \"line\": 0,";
                            "      \"severity\": \"High|Medium|Low|Critical\",";
                            "      \"type\": \"Performance|Maintainability|Security|Documentation\",";
                            "      \"description\": \"Description of the issue\",";
                            "      \"suggestion\": \"How to fix it\"";
                            "    }";
                            "  ],";
                            "  \"overall_quality\": 8.5,";
                            "  \"summary\": \"Overall assessment of the code quality\"";
                            "}";
                            "";
                            "Focus on real issues that would improve code quality, performance, or maintainability."
                        ]
                        |> String.concat Environment.NewLine
                    
                    let request = {
                        model = model
                        prompt = prompt
                        stream = false
                        options = Some (Map.ofList [
                            ("temperature", 0.3 :> obj)
                            ("top_p", 0.9 :> obj)
                        ])
                    }
                    
                    let json = JsonSerializer.Serialize(request)
                    let httpContent = new StringContent(json, Encoding.UTF8, "application/json")
                    
                    let! response = httpClient.PostAsync($"{baseUrl}/api/generate", httpContent) |> Async.AwaitTask
                    let! responseContent = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                    
                    if response.IsSuccessStatusCode then
                        match tryDeserialize<OllamaResponse>(responseContent) with
                        | None ->
                            logger.LogWarning("Received empty response from Ollama.")
                            return None
                        | Some ollamaResponse when String.IsNullOrWhiteSpace(ollamaResponse.response) ->
                            logger.LogWarning("Ollama response payload was empty.")
                            return None
                        | Some ollamaResponse ->
                            try
                                match tryDeserialize<OllamaAnalysis>(ollamaResponse.response) with
                                | None ->
                                    logger.LogWarning("Ollama analysis payload was null")
                                    return None
                                | Some analysisResult ->
                                    let issues = 
                                        analysisResult.issues
                                        |> Array.map (fun issue ->
                                            let severity = 
                                                match issue.severity with
                                                | "Critical" -> Severity.Critical
                                                | "High" -> Severity.High
                                                | "Medium" -> Severity.Medium
                                                | _ -> Severity.Low
                                            
                                            let patternType =
                                                match issue.issueType with
                                                | "Performance" -> PatternType.Performance
                                                | "Security" -> PatternType.Security
                                                | "Documentation" -> PatternType.Documentation
                                                | _ -> PatternType.Maintainability
                                            
                                            {
                                                Name = issue.description
                                                Description = issue.description
                                                PatternType = patternType
                                                Severity = severity
                                                Example = None
                                                Recommendation = issue.suggestion
                                            }
                                        )
                                        |> Array.toList

                                    let result = {
                                        FilePath = filePath
                                        Issues = issues
                                        OverallScore = analysisResult.overall_quality
                                        Recommendations = issues |> List.map (fun i -> i.Recommendation)
                                        AnalyzedAt = DateTime.UtcNow
                                    }

                                    logger.LogInformation("Ollama analysis completed: {IssueCount} issues found", issues.Length)
                                    return Some result
                            with
                            | ex ->
                                logger.LogWarning(ex, "Failed to parse Ollama response, using fallback analysis")
                                return None
                    else
                        logger.LogError("Ollama request failed: {StatusCode}", response.StatusCode)
                        return None
                        
                with
                | ex ->
                    logger.LogError(ex, "Error during Ollama analysis")
                    return None
            }
        
        /// Check if Ollama is available
        member this.IsAvailableAsync() =
            async {
                try
                    let! response = httpClient.GetAsync($"{baseUrl}/api/tags") |> Async.AwaitTask
                    return response.IsSuccessStatusCode
                with
                | _ -> return false
            }
        
        /// Get available models
        member this.GetAvailableModelsAsync() =
            async {
                try
                    let! response = httpClient.GetAsync($"{baseUrl}/api/tags") |> Async.AwaitTask
                    if response.IsSuccessStatusCode then
                        let! content = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                        let modelsResponse = JsonSerializer.Deserialize<OllamaModelsResponse>(content)
                        return modelsResponse.models |> Array.map (fun m -> m.name) |> Array.toList
                    else
                        return []
                with
                | _ -> return []
            }
