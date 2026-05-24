namespace TarsEngine.SelfImprovement

open System
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Module for analyzing code using Ollama
/// </summary>
module OllamaCodeAnalyzer =
    /// <summary>
    /// Issue model
    /// </summary>
    type CodeIssue = {
        severity: string
        description: string
        location: string
        suggestion: string
    }

    /// <summary>
    /// Improvement model
    /// </summary>
    type CodeImprovement = {
        ``type``: string
        description: string
        current_code: string
        improved_code: string
        rationale: string
    }

    /// <summary>
    /// Metric model
    /// </summary>
    type CodeMetric = {
        name: string
        value: string
        description: string
    }

    /// <summary>
    /// Code analysis result model
    /// </summary>
    type CodeAnalysisResult = {
        issues: CodeIssue[]
        improvements: CodeImprovement[]
        metrics: CodeMetric[]
    }

    /// <summary>
    /// Analyzes code using Ollama
    /// </summary>
    let analyzeCodeWithOllama (logger: ILogger) (filePath: string) (fileContent: string) (knowledgeBase: JsonElement) (ollamaEndpoint: string) (model: string) =
        task {
            try
                logger.LogInformation("Analyzing code using Ollama with model {Model}", model)

                // Extract knowledge context (simplified for now)
                let knowledgeContext = "No specific knowledge items available."

                // Create a prompt for code analysis
                let prompt =
                    $"You are an expert code reviewer and software architect specializing in C# and F#.\n\n" +
                    $"I'll provide you with code from a file and some knowledge context. Please analyze the code and suggest improvements.\n\n" +
                    $"File path: {filePath}\n\n" +
                    $"Knowledge context:\n{knowledgeContext}\n\n" +
                    $"Code to analyze:\n```\n{fileContent}\n```\n\n" +
                    $"Please provide your analysis in JSON format with the following structure:\n" +
                    $"{{\n" +
                    $"  \"issues\": [\n" +
                    $"    {{ \"severity\": \"high|medium|low\", \"description\": \"Issue description\", \"location\": \"Line number or method name\", \"suggestion\": \"How to fix it\" }}\n" +
                    $"  ],\n" +
                    $"  \"improvements\": [\n" +
                    $"    {{ \"type\": \"performance|readability|maintainability|security\", \"description\": \"Improvement description\", \"current_code\": \"Current code snippet\", \"improved_code\": \"Improved code snippet\", \"rationale\": \"Why this is better\" }}\n" +
                    $"  ],\n" +
                    $"  \"metrics\": [\n" +
                    $"    {{ \"name\": \"Metric name\", \"value\": \"Metric value\", \"description\": \"Metric description\" }}\n" +
                    $"  ]\n" +
                    $"}}\n\n" +
                    $"Only return the JSON, no other text."

                // Create HTTP client
                use client = new HttpClient()

                // Create request
                let request = {
                    model = model
                    prompt = prompt
                    stream = false
                    options = {|
                        temperature = 0.3
                        num_predict = 4000
                    |}
                }

                // Send request to Ollama
                let! response = client.PostAsJsonAsync($"{ollamaEndpoint}/api/generate", request)

                // Check if request was successful
                if not response.IsSuccessStatusCode then
                    logger.LogError("Failed to analyze code using Ollama: {StatusCode} - {ReasonPhrase}", response.StatusCode, response.ReasonPhrase)
                    return None
                else
                    // Parse response
                    let! result = response.Content.ReadFromJsonAsync<OllamaResponse>()

                    // Extract JSON from response
                    let jsonPattern = "\{[\s\S]*\}"
                    let jsonMatch = Regex.Match(result.response, jsonPattern)

                    if jsonMatch.Success then
                        let jsonResponse = jsonMatch.Value

                        // Parse JSON
                        let options = JsonSerializerOptions()
                        options.PropertyNameCaseInsensitive <- true

                        let analysisResult = JsonSerializer.Deserialize<CodeAnalysisResult>(jsonResponse, options)

                        // Log analysis results
                        logger.LogInformation("Code analysis completed for {FilePath}", filePath)
                        logger.LogInformation("Found {IssueCount} issues and {ImprovementCount} potential improvements",
                                             analysisResult.issues.Length,
                                             analysisResult.improvements.Length)

                        return Some analysisResult
                    else
                        logger.LogWarning("Failed to analyze code using Ollama: Invalid response format")
                        return None
            with ex ->
                logger.LogError(ex, "Error analyzing code using Ollama")
                return None
        }

    /// <summary>
    /// Generates improvement suggestions based on code analysis
    /// </summary>
    let generateImprovementSuggestions (analysisResult: CodeAnalysisResult option) (filePath: string) =
        match analysisResult with
        | None -> []
        | Some result ->
            // Convert improvements to a list of improvement suggestions
            result.improvements
            |> Array.map (fun improvement ->
                {
                    FilePath = filePath
                    Type = improvement.``type``
                    Description = improvement.description
                    CurrentCode = improvement.current_code
                    ImprovedCode = improvement.improved_code
                    Rationale = improvement.rationale
                    Confidence = 0.8  // Default confidence
                } : OllamaAnalysisResult)
            |> Array.toList
