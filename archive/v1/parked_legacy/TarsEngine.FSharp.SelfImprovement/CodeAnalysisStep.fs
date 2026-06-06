namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Threading.Tasks
open System.Net.Http
open System.Collections.Generic
open Microsoft.Extensions.Logging
open System.Text.Json
open System.Text.RegularExpressions
open ImprovementTypes
open ModelSelection

/// <summary>
/// Represents a step handler function
/// </summary>
type StepHandler = WorkflowState -> Task<StepResult>

/// <summary>
/// Represents a code improvement opportunity
/// </summary>
type ImprovementOpportunity = {
    /// <summary>
    /// The file path
    /// </summary>
    FilePath: string

    /// <summary>
    /// The pattern ID that can be applied
    /// </summary>
    PatternId: string

    /// <summary>
    /// The pattern name
    /// </summary>
    PatternName: string

    /// <summary>
    /// The confidence score (0-1)
    /// </summary>
    Confidence: float

    /// <summary>
    /// The line number where the pattern can be applied
    /// </summary>
    LineNumber: int option

    /// <summary>
    /// The code snippet that matches the pattern
    /// </summary>
    CodeSnippet: string option
}

/// <summary>
/// Module for the code analysis step in the autonomous improvement workflow
/// </summary>
module CodeAnalysisStep =
    /// <summary>
    /// The path to the knowledge base file
    /// </summary>
    let knowledgeBasePath = "knowledge_base.json"

    /// <summary>
    /// The path to the opportunities file
    /// </summary>
    let opportunitiesPath = "improvement_opportunities.json"

    let private severityToConfidence (severity: Severity) =
        match severity with
        | Severity.Critical -> 0.95
        | Severity.High -> 0.85
        | Severity.Medium -> 0.7
        | Severity.Low -> 0.55
        | _ -> 0.5

    let private sanitizeId (name: string) =
        if String.IsNullOrWhiteSpace(name) then
            $"op-{Guid.NewGuid():N}"
        else
            name.Trim().ToLowerInvariant().Replace(" ", "-")

    /// <summary>
    /// Gets all code files in a directory
    /// </summary>
    let getCodeFiles (directory: string) =
        if Directory.Exists(directory) then
            let extensions = [".cs"; ".fs"; ".fsx"; ".fsi"; ".xaml"; ".xml"; ".json"; ".md"]

            Directory.GetFiles(directory, "*.*", SearchOption.AllDirectories)
            |> Array.filter (fun file ->
                let ext = Path.GetExtension(file).ToLowerInvariant()
                extensions |> List.contains ext)
            |> Array.toList
        else
            []

    /// <summary>
    /// Loads the knowledge base
    /// </summary>
    let loadKnowledgeBase () =
        task {
            if File.Exists(knowledgeBasePath) then
                let! json = File.ReadAllTextAsync(knowledgeBasePath)
                return JsonSerializer.Deserialize<JsonElement>(json)
            else
                return JsonDocument.Parse("{}").RootElement
        }

    /// <summary>
    /// Matches a pattern against a file
    /// </summary>
    let matchPattern (pattern: JsonElement) (filePath: string) =
        task {
            try
                // Get the pattern details
                let mutable id = Unchecked.defaultof<JsonElement>
                let patternId =
                    if pattern.TryGetProperty("id", &id) then
                        id.GetString()
                    else
                        Guid.NewGuid().ToString()

                let mutable name = Unchecked.defaultof<JsonElement>
                let patternName =
                    if pattern.TryGetProperty("name", &name) then
                        name.GetString()
                    else
                        "Unnamed Pattern"

                let mutable patternProp = Unchecked.defaultof<JsonElement>
                let patternText =
                    if pattern.TryGetProperty("pattern", &patternProp) then
                        patternProp.GetString()
                    else
                        ""

                let mutable contextProp = Unchecked.defaultof<JsonElement>
                let context =
                    if pattern.TryGetProperty("context", &contextProp) then
                        contextProp.GetString()
                    else
                        ""

                // Check if the file extension matches the context
                let fileExt = Path.GetExtension(filePath).ToLowerInvariant()
                let isContextMatch =
                    match context.ToLowerInvariant() with
                    | "csharp" | "c#" -> fileExt = ".cs"
                    | "fsharp" | "f#" -> fileExt = ".fs" || fileExt = ".fsx" || fileExt = ".fsi"
                    | "xaml" -> fileExt = ".xaml"
                    | "xml" -> fileExt = ".xml"
                    | "json" -> fileExt = ".json"
                    | "markdown" | "md" -> fileExt = ".md"
                    | _ -> true

                if not isContextMatch || String.IsNullOrWhiteSpace(patternText) then
                    return []
                else
                    // Read the file content
                    let! content = File.ReadAllTextAsync(filePath)

                    // Create a regex for the pattern
                    // Note: This is a simplified approach. In a real implementation,
                    // you would need a more sophisticated pattern matching algorithm.
                    let regex = Regex(patternText, RegexOptions.IgnoreCase)
                    let matches = regex.Matches(content)

                    // Create opportunities for each match
                    return
                        matches
                        |> Seq.map (fun m ->
                            // Calculate the line number
                            let lineNumber =
                                content.Substring(0, m.Index).Split('\n').Length

                            // Create the opportunity
                            {
                                FilePath = filePath
                                PatternId = patternId
                                PatternName = patternName
                                Confidence = 0.8 // Fixed confidence for now
                                LineNumber = Some lineNumber
                                CodeSnippet = Some m.Value
                            })
                        |> Seq.toList
            with ex ->
                // Log the error and continue
                return []
        }

    /// <summary>
    /// Analyzes a file for improvement opportunities
    /// </summary>
    let analyzeFile (logger: ILogger) (patterns: JsonElement seq) (filePath: string) =
        task {
            try
                logger.LogInformation("Analyzing file: {FilePath}", filePath)

                // Match each pattern against the file
                let! opportunitiesLists =
                    patterns
                    |> Seq.map (fun pattern -> matchPattern pattern filePath)
                    |> Task.WhenAll

                // Flatten the lists
                let result = opportunitiesLists |> Array.collect List.toArray |> Array.toList
                return result
            with ex ->
                logger.LogError(ex, "Error analyzing file: {FilePath}", filePath)
                return []
        }

    /// <summary>
    /// Gets the code analysis step handler
    /// </summary>
    let getHandler (logger: ILogger) : WorkflowState -> Task<StepResult> =
        fun state ->
            task {
                logger.LogInformation("Starting code analysis step")

                // Get Ollama endpoint and model
                let defaultEndpoint =
                    let envVar = Environment.GetEnvironmentVariable("OLLAMA_BASE_URL")
                    if String.IsNullOrEmpty(envVar) then "http://localhost:11434" else envVar

                let selection =
                    ModelSelection.selectModel logger "local_reasoning" "llama3" (Some defaultEndpoint)

                let ollamaEndpoint = selection.Endpoint |> Option.defaultValue defaultEndpoint
                let ollamaModel = selection.ModelName

                // Check if we're using Docker / local inference
                let useOllama =
                    match Environment.GetEnvironmentVariable("OLLAMA_USE_DOCKER") with
                    | null
                    | "" ->
                        selection.Provider.IndexOf("ollama", StringComparison.OrdinalIgnoreCase) >= 0
                    | value -> value.Equals("true", StringComparison.OrdinalIgnoreCase)

                logger.LogInformation("Using code-analysis model {ModelName} from provider {Provider}", ollamaModel, selection.Provider)
                logger.LogInformation("LLM endpoint resolved to: {OllamaEndpoint}", ollamaEndpoint)
                logger.LogInformation("Ollama integration enabled: {UseOllama}", useOllama)
                selection.Notes
                |> Option.iter (fun note -> logger.LogInformation("Model recommendation note: {Note}", note))
                if selection.RequiresApiKey then
                    logger.LogInformation("Selected model requires API credentials; ensure the required environment variables are configured.")

                // Load the knowledge base
                let! knowledgeBase = loadKnowledgeBase()

                // Get the number of items in the knowledge base
                let itemCount =
                    match knowledgeBase.TryGetProperty("items") with
                    | true, items -> items.GetArrayLength()
                    | _ -> 0

                logger.LogInformation("Loaded knowledge base with {ItemCount} items", itemCount)

                let knowledgePatterns =
                    let mutable patternsProp = Unchecked.defaultof<JsonElement>
                    if knowledgeBase.TryGetProperty("patterns", &patternsProp) then
                        patternsProp.EnumerateArray() |> Seq.toList
                    else
                        []

                let httpClient =
                    if useOllama then new HttpClient()
                    else null

                let analyzerOpt =
                    if useOllama then
                        Some (OllamaCodeAnalyzer.OllamaCodeAnalyzer(httpClient, logger))
                    else
                        None

                let! ollamaAvailable =
                    match analyzerOpt with
                    | Some analyzer -> analyzer.IsAvailableAsync() |> Async.StartAsTask
                    | None -> Task.FromResult false

                let useOllamaAnalysis = useOllama && ollamaAvailable

                if useOllama && not useOllamaAnalysis then
                    logger.LogWarning("Ollama requested but endpoint was unavailable. Falling back to pattern-based analysis.")

                // Get all code files in the target directories
                let codeFiles =
                    state.TargetDirectories
                    |> List.collect getCodeFiles

                logger.LogInformation("Found {FileCount} code files in the target directories", codeFiles.Length)

                // Analyze each file
                let opportunities = ResizeArray<ImprovementOpportunity>()
                let mutable analyzedFiles = 0

                try
                    for filePath in codeFiles do
                        try
                            // Read the file content
                            let! content = File.ReadAllTextAsync(filePath)

                            // Baseline pattern recognition
                            let patternResult = PatternRecognition.analyzeFile filePath content
                            patternResult.Issues
                            |> List.iter (fun issue ->
                                opportunities.Add {
                                    FilePath = filePath
                                    PatternId = sanitizeId issue.Name
                                    PatternName = issue.Name
                                    Confidence = severityToConfidence issue.Severity
                                    LineNumber = None
                                    CodeSnippet = issue.Example
                                })

                            // Knowledge base driven patterns (if any)
                            if not (List.isEmpty knowledgePatterns) then
                                let! kbOpportunities = analyzeFile logger knowledgePatterns filePath
                                kbOpportunities |> List.iter opportunities.Add

                            // Ollama AI analysis
                            if useOllamaAnalysis then
                                match analyzerOpt with
                                | Some analyzer ->
                                    try
                                        logger.LogInformation("Analyzing file using Ollama: {FilePath}", filePath)
                                        let! aiResult = analyzer.AnalyzeCodeAsync(filePath, content, ollamaModel) |> Async.StartAsTask
                                        match aiResult with
                                        | Some analysis ->
                                            analysis.Issues
                                            |> List.iter (fun issue ->
                                                opportunities.Add {
                                                    FilePath = filePath
                                                    PatternId = "ollama-" + sanitizeId issue.Name
                                                    PatternName = issue.Name
                                                    Confidence = severityToConfidence issue.Severity
                                                    LineNumber = None
                                                    CodeSnippet = issue.Example
                                                })
                                        | None ->
                                            logger.LogDebug("Ollama did not return actionable issues for {FilePath}", filePath)
                                    with ex ->
                                        logger.LogError(ex, "Ollama analysis failed for {FilePath}", filePath)
                                | None -> ()

                            analyzedFiles <- analyzedFiles + 1
                        with ex ->
                            logger.LogError(ex, "Error analyzing file: {FilePath}", filePath)
                finally
                    if not (isNull httpClient) then
                        httpClient.Dispose()

                let distinctOpportunities =
                    opportunities
                    |> Seq.toList
                    |> List.distinctBy (fun op ->
                        let lineKey = op.LineNumber |> Option.map string |> Option.defaultValue "none"
                        let snippet = op.CodeSnippet |> Option.defaultValue ""
                        $"{op.FilePath}:{op.PatternId}:{lineKey}:{snippet}")

                logger.LogInformation("Found {OpportunityCount} improvement opportunities in {AnalyzedFiles} files",
                                     distinctOpportunities.Length, analyzedFiles)

                // Save the opportunities to a file
                let json = JsonSerializer.Serialize(distinctOpportunities, JsonSerializerOptions(WriteIndented = true))
                do! File.WriteAllTextAsync(opportunitiesPath, json)

                // Return the result data
                return Ok (Map.ofList [
                    "opportunities_path", opportunitiesPath
                    "opportunities_count", distinctOpportunities.Length.ToString()
                    "files_analyzed", analyzedFiles.ToString()
                    "analysis_time", DateTime.UtcNow.ToString("o")
                    "used_ollama", useOllamaAnalysis.ToString()
                    "model_provider", selection.Provider
                    "model_name", ollamaModel
                ])
            }
