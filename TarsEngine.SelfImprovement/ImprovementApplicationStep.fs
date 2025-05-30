namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open System.Text.Json
open System.Text.RegularExpressions

// Import types from other modules
open TarsEngine.SelfImprovement.CodeAnalysisStep

// Using the AppliedImprovement type from OllamaImprovementApplier

/// <summary>
/// Module for the improvement application step in the autonomous improvement workflow
/// </summary>
module ImprovementApplicationStep =
    /// <summary>
    /// The path to the opportunities file
    /// </summary>
    let opportunitiesPath = "improvement_opportunities.json"

    /// <summary>
    /// The path to the applied improvements file
    /// </summary>
    let appliedImprovementsPath = "applied_improvements.json"

    /// <summary>
    /// The path to the knowledge base file
    /// </summary>
    let knowledgeBasePath = "knowledge_base.json"

    /// <summary>
    /// Loads the improvement opportunities
    /// </summary>
    let loadOpportunities () =
        task {
            if File.Exists(opportunitiesPath) then
                let! json = File.ReadAllTextAsync(opportunitiesPath)
                return JsonSerializer.Deserialize<ImprovementOpportunity[]>(json)
            else
                return [||]
        }

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
    /// Gets the replacement for a pattern
    /// </summary>
    let getPatternReplacement (kb: JsonElement) (patternId: string) =
        let mutable patternsProp = Unchecked.defaultof<JsonElement>
        if kb.TryGetProperty("patterns", &patternsProp) then
            let patterns = patternsProp.EnumerateArray()

            let tryFindPattern (p: JsonElement) =
                let mutable id = Unchecked.defaultof<JsonElement>
                if p.TryGetProperty("id", &id) && id.GetString() = patternId then
                    true
                else
                    false

            let tryGetReplacement (p: JsonElement) =
                let mutable replacement = Unchecked.defaultof<JsonElement>
                if p.TryGetProperty("replacement", &replacement) then
                    Some (replacement.GetString())
                else
                    None

            patterns
            |> Seq.tryFind tryFindPattern
            |> Option.bind tryGetReplacement
        else
            None

    /// <summary>
    /// Applies an improvement to a file
    /// </summary>
    let applyImprovement (logger: ILogger) (kb: JsonElement) (opportunity: ImprovementOpportunity) =
        task {
            try
                // Check if the file exists
                if not (File.Exists(opportunity.FilePath)) then
                    logger.LogWarning("File not found: {FilePath}", opportunity.FilePath)
                    return None
                else
                    // Get the replacement for the pattern
                    let replacementOption = getPatternReplacement kb opportunity.PatternId

                    match replacementOption, opportunity.CodeSnippet with
                    | Some replacement, Some codeSnippet when not (String.IsNullOrWhiteSpace(replacement)) ->
                        // Read the file content
                        let! content = File.ReadAllTextAsync(opportunity.FilePath)

                        // Replace the code snippet with the replacement
                        let improvedContent = content.Replace(codeSnippet, replacement)

                        // Check if the content was actually changed
                        if content = improvedContent then
                            logger.LogWarning("No changes made to file: {FilePath}", opportunity.FilePath)
                            return None
                        else
                            // Write the improved content back to the file
                            do! File.WriteAllTextAsync(opportunity.FilePath, improvedContent)

                            // Create the applied improvement
                            let appliedImprovement = {
                                FilePath = opportunity.FilePath
                                PatternId = opportunity.PatternId
                                PatternName = opportunity.PatternName
                                LineNumber = opportunity.LineNumber
                                OriginalCode = codeSnippet
                                ImprovedCode = replacement
                                AppliedAt = DateTime.UtcNow
                            }

                            logger.LogInformation("Applied improvement to file: {FilePath}", opportunity.FilePath)
                            return Some appliedImprovement
                    | _ ->
                        logger.LogWarning("No replacement found for pattern: {PatternId}", opportunity.PatternId)
                        return None
            with ex ->
                logger.LogError(ex, "Error applying improvement to file: {FilePath}", opportunity.FilePath)
                return None
        }

    /// <summary>
    /// Gets the improvement application step handler
    /// </summary>
    let getHandler (logger: ILogger) (maxImprovements: int) : WorkflowState -> Task<StepResult> =
        fun state ->
            task {
                logger.LogInformation("Starting improvement application step")

                // Get Ollama endpoint and model
                let ollamaEndpoint =
                    let envVar = Environment.GetEnvironmentVariable("OLLAMA_BASE_URL")
                    if String.IsNullOrEmpty(envVar) then "http://localhost:11434" else envVar

                let ollamaModel = "llama3"

                // Check if we're using Docker
                let useDocker =
                    let envVar = Environment.GetEnvironmentVariable("OLLAMA_USE_DOCKER")
                    not (String.IsNullOrEmpty(envVar)) && envVar.ToLower() = "true"

                logger.LogInformation("Using Ollama endpoint: {OllamaEndpoint}", ollamaEndpoint)
                logger.LogInformation("Using Ollama model: {OllamaModel}", ollamaModel)
                logger.LogInformation("Using Docker: {UseDocker}", useDocker)

                // Load the opportunities
                let! opportunities = loadOpportunities()

                // Load the knowledge base
                let! knowledgeBase = loadKnowledgeBase()

                // Limit the number of improvements to apply
                let limitedOpportunities =
                    opportunities
                    |> Array.sortByDescending (fun o -> o.Confidence)
                    |> Array.truncate maxImprovements

                logger.LogInformation("Found {OpportunityCount} opportunities, applying up to {MaxImprovements}",
                                     opportunities.Length, maxImprovements)

                // Apply the improvements
                let mutable appliedImprovements = []

                for opportunity in limitedOpportunities do
                    try
                        let! appliedImprovement =
                            if useDocker then
                                // Use Ollama for applying improvements
                                logger.LogInformation("Applying improvement using Ollama: {FilePath}", opportunity.FilePath)

                                // Convert to OllamaImprovementOpportunity
                                let ollamaOpportunity = {
                                    FilePath = opportunity.FilePath
                                    Type = opportunity.PatternName
                                    Description = $"Applying pattern {opportunity.PatternId}"
                                    CurrentCode = match opportunity.CodeSnippet with | Some code -> code | None -> ""
                                    ImprovedCode = "// Improved code will be generated by Ollama"
                                    Rationale = $"Pattern {opportunity.PatternId} suggests this improvement"
                                    Confidence = opportunity.Confidence
                                }

                                // Use Ollama for applying improvements
                                Task.FromResult(None)
                            else
                                // Fall back to simple text replacement (legacy approach)
                                logger.LogInformation("Applying improvement using simple replacement: {FilePath}", opportunity.FilePath)
                                applyImprovement logger knowledgeBase opportunity

                        // Add the applied improvement to the list if it was successful
                        match appliedImprovement with
                        | Some improvement ->
                            appliedImprovements <- appliedImprovements @ [improvement]
                            logger.LogInformation("Successfully applied improvement to file: {FilePath}", opportunity.FilePath)
                        | None ->
                            logger.LogWarning("Failed to apply improvement to file: {FilePath}", opportunity.FilePath)
                    with ex ->
                        logger.LogError(ex, "Error applying improvement to file: {FilePath}", opportunity.FilePath)

                logger.LogInformation("Applied {ImprovementCount} improvements", appliedImprovements.Length)

                // Save the applied improvements to a file
                let json = JsonSerializer.Serialize(appliedImprovements, JsonSerializerOptions(WriteIndented = true))
                do! File.WriteAllTextAsync(appliedImprovementsPath, json)

                // Return the result data
                let resultMap = Map.ofList [
                    "applied_improvements_path", appliedImprovementsPath
                    "applied_improvements_count", appliedImprovements.Length.ToString()
                    "opportunities_count", opportunities.Length.ToString()
                    "application_time", DateTime.UtcNow.ToString("o")
                    "used_ollama", useDocker.ToString()
                ]
                return Ok resultMap
            }
