namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Net.Http
open System.Net.Http.Json
open System.Text.Json
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open System.Collections.Generic

// Using the AppliedImprovement type from ImprovementTypes.fs

/// <summary>
/// Module for applying improvements using Ollama and pattern-based suggestions
/// </summary>
module OllamaImprovementApplier =

    /// <summary>
    /// Applies an improvement to a file using pattern-based suggestions
    /// </summary>
    let applyPatternBasedImprovement (logger: ILogger) (filePath: string) (patternMatch: PatternMatch) =
        task {
            try
                logger.LogInformation("Applying pattern-based improvement to file {FilePath} for pattern {PatternId}", filePath, patternMatch.PatternId)

                // Check if the file exists
                if not (File.Exists(filePath)) then
                    logger.LogWarning("File not found: {FilePath}", filePath)
                    return None
                else
                    // Read the file content
                    let! fileContent = File.ReadAllTextAsync(filePath)

                    // Create a backup of the original file
                    let backupPath = $"{filePath}.bak"
                    do! File.WriteAllTextAsync(backupPath, fileContent)
                    logger.LogInformation("Created backup of file at {BackupPath}", backupPath)

                    // Get the context (the line containing the match)
                    let contextLine = patternMatch.Context

                    // Find the suggestion for this pattern
                    let suggestionOption = ImprovementSuggestions.getSuggestion patternMatch.PatternId contextLine

                    match suggestionOption with
                    | None ->
                        logger.LogWarning("No improvement suggestion found for pattern {PatternId}", patternMatch.PatternId)
                        return None
                    | Some suggestion ->
                        // Apply the suggestion
                        let improvedLine = ImprovementSuggestions.applySuggestion suggestion contextLine

                        // Replace the line in the file content
                        let lines = fileContent.Split('\n')
                        if patternMatch.LineNumber > 0 && patternMatch.LineNumber <= lines.Length then
                            lines.[patternMatch.LineNumber - 1] <- improvedLine
                            let updatedContent = String.Join('\n', lines)

                            // Check if the content was actually changed
                            if fileContent = updatedContent then
                                logger.LogWarning("No changes made to file: {FilePath}", filePath)
                                return None
                            else
                                // Write the updated content back to the file
                                do! File.WriteAllTextAsync(filePath, updatedContent)

                                // Get the pattern name
                                let pattern = PatternRecognition.commonPatterns |> List.find (fun p -> p.Id = patternMatch.PatternId)

                                // Create the applied improvement
                                let appliedImprovement = {
                                    FilePath = filePath
                                    PatternId = patternMatch.PatternId
                                    PatternName = pattern.Name
                                    LineNumber = Some patternMatch.LineNumber
                                    OriginalCode = contextLine
                                    ImprovedCode = improvedLine
                                    AppliedAt = DateTime.UtcNow
                                }

                                logger.LogInformation("Successfully applied pattern-based improvement to file: {FilePath}", filePath)
                                return Some appliedImprovement
                        else
                            logger.LogWarning("Invalid line number {LineNumber} for file: {FilePath}", patternMatch.LineNumber, filePath)
                            return None
            with ex ->
                logger.LogError(ex, "Error applying pattern-based improvement to file: {FilePath}", filePath)
                return None
        }

    /// <summary>
    /// Applies an improvement to a file using Ollama
    /// </summary>
    let applyImprovementWithOllama (logger: ILogger) (opportunity: OllamaImprovementOpportunity) (ollamaEndpoint: string) (model: string) =
        task {
            try
                logger.LogInformation("Applying improvement to file {FilePath} using Ollama with model {Model}", opportunity.FilePath, model)

                // Check if the file exists
                if not (File.Exists(opportunity.FilePath)) then
                    logger.LogWarning("File not found: {FilePath}", opportunity.FilePath)
                    return None
                else
                    // Read the file content
                    let! fileContent = File.ReadAllTextAsync(opportunity.FilePath)

                    // Create a backup of the original file
                    let backupPath = $"{opportunity.FilePath}.bak"
                    do! File.WriteAllTextAsync(backupPath, fileContent)
                    logger.LogInformation("Created backup of file at {BackupPath}", backupPath)

                    // Check if the current code snippet exists in the file
                    if not (fileContent.Contains(opportunity.CurrentCode)) then
                        logger.LogWarning("Current code snippet not found in file: {FilePath}", opportunity.FilePath)
                        return None
                    else
                        // Create a prompt for applying the improvement
                        let prompt =
                            $"You are an expert software developer tasked with applying a code improvement.\n\n" +
                            $"I'll provide you with:\n" +
                            $"1. The file path\n" +
                            $"2. The current file content\n" +
                            $"3. The improvement to apply\n\n" +
                            $"Your task is to apply the improvement to the file content and return the entire updated file content.\n\n" +
                            $"File path: {opportunity.FilePath}\n\n" +
                            $"Improvement details:\n" +
                            $"- Type: {opportunity.Type}\n" +
                            $"- Description: {opportunity.Description}\n" +
                            $"- Current code: ```\n{opportunity.CurrentCode}\n```\n" +
                            $"- Improved code: ```\n{opportunity.ImprovedCode}\n```\n" +
                            $"- Rationale: {opportunity.Rationale}\n\n" +
                            $"Current file content:\n```\n{fileContent}\n```\n\n" +
                            $"Please return the entire updated file content with the improvement applied. Only return the updated file content, no explanations or other text."

                        // Create HTTP client
                        use client = new HttpClient()

                        // Create request
                        let request = {
                            model = model
                            prompt = prompt
                            stream = false
                            options = {|
                                temperature = 0.2  // Lower temperature for more deterministic output
                                num_predict = 8000  // Higher token limit for larger files
                            |}
                        }

                        // Send request to Ollama
                        let! response = client.PostAsJsonAsync($"{ollamaEndpoint}/api/generate", request)

                        // Check if request was successful
                        if not response.IsSuccessStatusCode then
                            logger.LogError("Failed to apply improvement using Ollama: {StatusCode} - {ReasonPhrase}", response.StatusCode, response.ReasonPhrase)
                            return None
                        else
                            // Parse response
                            let! result = response.Content.ReadFromJsonAsync<OllamaResponse>()

                            // Extract the updated file content
                            let updatedContent =
                                // Try to extract code block if present
                                let codeBlockPattern = "```(?:csharp|fsharp)?\s*\n([\s\S]*?)\n```"
                                let codeBlockMatch = Regex.Match(result.response, codeBlockPattern)

                                if codeBlockMatch.Success then
                                    codeBlockMatch.Groups.[1].Value
                                else
                                    // If no code block, use the entire response
                                    result.response

                            // Check if the content was actually changed
                            if fileContent = updatedContent then
                                logger.LogWarning("No changes made to file: {FilePath}", opportunity.FilePath)
                                return None
                            else
                                // Write the updated content back to the file
                                do! File.WriteAllTextAsync(opportunity.FilePath, updatedContent)

                                // Create the applied improvement
                                let appliedImprovement = {
                                    FilePath = opportunity.FilePath
                                    PatternId = "ollama-generated"
                                    PatternName = opportunity.Type
                                    LineNumber = None
                                    OriginalCode = opportunity.CurrentCode
                                    ImprovedCode = opportunity.ImprovedCode
                                    AppliedAt = DateTime.UtcNow
                                }

                                logger.LogInformation("Successfully applied improvement to file: {FilePath}", opportunity.FilePath)
                                return Some appliedImprovement
            with ex ->
                logger.LogError(ex, "Error applying improvement to file: {FilePath}", opportunity.FilePath)
                return None
        }
