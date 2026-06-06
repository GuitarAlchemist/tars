namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Text.Json
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open FSharp.SystemTextJson

/// <summary>
/// Module for extracting knowledge from exploration files
/// </summary>
module KnowledgeExtractor =
    /// <summary>
    /// Extracts concepts from text content
    /// </summary>
    let extractConcepts (content: string) =
        // Simple pattern matching for concepts
        // In a real implementation, this would use more sophisticated NLP techniques
        let conceptPatterns = [
            @"concept\s+of\s+([^\.]+)"
            @"([^\.]+)\s+is\s+defined\s+as"
            @"([^\.]+)\s+refers\s+to"
        ]

        conceptPatterns
        |> List.collect (fun pattern ->
            let matches = Regex.Matches(content, pattern, RegexOptions.IgnoreCase)
            matches
            |> Seq.cast<Match>
            |> Seq.map (fun m ->
                {
                    Id = Guid.NewGuid().ToString()
                    Type = "Concept"
                    Content = m.Groups.[1].Value.Trim()
                    Source = ""  // Will be set by the caller
                    SourceType = KnowledgeSourceType.Documentation  // Will be set by the caller
                    Confidence = 0.8
                    Tags = ["concept"]
                    RelatedItems = []
                    ExtractedAt = DateTime.UtcNow
                })
            |> Seq.toList)

    /// <summary>
    /// Extracts insights from text content
    /// </summary>
    let extractInsights (content: string) =
        // Simple pattern matching for insights
        // In a real implementation, this would use more sophisticated NLP techniques
        let insightPatterns = [
            @"I\s+realized\s+that\s+([^\.]+)"
            @"key\s+insight\s+is\s+that\s+([^\.]+)"
            @"important\s+to\s+note\s+that\s+([^\.]+)"
        ]

        insightPatterns
        |> List.collect (fun pattern ->
            let matches = Regex.Matches(content, pattern, RegexOptions.IgnoreCase)
            matches
            |> Seq.cast<Match>
            |> Seq.map (fun m ->
                {
                    Id = Guid.NewGuid().ToString()
                    Type = "Insight"
                    Content = m.Groups.[1].Value.Trim()
                    Source = ""  // Will be set by the caller
                    SourceType = KnowledgeSourceType.Documentation  // Will be set by the caller
                    Confidence = 0.7
                    Tags = ["insight"]
                    RelatedItems = []
                    ExtractedAt = DateTime.UtcNow
                })
            |> Seq.toList)

    /// <summary>
    /// Extracts code patterns from text content
    /// </summary>
    let extractCodePatterns (content: string) =
        // Extract code blocks
        let codeBlockPattern = @"```(?:csharp|fsharp)?\s*\n([\s\S]*?)\n```"
        let matches = Regex.Matches(content, codeBlockPattern)

        matches
        |> Seq.cast<Match>
        |> Seq.map (fun m ->
            {
                Id = Guid.NewGuid().ToString()
                Type = "CodePattern"
                Content = m.Groups.[1].Value.Trim()
                Source = ""  // Will be set by the caller
                SourceType = KnowledgeSourceType.Documentation  // Will be set by the caller
                Confidence = 0.9
                Tags = ["code"]
                RelatedItems = []
                ExtractedAt = DateTime.UtcNow
            })
        |> Seq.toList

    /// <summary>
    /// Extracts knowledge from a file
    /// </summary>
    let extractKnowledgeFromFile (logger: ILogger) (filePath: string) =
        task {
            try
                logger.LogInformation("Extracting knowledge from {FilePath}", filePath)

                // Read the file content
                let! content = ExplorationFileProcessor.readFileContent filePath

                // Determine the source type
                let sourceType = ExplorationFileProcessor.determineSourceType filePath

                // Extract knowledge items
                let concepts = extractConcepts content
                let insights = extractInsights content
                let codePatterns = extractCodePatterns content

                // Combine all items and set the source
                let items =
                    List.concat [concepts; insights; codePatterns]
                    |> List.map (fun item ->
                        { item with
                            Source = filePath
                            SourceType = sourceType })

                logger.LogInformation("Extracted {ItemCount} knowledge items from {FilePath}", items.Length, filePath)
                return items
            with
            | ex ->
                logger.LogError(ex, "Error extracting knowledge from {FilePath}", filePath)
                return []
        }

    /// <summary>
    /// Saves the knowledge base to a file
    /// </summary>
    let saveKnowledgeBase (knowledgeBase: KnowledgeBase) (filePath: string) =
        task {
            try
                let options = JsonSerializerOptions()
                options.WriteIndented <- true

                // Serialize to JSON
                let json = JsonSerializer.Serialize(knowledgeBase, typeof<KnowledgeBase>, options)
                do! File.WriteAllTextAsync(filePath, json)
                return true
            with
            | ex ->
                return false
        }

    /// <summary>
    /// Loads the knowledge base from a file
    /// </summary>
    let loadKnowledgeBase (filePath: string) =
        task {
            try
                if File.Exists(filePath) then
                    let! json = File.ReadAllTextAsync(filePath)

                    let options = JsonSerializerOptions()

                    let result = JsonSerializer.Deserialize<KnowledgeBase>(json, options)
                    return
                        if isNull (box result) then
                            {
                                Items = []
                                LastUpdated = DateTime.UtcNow
                                Version = "1.0"
                                Statistics = Map.empty
                            }
                        else
                            result
                else
                    return {
                        Items = []
                        LastUpdated = DateTime.UtcNow
                        Version = "1.0"
                        Statistics = Map.empty
                    }
            with
            | ex ->
                return {
                    Items = []
                    LastUpdated = DateTime.UtcNow
                    Version = "1.0"
                    Statistics = Map.empty
                }
        }
