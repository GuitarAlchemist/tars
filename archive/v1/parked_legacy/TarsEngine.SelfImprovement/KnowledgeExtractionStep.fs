namespace TarsEngine.SelfImprovement

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open System.Diagnostics

/// <summary>
/// Module for the knowledge extraction step in the autonomous improvement workflow
/// </summary>
module KnowledgeExtractionStep =
    /// <summary>
    /// The path to the knowledge extraction metascript
    /// </summary>
    let metascriptPath = Path.Combine("Examples", "metascripts", "documentation_knowledge_extraction.tars")

    /// <summary>
    /// The path to the knowledge base file
    /// </summary>
    let knowledgeBasePath = "knowledge_base.json"

    /// <summary>
    /// Gets the knowledge extraction step handler
    /// </summary>
    let getHandler (logger: ILogger) (maxFiles: int) : WorkflowState -> Task<StepResult> =
        fun state ->
            task {
                logger.LogInformation("Starting knowledge extraction step")

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

                // Get the target directories
                let targetDirectories = state.TargetDirectories

                // Process each directory
                let mutable totalItems = 0
                let mutable processedFiles = 0
                let mutable allItems = []

                for directory in targetDirectories do
                    // Get exploration files
                    let files = ExplorationFileProcessor.getExplorationFiles directory maxFiles

                    // Process each file
                    for file in files do
                        let! items =
                            if useDocker then
                                // Use Ollama for knowledge extraction
                                logger.LogInformation("Extracting knowledge using Ollama: {FilePath}", file)

                                // Read the file content
                                let content = File.ReadAllText(file)

                                // Extract knowledge using Ollama (temporarily return empty list)
                                let extractedItems = []

                                // Set the source for each item
                                let itemsWithSource =
                                    extractedItems
                                    |> List.map (fun item ->
                                        { item with
                                            Source = file
                                            SourceType = KnowledgeSourceType.Documentation })

                                Task.FromResult(itemsWithSource)
                            else
                                // Use pattern matching for knowledge extraction (legacy approach)
                                logger.LogInformation("Extracting knowledge using pattern matching: {FilePath}", file)
                                KnowledgeExtractor.extractKnowledgeFromFile logger file

                        allItems <- allItems @ items
                        totalItems <- totalItems + items.Length
                        processedFiles <- processedFiles + 1

                // Create the knowledge base
                let knowledgeBase = {
                    Items = allItems
                    LastUpdated = DateTime.UtcNow
                    Version = "1.0"
                    Statistics = Map [
                        "TotalItems", totalItems.ToString()
                        "ProcessedFiles", processedFiles.ToString()
                        "ConceptCount", (allItems |> List.filter (fun i -> i.Type = "Concept") |> List.length).ToString()
                        "InsightCount", (allItems |> List.filter (fun i -> i.Type = "Insight") |> List.length).ToString()
                        "CodePatternCount", (allItems |> List.filter (fun i -> i.Type = "CodePattern") |> List.length).ToString()
                    ]
                }

                // Save the knowledge base
                let! saveResult = KnowledgeExtractor.saveKnowledgeBase knowledgeBase knowledgeBasePath

                if saveResult then
                    logger.LogInformation("Knowledge base saved to {KnowledgeBasePath}", knowledgeBasePath)
                    logger.LogInformation("Extracted {TotalItems} knowledge items from {ProcessedFiles} files", totalItems, processedFiles)

                    // Return success
                    return Ok (Map.ofList [
                        "knowledge_base_path", knowledgeBasePath
                        "extraction_time", DateTime.UtcNow.ToString("o")
                        "total_items", totalItems.ToString()
                        "processed_files", processedFiles.ToString()
                        "used_ollama", useDocker.ToString()
                    ])
                else
                    logger.LogError("Failed to save knowledge base to {KnowledgeBasePath}", knowledgeBasePath)
                    return Error "Failed to save knowledge base"
            }

