namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open System.Diagnostics
open OpenSearchIntegration
open ModelSelection

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
                let defaultEndpoint =
                    let envVar = Environment.GetEnvironmentVariable("OLLAMA_BASE_URL")
                    if String.IsNullOrEmpty(envVar) then "http://localhost:11434" else envVar

                let selection =
                    ModelSelection.selectModel logger "local_reasoning" "llama3" (Some defaultEndpoint)

                let ollamaEndpoint = selection.Endpoint |> Option.defaultValue defaultEndpoint
                let ollamaModel = selection.ModelName

                // Check if we're using Docker
                let useOllama =
                    let envVar = Environment.GetEnvironmentVariable("OLLAMA_USE_DOCKER")
                    not (String.IsNullOrEmpty(envVar)) && envVar.ToLower() = "true"

                logger.LogInformation("Using knowledge extraction model {ModelName} from provider {Provider}", ollamaModel, selection.Provider)
                logger.LogInformation("LLM endpoint resolved to: {OllamaEndpoint}", ollamaEndpoint)
                selection.Notes
                |> Option.iter (fun note -> logger.LogInformation("Model recommendation note: {Note}", note))
                if selection.RequiresApiKey then
                    logger.LogInformation("Selected model requires API credentials; ensure the required environment variables are configured.")
                logger.LogInformation("Using Ollama integration: {UseOllama}", useOllama)

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
                            if useOllama then
                                task {
                                    logger.LogInformation("Extracting knowledge using Ollama: {FilePath}", file)

                                    // Read the file content with consistent encoding handling
                                    let! content = ExplorationFileProcessor.readFileContent file

                                    let! extractedItems =
                                        OllamaKnowledgeExtractor.extractKnowledgeWithOllama logger content ollamaEndpoint ollamaModel

                                    let sourceType = ExplorationFileProcessor.determineSourceType file

                                    return
                                        extractedItems
                                        |> List.map (fun item ->
                                            { item with
                                                Source = file
                                                SourceType = sourceType })
                                }
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

                    let! openSearchResult =
                        match OpenSearchIntegration.tryLoadConfig () with
                        | Some config ->
                            task {
                                let! health = OpenSearchIntegration.getClusterHealthAsync logger config
                                match health with
                                | Ok status ->
                                    logger.LogInformation("OpenSearch cluster status {Status} (nodes: {Nodes}, active shards: {Shards})",
                                                         status.Status,
                                                         status.NodeCount |> Option.map string |> Option.defaultValue "unknown",
                                                         status.ActiveShards |> Option.map string |> Option.defaultValue "unknown")

                                    logger.LogInformation("Indexing extracted knowledge items into OpenSearch index {IndexName}", config.IndexName)
                                    let! outcome = OpenSearchIntegration.indexKnowledgeItemsAsync logger config allItems

                                    if outcome.Indexed > 0 then
                                        logger.LogInformation("Indexed {Indexed} knowledge items into OpenSearch ({IndexName})", outcome.Indexed, config.IndexName)
                                    if not outcome.Failed.IsEmpty then
                                        logger.LogWarning("OpenSearch indexing failed for {FailureCount} items", outcome.Failed.Length)

                                    return Ok (Some (config, status, outcome))
                                | Error reason ->
                                    return Error reason
                            }
                        | None ->
                            logger.LogInformation("OPENSEARCH_URL not configured; skipping OpenSearch indexing.")
                            task { return Ok None }

                    let baseEntries = [
                        "knowledge_base_path", knowledgeBasePath
                        "extraction_time", DateTime.UtcNow.ToString("o")
                        "total_items", totalItems.ToString()
                        "processed_files", processedFiles.ToString()
                        "used_ollama", useOllama.ToString()
                        "model_provider", selection.Provider
                        "model_name", ollamaModel
                    ]

                    match openSearchResult with
                    | Ok openSearchData ->
                        let entries =
                            match openSearchData with
                            | Some (config, status, outcome) ->
                                baseEntries @ [
                                    "opensearch_index", config.IndexName
                                    "opensearch_indexed", outcome.Indexed.ToString()
                                    "opensearch_failed", outcome.Failed.Length.ToString()
                                    "opensearch_status", status.Status
                                    "opensearch_nodes", status.NodeCount |> Option.map string |> Option.defaultValue "unknown"
                                    "opensearch_active_shards", status.ActiveShards |> Option.map string |> Option.defaultValue "unknown"
                                ]
                            | None -> baseEntries

                        // Return success
                        return Ok (Map.ofList entries)
                    | Error reason ->
                        logger.LogError("OpenSearch health check failed: {Reason}", reason)
                        return Error $"OpenSearch health check failed: {reason}"
                else
                    logger.LogError("Failed to save knowledge base to {KnowledgeBasePath}", knowledgeBasePath)
                    return Error "Failed to save knowledge base"
            }

