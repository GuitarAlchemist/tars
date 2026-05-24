namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.OnDemandSearch

/// Web Search Command for TARS CLI
type WebSearchCommand(
    logger: ILogger<WebSearchCommand>,
    searchService: IOnDemandSearchService) =
    
    let mutable searchStats = {| 
        totalSearches = 0
        successfulSearches = 0
        averageLatency = 0.0
        cacheHits = 0
    |}
    
    interface ICommand with
        member _.Name = "web"
        
        member _.Description = "Web search capabilities with multiple providers and intelligent caching"
        
        member _.Usage = """
Usage: tars web <subcommand> [options]

Subcommands:
  search <query>                       - Basic web search using available providers
  search-fast <query>                  - Accelerated search with aggressive caching
  search-academic <query>              - Academic/research focused search
  search-technical <query>             - Technical/programming focused search
  search-adaptive <query>              - Adaptive search with intent detection
  stats                                - Show web search statistics
  providers                            - Show available search providers
  help                                 - Show this help
"""
        
        member _.Examples = [
            "tars web search \"artificial intelligence\""
            "tars web search-technical \"F# functional programming\""
            "tars web search-academic \"machine learning algorithms\""
            "tars web search-adaptive \"best practices microservices\""
            "tars web stats"
            "tars web providers"
        ]
        
        member _.ValidateOptions(_) = true
        
        member self.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | [] -> 
                        return self.ShowHelp()
                    
                    // Basic web search
                    | "search" :: queryParts when queryParts.Length > 0 ->
                        let query = String.Join(" ", queryParts)
                        return! self.PerformWebSearch(query, "basic")
                    
                    // Fast search with caching
                    | "search-fast" :: queryParts when queryParts.Length > 0 ->
                        let query = String.Join(" ", queryParts)
                        return! self.PerformWebSearch(query, "fast")

                    // Academic search
                    | "search-academic" :: queryParts when queryParts.Length > 0 ->
                        let query = String.Join(" ", queryParts)
                        return! self.PerformAcademicSearch(query)

                    // Technical search
                    | "search-technical" :: queryParts when queryParts.Length > 0 ->
                        let query = String.Join(" ", queryParts)
                        return! self.PerformTechnicalSearch(query)

                    // Adaptive search
                    | "search-adaptive" :: queryParts when queryParts.Length > 0 ->
                        let query = String.Join(" ", queryParts)
                        return! self.PerformAdaptiveSearch(query)
                    
                    // Statistics
                    | "stats" :: _ ->
                        return! self.ShowSearchStats()
                    
                    // Providers
                    | "providers" :: _ ->
                        return! self.ShowProviders()
                    
                    // Help
                    | "help" :: _ ->
                        return self.ShowHelp()
                    
                    | unknown :: _ ->
                        logger.LogWarning("Unknown web search subcommand: {Command}", unknown)
                        return { Success = false; ExitCode = 1; Message = $"Unknown subcommand: {unknown}. Use 'tars web help' for available commands." }
                with
                | ex ->
                    logger.LogError(ex, "Error executing web search command")
                    return { Success = false; ExitCode = 1; Message = $"Error: {ex.Message}" }
            }
    
    member private this.ShowHelp() =
        let helpText = """
🌐 TARS Web Search System

Advanced web search capabilities with multiple providers, intelligent caching,
and specialized search modes for different types of queries.

Search Types:
• Basic Search      - General web search across multiple providers
• Fast Search       - Optimized search with aggressive caching
• Academic Search   - Research-focused search (arXiv, PubMed, etc.)
• Technical Search  - Programming and technical documentation
• Adaptive Search   - Intelligent search with automatic intent detection

Available Commands:
  search <query>                       - Basic web search
  search-fast <query>                  - Fast cached search
  search-academic <query>              - Academic/research search
  search-technical <query>             - Technical/programming search
  search-adaptive <query>              - Adaptive search with intent detection
  stats                                - Search performance statistics
  providers                            - Available search providers
  help                                 - This help message

Examples:
  tars web search "machine learning trends 2024"
  tars web search-technical "F# async programming patterns"
  tars web search-academic "neural network architectures"
  tars web search-adaptive "best practices for microservices"
  tars web stats
"""
        { Success = true; ExitCode = 0; Message = helpText }
    
    member private this.PerformWebSearch(query: string, searchType: string) =
        task {
            let startTime = DateTime.UtcNow
            logger.LogInformation("Performing {SearchType} web search for: {Query}", searchType, query)
            
            try
                let! results = searchService.SearchWebAsync query 10
                let latency = (DateTime.UtcNow - startTime).TotalMilliseconds
                
                // Update statistics
                searchStats <- {| 
                    totalSearches = searchStats.totalSearches + 1
                    successfulSearches = searchStats.successfulSearches + 1
                    averageLatency = (searchStats.averageLatency + latency) / 2.0
                    cacheHits = searchStats.cacheHits + (if searchType = "fast" then 1 else 0)
                |}
                
                let resultText = 
                    if results.Length > 0 then
                        let formattedResults = 
                            results 
                            |> Array.take (min 5 results.Length)
                            |> Array.mapi (fun i result -> 
                                $"{i + 1}. {result.Title}\n   Source: {result.Source} | Relevance: {result.Relevance:F2}\n   {result.Description}\n   URL: {result.Url}\n")
                            |> String.concat "\n"
                        
                        $"""🌐 Web Search Results for: "{query}"
Search Type: {searchType}
Found {results.Length} results in {latency:F1}ms

{formattedResults}
{if results.Length > 5 then $"... and {results.Length - 5} more results" else ""}"""
                    else
                        $"🌐 No results found for: \"{query}\""
                
                return { Success = true; ExitCode = 0; Message = resultText }
            with
            | ex ->
                logger.LogError(ex, "Web search failed for query: {Query}", query)
                searchStats <- {| searchStats with totalSearches = searchStats.totalSearches + 1 |}
                return { Success = false; ExitCode = 1; Message = $"❌ Web search failed: {ex.Message}" }
        }
    
    member private this.PerformAcademicSearch(query: string) =
        task {
            let startTime = DateTime.UtcNow
            logger.LogInformation("Performing academic search for: {Query}", query)
            
            try
                let domains = [|"arxiv"; "pubmed"|]
                let! results = searchService.SearchAcademicAsync query domains
                let latency = (DateTime.UtcNow - startTime).TotalMilliseconds
                
                searchStats <- {| 
                    totalSearches = searchStats.totalSearches + 1
                    successfulSearches = searchStats.successfulSearches + 1
                    averageLatency = (searchStats.averageLatency + latency) / 2.0
                    cacheHits = searchStats.cacheHits
                |}
                
                let resultText = 
                    if results.Length > 0 then
                        let formattedResults = 
                            results 
                            |> Array.take (min 5 results.Length)
                            |> Array.mapi (fun i result -> 
                                $"{i + 1}. {result.Title}\n   Source: {result.Source} | Credibility: {result.Credibility:F2}\n   {result.Description}\n   URL: {result.Url}\n")
                            |> String.concat "\n"
                        
                        $"""📚 Academic Search Results for: "{query}"
Searched: arXiv, PubMed
Found {results.Length} academic results in {latency:F1}ms

{formattedResults}
{if results.Length > 5 then $"... and {results.Length - 5} more results" else ""}"""
                    else
                        $"📚 No academic results found for: \"{query}\""
                
                return { Success = true; ExitCode = 0; Message = resultText }
            with
            | ex ->
                logger.LogError(ex, "Academic search failed for query: {Query}", query)
                searchStats <- {| searchStats with totalSearches = searchStats.totalSearches + 1 |}
                return { Success = false; ExitCode = 1; Message = $"❌ Academic search failed: {ex.Message}" }
        }
    
    member private this.PerformTechnicalSearch(query: string) =
        task {
            let startTime = DateTime.UtcNow
            logger.LogInformation("Performing technical search for: {Query}", query)
            
            try
                // Enhance query for technical search
                let technicalQuery = query + " programming technical documentation"
                let! webResults = searchService.SearchWebAsync technicalQuery 10
                let latency = (DateTime.UtcNow - startTime).TotalMilliseconds
                
                searchStats <- {| 
                    totalSearches = searchStats.totalSearches + 1
                    successfulSearches = searchStats.successfulSearches + 1
                    averageLatency = (searchStats.averageLatency + latency) / 2.0
                    cacheHits = searchStats.cacheHits
                |}
                
                let resultText = 
                    if webResults.Length > 0 then
                        let formattedResults = 
                            webResults 
                            |> Array.take (min 5 webResults.Length)
                            |> Array.mapi (fun i result ->
                                sprintf "%d. %s\n   Source: %s | Relevance: %.2f\n   %s\n   URL: %s\n" (i + 1) result.Title result.Source result.Relevance result.Description result.Url)
                            |> String.concat "\n"
                        
                        sprintf "💻 Technical Search Results for: \"%s\"\nEnhanced Query: \"%s\"\nFound %d technical results in %.1fms\n\n%s\n%s"
                            query technicalQuery webResults.Length latency formattedResults
                            (if webResults.Length > 5 then sprintf "... and %d more results" (webResults.Length - 5) else "")
                    else
                        sprintf "💻 No technical results found for: \"%s\"" query
                
                return { Success = true; ExitCode = 0; Message = resultText }
            with
            | ex ->
                logger.LogError(ex, "Technical search failed for query: {Query}", query)
                searchStats <- {| searchStats with totalSearches = searchStats.totalSearches + 1 |}
                return { Success = false; ExitCode = 1; Message = $"❌ Technical search failed: {ex.Message}" }
        }
    
    member private this.PerformAdaptiveSearch(query: string) =
        task {
            let startTime = DateTime.UtcNow
            logger.LogInformation("Performing adaptive search for: {Query}", query)
            
            try
                // Use the adaptive search capability from OnDemandKnowledgeSearch
                let searchQuery = {
                    Query = query
                    Intent = None  // Let the service detect intent
                    Domain = None
                    Context = Map.empty
                    MaxResults = 15
                    QualityThreshold = 0.7
                    Providers = None
                }
                
                let! searchResults = searchService.SearchAsync searchQuery SearchStrategy.Adaptive
                let latency = (DateTime.UtcNow - startTime).TotalMilliseconds
                
                searchStats <- {| 
                    totalSearches = searchStats.totalSearches + 1
                    successfulSearches = searchStats.successfulSearches + 1
                    averageLatency = (searchStats.averageLatency + latency) / 2.0
                    cacheHits = searchStats.cacheHits
                |}
                
                let resultText = 
                    if searchResults.Results.Length > 0 then
                        let formattedResults = 
                            searchResults.Results 
                            |> Array.take (min 5 searchResults.Results.Length)
                            |> Array.mapi (fun i result -> 
                                $"{i + 1}. {result.Title}\n   Source: {result.Source} | Quality: {result.Relevance:F2}\n   {result.Description}\n   URL: {result.Url}\n")
                            |> String.concat "\n"
                        
                        $"""🎯 Adaptive Search Results for: "{query}"
Quality Score: {searchResults.QualityScore:F2}
Found {searchResults.Results.Length} results in {latency:F1}ms

{formattedResults}
{if searchResults.Results.Length > 5 then $"... and {searchResults.Results.Length - 5} more results" else ""}

Search Strategy: Adaptive with automatic intent detection"""
                    else
                        $"🎯 No adaptive search results found for: \"{query}\""
                
                return { Success = true; ExitCode = 0; Message = resultText }
            with
            | ex ->
                logger.LogError(ex, "Adaptive search failed for query: {Query}", query)
                searchStats <- {| searchStats with totalSearches = searchStats.totalSearches + 1 |}
                return { Success = false; ExitCode = 1; Message = $"❌ Adaptive search failed: {ex.Message}" }
        }
    
    member private this.ShowSearchStats() =
        task {
            try
                let! providerStatus = searchService.GetProviderStatusAsync()
                let successRate = if searchStats.totalSearches > 0 then (float searchStats.successfulSearches / float searchStats.totalSearches) * 100.0 else 0.0
                let cacheHitRate = if searchStats.totalSearches > 0 then (float searchStats.cacheHits / float searchStats.totalSearches) * 100.0 else 0.0
                
                let statsText =
                    sprintf "┌─────────────────────────────────────────────────────────┐\n│ Web Search Performance Statistics                       │\n├─────────────────────────────────────────────────────────┤\n│ Total Searches: %d\n│ Successful Searches: %d\n│ Success Rate: %.1f%%\n│ Average Latency: %.1fms\n│ Cache Hits: %d\n│ Cache Hit Rate: %.1f%%\n│ Available Providers: %d\n│ Total Providers: %d\n└─────────────────────────────────────────────────────────┘"
                        searchStats.totalSearches
                        searchStats.successfulSearches
                        successRate
                        searchStats.averageLatency
                        searchStats.cacheHits
                        cacheHitRate
                        (providerStatus |> Map.filter (fun _ v -> v) |> Map.count)
                        providerStatus.Count
                
                return { Success = true; ExitCode = 0; Message = statsText }
            with
            | ex ->
                logger.LogError(ex, "Failed to get search statistics")
                return { Success = false; ExitCode = 1; Message = sprintf "❌ Failed to get statistics: %s" ex.Message }
        }
    
    member private this.ShowProviders() =
        task {
            try
                let! providerStatus = searchService.GetProviderStatusAsync()
                
                let providerList = 
                    providerStatus
                    |> Map.toList
                    |> List.map (fun (name, isAvailable) -> 
                        let status = if isAvailable then "✅ Available" else "❌ Unavailable"
                        $"• {name}: {status}")
                    |> String.concat "\n"
                
                let providersText = $"""
🔍 Available Search Providers

{providerList}

Provider Types:
• Web Search: DuckDuckGo, Google (if configured)
• Academic: arXiv, PubMed
• Technical: GitHub, StackOverflow (if configured)
• Knowledge: Wikidata, DBpedia

Note: Some providers require API keys to be configured."""
                
                return { Success = true; ExitCode = 0; Message = providersText }
            with
            | ex ->
                logger.LogError(ex, "Failed to get provider status")
                return { Success = false; ExitCode = 1; Message = $"❌ Failed to get providers: {ex.Message}" }
        }
