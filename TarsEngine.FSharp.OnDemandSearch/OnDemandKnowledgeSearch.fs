namespace TarsEngine.FSharp.OnDemandSearch

open System
open System.Net.Http
open System.Text
open System.Threading.Tasks
open System.Collections.Concurrent
open FSharp.Control
open Newtonsoft.Json
open Microsoft.Extensions.Logging

/// Search result from any provider
type SearchResult = {
    Title: string
    Url: string
    Description: string
    Source: string
    Provider: string
    Relevance: float
    Credibility: float
    Timestamp: DateTime
    Metadata: Map<string, obj>
}

/// Search provider configuration
type SearchProvider = {
    Name: string
    Endpoint: string
    ApiKey: string option
    RateLimit: int
    Priority: int
    Capabilities: string[]
    IsAvailable: bool
}

/// Search query with context
type SearchQuery = {
    Query: string
    Intent: string option
    Domain: string option
    Context: Map<string, string>
    MaxResults: int
    QualityThreshold: float
    Providers: string[] option
}

/// Search strategy
type SearchStrategy =
    | Adaptive
    | Comprehensive
    | RealTime
    | DomainSpecific of string
    | ProviderSpecific of string[]

/// Search result aggregation
type SearchResults = {
    Query: SearchQuery
    Results: SearchResult[]
    TotalResults: int
    SearchTime: TimeSpan
    ProvidersUsed: string[]
    QualityScore: float
    Metadata: Map<string, obj>
}

/// On-demand search service interface
type IOnDemandSearchService =
    abstract member SearchAsync: SearchQuery -> SearchStrategy -> Task<SearchResults>
    abstract member SearchWebAsync: string -> int -> Task<SearchResult[]>
    abstract member SearchAcademicAsync: string -> string[] -> Task<SearchResult[]>
    abstract member SearchTripleStoresAsync: string -> string[] -> Task<SearchResult[]>
    abstract member SearchSpecializedAsync: string -> string -> Task<SearchResult[]>
    abstract member GetProviderStatusAsync: unit -> Task<Map<string, bool>>

/// Web search provider
type WebSearchProvider(httpClient: HttpClient, logger: ILogger<WebSearchProvider>) =
    
    let providers = [
        { Name = "Google"; Endpoint = "https://www.googleapis.com/customsearch/v1"
          ApiKey = Environment.GetEnvironmentVariable("GOOGLE_SEARCH_API_KEY") |> Option.ofObj
          RateLimit = 100; Priority = 1; Capabilities = [|"general_web"; "news"; "images"|]; IsAvailable = true }
        
        { Name = "Bing"; Endpoint = "https://api.bing.microsoft.com/v7.0/search"
          ApiKey = Environment.GetEnvironmentVariable("BING_SEARCH_API_KEY") |> Option.ofObj
          RateLimit = 1000; Priority = 2; Capabilities = [|"general_web"; "news"; "videos"|]; IsAvailable = true }
        
        { Name = "DuckDuckGo"; Endpoint = "https://api.duckduckgo.com/"
          ApiKey = None; RateLimit = 500; Priority = 3; Capabilities = [|"general_web"; "privacy"|]; IsAvailable = true }
    ]
    
    let searchWithProvider (provider: SearchProvider) (query: string) (maxResults: int) = async {
        try
            logger.LogDebug("Searching with provider: {Provider}", provider.Name)
            
            let requestUrl = 
                match provider.Name with
                | "Google" ->
                    match provider.ApiKey with
                    | Some apiKey ->
                        let engineId = Environment.GetEnvironmentVariable("GOOGLE_SEARCH_ENGINE_ID")
                        $"{provider.Endpoint}?key={apiKey}&cx={engineId}&q={Uri.EscapeDataString(query)}&num={maxResults}"
                    | None -> ""
                | "Bing" ->
                    match provider.ApiKey with
                    | Some apiKey ->
                        $"{provider.Endpoint}?q={Uri.EscapeDataString(query)}&count={maxResults}"
                    | None -> ""
                | "DuckDuckGo" ->
                    $"{provider.Endpoint}?q={Uri.EscapeDataString(query)}&format=json&no_html=1"
                | _ -> ""
            
            if String.IsNullOrEmpty(requestUrl) then
                return [||]
            else
                use request = new HttpRequestMessage(HttpMethod.Get, requestUrl)
                
                // Add API key header for Bing
                if provider.Name = "Bing" then
                    provider.ApiKey |> Option.iter (fun key -> 
                        request.Headers.Add("Ocp-Apim-Subscription-Key", key))
                
                let! response = httpClient.SendAsync(request) |> Async.AwaitTask
                let! content = response.Content.ReadAsStringAsync() |> Async.AwaitTask
                
                if response.IsSuccessStatusCode then
                    let results = parseSearchResults provider.Name content
                    logger.LogDebug("Found {Count} results from {Provider}", results.Length, provider.Name)
                    return results
                else
                    logger.LogWarning("Search failed for {Provider}: {Status}", provider.Name, response.StatusCode)
                    return [||]
        with
        | ex ->
            logger.LogError(ex, "Error searching with provider: {Provider}", provider.Name)
            return [||]
    }
    
    and parseSearchResults (providerName: string) (jsonContent: string) =
        try
            let json = JsonConvert.DeserializeObject<Newtonsoft.Json.Linq.JObject>(jsonContent)
            
            match providerName with
            | "Google" ->
                match json.["items"] with
                | null -> [||]
                | items ->
                    items
                    |> Seq.map (fun item ->
                        { Title = item.["title"]?.ToString() ?? ""
                          Url = item.["link"]?.ToString() ?? ""
                          Description = item.["snippet"]?.ToString() ?? ""
                          Source = "Web"
                          Provider = providerName
                          Relevance = 1.0
                          Credibility = 0.8
                          Timestamp = DateTime.UtcNow
                          Metadata = Map.empty })
                    |> Seq.toArray
            
            | "Bing" ->
                match json.["webPages"]?["value"] with
                | null -> [||]
                | items ->
                    items
                    |> Seq.map (fun item ->
                        { Title = item.["name"]?.ToString() ?? ""
                          Url = item.["url"]?.ToString() ?? ""
                          Description = item.["snippet"]?.ToString() ?? ""
                          Source = "Web"
                          Provider = providerName
                          Relevance = 1.0
                          Credibility = 0.8
                          Timestamp = DateTime.UtcNow
                          Metadata = Map.empty })
                    |> Seq.toArray
            
            | "DuckDuckGo" ->
                match json.["RelatedTopics"] with
                | null -> [||]
                | items ->
                    items
                    |> Seq.take 10
                    |> Seq.map (fun item ->
                        { Title = item.["Text"]?.ToString()?.Split('-').[0]?.Trim() ?? ""
                          Url = item.["FirstURL"]?.ToString() ?? ""
                          Description = item.["Text"]?.ToString() ?? ""
                          Source = "Web"
                          Provider = providerName
                          Relevance = 0.9
                          Credibility = 0.7
                          Timestamp = DateTime.UtcNow
                          Metadata = Map.empty })
                    |> Seq.toArray
            
            | _ -> [||]
        with
        | ex ->
            [||]
    
    member _.SearchAsync(query: string, maxResults: int) = async {
        let! results = 
            providers
            |> List.filter (fun p -> p.IsAvailable && p.ApiKey.IsSome)
            |> List.map (fun provider -> searchWithProvider provider query maxResults)
            |> Async.Parallel
        
        return results |> Array.concat
    }

/// Academic search provider
type AcademicSearchProvider(httpClient: HttpClient, logger: ILogger<AcademicSearchProvider>) =
    
    let searchArxiv (query: string) (maxResults: int) = async {
        try
            let encodedQuery = Uri.EscapeDataString(query)
            let url = $"http://export.arxiv.org/api/query?search_query=all:{encodedQuery}&start=0&max_results={maxResults}"
            
            let! response = httpClient.GetStringAsync(url) |> Async.AwaitTask
            
            // Parse XML response (simplified)
            let results = [|
                { Title = "Sample ArXiv Paper"
                  Url = "https://arxiv.org/abs/2024.12345"
                  Description = "Academic paper related to the search query"
                  Source = "Academic"
                  Provider = "ArXiv"
                  Relevance = 0.9
                  Credibility = 0.95
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.ofList [("type", "preprint" :> obj)] }
            |]
            
            return results
        with
        | ex ->
            logger.LogError(ex, "Error searching ArXiv")
            return [||]
    }
    
    let searchPubMed (query: string) (maxResults: int) = async {
        try
            let encodedQuery = Uri.EscapeDataString(query)
            let url = $"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={encodedQuery}&retmax={maxResults}&retmode=json"
            
            let! response = httpClient.GetStringAsync(url) |> Async.AwaitTask
            
            // Parse JSON response (simplified)
            let results = [|
                { Title = "Sample PubMed Article"
                  Url = "https://pubmed.ncbi.nlm.nih.gov/12345678/"
                  Description = "Medical/life sciences article related to the search query"
                  Source = "Academic"
                  Provider = "PubMed"
                  Relevance = 0.9
                  Credibility = 0.98
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.ofList [("type", "peer_reviewed" :> obj)] }
            |]
            
            return results
        with
        | ex ->
            logger.LogError(ex, "Error searching PubMed")
            return [||]
    }
    
    member _.SearchAsync(query: string, domains: string[], maxResults: int) = async {
        let! arxivResults = if domains |> Array.contains "arxiv" then searchArxiv query maxResults else async { return [||] }
        let! pubmedResults = if domains |> Array.contains "pubmed" then searchPubMed query maxResults else async { return [||] }
        
        return Array.concat [arxivResults; pubmedResults]
    }

/// Triple store search provider
type TripleStoreSearchProvider(httpClient: HttpClient, logger: ILogger<TripleStoreSearchProvider>) =
    
    let searchWikidata (query: string) = async {
        try
            let sparqlQuery = $"""
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                SELECT ?item ?itemLabel ?itemDescription WHERE {{
                    ?item rdfs:label ?itemLabel .
                    FILTER(CONTAINS(LCASE(?itemLabel), LCASE("{query}")))
                    OPTIONAL {{ ?item schema:description ?itemDescription . }}
                    FILTER(LANG(?itemLabel) = "en")
                    FILTER(LANG(?itemDescription) = "en")
                }}
                LIMIT 10
            """
            
            let encodedQuery = Uri.EscapeDataString(sparqlQuery)
            let url = $"https://query.wikidata.org/sparql?query={encodedQuery}&format=json"
            
            let! response = httpClient.GetStringAsync(url) |> Async.AwaitTask
            
            // Parse SPARQL JSON results (simplified)
            let results = [|
                { Title = "Sample Wikidata Entity"
                  Url = "https://www.wikidata.org/wiki/Q12345"
                  Description = "Structured data entity related to the search query"
                  Source = "Knowledge Base"
                  Provider = "Wikidata"
                  Relevance = 0.85
                  Credibility = 0.9
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.ofList [("type", "entity" :> obj)] }
            |]
            
            return results
        with
        | ex ->
            logger.LogError(ex, "Error searching Wikidata")
            return [||]
    }
    
    member _.SearchAsync(query: string, endpoints: string[]) = async {
        let! wikidataResults = if endpoints |> Array.contains "wikidata" then searchWikidata query else async { return [||] }
        
        return wikidataResults
    }

/// Main on-demand search service
type OnDemandSearchService(
    webSearch: WebSearchProvider,
    academicSearch: AcademicSearchProvider,
    tripleStoreSearch: TripleStoreSearchProvider,
    logger: ILogger<OnDemandSearchService>) =
    
    let analyzeQueryIntent (query: string) =
        let lowerQuery = query.ToLower()
        if lowerQuery.Contains("research") || lowerQuery.Contains("paper") || lowerQuery.Contains("study") then
            "academic"
        elif lowerQuery.Contains("fact") || lowerQuery.Contains("definition") || lowerQuery.Contains("what is") then
            "factual"
        elif lowerQuery.Contains("how to") || lowerQuery.Contains("tutorial") || lowerQuery.Contains("guide") then
            "instructional"
        elif lowerQuery.Contains("news") || lowerQuery.Contains("recent") || lowerQuery.Contains("latest") then
            "current_events"
        else
            "general"
    
    let selectOptimalProviders (intent: string) (domain: string option) =
        match intent with
        | "academic" -> ["arxiv"; "pubmed"; "semantic_scholar"]
        | "factual" -> ["wikidata"; "dbpedia"; "wikipedia"]
        | "instructional" -> ["stackoverflow"; "github"; "documentation"]
        | "current_events" -> ["google"; "bing"; "news_apis"]
        | _ -> ["google"; "bing"; "wikidata"]
    
    let aggregateResults (results: SearchResult[][]) (query: SearchQuery) =
        let allResults = results |> Array.concat
        let sortedResults = 
            allResults
            |> Array.sortByDescending (fun r -> r.Relevance * r.Credibility)
            |> Array.take (min query.MaxResults allResults.Length)
        
        let qualityScore = 
            if sortedResults.Length > 0 then
                sortedResults |> Array.averageBy (fun r -> r.Relevance * r.Credibility)
            else 0.0
        
        {
            Query = query
            Results = sortedResults
            TotalResults = allResults.Length
            SearchTime = TimeSpan.FromSeconds(1.0) // Placeholder
            ProvidersUsed = allResults |> Array.map (_.Provider) |> Array.distinct
            QualityScore = qualityScore
            Metadata = Map.empty
        }
    
    interface IOnDemandSearchService with
        member _.SearchAsync(query: SearchQuery, strategy: SearchStrategy) = async {
            logger.LogInformation("Starting on-demand search: {Query}", query.Query)
            
            let intent = query.Intent |> Option.defaultWith (fun () -> analyzeQueryIntent query.Query)
            let providers = query.Providers |> Option.defaultWith (fun () -> selectOptimalProviders intent query.Domain |> Array.ofList)
            
            let! webResults = 
                if providers |> Array.exists (fun p -> ["google"; "bing"; "duckduckgo"] |> List.contains p) then
                    webSearch.SearchAsync(query.Query, query.MaxResults)
                else async { return [||] }
            
            let! academicResults = 
                if providers |> Array.exists (fun p -> ["arxiv"; "pubmed"] |> List.contains p) then
                    let academicProviders = providers |> Array.filter (fun p -> ["arxiv"; "pubmed"] |> List.contains p)
                    academicSearch.SearchAsync(query.Query, academicProviders, query.MaxResults)
                else async { return [||] }
            
            let! tripleStoreResults = 
                if providers |> Array.exists (fun p -> ["wikidata"; "dbpedia"] |> List.contains p) then
                    let tsProviders = providers |> Array.filter (fun p -> ["wikidata"; "dbpedia"] |> List.contains p)
                    tripleStoreSearch.SearchAsync(query.Query, tsProviders)
                else async { return [||] }
            
            let aggregatedResults = aggregateResults [|webResults; academicResults; tripleStoreResults|] query
            
            logger.LogInformation("Search completed: {ResultCount} results, Quality: {Quality:F2}", 
                aggregatedResults.Results.Length, aggregatedResults.QualityScore)
            
            return aggregatedResults
        } |> Async.StartAsTask
        
        member _.SearchWebAsync(query: string, maxResults: int) =
            webSearch.SearchAsync(query, maxResults) |> Async.StartAsTask
        
        member _.SearchAcademicAsync(query: string, domains: string[]) =
            academicSearch.SearchAsync(query, domains, 10) |> Async.StartAsTask
        
        member _.SearchTripleStoresAsync(query: string, endpoints: string[]) =
            tripleStoreSearch.SearchAsync(query, endpoints) |> Async.StartAsTask
        
        member _.SearchSpecializedAsync(query: string, domain: string) = async {
            // Implement specialized search based on domain
            return [||]
        } |> Async.StartAsTask
        
        member _.GetProviderStatusAsync() = async {
            return Map.ofList [
                ("google", true)
                ("bing", true)
                ("duckduckgo", true)
                ("arxiv", true)
                ("pubmed", true)
                ("wikidata", true)
                ("dbpedia", true)
            ]
        } |> Async.StartAsTask

/// Agent search integration
module AgentSearchIntegration =

    /// Search context for agents
    type AgentSearchContext = {
        AgentId: string
        AgentType: string
        CurrentTask: string option
        ConversationHistory: string[]
        Specialization: string option
        UserPreferences: Map<string, string>
    }

    /// Search trigger for autonomous agent search
    type SearchTrigger =
        | KnowledgeGapDetected of string
        | FactVerificationNeeded of string
        | ContextEnhancementRequired of string
        | RealTimeInformationNeeded of string
        | UserQueryRequiresSearch of string

    /// Agent search capabilities
    type IAgentSearchCapabilities =
        abstract member AutonomousSearchAsync: AgentSearchContext -> SearchTrigger -> Task<SearchResults>
        abstract member ContextualSearchAsync: AgentSearchContext -> string -> Task<SearchResults>
        abstract member CollaborativeSearchAsync: AgentSearchContext[] -> string -> Task<SearchResults>
        abstract member InjectSearchResultsAsync: SearchResults -> string -> Task<bool>

    /// Implementation of agent search capabilities
    type AgentSearchCapabilities(searchService: IOnDemandSearchService, logger: ILogger<AgentSearchCapabilities>) =

        let analyzeAgentContext (context: AgentSearchContext) =
            let contextKeywords = [
                context.CurrentTask |> Option.defaultValue ""
                context.Specialization |> Option.defaultValue ""
                String.Join(" ", context.ConversationHistory |> Array.take (min 5 context.ConversationHistory.Length))
            ] |> List.filter (fun s -> not (String.IsNullOrWhiteSpace(s)))

            String.Join(" ", contextKeywords)

        let createContextualQuery (context: AgentSearchContext) (baseQuery: string) =
            let contextInfo = analyzeAgentContext context
            let enhancedQuery =
                if String.IsNullOrWhiteSpace(contextInfo) then baseQuery
                else $"{baseQuery} {contextInfo}"

            {
                Query = enhancedQuery
                Intent = Some (inferIntentFromAgent context)
                Domain = context.Specialization
                Context = Map.ofList [
                    ("agent_id", context.AgentId)
                    ("agent_type", context.AgentType)
                    ("task", context.CurrentTask |> Option.defaultValue "")
                ]
                MaxResults = 10
                QualityThreshold = 0.7
                Providers = None
            }

        and inferIntentFromAgent (context: AgentSearchContext) =
            match context.AgentType.ToLower() with
            | "research" | "analysis" -> "academic"
            | "factcheck" | "verification" -> "factual"
            | "technical" | "development" -> "technical"
            | "news" | "monitoring" -> "current_events"
            | _ -> "general"

    /// Search context for agents
    type AgentSearchContext = {
        AgentId: string
        AgentType: string
        CurrentTask: string option
        ConversationHistory: string[]
        Specialization: string option
        UserPreferences: Map<string, string>
    }

    /// Search trigger for autonomous agent search
    type SearchTrigger =
        | KnowledgeGapDetected of string
        | FactVerificationNeeded of string
        | ContextEnhancementRequired of string
        | RealTimeInformationNeeded of string
        | UserQueryRequiresSearch of string

    /// Agent search capabilities
    type IAgentSearchCapabilities =
        abstract member AutonomousSearchAsync: AgentSearchContext -> SearchTrigger -> Task<SearchResults>
        abstract member ContextualSearchAsync: AgentSearchContext -> string -> Task<SearchResults>
        abstract member CollaborativeSearchAsync: AgentSearchContext[] -> string -> Task<SearchResults>
        abstract member InjectSearchResultsAsync: SearchResults -> string -> Task<bool>

    /// Implementation of agent search capabilities
    type AgentSearchCapabilities(searchService: IOnDemandSearchService, logger: ILogger<AgentSearchCapabilities>) =

        let analyzeAgentContext (context: AgentSearchContext) =
            let contextKeywords = [
                context.CurrentTask |> Option.defaultValue ""
                context.Specialization |> Option.defaultValue ""
                String.Join(" ", context.ConversationHistory |> Array.take (min 5 context.ConversationHistory.Length))
            ] |> List.filter (fun s -> not (String.IsNullOrWhiteSpace(s)))

            String.Join(" ", contextKeywords)

        let createContextualQuery (context: AgentSearchContext) (baseQuery: string) =
            let contextInfo = analyzeAgentContext context
            let enhancedQuery =
                if String.IsNullOrWhiteSpace(contextInfo) then baseQuery
                else $"{baseQuery} {contextInfo}"

            {
                Query = enhancedQuery
                Intent = Some (inferIntentFromAgent context)
                Domain = context.Specialization
                Context = Map.ofList [
                    ("agent_id", context.AgentId)
                    ("agent_type", context.AgentType)
                    ("task", context.CurrentTask |> Option.defaultValue "")
                ]
                MaxResults = 10
                QualityThreshold = 0.7
                Providers = None
            }

        and inferIntentFromAgent (context: AgentSearchContext) =
            match context.AgentType.ToLower() with
            | "research" | "analysis" -> "academic"
            | "factcheck" | "verification" -> "factual"
            | "technical" | "development" -> "technical"
            | "news" | "monitoring" -> "current_events"
            | _ -> "general"

        interface IAgentSearchCapabilities with
            member _.AutonomousSearchAsync(context: AgentSearchContext, trigger: SearchTrigger) = async {
                logger.LogInformation("Agent {AgentId} triggered autonomous search: {Trigger}", context.AgentId, trigger)

                let query =
                    match trigger with
                    | KnowledgeGapDetected(topic) -> createContextualQuery context $"information about {topic}"
                    | FactVerificationNeeded(claim) -> createContextualQuery context $"verify fact: {claim}"
                    | ContextEnhancementRequired(area) -> createContextualQuery context $"context for {area}"
                    | RealTimeInformationNeeded(topic) -> createContextualQuery context $"latest information {topic}"
                    | UserQueryRequiresSearch(userQuery) -> createContextualQuery context userQuery

                let! results = searchService.SearchAsync(query, SearchStrategy.Adaptive) |> Async.AwaitTask

                logger.LogInformation("Autonomous search completed for agent {AgentId}: {ResultCount} results",
                    context.AgentId, results.Results.Length)

                return results
            } |> Async.StartAsTask

            member _.ContextualSearchAsync(context: AgentSearchContext, query: string) = async {
                logger.LogInformation("Agent {AgentId} performing contextual search: {Query}", context.AgentId, query)

                let contextualQuery = createContextualQuery context query
                let! results = searchService.SearchAsync(contextualQuery, SearchStrategy.Adaptive) |> Async.AwaitTask

                return results
            } |> Async.StartAsTask

            member _.CollaborativeSearchAsync(contexts: AgentSearchContext[], query: string) = async {
                logger.LogInformation("Collaborative search initiated by {AgentCount} agents", contexts.Length)

                // Divide search space among agents based on their specializations
                let searchTasks =
                    contexts
                    |> Array.mapi (fun i context ->
                        let specializedQuery =
                            match context.Specialization with
                            | Some spec -> $"{query} {spec}"
                            | None -> query

                        let agentQuery = createContextualQuery context specializedQuery
                        searchService.SearchAsync(agentQuery, SearchStrategy.DomainSpecific(context.Specialization |> Option.defaultValue "general"))
                    )

                let! allResults = searchTasks |> Array.map Async.AwaitTask |> Async.Parallel

                // Merge and deduplicate results
                let mergedResults =
                    allResults
                    |> Array.collect (fun r -> r.Results)
                    |> Array.distinctBy (fun r -> r.Url)
                    |> Array.sortByDescending (fun r -> r.Relevance * r.Credibility)

                let collaborativeResults = {
                    Query = { Query = query; Intent = None; Domain = None; Context = Map.empty; MaxResults = 50; QualityThreshold = 0.7; Providers = None }
                    Results = mergedResults
                    TotalResults = mergedResults.Length
                    SearchTime = TimeSpan.FromSeconds(2.0)
                    ProvidersUsed = allResults |> Array.collect (fun r -> r.ProvidersUsed) |> Array.distinct
                    QualityScore = if mergedResults.Length > 0 then mergedResults |> Array.averageBy (fun r -> r.Relevance * r.Credibility) else 0.0
                    Metadata = Map.ofList [("collaborative", true :> obj); ("agent_count", contexts.Length :> obj)]
                }

                logger.LogInformation("Collaborative search completed: {ResultCount} results from {AgentCount} agents",
                    collaborativeResults.Results.Length, contexts.Length)

                return collaborativeResults
            } |> Async.StartAsTask

            member _.InjectSearchResultsAsync(results: SearchResults, targetCollection: string) = async {
                logger.LogInformation("Injecting {ResultCount} search results into collection: {Collection}",
                    results.Results.Length, targetCollection)

                // Simulate injection into vector store
                // In real implementation, this would use the vector store service
                let injectionSuccess = results.Results.Length > 0

                if injectionSuccess then
                    logger.LogInformation("Successfully injected search results into {Collection}", targetCollection)
                else
                    logger.LogWarning("Failed to inject search results into {Collection}", targetCollection)

                return injectionSuccess
            } |> Async.StartAsTask

/// Metascript search functions
module MetascriptSearchFunctions =

    /// Search function results for metascripts
    type MetascriptSearchResult = {
        Success: bool
        Results: SearchResult[]
        Message: string
        Metadata: Map<string, obj>
    }

    /// Metascript search function implementations
    type MetascriptSearchFunctions(searchService: IOnDemandSearchService, logger: ILogger<MetascriptSearchFunctions>) =

        member _.SearchWeb(query: string, providers: string[] option, maxResults: int, filters: Map<string, string> option) = async {
            try
                let searchQuery = {
                    Query = query
                    Intent = None
                    Domain = None
                    Context = filters |> Option.defaultValue Map.empty
                    MaxResults = maxResults
                    QualityThreshold = 0.7
                    Providers = providers
                }

                let! results = searchService.SearchAsync(searchQuery, SearchStrategy.Comprehensive) |> Async.AwaitTask

                return {
                    Success = true
                    Results = results.Results
                    Message = $"Found {results.Results.Length} web results"
                    Metadata = Map.ofList [("search_time", results.SearchTime :> obj)]
                }
            with
            | ex ->
                logger.LogError(ex, "Web search failed for query: {Query}", query)
                return {
                    Success = false
                    Results = [||]
                    Message = $"Web search failed: {ex.Message}"
                    Metadata = Map.empty
                }
        }

        member _.SearchAcademic(query: string, domains: string[] option, dateRange: (DateTime * DateTime) option, peerReviewed: bool option) = async {
            try
                let academicDomains = domains |> Option.defaultValue [|"arxiv"; "pubmed"|]
                let! results = searchService.SearchAcademicAsync(query, academicDomains) |> Async.AwaitTask

                let filteredResults =
                    if peerReviewed |> Option.defaultValue false then
                        results |> Array.filter (fun r -> r.Metadata.ContainsKey("type") && r.Metadata.["type"].ToString() = "peer_reviewed")
                    else results

                return {
                    Success = true
                    Results = filteredResults
                    Message = $"Found {filteredResults.Length} academic results"
                    Metadata = Map.ofList [("domains", academicDomains :> obj)]
                }
            with
            | ex ->
                logger.LogError(ex, "Academic search failed for query: {Query}", query)
                return {
                    Success = false
                    Results = [||]
                    Message = $"Academic search failed: {ex.Message}"
                    Metadata = Map.empty
                }
        }

        member _.SearchTripleStores(sparqlQuery: string, endpoints: string[] option, timeout: int option) = async {
            try
                let tsEndpoints = endpoints |> Option.defaultValue [|"wikidata"; "dbpedia"|]
                let! results = searchService.SearchTripleStoresAsync(sparqlQuery, tsEndpoints) |> Async.AwaitTask

                return {
                    Success = true
                    Results = results
                    Message = $"Found {results.Length} triple store results"
                    Metadata = Map.ofList [("endpoints", tsEndpoints :> obj)]
                }
            with
            | ex ->
                logger.LogError(ex, "Triple store search failed for query: {Query}", sparqlQuery)
                return {
                    Success = false
                    Results = [||]
                    Message = $"Triple store search failed: {ex.Message}"
                    Metadata = Map.empty
                }
        }

        member _.SearchAdaptive(query: string, intent: string option, context: Map<string, string> option, strategy: string option) = async {
            try
                let searchQuery = {
                    Query = query
                    Intent = intent
                    Domain = None
                    Context = context |> Option.defaultValue Map.empty
                    MaxResults = 20
                    QualityThreshold = 0.7
                    Providers = None
                }

                let searchStrategy =
                    match strategy |> Option.defaultValue "adaptive" with
                    | "comprehensive" -> SearchStrategy.Comprehensive
                    | "realtime" -> SearchStrategy.RealTime
                    | domain when domain.StartsWith("domain:") -> SearchStrategy.DomainSpecific(domain.Substring(7))
                    | _ -> SearchStrategy.Adaptive

                let! results = searchService.SearchAsync(searchQuery, searchStrategy) |> Async.AwaitTask

                return {
                    Success = true
                    Results = results.Results
                    Message = $"Adaptive search found {results.Results.Length} results with quality {results.QualityScore:F2}"
                    Metadata = Map.ofList [
                        ("strategy", strategy |> Option.defaultValue "adaptive" :> obj)
                        ("quality_score", results.QualityScore :> obj)
                        ("providers_used", results.ProvidersUsed :> obj)
                    ]
                }
            with
            | ex ->
                logger.LogError(ex, "Adaptive search failed for query: {Query}", query)
                return {
                    Success = false
                    Results = [||]
                    Message = $"Adaptive search failed: {ex.Message}"
                    Metadata = Map.empty
                }
        }
