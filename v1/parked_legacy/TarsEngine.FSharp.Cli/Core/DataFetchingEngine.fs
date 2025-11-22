namespace TarsEngine.FSharp.Cli.Core

open System
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// TARS Autonomous Data Fetching Engine with triple store and SPARQL support
module DataFetchingEngine =
    
    /// Supported data source types
    type DataSourceType =
        | TripleStore of endpoint: string
        | SparqlEndpoint of endpoint: string
        | RestApi of baseUrl: string
        | GraphQL of endpoint: string
        | Database of connectionString: string
        | FileSystem of path: string
        | WebScraping of url: string

    /// Query types for different data sources
    type QueryType =
        | SparqlQuery of query: string
        | SqlQuery of query: string
        | RestQuery of endpoint: string * method: string * headers: Map<string, string>
        | GraphQLQuery of query: string * variables: Map<string, obj>
        | FileQuery of pattern: string

    /// Data fetch result
    type DataFetchResult = {
        Source: DataSourceType
        Query: QueryType
        Data: string
        Metadata: Map<string, obj>
        FetchTime: DateTime
        ExecutionTime: TimeSpan
        RecordCount: int option
        Success: bool
        ErrorMessage: string option
    }

    /// SPARQL query builder for common patterns
    module SparqlQueryBuilder =
        
        /// Build a SELECT query with common prefixes
        let buildSelectQuery (prefixes: (string * string) list) (selectClause: string) (whereClause: string) (limit: int option) =
            let prefixLines = 
                prefixes 
                |> List.map (fun (prefix, uri) -> $"PREFIX {prefix}: <{uri}>")
                |> String.concat "\n"
            
            let limitClause = 
                match limit with
                | Some n -> $"\nLIMIT {n}"
                | None -> ""
            
            $"{prefixLines}\nSELECT {selectClause}\nWHERE {{\n{whereClause}\n}}{limitClause}"

        /// Common RDF prefixes
        let commonPrefixes = [
            ("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
            ("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
            ("owl", "http://www.w3.org/2002/07/owl#")
            ("foaf", "http://xmlns.com/foaf/0.1/")
            ("dc", "http://purl.org/dc/elements/1.1/")
            ("dbo", "http://dbpedia.org/ontology/")
            ("dbr", "http://dbpedia.org/resource/")
        ]

        /// Build a query to find all classes
        let findAllClasses (limit: int option) =
            buildSelectQuery commonPrefixes "DISTINCT ?class ?label" 
                "?class a owl:Class .\n  OPTIONAL { ?class rdfs:label ?label }" limit

        /// Build a query to find all properties
        let findAllProperties (limit: int option) =
            buildSelectQuery commonPrefixes "DISTINCT ?property ?label" 
                "?property a rdf:Property .\n  OPTIONAL { ?property rdfs:label ?label }" limit

        /// Build a query to explore a specific resource
        let exploreResource (resourceUri: string) =
            buildSelectQuery commonPrefixes "?predicate ?object" 
                $"<{resourceUri}> ?predicate ?object" (Some 100)

    /// Triple store client for SPARQL operations
    type TripleStoreClient(endpoint: string, httpClient: HttpClient, logger: ILogger option) =
        
        /// Execute SPARQL SELECT query
        member this.ExecuteSelectQuery(query: string) : Task<DataFetchResult> =
            task {
                let startTime = DateTime.Now
                try
                    let encodedQuery = Uri.EscapeDataString(query)
                    let requestUrl = $"{endpoint}?query={encodedQuery}&format=application/sparql-results+json"
                    
                    logger |> Option.iter (fun l -> l.LogInformation($"Executing SPARQL query: {query}"))
                    
                    let! response = httpClient.GetAsync(requestUrl)
                    let! content = response.Content.ReadAsStringAsync()
                    
                    if response.IsSuccessStatusCode then
                        // Parse SPARQL JSON results to count records
                        let recordCount = 
                            try
                                let jsonDoc = JsonDocument.Parse(content)
                                let bindings = jsonDoc.RootElement.GetProperty("results").GetProperty("bindings")
                                Some (bindings.GetArrayLength())
                            with
                            | _ -> None
                        
                        let executionTime = DateTime.Now - startTime
                        return {
                            Source = SparqlEndpoint endpoint
                            Query = SparqlQuery query
                            Data = content
                            Metadata = Map [
                                ("contentType",
                                    match response.Content.Headers.ContentType with
                                    | null -> "unknown"
                                    | ct -> ct.ToString())
                                ("statusCode", response.StatusCode.ToString())
                            ]
                            FetchTime = startTime
                            ExecutionTime = executionTime
                            RecordCount = recordCount
                            Success = true
                            ErrorMessage = None
                        }
                    else
                        let executionTime = DateTime.Now - startTime
                        return {
                            Source = SparqlEndpoint endpoint
                            Query = SparqlQuery query
                            Data = content
                            Metadata = Map [("statusCode", response.StatusCode.ToString())]
                            FetchTime = startTime
                            ExecutionTime = executionTime
                            RecordCount = None
                            Success = false
                            ErrorMessage = Some $"HTTP {response.StatusCode}: {content}"
                        }
                with
                | ex ->
                    let executionTime = DateTime.Now - startTime
                    logger |> Option.iter (fun l -> l.LogError(ex, "SPARQL query execution failed"))
                    return {
                        Source = SparqlEndpoint endpoint
                        Query = SparqlQuery query
                        Data = ""
                        Metadata = Map.empty
                        FetchTime = startTime
                        ExecutionTime = executionTime
                        RecordCount = None
                        Success = false
                        ErrorMessage = Some ex.Message
                    }
            }

        /// Execute SPARQL ASK query
        member this.ExecuteAskQuery(query: string) : Task<bool * DataFetchResult> =
            task {
                let! result = this.ExecuteSelectQuery(query)
                if result.Success then
                    try
                        let jsonDoc = JsonDocument.Parse(result.Data)
                        let askResult = jsonDoc.RootElement.GetProperty("boolean").GetBoolean()
                        return (askResult, result)
                    with
                    | _ -> return (false, { result with Success = false; ErrorMessage = Some "Failed to parse ASK result" })
                else
                    return (false, result)
            }

        /// Get endpoint information
        member this.GetEndpointInfo() : Task<DataFetchResult> =
            task {
                let infoQuery = """
                    SELECT (COUNT(*) as ?triples) WHERE { ?s ?p ?o }
                """
                return! this.ExecuteSelectQuery(infoQuery)
            }

    /// REST API client for general data fetching
    type RestApiClient(baseUrl: string, httpClient: HttpClient, logger: ILogger option) =
        
        /// Execute REST API call
        member this.ExecuteRequest(endpoint: string, method: string, headers: Map<string, string>, body: string option) : Task<DataFetchResult> =
            task {
                let startTime = DateTime.Now
                try
                    let requestUrl = if endpoint.StartsWith("http") then endpoint else $"{baseUrl.TrimEnd('/')}/{endpoint.TrimStart('/')}"
                    
                    let request = new HttpRequestMessage()
                    request.RequestUri <- Uri(requestUrl)
                    request.Method <- HttpMethod(method.ToUpper())
                    
                    // Add headers
                    for kvp in headers do
                        request.Headers.TryAddWithoutValidation(kvp.Key, kvp.Value) |> ignore
                    
                    // Add body if provided
                    match body with
                    | Some content -> request.Content <- new StringContent(content, Encoding.UTF8, "application/json")
                    | None -> ()
                    
                    logger |> Option.iter (fun l -> l.LogInformation($"Executing REST request: {method} {requestUrl}"))
                    
                    let! response = httpClient.SendAsync(request)
                    let! content = response.Content.ReadAsStringAsync()
                    
                    let executionTime = DateTime.Now - startTime
                    return {
                        Source = RestApi baseUrl
                        Query = RestQuery (endpoint, method, headers)
                        Data = content
                        Metadata = Map [
                            ("contentType",
                                match response.Content.Headers.ContentType with
                                | null -> "unknown"
                                | ct -> ct.ToString())
                            ("statusCode", response.StatusCode.ToString())
                            ("contentLength",
                                match response.Content.Headers.ContentLength with
                                | cl when cl.HasValue -> cl.Value.ToString()
                                | _ -> "unknown")
                        ]
                        FetchTime = startTime
                        ExecutionTime = executionTime
                        RecordCount = None
                        Success = response.IsSuccessStatusCode
                        ErrorMessage = if response.IsSuccessStatusCode then None else Some $"HTTP {response.StatusCode}: {content}"
                    }
                with
                | ex ->
                    let executionTime = DateTime.Now - startTime
                    logger |> Option.iter (fun l -> l.LogError(ex, "REST API request failed"))
                    return {
                        Source = RestApi baseUrl
                        Query = RestQuery (endpoint, method, headers)
                        Data = ""
                        Metadata = Map.empty
                        FetchTime = startTime
                        ExecutionTime = executionTime
                        RecordCount = None
                        Success = false
                        ErrorMessage = Some ex.Message
                    }
            }

    /// Main data fetching orchestrator
    type DataFetchingOrchestrator(httpClient: HttpClient, logger: ILogger option) =
        
        /// Execute data fetch based on source type and query
        member this.FetchData(source: DataSourceType, query: QueryType) : Task<DataFetchResult> =
            task {
                match source, query with
                | SparqlEndpoint endpoint, SparqlQuery sparqlQuery ->
                    let client = TripleStoreClient(endpoint, httpClient, logger)
                    return! client.ExecuteSelectQuery(sparqlQuery)
                
                | TripleStore endpoint, SparqlQuery sparqlQuery ->
                    // Assume Fuseki-style endpoint
                    let sparqlEndpoint = $"{endpoint.TrimEnd('/')}/sparql"
                    let client = TripleStoreClient(sparqlEndpoint, httpClient, logger)
                    return! client.ExecuteSelectQuery(sparqlQuery)
                
                | RestApi baseUrl, RestQuery (endpoint, method, headers) ->
                    let client = RestApiClient(baseUrl, httpClient, logger)
                    return! client.ExecuteRequest(endpoint, method, headers, None)
                
                | _ ->
                    return {
                        Source = source
                        Query = query
                        Data = ""
                        Metadata = Map.empty
                        FetchTime = DateTime.Now
                        ExecutionTime = TimeSpan.Zero
                        RecordCount = None
                        Success = false
                        ErrorMessage = Some "Unsupported source/query combination"
                    }
            }

        /// Discover available data sources
        member this.DiscoverDataSources(endpoints: string list) : Task<DataFetchResult list> =
            task {
                let tasks = 
                    endpoints 
                    |> List.map (fun endpoint ->
                        task {
                            try
                                // Try SPARQL endpoint
                                let client = TripleStoreClient(endpoint, httpClient, logger)
                                return! client.GetEndpointInfo()
                            with
                            | _ ->
                                // Fallback to simple HTTP check
                                let! response = httpClient.GetAsync(endpoint)
                                let! content = response.Content.ReadAsStringAsync()
                                return {
                                    Source = SparqlEndpoint endpoint
                                    Query = SparqlQuery "discovery"
                                    Data = content
                                    Metadata = Map [("statusCode", response.StatusCode.ToString())]
                                    FetchTime = DateTime.Now
                                    ExecutionTime = TimeSpan.Zero
                                    RecordCount = None
                                    Success = response.IsSuccessStatusCode
                                    ErrorMessage = if response.IsSuccessStatusCode then None else Some "Endpoint not accessible"
                                }
                        })
                
                let! results = Task.WhenAll(tasks)
                return results |> Array.toList
            }

    /// Well-known SPARQL endpoints for discovery
    let wellKnownEndpoints = [
        "https://dbpedia.org/sparql"  // DBpedia
        "https://query.wikidata.org/sparql"  // Wikidata
        "http://localhost:3030/ds/sparql"  // Local Fuseki
        "http://localhost:8890/sparql"  // Local Virtuoso
    ]

    /// Create data fetching orchestrator with default HTTP client
    let createOrchestrator (logger: ILogger option) =
        let httpClient = new HttpClient()
        httpClient.Timeout <- TimeSpan.FromMinutes(5.0)
        httpClient.DefaultRequestHeaders.Add("User-Agent", "TARS-DataFetcher/1.0")
        DataFetchingOrchestrator(httpClient, logger)
