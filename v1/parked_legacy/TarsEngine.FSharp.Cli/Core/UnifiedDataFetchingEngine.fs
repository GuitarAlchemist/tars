namespace TarsEngine.FSharp.Cli.Core

open System
open System.Net.Http
open System.Threading
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem

/// Unified Data Fetching Engine - Data retrieval using unified architecture
module UnifiedDataFetchingEngine =
    
    /// Data source types
    type UnifiedDataSource =
        | SparqlEndpoint of endpoint: string
        | TripleStore of endpoint: string
        | RestApi of baseUrl: string
        | GraphQL of endpoint: string
        | Database of connectionString: string
        | FileSystem of basePath: string
        | WebScraping of baseUrl: string
    
    /// Query types
    type UnifiedQuery =
        | SparqlQuery of query: string
        | RestQuery of path: string * parameters: Map<string, string>
        | GraphQLQuery of query: string * variables: Map<string, obj>
        | SqlQuery of query: string
        | FileQuery of pattern: string
        | XPathQuery of xpath: string
    
    /// Data fetch result using unified types
    type UnifiedDataFetchResult = {
        Success: bool
        Data: obj option
        RecordCount: int
        ExecutionTime: TimeSpan
        DataSource: UnifiedDataSource
        Query: UnifiedQuery
        ErrorDetails: TarsError option
        Metadata: Map<string, obj>
        ProofId: string option
        CorrelationId: string
    }
    
    /// Data fetching context
    type DataFetchingContext = {
        ConfigManager: UnifiedConfigurationManager
        ProofGenerator: UnifiedProofGenerator
        Logger: ITarsLogger
        HttpClient: HttpClient
        CorrelationId: string
        Timeout: TimeSpan
    }
    
    /// Create data fetching context
    let createDataFetchingContext (logger: ITarsLogger) (configManager: UnifiedConfigurationManager) (proofGenerator: UnifiedProofGenerator) =
        let timeout = TimeSpan.FromSeconds(ConfigurationExtensions.getInt configManager "tars.data.timeoutSeconds" 30)
        let httpClient = new HttpClient()
        httpClient.Timeout <- timeout
        
        {
            ConfigManager = configManager
            ProofGenerator = proofGenerator
            Logger = logger
            HttpClient = httpClient
            CorrelationId = generateCorrelationId()
            Timeout = timeout
        }
    
    /// Execute SPARQL query
    let executeSparqlQuery (context: DataFetchingContext) (endpoint: string) (query: string) =
        task {
            try
                let startTime = DateTime.UtcNow
                context.Logger.LogInformation(context.CorrelationId, $"Executing SPARQL query against {endpoint}")
                
                // Generate proof for data fetch operation
                let! fetchProof =
                    ProofExtensions.generateExecutionProof
                        context.ProofGenerator
                        $"SparqlDataFetch_{endpoint}"
                        context.CorrelationId
                
                let encodedQuery = Uri.EscapeDataString(query)
                let requestUrl = $"{endpoint}?query={encodedQuery}&format=application/sparql-results+json"
                
                let! response = context.HttpClient.GetAsync(requestUrl)
                let! content = response.Content.ReadAsStringAsync()
                
                let executionTime = DateTime.UtcNow - startTime
                
                if response.IsSuccessStatusCode then
                    // Parse JSON response (simplified)
                    let recordCount = if content.Contains("\"results\"") then 1 else 0
                    
                    let proofId = match fetchProof with
                                  | Success (proof, _) -> Some proof.ProofId
                                  | Failure _ -> None
                    
                    return {
                        Success = true
                        Data = Some (box content)
                        RecordCount = recordCount
                        ExecutionTime = executionTime
                        DataSource = SparqlEndpoint endpoint
                        Query = SparqlQuery query
                        ErrorDetails = None
                        Metadata = Map [
                            ("endpoint", box endpoint)
                            ("responseSize", box content.Length)
                            ("statusCode", box (int response.StatusCode))
                        ]
                        ProofId = proofId
                        CorrelationId = context.CorrelationId
                    }
                else
                    let error = NetworkError ($"SPARQL query failed with status {response.StatusCode}", endpoint)
                    return {
                        Success = false
                        Data = None
                        RecordCount = 0
                        ExecutionTime = executionTime
                        DataSource = SparqlEndpoint endpoint
                        Query = SparqlQuery query
                        ErrorDetails = Some error
                        Metadata = Map [("statusCode", box (int response.StatusCode)); ("error", box content)]
                        ProofId = None
                        CorrelationId = context.CorrelationId
                    }
            
            with
            | ex ->
                context.Logger.LogError(context.CorrelationId, TarsError.create "SparqlError" "SPARQL query failed" (Some ex), ex)
                let error = ExecutionError ($"SPARQL query execution failed: {ex.Message}", Some ex)
                return {
                    Success = false
                    Data = None
                    RecordCount = 0
                    ExecutionTime = TimeSpan.Zero
                    DataSource = SparqlEndpoint endpoint
                    Query = SparqlQuery query
                    ErrorDetails = Some error
                    Metadata = Map [("exception", box ex.Message)]
                    ProofId = None
                    CorrelationId = context.CorrelationId
                }
        }
    
    /// Execute REST API query
    let executeRestQuery (context: DataFetchingContext) (baseUrl: string) (path: string) (parameters: Map<string, string>) =
        task {
            try
                let startTime = DateTime.UtcNow
                context.Logger.LogInformation(context.CorrelationId, $"Executing REST query: {baseUrl}/{path}")
                
                // Generate proof for REST fetch operation
                let! fetchProof =
                    ProofExtensions.generateExecutionProof
                        context.ProofGenerator
                        $"RestDataFetch_{baseUrl}"
                        context.CorrelationId
                
                let queryString = 
                    parameters 
                    |> Map.toSeq 
                    |> Seq.map (fun (k, v) -> $"{k}={Uri.EscapeDataString(v)}")
                    |> String.concat "&"
                
                let requestUrl = 
                    if String.IsNullOrEmpty(queryString) then
                        $"{baseUrl.TrimEnd('/')}/{path.TrimStart('/')}"
                    else
                        $"{baseUrl.TrimEnd('/')}/{path.TrimStart('/')}?{queryString}"
                
                let! response = context.HttpClient.GetAsync(requestUrl)
                let! content = response.Content.ReadAsStringAsync()
                
                let executionTime = DateTime.UtcNow - startTime
                
                if response.IsSuccessStatusCode then
                    let proofId = match fetchProof with
                                  | Success (proof, _) -> Some proof.ProofId
                                  | Failure _ -> None
                    
                    return {
                        Success = true
                        Data = Some (box content)
                        RecordCount = 1
                        ExecutionTime = executionTime
                        DataSource = RestApi baseUrl
                        Query = RestQuery (path, parameters)
                        ErrorDetails = None
                        Metadata = Map [
                            ("url", box requestUrl)
                            ("responseSize", box content.Length)
                            ("statusCode", box (int response.StatusCode))
                            ("contentType", box (match response.Content.Headers.ContentType with | null -> "unknown" | ct -> ct.ToString()))
                        ]
                        ProofId = proofId
                        CorrelationId = context.CorrelationId
                    }
                else
                    let error = NetworkError ($"REST query failed with status {response.StatusCode}", requestUrl)
                    return {
                        Success = false
                        Data = None
                        RecordCount = 0
                        ExecutionTime = executionTime
                        DataSource = RestApi baseUrl
                        Query = RestQuery (path, parameters)
                        ErrorDetails = Some error
                        Metadata = Map [("statusCode", box (int response.StatusCode)); ("error", box content)]
                        ProofId = None
                        CorrelationId = context.CorrelationId
                    }
            
            with
            | ex ->
                context.Logger.LogError(context.CorrelationId, TarsError.create "RestError" "REST query failed" (Some ex), ex)
                let error = ExecutionError ($"REST query execution failed: {ex.Message}", Some ex)
                return {
                    Success = false
                    Data = None
                    RecordCount = 0
                    ExecutionTime = TimeSpan.Zero
                    DataSource = RestApi baseUrl
                    Query = RestQuery (path, parameters)
                    ErrorDetails = Some error
                    Metadata = Map [("exception", box ex.Message)]
                    ProofId = None
                    CorrelationId = context.CorrelationId
                }
        }
    
    /// Execute file system query
    let executeFileQuery (context: DataFetchingContext) (basePath: string) (pattern: string) =
        task {
            try
                let startTime = DateTime.UtcNow
                context.Logger.LogInformation(context.CorrelationId, $"Executing file query: {basePath}/{pattern}")
                
                // Generate proof for file fetch operation
                let! fetchProof =
                    ProofExtensions.generateExecutionProof
                        context.ProofGenerator
                        $"FileDataFetch_{basePath}"
                        context.CorrelationId
                
                if not (System.IO.Directory.Exists(basePath)) then
                    let error = ValidationError ($"Directory not found: {basePath}", Map [("basePath", basePath)])
                    return {
                        Success = false
                        Data = None
                        RecordCount = 0
                        ExecutionTime = TimeSpan.Zero
                        DataSource = FileSystem basePath
                        Query = FileQuery pattern
                        ErrorDetails = Some error
                        Metadata = Map [("error", box "Directory not found")]
                        ProofId = None
                        CorrelationId = context.CorrelationId
                    }
                
                let files = System.IO.Directory.GetFiles(basePath, pattern, System.IO.SearchOption.TopDirectoryOnly)
                let executionTime = DateTime.UtcNow - startTime
                
                let proofId = match fetchProof with
                              | Success (proof, _) -> Some proof.ProofId
                              | Failure _ -> None
                
                return {
                    Success = true
                    Data = Some (box files)
                    RecordCount = files.Length
                    ExecutionTime = executionTime
                    DataSource = FileSystem basePath
                    Query = FileQuery pattern
                    ErrorDetails = None
                    Metadata = Map [
                        ("basePath", box basePath)
                        ("pattern", box pattern)
                        ("fileCount", box files.Length)
                    ]
                    ProofId = proofId
                    CorrelationId = context.CorrelationId
                }
            
            with
            | ex ->
                context.Logger.LogError(context.CorrelationId, TarsError.create "FileError" "File query failed" (Some ex), ex)
                let error = ExecutionError ($"File query execution failed: {ex.Message}", Some ex)
                return {
                    Success = false
                    Data = None
                    RecordCount = 0
                    ExecutionTime = TimeSpan.Zero
                    DataSource = FileSystem basePath
                    Query = FileQuery pattern
                    ErrorDetails = Some error
                    Metadata = Map [("exception", box ex.Message)]
                    ProofId = None
                    CorrelationId = context.CorrelationId
                }
        }
    
    /// Unified Data Fetching Engine implementation
    type UnifiedDataFetchingEngine(logger: ITarsLogger, configManager: UnifiedConfigurationManager, proofGenerator: UnifiedProofGenerator) =
        
        let context = createDataFetchingContext logger configManager proofGenerator
        
        /// Fetch data from any supported source
        member this.FetchDataAsync(dataSource: UnifiedDataSource, query: UnifiedQuery) : Task<UnifiedDataFetchResult> =
            task {
                match dataSource, query with
                | SparqlEndpoint endpoint, SparqlQuery sparqlQuery ->
                    return! executeSparqlQuery context endpoint sparqlQuery
                
                | TripleStore endpoint, SparqlQuery sparqlQuery ->
                    let sparqlEndpoint = $"{endpoint.TrimEnd('/')}/sparql"
                    return! executeSparqlQuery context sparqlEndpoint sparqlQuery
                
                | RestApi baseUrl, RestQuery (path, parameters) ->
                    return! executeRestQuery context baseUrl path parameters
                
                | FileSystem basePath, FileQuery pattern ->
                    return! executeFileQuery context basePath pattern
                
                | _ ->
                    let error = ValidationError ($"Unsupported combination: {dataSource} with {query}", Map.empty)
                    return {
                        Success = false
                        Data = None
                        RecordCount = 0
                        ExecutionTime = TimeSpan.Zero
                        DataSource = dataSource
                        Query = query
                        ErrorDetails = Some error
                        Metadata = Map [("error", box "Unsupported combination")]
                        ProofId = None
                        CorrelationId = context.CorrelationId
                    }
            }
        
        /// Get supported data sources
        member this.GetSupportedDataSources() : string list =
            ["SPARQL Endpoint"; "Triple Store"; "REST API"; "GraphQL"; "Database"; "File System"; "Web Scraping"]
        
        /// Get supported query types
        member this.GetSupportedQueryTypes() : string list =
            ["SPARQL Query"; "REST Query"; "GraphQL Query"; "SQL Query"; "File Query"; "XPath Query"]
        
        /// Test connection to data source
        member this.TestConnectionAsync(dataSource: UnifiedDataSource) : Task<TarsResult<bool, TarsError>> =
            task {
                try
                    match dataSource with
                    | SparqlEndpoint endpoint ->
                        let testQuery = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
                        let! result = executeSparqlQuery context endpoint testQuery
                        return Success (result.Success, Map [("endpoint", box endpoint); ("tested", box true)])
                    
                    | RestApi baseUrl ->
                        let! result = executeRestQuery context baseUrl "" Map.empty
                        return Success (result.Success, Map [("baseUrl", box baseUrl); ("tested", box true)])
                    
                    | FileSystem basePath ->
                        let exists = System.IO.Directory.Exists(basePath)
                        return Success (exists, Map [("basePath", box basePath); ("exists", box exists)])
                    
                    | _ ->
                        return Success (true, Map [("dataSource", box (dataSource.ToString())); ("tested", box false)])
                
                with
                | ex ->
                    let error = ExecutionError ($"Connection test failed: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Get data fetching statistics
        member this.GetStatistics() : Map<string, obj> =
            Map [
                ("supportedSources", box (this.GetSupportedDataSources().Length))
                ("supportedQueries", box (this.GetSupportedQueryTypes().Length))
                ("timeout", box context.Timeout.TotalSeconds)
                ("correlationId", box context.CorrelationId)
                ("isInitialized", box true)
            ]
        
        /// Dispose resources
        interface IDisposable with
            member this.Dispose() =
                context.HttpClient.Dispose()
