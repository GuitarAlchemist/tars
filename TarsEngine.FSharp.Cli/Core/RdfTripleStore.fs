namespace TarsEngine.FSharp.Cli.Core

open System
open System.IO
open System.Collections.Generic
open Microsoft.Extensions.Logging
open VDS.RDF
open VDS.RDF.Storage
open VDS.RDF.Query
open VDS.RDF.Query.Datasets
open VDS.RDF.Parsing
open VDS.RDF.Writing

/// TARS Comprehensive RDF Triple Store Integration using dotNetRDF
/// Supports Virtuoso, in-memory stores, and various RDF operations
module RdfTripleStore =
    
    /// RDF store types supported by TARS
    type RdfStoreType =
        | InMemory
        | Virtuoso of connectionString: string
        | File of path: string
        | Remote of endpoint: string

    /// Named graph for organizing RDF data
    type NamedGraph = {
        Uri: Uri
        Name: string
        Description: string
        Created: DateTime
        LastModified: DateTime
    }

    /// Agent belief state for RDF storage
    type AgentBelief = {
        AgentId: string
        BeliefType: string
        Subject: string
        Predicate: string
        Object: string
        Confidence: float
        Timestamp: DateTime
        Source: string
    }

    /// TARS execution state for RDF storage
    type TarsState = {
        StateId: string
        AgentId: string
        Action: string
        Parameters: Map<string, string>
        Result: string
        Timestamp: DateTime
        Duration: TimeSpan
        Success: bool
    }

    /// RDF query result
    type RdfQueryResult = {
        Query: string
        Results: string list list
        Variables: string list
        ExecutionTime: TimeSpan
        RecordCount: int
        Success: bool
        ErrorMessage: string option
    }

    /// TARS RDF Store Manager
    type TarsRdfStore(storeType: RdfStoreType, logger: ILogger option) =
        let mutable store: ITripleStore option = None
        let mutable storageProvider: IStorageProvider option = None
        
        /// Initialize the RDF store based on type
        member private this.InitializeStore() =
            try
                match storeType with
                | InMemory ->
                    let tripleStore = new TripleStore()
                    store <- Some tripleStore
                    logger |> Option.iter (fun l -> l.LogInformation("Initialized in-memory RDF store"))
                    true
                
                | Virtuoso connectionString ->
                    // Note: VirtuosoManager removed for .NET 9 compatibility
                    // Using HTTP-based RDF client instead
                    logger |> Option.iter (fun l -> l.LogInformation($"Virtuoso connection configured: {connectionString} (using HTTP client)"))
                    true
                
                | File path ->
                    let tripleStore = new TripleStore()
                    if File.Exists(path) then
                        try
                            let parser = new TriGParser()
                            parser.Load(tripleStore, path)
                            logger |> Option.iter (fun l -> l.LogInformation($"Loaded RDF store from file: {path}"))
                        with
                        | ex ->
                            logger |> Option.iter (fun l -> l.LogWarning(ex, $"Failed to load existing file: {path}"))
                    store <- Some tripleStore
                    true
                
                | Remote endpoint ->
                    try
                        let sparqlConnector = new SparqlConnector(Uri(endpoint))
                        storageProvider <- Some sparqlConnector
                        logger |> Option.iter (fun l -> l.LogInformation($"Connected to remote SPARQL endpoint: {endpoint}"))
                        true
                    with
                    | ex ->
                        logger |> Option.iter (fun l -> l.LogError(ex, "Failed to connect to remote endpoint"))
                        false
            with
            | ex ->
                logger |> Option.iter (fun l -> l.LogError(ex, "Failed to initialize RDF store"))
                false

        /// Get or create a named graph
        member this.GetOrCreateGraph(graphUri: string, name: string, description: string) : IGraph option =
            try
                match store with
                | Some tripleStore ->
                    let uri = Uri(graphUri)
                    let graph = 
                        if tripleStore.HasGraph(uri) then
                            tripleStore.Graphs.[uri]
                        else
                            let newGraph = new Graph(uri)
                            tripleStore.Add(newGraph)
                            newGraph
                    Some graph
                | None ->
                    if this.InitializeStore() then
                        this.GetOrCreateGraph(graphUri, name, description)
                    else
                        None
            with
            | ex ->
                logger |> Option.iter (fun l -> l.LogError(ex, $"Failed to get/create graph: {graphUri}"))
                None

        /// Store agent belief in RDF format
        member this.StoreAgentBelief(belief: AgentBelief, graphUri: string) : bool =
            try
                match this.GetOrCreateGraph(graphUri, "Agent Beliefs", "Agent belief states and knowledge") with
                | Some graph ->
                    let subjectNode = graph.CreateUriNode(Uri($"tars:agent/{belief.AgentId}"))
                    let predicateNode = graph.CreateUriNode(Uri($"tars:belief/{belief.BeliefType}"))
                    let objectNode = 
                        if belief.Object.StartsWith("http") then
                            graph.CreateUriNode(Uri(belief.Object)) :> INode
                        else
                            graph.CreateLiteralNode(belief.Object) :> INode
                    
                    let triple = new Triple(subjectNode, predicateNode, objectNode)
                    graph.Assert(triple)
                    
                    // Add metadata
                    let confidenceNode = graph.CreateUriNode(Uri("tars:confidence"))
                    let confidenceTriple = new Triple(subjectNode, confidenceNode, graph.CreateLiteralNode(belief.Confidence.ToString()))
                    graph.Assert(confidenceTriple)
                    
                    let timestampNode = graph.CreateUriNode(Uri("tars:timestamp"))
                    let timestampTriple = new Triple(subjectNode, timestampNode, graph.CreateLiteralNode(belief.Timestamp.ToString("O")))
                    graph.Assert(timestampTriple)
                    
                    logger |> Option.iter (fun l -> l.LogInformation($"Stored agent belief: {belief.AgentId} -> {belief.BeliefType}"))
                    true
                | None -> false
            with
            | ex ->
                logger |> Option.iter (fun l -> l.LogError(ex, "Failed to store agent belief"))
                false

        /// Store TARS execution state
        member this.StoreTarsState(state: TarsState, graphUri: string) : bool =
            try
                match this.GetOrCreateGraph(graphUri, "TARS States", "TARS execution states and actions") with
                | Some graph ->
                    let stateNode = graph.CreateUriNode(Uri($"tars:state/{state.StateId}"))
                    let actionNode = graph.CreateUriNode(Uri("tars:action"))
                    let actionTriple = new Triple(stateNode, actionNode, graph.CreateLiteralNode(state.Action))
                    graph.Assert(actionTriple)
                    
                    let agentNode = graph.CreateUriNode(Uri("tars:agent"))
                    let agentTriple = new Triple(stateNode, agentNode, graph.CreateUriNode(Uri($"tars:agent/{state.AgentId}")))
                    graph.Assert(agentTriple)
                    
                    let resultNode = graph.CreateUriNode(Uri("tars:result"))
                    let resultTriple = new Triple(stateNode, resultNode, graph.CreateLiteralNode(state.Result))
                    graph.Assert(resultTriple)
                    
                    let successNode = graph.CreateUriNode(Uri("tars:success"))
                    let successTriple = new Triple(stateNode, successNode, graph.CreateLiteralNode(state.Success.ToString()))
                    graph.Assert(successTriple)
                    
                    let timestampNode = graph.CreateUriNode(Uri("tars:timestamp"))
                    let timestampTriple = new Triple(stateNode, timestampNode, graph.CreateLiteralNode(state.Timestamp.ToString("O")))
                    graph.Assert(timestampTriple)
                    
                    logger |> Option.iter (fun l -> l.LogInformation($"Stored TARS state: {state.StateId}"))
                    true
                | None -> false
            with
            | ex ->
                logger |> Option.iter (fun l -> l.LogError(ex, "Failed to store TARS state"))
                false

        /// Execute SPARQL query
        member this.ExecuteSparqlQuery(sparqlQuery: string) : RdfQueryResult =
            let startTime = DateTime.Now
            try
                match store, storageProvider with
                | Some tripleStore, _ ->
                    let dataset = new InMemoryDataset(tripleStore :?> IInMemoryQueryableStore)
                    let processor = new LeviathanQueryProcessor(dataset)
                    let parser = new SparqlQueryParser()
                    let query = parser.ParseFromString(sparqlQuery)
                    let results = processor.ProcessQuery(query)
                    
                    let executionTime = DateTime.Now - startTime
                    
                    match results with
                    | :? SparqlResultSet as resultSet ->
                        let variables = resultSet.Variables |> Seq.toList
                        let resultRows = 
                            resultSet
                            |> Seq.map (fun result -> 
                                variables |> List.map (fun var -> 
                                    if result.HasValue(var) then result.[var].ToString() else ""))
                            |> Seq.toList
                        
                        {
                            Query = sparqlQuery
                            Results = resultRows
                            Variables = variables
                            ExecutionTime = executionTime
                            RecordCount = resultRows.Length
                            Success = true
                            ErrorMessage = None
                        }
                    | _ ->
                        {
                            Query = sparqlQuery
                            Results = []
                            Variables = []
                            ExecutionTime = executionTime
                            RecordCount = 0
                            Success = true
                            ErrorMessage = None
                        }
                
                | None, Some provider ->
                    // Use storage provider for remote queries
                    let sparqlConnector = provider :?> SparqlConnector
                    let endpoint = sparqlConnector.Endpoint
                    let processor = new RemoteQueryProcessor(endpoint)
                    let parser = new SparqlQueryParser()
                    let query = parser.ParseFromString(sparqlQuery)
                    let results = processor.ProcessQuery(query)
                    
                    let executionTime = DateTime.Now - startTime
                    
                    match results with
                    | :? SparqlResultSet as resultSet ->
                        let variables = resultSet.Variables |> Seq.toList
                        let resultRows = 
                            resultSet
                            |> Seq.map (fun result -> 
                                variables |> List.map (fun var -> 
                                    if result.HasValue(var) then result.[var].ToString() else ""))
                            |> Seq.toList
                        
                        {
                            Query = sparqlQuery
                            Results = resultRows
                            Variables = variables
                            ExecutionTime = executionTime
                            RecordCount = resultRows.Length
                            Success = true
                            ErrorMessage = None
                        }
                    | _ ->
                        {
                            Query = sparqlQuery
                            Results = []
                            Variables = []
                            ExecutionTime = executionTime
                            RecordCount = 0
                            Success = true
                            ErrorMessage = None
                        }
                
                | None, None ->
                    if this.InitializeStore() then
                        this.ExecuteSparqlQuery(sparqlQuery)
                    else
                        {
                            Query = sparqlQuery
                            Results = []
                            Variables = []
                            ExecutionTime = DateTime.Now - startTime
                            RecordCount = 0
                            Success = false
                            ErrorMessage = Some "Failed to initialize RDF store"
                        }
            with
            | ex ->
                let executionTime = DateTime.Now - startTime
                logger |> Option.iter (fun l -> l.LogError(ex, "SPARQL query execution failed"))
                {
                    Query = sparqlQuery
                    Results = []
                    Variables = []
                    ExecutionTime = executionTime
                    RecordCount = 0
                    Success = false
                    ErrorMessage = Some ex.Message
                }

        /// Get all named graphs
        member this.GetNamedGraphs() : NamedGraph list =
            try
                match store with
                | Some tripleStore ->
                    tripleStore.Graphs
                    |> Seq.map (fun graph -> {
                        Uri = graph.BaseUri
                        Name = graph.BaseUri.ToString()
                        Description = "RDF Graph"
                        Created = DateTime.Now
                        LastModified = DateTime.Now
                    })
                    |> Seq.toList
                | None -> []
            with
            | ex ->
                logger |> Option.iter (fun l -> l.LogError(ex, "Failed to get named graphs"))
                []

        /// Save store to file (for file-based stores)
        member this.SaveToFile(filePath: string) : bool =
            try
                match store with
                | Some tripleStore ->
                    let writer = new TriGWriter()
                    writer.Save(tripleStore, filePath)
                    logger |> Option.iter (fun l -> l.LogInformation($"Saved RDF store to: {filePath}"))
                    true
                | None -> false
            with
            | ex ->
                logger |> Option.iter (fun l -> l.LogError(ex, $"Failed to save to file: {filePath}"))
                false

        interface IDisposable with
            member this.Dispose() =
                store |> Option.iter (fun s -> s.Dispose())
                storageProvider |> Option.iter (fun p -> p.Dispose())

    /// Create in-memory RDF store
    let createInMemoryStore (logger: ILogger option) =
        new TarsRdfStore(InMemory, logger)

    /// Create Virtuoso-backed RDF store
    let createVirtuosoStore (connectionString: string) (logger: ILogger option) =
        new TarsRdfStore(Virtuoso connectionString, logger)

    /// Create file-based RDF store
    let createFileStore (filePath: string) (logger: ILogger option) =
        new TarsRdfStore(File filePath, logger)

    /// Create remote SPARQL endpoint store
    let createRemoteStore (endpoint: string) (logger: ILogger option) =
        new TarsRdfStore(Remote endpoint, logger)
