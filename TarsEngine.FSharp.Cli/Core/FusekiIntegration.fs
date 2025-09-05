namespace TarsEngine.FSharp.Cli.Core

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks
open System.Text.Json
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core.DataFetchingEngine

/// TARS Apache Jena Fuseki Integration for local triple store management
module FusekiIntegration =
    
    /// Fuseki server configuration
    type FusekiConfig = {
        Port: int
        DataDirectory: string
        ServerJar: string
        JavaPath: string
        MaxMemory: string
        Datasets: string list
        EnableUI: bool
        EnableCORS: bool
        LogLevel: string
    }

    /// Fuseki server status
    type FusekiStatus =
        | NotInstalled
        | Stopped
        | Starting
        | Running of port: int
        | Error of message: string

    /// Dataset configuration for Fuseki
    type DatasetConfig = {
        Name: string
        Type: string // "tdb2", "mem", "file"
        Location: string option
        Unionable: bool
        ReadOnly: bool
    }

    /// RDF data format
    type RdfFormat =
        | Turtle
        | NTriples
        | RdfXml
        | JsonLd
        | NQuads

    /// Fuseki management operations
    type FusekiOperation =
        | Install
        | Start
        | Stop
        | Restart
        | Status
        | CreateDataset of DatasetConfig
        | DeleteDataset of string
        | LoadData of dataset: string * file: string * format: RdfFormat
        | BackupDataset of dataset: string * backupPath: string
        | RestoreDataset of dataset: string * backupPath: string

    /// Default Fuseki configuration for TARS
    let defaultConfig = {
        Port = 3030
        DataDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "fuseki")
        ServerJar = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "fuseki", "fuseki-server.jar")
        JavaPath = "java"
        MaxMemory = "2G"
        Datasets = ["tars-knowledge"; "tars-codebase"; "tars-cache"]
        EnableUI = true
        EnableCORS = true
        LogLevel = "INFO"
    }

    /// Fuseki server manager
    type FusekiManager(config: FusekiConfig, logger: ILogger option) =
        let mutable serverProcess: Process option = None
        let mutable currentStatus = NotInstalled

        /// Check if Java is available
        member private this.CheckJavaAvailability() : Task<bool> =
            task {
                try
                    let startInfo = ProcessStartInfo()
                    startInfo.FileName <- config.JavaPath
                    startInfo.Arguments <- "-version"
                    startInfo.UseShellExecute <- false
                    startInfo.RedirectStandardOutput <- true
                    startInfo.RedirectStandardError <- true
                    startInfo.CreateNoWindow <- true

                    use proc = Process.Start(startInfo)
                    let! _ = proc.WaitForExitAsync()
                    return proc.ExitCode = 0
                with
                | _ -> return false
            }

        /// Download Fuseki server if not present
        member private this.DownloadFuseki() : Task<bool> =
            task {
                try
                    let fusekiDir = Path.GetDirectoryName(config.ServerJar)
                    if not (Directory.Exists(fusekiDir)) then
                        Directory.CreateDirectory(fusekiDir) |> ignore

                    if not (File.Exists(config.ServerJar)) then
                        logger |> Option.iter (fun l -> l.LogInformation("Downloading Apache Jena Fuseki..."))
                        
                        // Note: In production, this would download from Apache mirrors
                        // For now, we'll create a placeholder that indicates manual installation needed
                        let readmeContent = """
# TARS Fuseki Integration

To complete the Fuseki integration, please:

1. Download Apache Jena Fuseki from: https://jena.apache.org/download/
2. Extract the fuseki-server.jar to this directory
3. Restart TARS

The jar file should be named: fuseki-server.jar
"""
                        File.WriteAllText(Path.Combine(fusekiDir, "README.md"), readmeContent)
                        return false
                    else
                        return true
                with
                | ex ->
                    logger |> Option.iter (fun l -> l.LogError(ex, "Failed to setup Fuseki"))
                    return false
            }

        /// Start Fuseki server
        member this.StartServer() : Task<FusekiStatus> =
            task {
                try
                    let! javaAvailable = this.CheckJavaAvailability()
                    if not javaAvailable then
                        currentStatus <- Error "Java not found. Please install Java 11 or later."
                        return currentStatus
                    else
                        let! fusekiReady = this.DownloadFuseki()
                        if not fusekiReady then
                            currentStatus <- Error "Fuseki server jar not found. Please download manually."
                            return currentStatus
                        else
                            if not (Directory.Exists(config.DataDirectory)) then
                                Directory.CreateDirectory(config.DataDirectory) |> ignore

                            currentStatus <- Starting
                            logger |> Option.iter (fun l -> l.LogInformation($"Starting Fuseki server on port {config.Port}..."))

                            let startInfo = ProcessStartInfo()
                            startInfo.FileName <- config.JavaPath
                            startInfo.Arguments <- $"-Xmx{config.MaxMemory} -jar \"{config.ServerJar}\" --port={config.Port} --loc=\"{config.DataDirectory}\""

                            if config.EnableCORS then
                                startInfo.Arguments <- startInfo.Arguments + " --cors"

                            startInfo.UseShellExecute <- false
                            startInfo.RedirectStandardOutput <- true
                            startInfo.RedirectStandardError <- true
                            startInfo.CreateNoWindow <- true

                            let proc = Process.Start(startInfo)
                            serverProcess <- Some proc

                            // Wait a bit for server to start
                            do! Task.Delay(3000)

                            if proc.HasExited then
                                currentStatus <- Error $"Fuseki failed to start. Exit code: {proc.ExitCode}"
                            else
                                currentStatus <- Running config.Port
                                logger |> Option.iter (fun l -> l.LogInformation($"Fuseki server started successfully on port {config.Port}"))

                            return currentStatus
                with
                | ex ->
                    currentStatus <- Error ex.Message
                    logger |> Option.iter (fun l -> l.LogError(ex, "Failed to start Fuseki server"))
                    return currentStatus
            }

        /// Stop Fuseki server
        member this.StopServer() : Task<FusekiStatus> =
            task {
                try
                    match serverProcess with
                    | Some proc when not proc.HasExited ->
                        logger |> Option.iter (fun l -> l.LogInformation("Stopping Fuseki server..."))
                        proc.Kill()
                        let! _ = proc.WaitForExitAsync()
                        serverProcess <- None
                        currentStatus <- Stopped
                        logger |> Option.iter (fun l -> l.LogInformation("Fuseki server stopped"))
                    | _ ->
                        currentStatus <- Stopped

                    return currentStatus
                with
                | ex ->
                    currentStatus <- Error ex.Message
                    logger |> Option.iter (fun l -> l.LogError(ex, "Failed to stop Fuseki server"))
                    return currentStatus
            }

        /// Get current server status
        member this.GetStatus() : Task<FusekiStatus> =
            task {
                try
                    match serverProcess with
                    | Some proc when not proc.HasExited ->
                        // Verify server is actually responding
                        let httpClient = new System.Net.Http.HttpClient()
                        let! response = httpClient.GetAsync($"http://localhost:{config.Port}/$/ping")
                        if response.IsSuccessStatusCode then
                            currentStatus <- Running config.Port
                        else
                            currentStatus <- Error "Server not responding"
                    | _ ->
                        if File.Exists(config.ServerJar) then
                            currentStatus <- Stopped
                        else
                            currentStatus <- NotInstalled

                    return currentStatus
                with
                | _ ->
                    currentStatus <- Error "Status check failed"
                    return currentStatus
            }

        /// Create a new dataset
        member this.CreateDataset(dataset: DatasetConfig) : Task<bool> =
            task {
                try
                    let httpClient = new System.Net.Http.HttpClient()
                    let createUrl = $"http://localhost:{config.Port}/$/datasets"
                    
                    let datasetJson = JsonSerializer.Serialize({|
                        dbName = dataset.Name
                        dbType = dataset.Type
                    |})
                    
                    let content = new System.Net.Http.StringContent(datasetJson, System.Text.Encoding.UTF8, "application/json")
                    let! response = httpClient.PostAsync(createUrl, content)
                    
                    if response.IsSuccessStatusCode then
                        logger |> Option.iter (fun l -> l.LogInformation($"Created dataset: {dataset.Name}"))
                        return true
                    else
                        logger |> Option.iter (fun l -> l.LogWarning($"Failed to create dataset {dataset.Name}: {response.StatusCode}"))
                        return false
                with
                | ex ->
                    logger |> Option.iter (fun l -> l.LogError(ex, $"Error creating dataset {dataset.Name}"))
                    return false
            }

        /// Load RDF data into a dataset
        member this.LoadData(datasetName: string, rdfData: string, format: RdfFormat) : Task<bool> =
            task {
                try
                    let httpClient = new System.Net.Http.HttpClient()
                    let uploadUrl = $"http://localhost:{config.Port}/{datasetName}/data"
                    
                    let contentType = 
                        match format with
                        | Turtle -> "text/turtle"
                        | NTriples -> "application/n-triples"
                        | RdfXml -> "application/rdf+xml"
                        | JsonLd -> "application/ld+json"
                        | NQuads -> "application/n-quads"
                    
                    let content = new System.Net.Http.StringContent(rdfData, System.Text.Encoding.UTF8, contentType)
                    let! response = httpClient.PostAsync(uploadUrl, content)
                    
                    if response.IsSuccessStatusCode then
                        logger |> Option.iter (fun l -> l.LogInformation($"Loaded data into dataset: {datasetName}"))
                        return true
                    else
                        logger |> Option.iter (fun l -> l.LogWarning($"Failed to load data into {datasetName}: {response.StatusCode}"))
                        return false
                with
                | ex ->
                    logger |> Option.iter (fun l -> l.LogError(ex, $"Error loading data into {datasetName}"))
                    return false
            }

        /// Execute SPARQL query on local Fuseki
        member this.ExecuteQuery(datasetName: string, sparqlQuery: string) : Task<DataFetchResult> =
            task {
                let startTime = DateTime.Now
                try
                    let httpClient = new System.Net.Http.HttpClient()
                    let queryUrl = $"http://localhost:{config.Port}/{datasetName}/sparql"
                    
                    let encodedQuery = Uri.EscapeDataString(sparqlQuery)
                    let requestUrl = $"{queryUrl}?query={encodedQuery}&format=application/sparql-results+json"
                    
                    let! response = httpClient.GetAsync(requestUrl)
                    let! content = response.Content.ReadAsStringAsync()
                    
                    let executionTime = DateTime.Now - startTime
                    
                    if response.IsSuccessStatusCode then
                        return {
                            Source = TripleStore $"http://localhost:{config.Port}"
                            Query = SparqlQuery sparqlQuery
                            Data = content
                            Metadata = Map [
                                ("dataset", datasetName)
                                ("contentType",
                                    match response.Content.Headers.ContentType with
                                    | null -> "unknown"
                                    | ct -> ct.ToString())
                            ]
                            FetchTime = startTime
                            ExecutionTime = executionTime
                            RecordCount = None // Could parse JSON to count
                            Success = true
                            ErrorMessage = None
                        }
                    else
                        return {
                            Source = TripleStore $"http://localhost:{config.Port}"
                            Query = SparqlQuery sparqlQuery
                            Data = content
                            Metadata = Map [("dataset", datasetName)]
                            FetchTime = startTime
                            ExecutionTime = executionTime
                            RecordCount = None
                            Success = false
                            ErrorMessage = Some $"HTTP {response.StatusCode}: {content}"
                        }
                with
                | ex ->
                    let executionTime = DateTime.Now - startTime
                    return {
                        Source = TripleStore $"http://localhost:{config.Port}"
                        Query = SparqlQuery sparqlQuery
                        Data = ""
                        Metadata = Map [("dataset", datasetName)]
                        FetchTime = startTime
                        ExecutionTime = executionTime
                        RecordCount = None
                        Success = false
                        ErrorMessage = Some ex.Message
                    }
            }

        interface IDisposable with
            member this.Dispose() =
                this.StopServer() |> ignore

    /// Create Fuseki manager with default configuration
    let createFusekiManager (logger: ILogger option) =
        new FusekiManager(defaultConfig, logger)

    /// Create Fuseki manager with custom configuration
    let createFusekiManagerWithConfig (config: FusekiConfig) (logger: ILogger option) =
        new FusekiManager(config, logger)
