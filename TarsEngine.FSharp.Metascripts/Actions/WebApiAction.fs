namespace TarsEngine.FSharp.Metascripts.Actions

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.DataSources.Closures
open TarsEngine.FSharp.Metascripts.Core

/// Web API action for creating and running endpoints on-the-fly from metascripts
type WebApiAction(logger: ILogger<WebApiAction>) =
    
    let webApiFactory = WebApiClosureFactory()
    let mutable runningProcesses = Map.empty<string, Process>
    
    interface IMetascriptAction with
        member _.Name = "webapi"
        member _.Description = "Create and run REST/GraphQL endpoints on-the-fly"
        
        member _.Execute(context: MetascriptContext, parameters: Map<string, obj>) =
            async {
                try
                    let actionType = parameters.TryFind("type") |> Option.map (fun x -> x.ToString()) |> Option.defaultValue "create"
                    
                    match actionType.ToLower() with
                    | "create" -> return! this.CreateEndpoint(context, parameters)
                    | "start" -> return! this.StartEndpoint(context, parameters)
                    | "stop" -> return! this.StopEndpoint(context, parameters)
                    | "status" -> return! this.GetEndpointStatus(context, parameters)
                    | "list" -> return! this.ListEndpoints(context, parameters)
                    | _ -> 
                        logger.LogWarning("Unknown webapi action type: {Type}", actionType)
                        return Error $"Unknown webapi action type: {actionType}"
                        
                with
                | ex ->
                    logger.LogError(ex, "Error executing webapi action")
                    return Error ex.Message
            }
    
    /// Creates a new endpoint from metascript configuration
    member _.CreateEndpoint(context: MetascriptContext, parameters: Map<string, obj>) =
        async {
            try
                let endpointType = parameters.TryFind("endpoint_type") |> Option.map (fun x -> x.ToString()) |> Option.defaultValue "REST_ENDPOINT"
                let name = parameters.TryFind("name") |> Option.map (fun x -> x.ToString()) |> Option.defaultValue "TarsEndpoint"
                let port = parameters.TryFind("port") |> Option.map (fun x -> int (x.ToString())) |> Option.defaultValue 5000
                let autoStart = parameters.TryFind("auto_start") |> Option.map (fun x -> bool.Parse(x.ToString())) |> Option.defaultValue true
                
                logger.LogInformation("Creating {Type} endpoint: {Name} on port {Port}", endpointType, name, port)
                
                // Parse endpoint configuration
                let config = this.ParseEndpointConfig(parameters, port)
                
                // Generate the endpoint using closure factory
                let outputDir = Path.Combine("output", "live-endpoints", name.ToLower())
                let closure = webApiFactory.CreateClosure(endpointType, name, config)
                
                let! result = closure outputDir
                
                match result with
                | Ok info ->
                    logger.LogInformation("Endpoint {Name} created successfully at {OutputDir}", name, outputDir)
                    
                    // Auto-start if requested
                    if autoStart then
                        let! startResult = this.StartEndpointProcess(name, outputDir, port)
                        match startResult with
                        | Ok processInfo ->
                            return Ok {|
                                Action = "create"
                                Name = name
                                Type = endpointType
                                Port = port
                                OutputDirectory = outputDir
                                Status = "running"
                                ProcessId = processInfo.ProcessId
                                BaseUrl = $"http://localhost:{port}"
                                SwaggerUrl = $"http://localhost:{port}/swagger"
                                GraphQLUrl = $"http://localhost:{port}/graphql"
                                HealthUrl = $"http://localhost:{port}/health"
                                CreatedAt = DateTime.UtcNow
                            |}
                        | Error error ->
                            return Ok {|
                                Action = "create"
                                Name = name
                                Type = endpointType
                                Port = port
                                OutputDirectory = outputDir
                                Status = "created_not_started"
                                Error = error
                                CreatedAt = DateTime.UtcNow
                            |}
                    else
                        return Ok {|
                            Action = "create"
                            Name = name
                            Type = endpointType
                            Port = port
                            OutputDirectory = outputDir
                            Status = "created"
                            CreatedAt = DateTime.UtcNow
                        |}
                        
                | Error error ->
                    logger.LogError("Failed to create endpoint {Name}: {Error}", name, error)
                    return Error $"Failed to create endpoint {name}: {error}"
                    
            with
            | ex ->
                logger.LogError(ex, "Error creating endpoint")
                return Error ex.Message
        }
    
    /// Starts an existing endpoint
    member _.StartEndpoint(context: MetascriptContext, parameters: Map<string, obj>) =
        async {
            try
                let name = parameters.TryFind("name") |> Option.map (fun x -> x.ToString()) |> Option.defaultValue "TarsEndpoint"
                let port = parameters.TryFind("port") |> Option.map (fun x -> int (x.ToString())) |> Option.defaultValue 5000
                let outputDir = Path.Combine("output", "live-endpoints", name.ToLower())
                
                let! result = this.StartEndpointProcess(name, outputDir, port)
                
                match result with
                | Ok processInfo ->
                    return Ok {|
                        Action = "start"
                        Name = name
                        Port = port
                        Status = "running"
                        ProcessId = processInfo.ProcessId
                        BaseUrl = $"http://localhost:{port}"
                        StartedAt = DateTime.UtcNow
                    |}
                | Error error ->
                    return Error error
                    
            with
            | ex ->
                logger.LogError(ex, "Error starting endpoint")
                return Error ex.Message
        }
    
    /// Stops a running endpoint
    member _.StopEndpoint(context: MetascriptContext, parameters: Map<string, obj>) =
        async {
            try
                let name = parameters.TryFind("name") |> Option.map (fun x -> x.ToString()) |> Option.defaultValue "TarsEndpoint"
                
                match runningProcesses.TryFind(name) with
                | Some process ->
                    if not process.HasExited then
                        process.Kill()
                        process.WaitForExit(5000) |> ignore
                    
                    runningProcesses <- runningProcesses.Remove(name)
                    
                    return Ok {|
                        Action = "stop"
                        Name = name
                        Status = "stopped"
                        StoppedAt = DateTime.UtcNow
                    |}
                | None ->
                    return Error $"No running process found for endpoint: {name}"
                    
            with
            | ex ->
                logger.LogError(ex, "Error stopping endpoint")
                return Error ex.Message
        }
    
    /// Gets status of endpoints
    member _.GetEndpointStatus(context: MetascriptContext, parameters: Map<string, obj>) =
        async {
            try
                let name = parameters.TryFind("name") |> Option.map (fun x -> x.ToString())
                
                match name with
                | Some endpointName ->
                    // Get status for specific endpoint
                    match runningProcesses.TryFind(endpointName) with
                    | Some process ->
                        let status = if process.HasExited then "stopped" else "running"
                        return Ok {|
                            Action = "status"
                            Name = endpointName
                            Status = status
                            ProcessId = if not process.HasExited then Some process.Id else None
                            CheckedAt = DateTime.UtcNow
                        |}
                    | None ->
                        return Ok {|
                            Action = "status"
                            Name = endpointName
                            Status = "not_running"
                            CheckedAt = DateTime.UtcNow
                        |}
                | None ->
                    // Get status for all endpoints
                    let statuses = 
                        runningProcesses
                        |> Map.map (fun name process ->
                            {|
                                Name = name
                                Status = if process.HasExited then "stopped" else "running"
                                ProcessId = if not process.HasExited then Some process.Id else None
                            |}
                        )
                        |> Map.values
                        |> Seq.toList
                    
                    return Ok {|
                        Action = "status"
                        Endpoints = statuses
                        CheckedAt = DateTime.UtcNow
                    |}
                    
            with
            | ex ->
                logger.LogError(ex, "Error getting endpoint status")
                return Error ex.Message
        }
    
    /// Lists all available endpoints
    member _.ListEndpoints(context: MetascriptContext, parameters: Map<string, obj>) =
        async {
            try
                let endpointsDir = "output/live-endpoints"
                
                let endpoints = 
                    if Directory.Exists(endpointsDir) then
                        Directory.GetDirectories(endpointsDir)
                        |> Array.map (fun dir ->
                            let name = Path.GetFileName(dir)
                            let isRunning = runningProcesses.ContainsKey(name)
                            {|
                                Name = name
                                Directory = dir
                                Status = if isRunning then "running" else "stopped"
                                HasProject = File.Exists(Path.Combine(dir, name + ".fsproj"))
                            |}
                        )
                        |> Array.toList
                    else
                        []
                
                return Ok {|
                    Action = "list"
                    Endpoints = endpoints
                    Count = endpoints.Length
                    ListedAt = DateTime.UtcNow
                |}
                
            with
            | ex ->
                logger.LogError(ex, "Error listing endpoints")
                return Error ex.Message
        }
    
    /// Starts an endpoint process
    member _.StartEndpointProcess(name: string, outputDir: string, port: int) =
        async {
            try
                if not (Directory.Exists(outputDir)) then
                    return Error $"Endpoint directory not found: {outputDir}"
                
                // Find the project file
                let projectFiles = Directory.GetFiles(outputDir, "*.fsproj")
                if projectFiles.Length = 0 then
                    return Error $"No .fsproj file found in {outputDir}"
                
                let projectFile = projectFiles.[0]
                
                // Start the dotnet run process
                let startInfo = ProcessStartInfo()
                startInfo.FileName <- "dotnet"
                startInfo.Arguments <- $"run --project \"{projectFile}\" --urls http://localhost:{port}"
                startInfo.WorkingDirectory <- outputDir
                startInfo.UseShellExecute <- false
                startInfo.RedirectStandardOutput <- true
                startInfo.RedirectStandardError <- true
                startInfo.CreateNoWindow <- true
                
                let process = Process.Start(startInfo)
                
                // Store the process
                runningProcesses <- runningProcesses.Add(name, process)
                
                // Give it a moment to start
                do! Async.Sleep(2000)
                
                if process.HasExited then
                    let error = process.StandardError.ReadToEnd()
                    return Error $"Process exited immediately: {error}"
                else
                    logger.LogInformation("Endpoint {Name} started on port {Port} with PID {ProcessId}", name, port, process.Id)
                    return Ok {|
                        ProcessId = process.Id
                        Port = port
                        BaseUrl = $"http://localhost:{port}"
                    |}
                    
            with
            | ex ->
                logger.LogError(ex, "Error starting endpoint process")
                return Error ex.Message
        }
    
    /// Parses endpoint configuration from metascript parameters
    member _.ParseEndpointConfig(parameters: Map<string, obj>, port: int) =
        let mutable config = Map.empty<string, obj>
        
        // Basic configuration
        config <- config.Add("base_url", box $"http://localhost:{port}")
        
        // Parse endpoints if provided
        if parameters.ContainsKey("endpoints") then
            config <- config.Add("endpoints", parameters.["endpoints"])
        
        // Parse GraphQL schema if provided
        if parameters.ContainsKey("graphql") then
            config <- config.Add("graphql", parameters.["graphql"])
        
        // Parse authentication if provided
        if parameters.ContainsKey("auth") then
            config <- config.Add("auth", parameters.["auth"])
        
        // Parse CORS if provided
        if parameters.ContainsKey("cors") then
            config <- config.Add("cors", parameters.["cors"])
        
        // Default values
        if not (config.ContainsKey("title")) then
            config <- config.Add("title", box "TARS Live Endpoint")
        
        if not (config.ContainsKey("description")) then
            config <- config.Add("description", box "Live endpoint created by TARS metascript")
        
        if not (config.ContainsKey("version")) then
            config <- config.Add("version", box "1.0.0")
        
        config
    
    /// Cleanup method to stop all running processes
    member _.StopAllEndpoints() =
        for kvp in runningProcesses do
            try
                if not kvp.Value.HasExited then
                    kvp.Value.Kill()
                    kvp.Value.WaitForExit(5000) |> ignore
            with
            | ex -> logger.LogWarning(ex, "Error stopping process for endpoint {Name}", kvp.Key)
        
        runningProcesses <- Map.empty
    
    interface IDisposable with
        member this.Dispose() = this.StopAllEndpoints()
