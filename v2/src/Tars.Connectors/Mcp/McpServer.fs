namespace Tars.Connectors.Mcp

open System
open System.Text.Json
open System.Threading.Tasks
open System.Collections.Concurrent
open Tars.Core

/// Enhanced MCP Server with progress notifications, subagent support, and knowledge graph
type McpServer(registry: IToolRegistry, ?knowledgeGraph: TemporalKnowledgeGraph.TemporalGraph) =
    let instanceId = Guid.NewGuid().ToString().Substring(0, 8)
    let startTime = DateTime.UtcNow
    
    // Attempt to get git commit hash or use a placeholder
    let gitCommit = 
        match Environment.GetEnvironmentVariable("GIT_COMMIT") with
        | null -> "8bddcb89" // Updated from git rev-parse HEAD
        | s -> s.Substring(0, 8)

    let serializerOptions =
        JsonSerializerOptions(WriteIndented = false, PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

    let log (msg: string) =
        Console.Error.WriteLine($"[TARS MCP] {msg}")

    // Active tasks with progress reporters
    let activeTasks = ConcurrentDictionary<Guid, ProgressReporter>()

    // Subagent manager (initialized lazily when first used)
    let mutable subagentManager: SubagentManager option = None

    let writeResponse (response: JsonRpcResponse) =
        task {
            let json = JsonSerializer.Serialize(response, serializerOptions)
            Console.Out.WriteLine(json)
            do! Console.Out.FlushAsync()
        }

    /// Send a JSON-RPC notification (no id, no response expected)
    let writeNotification (method: string) (params': obj) =
        task {
            let notification =
                {| jsonrpc = "2.0"
                   method = method
                   ``params`` = params' |}

            let json = JsonSerializer.Serialize(notification, serializerOptions)
            Console.Out.WriteLine(json)
            do! Console.Out.FlushAsync()
        }

    let createError (id: int) (code: int) (msg: string) =
        { JsonRpc = "2.0"
          Result = None
          Error =
            Some
                { Code = code
                  Message = msg
                  Data = None }
          Id = id }

    let createSuccessResponse (id: int) (result: obj) =
        { JsonRpc = "2.0"
          Result = Some(JsonSerializer.SerializeToElement(result, serializerOptions))
          Error = None
          Id = id }

    // Progress observer that sends notifications to IDE
    let createProgressObserver () =
        { new IProgressObserver with
            member _.OnProgress(update) =
                writeNotification "progress/update" update
                |> Async.AwaitTask
                |> Async.RunSynchronously }

    let handleInitialize (id: int) =
        let result =
            { ProtocolVersion = "2024-11-05"
              Capabilities =
                { Logging = Some(obj ())
                  Prompts = None
                  Resources = Some(obj ()) // Enable resources for knowledge graph
                  Tools = Some(obj ()) }
              ServerInfo = { Name = "TARS"; Version = "2.0.0" } }

        createSuccessResponse id result

    let handleListTools (id: int) =
        let tools = registry.GetAll()

        let mcpTools =
            tools
            |> List.map (fun t ->
                let schema =
                    {| ``type`` = "object"
                       properties =
                        {| arguments =
                            {| ``type`` = "string"
                               description = "Arguments for the tool (JSON string or plain text)" |} |} |}

                { Name = t.Name
                  Description = Some t.Description
                  InputSchema = JsonSerializer.SerializeToElement(schema, serializerOptions) })

        let result = { Tools = mcpTools; NextCursor = None }
        createSuccessResponse id result

    let handleCallTool (id: int) (params': JsonElement) =
        task {
            try
                let nameProp = params'.GetProperty("name").GetString()
                let mutable argsElement = Unchecked.defaultof<JsonElement>

                let argsProp =
                    if params'.TryGetProperty("arguments", &argsElement) then
                        Some argsElement
                    else
                        None

                match registry.Get(nameProp) with
                | Some tool ->
                    let input =
                        match argsProp with
                        | Some args ->
                            if args.ValueKind = JsonValueKind.String then
                                args.GetString()
                            else
                                args.GetRawText()
                        | None -> ""

                    let correlationId = Guid.NewGuid().ToString().Substring(0, 8)
                    let meta = 
                        Map [ ("instance_id", instanceId :> obj)
                              ("correlation_id", correlationId :> obj) ]
                        |> Some

                    log $"Executing tool '{nameProp}' [CID: {correlationId}] with input: {input}"
                    let! result = Async.StartAsTask(tool.Execute(input))

                    match result with
                    | Result.Ok output ->
                        let content =
                            [ { Type = "text"
                                Text = Some output
                                Data = None
                                MimeType = None
                                Resource = None } ]

                        let callResult =
                            { Content = content
                              IsError = Some false
                              Meta = meta }

                        return createSuccessResponse id callResult

                    | Result.Error err ->
                        let content =
                            [ { Type = "text"
                                Text = Some err
                                Data = None
                                MimeType = None
                                Resource = None } ]

                        let callResult =
                            { Content = content
                              IsError = Some true
                              Meta = meta }

                        return createSuccessResponse id callResult

                | None -> return createError id -32601 $"Tool not found: {nameProp}"
            with ex ->
                log $"Error calling tool: {ex.Message}"
                return createError id -32603 $"Internal error: {ex.Message}"
        }

    // ========== TASK MANAGEMENT ==========

    let handleTasksCreate (id: int) (params': JsonElement) =
        let goal = params'.GetProperty("goal").GetString()
        let taskId = Guid.NewGuid()
        let reporter = ProgressReporter(taskId, goal)
        reporter.AddObserver(createProgressObserver ())
        activeTasks.[taskId] <- reporter

        reporter.SetMode(TaskMode.Planning)
        reporter.SetStatus("Task created, awaiting execution")

        log $"Created task {taskId}: {goal}"
        createSuccessResponse id {| taskId = taskId.ToString() |}

    let handleTasksList (id: int) =
        let tasks =
            activeTasks
            |> Seq.map (fun kv ->
                let update = kv.Value.GetCurrentUpdate()

                {| taskId = update.TaskId.ToString()
                   taskName = update.TaskName
                   mode = update.Mode
                   status = update.Status
                   progress = update.Progress |})
            |> Seq.toList

        createSuccessResponse id {| tasks = tasks |}

    let handleTasksCancel (id: int) (params': JsonElement) =
        let taskIdStr = params'.GetProperty("taskId").GetString()

        match Guid.TryParse(taskIdStr) with
        | true, taskId ->
            match activeTasks.TryRemove(taskId) with
            | true, _ ->
                log $"Cancelled task {taskId}"
                createSuccessResponse id {| cancelled = true |}
            | false, _ -> createError id -32602 $"Task not found: {taskId}"
        | false, _ -> createError id -32602 $"Invalid task ID: {taskIdStr}"

    // ========== SUBAGENT MANAGEMENT ==========

    let getSubagentManager () =
        match subagentManager with
        | Some mgr -> mgr
        | None ->
            let mgr =
                SubagentManager(fun req ct obs ->
                    task {
                        // Placeholder implementation - will be connected to Evolution engine
                        do! Task.Delay(1000, ct)

                        return
                            { Id = Guid.NewGuid()
                              Success = true
                              Output = $"Completed research on: {req.Goal}"
                              Artifacts = []
                              Duration = TimeSpan.FromSeconds(1.0)
                              Error = None }
                    })

            subagentManager <- Some mgr
            mgr

    let handleSubagentsSpawn (id: int) (params': JsonElement) =
        let goal = params'.GetProperty("goal").GetString()
        let mutable maxDuration = 30
        let mutable durationElem = Unchecked.defaultof<JsonElement>

        if params'.TryGetProperty("maxDurationMinutes", &durationElem) then
            maxDuration <- durationElem.GetInt32()

        let request =
            { Goal = goal
              MaxDurationMinutes = maxDuration
              AllowTools = None
              ParentTaskId = None
              AgentHint = Some "research" }

        let mgr = getSubagentManager ()
        let subagentId = mgr.Spawn(request, createProgressObserver ())

        log $"Spawned subagent {subagentId}: {goal}"
        createSuccessResponse id {| subagentId = subagentId.ToString() |}

    let handleSubagentsList (id: int) =
        let mgr = getSubagentManager ()

        let subagents =
            mgr.ListActive()
            |> List.map (fun s ->
                {| id = s.Id.ToString()
                   name = s.Name
                   goal = s.Goal
                   status = s.Status
                   progress = s.Progress |})

        createSuccessResponse id {| subagents = subagents |}

    let handleSubagentsCancel (id: int) (params': JsonElement) =
        let subagentIdStr = params'.GetProperty("subagentId").GetString()

        match Guid.TryParse(subagentIdStr) with
        | true, subagentId ->
            let mgr = getSubagentManager ()
            let cancelled = mgr.Cancel(subagentId)
            createSuccessResponse id {| cancelled = cancelled |}
        | false, _ -> createError id -32602 $"Invalid subagent ID: {subagentIdStr}"

    // ========== KNOWLEDGE GRAPH ==========

    let handleKnowledgeEntities (id: int) =
        match knowledgeGraph with
        | Some kg ->
            let entities =
                kg.GetNodes()
                |> List.map (fun e ->
                    let entityId = TarsEntity.getId e

                    let (entityType, name) =
                        match e with
                        | TarsEntity.ConceptE c -> ("concept", c.Name)
                        | TarsEntity.AgentBeliefE b -> ("belief", b.Statement.Substring(0, min 50 b.Statement.Length))
                        | TarsEntity.CodePatternE p -> ("pattern", p.Name)
                        | TarsEntity.CodeModuleE m -> ("module", m.Path)
                        | TarsEntity.GrammarRuleE g -> ("grammar", g.Name)
                        | TarsEntity.AnomalyE a -> ("anomaly", a.Location)
                        | TarsEntity.EpisodeE e -> ("episode", TarsEntity.getId (TarsEntity.EpisodeE e))
                        | TarsEntity.FileE pf -> ("file", pf)
                        | TarsEntity.FunctionE f -> ("function", f)

                    {| id = entityId
                       entityType = entityType
                       name = name |})

            createSuccessResponse id {| entities = entities |}
        | None -> createError id -32603 "Knowledge graph not available"

    let handleKnowledgeSubgraph (id: int) (params': JsonElement) =
        match knowledgeGraph with
        | Some kg ->
            let entityId = params'.GetProperty("entityId").GetString()
            let mutable depthElem = Unchecked.defaultof<JsonElement>

            let depth =
                if params'.TryGetProperty("depth", &depthElem) then
                    depthElem.GetInt32()
                else
                    2

            // Get current snapshot of facts
            let facts = kg.GetSnapshot(DateTime.UtcNow)

            // Filter to facts involving the entity (simplified)
            let relatedFacts =
                facts
                |> List.filter (fun f ->
                    let sourceId = TarsEntity.getId (TarsFact.source f)
                    let targetId = TarsFact.target f |> Option.map TarsEntity.getId
                    sourceId = entityId || targetId = Some entityId)
                |> List.map (fun f ->
                    let sourceId = TarsEntity.getId (TarsFact.source f)

                    let targetId =
                        TarsFact.target f |> Option.map TarsEntity.getId |> Option.defaultValue ""

                    let factType =
                        match f with
                        | TarsFact.Implements _ -> "implements"
                        | TarsFact.DependsOn _ -> "depends_on"
                        | TarsFact.Contradicts _ -> "contradicts"
                        | TarsFact.EvolvedFrom _ -> "evolved_from"
                        | TarsFact.BelongsTo _ -> "belongs_to"
                        | TarsFact.SimilarTo _ -> "similar_to"
                        | TarsFact.DerivedFrom _ -> "derived_from"
                        | TarsFact.Contains _ -> "contains"

                    {| source = sourceId
                       target = targetId
                       factType = factType |})

            createSuccessResponse
                id
                {| facts = relatedFacts
                   depth = depth |}
        | None -> createError id -32603 "Knowledge graph not available"

    let handleKnowledgeSearch (id: int) (params': JsonElement) =
        match knowledgeGraph with
        | Some kg ->
            let query = params'.GetProperty("query").GetString().ToLowerInvariant()

            let entities =
                kg.GetNodes()
                |> List.choose (fun e ->
                    let entityId = TarsEntity.getId e

                    let name =
                        match e with
                        | TarsEntity.ConceptE c -> c.Name
                        | TarsEntity.AgentBeliefE b -> b.Statement
                        | TarsEntity.CodePatternE p -> p.Name
                        | TarsEntity.CodeModuleE m -> m.Path
                        | TarsEntity.GrammarRuleE g -> g.Name
                        | TarsEntity.AnomalyE a -> a.Location
                        | TarsEntity.EpisodeE e -> "Episode " + TarsEntity.getId (TarsEntity.EpisodeE e)
                        | TarsEntity.FileE pf -> pf
                        | TarsEntity.FunctionE f -> f

                    if name.ToLowerInvariant().Contains(query) then
                        Some {| id = entityId; name = name |}
                    else
                        None)
                |> List.truncate 20

            createSuccessResponse id {| results = entities |}
        | None -> createError id -32603 "Knowledge graph not available"

    member _.InstanceId = instanceId
    member _.StartTime = startTime

    member _.GetInfo() =
        let tools = registry.GetAll()
        let uptime = DateTime.UtcNow - startTime
        
        {| instance_id = instanceId
           git_commit = gitCommit
           startup_time = startTime
           uptime = uptime.ToString("d\\.hh\\:mm\\:ss")
           tool_count = tools.Length
           graphiti_status = if knowledgeGraph.IsSome then "connected" else "disconnected"
           degraded_mode_enabled = true // Falback mechanism is implemented in memory search
        |}

    member this.HandleRequest(line: string) =
        task {
            if String.IsNullOrWhiteSpace(line) then
                return None
            else
                let request = JsonSerializer.Deserialize<JsonRpcRequest>(line, serializerOptions)

                match request.Id with
                | Some id ->
                    let! resp =
                        task {
                            match request.Method with
                            | "initialize" -> return handleInitialize id
                            | "tools/list" -> return handleListTools id
                            | "tools/call" ->
                                match request.Params with
                                | Some p -> return! handleCallTool id p
                                | None -> return createError id -32602 "Missing params"
                            | "ping" -> return (createSuccessResponse id {| pong = true |})
                            | "tasks/create" ->
                                match request.Params with
                                | Some p -> return (handleTasksCreate id p)
                                | None -> return (createError id -32602 "Missing params")
                            | "tasks/list" -> return (handleTasksList id)
                            | "tasks/cancel" ->
                                match request.Params with
                                | Some p -> return (handleTasksCancel id p)
                                | None -> return (createError id -32602 "Missing params")
                            | "subagents/spawn" ->
                                match request.Params with
                                | Some p -> return (handleSubagentsSpawn id p)
                                | None -> return (createError id -32602 "Missing params")
                            | "subagents/list" -> return (handleSubagentsList id)
                            | "subagents/cancel" ->
                                match request.Params with
                                | Some p -> return (handleSubagentsCancel id p)
                                | None -> return (createError id -32602 "Missing params")
                            | "knowledge/entities" -> return (handleKnowledgeEntities id)
                            | "knowledge/subgraph" ->
                                match request.Params with
                                | Some p -> return (handleKnowledgeSubgraph id p)
                                | None -> return (createError id -32602 "Missing params")
                            | "knowledge/search" ->
                                match request.Params with
                                | Some p -> return (handleKnowledgeSearch id p)
                                | None -> return (createError id -32602 "Missing params")
                            | _ -> return (createError id -32601 $"Method not found: {request.Method}")
                        }

                    return Some(JsonSerializer.Serialize(resp, serializerOptions))

                | None ->
                    match request.Method with
                    | "notifications/initialized" -> log "Client initialized."
                    | _ -> log $"Received notification: {request.Method}"

                    return None
        }

    member this.RunAsync() =
        task {
            log "TARS MCP Server Interface Started (Agentic AI Mode)"
            let mutable running = true

            while running do
                try
                    let! line = Console.In.ReadLineAsync()

                    if isNull line then
                        running <- false
                    else
                        let! response = this.HandleRequest(line)

                        match response with
                        | Some resp ->
                            Console.Out.WriteLine(resp)
                            do! Console.Out.FlushAsync()
                        | None -> ()
                with ex ->
                    log $"Error handling request: {ex.Message}"
        }

    /// Send a progress update notification to the client
    member this.SendProgress(update: TaskProgressUpdate) =
        writeNotification "progress/update" update
        |> Async.AwaitTask
        |> Async.RunSynchronously

    /// Get a progress reporter for a task
    member this.GetReporter(taskId: Guid) =
        match activeTasks.TryGetValue(taskId) with
        | true, reporter -> Some reporter
        | false, _ -> None
