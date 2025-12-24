module Tars.Interface.Ui.Main

open System
open Bolero
open Bolero.Html
open Microsoft.AspNetCore.Components
open Elmish
open System.Threading.Tasks
open Tars.Core
open Tars.Kernel
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Llm.Routing
open Tars.Cortex
open Tars.Cortex.Patterns

// ============================================================================
// Model
// ============================================================================

type Page =
    | Dashboard
    | Chat
    | Agents
    | Knowledge
    | Evolution
    | Tools
    | Reasoning // GoT/WoT Demo


type ChatMessage =
    { MsgId: Guid
      Role: string // "user" or "assistant"
      Content: string
      Timestamp: DateTime
      ToolCall: string option
      WoTMetadata: Map<string, string> option }

type Message =
    | NavigateTo of Page
    | EventReceived of SemanticMessage<obj>
    | ClearEvents
    | SetChatInput of string
    | SendChatMessage
    | ReceiveChatResponse of string
    | LoadAgents
    | AgentsLoaded of Agent list
    | LoadKnowledge
    | KnowledgeLoaded of TarsFact list
    | ScanCodebase
    | ScanCodebaseComplete of int
    | StartEvolution
    | EvolutionProgress of int * int * string
    | EvolutionComplete of int * int
    | LoadTools
    | ToolsLoaded of Tool list
    | TestTool of string
    | TestToolResult of string * string
    | SetKnowledgeFilter of string
    // GoT/WoT Messages
    | SetGoTInput of string
    | StartGoT
    | GoTProgress of string
    | GoTComplete of string
    | SelectPattern of AgenticWorkflowPattern
    | RefreshCognitiveState
    | CognitiveStateUpdated of CognitiveState
    | Error of exn

/// GoT reasoning state
type GoTState =
    { Input: string
      IsRunning: bool
      Progress: string list
      Result: string option
      NodesExplored: int
      Pattern: AgenticWorkflowPattern
      SelectedBranchingFactor: int
      SelectedMaxDepth: int }


type EvolutionState =
    { IsRunning: bool
      Generation: int
      TasksCompleted: int
      CurrentTask: string option
      Log: string list }

type Model =
    { CurrentPage: Page
      Events: SemanticMessage<obj> list
      ChatMessages: ChatMessage list
      ChatInput: string
      IsLoading: bool
      IsScanning: bool
      ScanResult: string option
      Agents: Agent list
      Evolution: EvolutionState
      GoT: GoTState // GoT/WoT Demo State
      Tools: Tool list
      ToolTestResults: Map<string, string>
      CognitiveState: CognitiveState option
      Knowledge: TarsFact list
      KnowledgeFilter: string
      Error: string option }

let initModel =
    { CurrentPage = Chat // Start on chat page
      Events = []
      ChatMessages =
        [ { MsgId = Guid.NewGuid()
            Role = "assistant"
            Content = "Hello! I'm TARS, your autonomous reasoning assistant. How can I help you today?"
            Timestamp = DateTime.Now
            ToolCall = None
            WoTMetadata = None } ]
      ChatInput = ""
      IsLoading = false
      IsScanning = false
      ScanResult = None
      Agents = []
      Knowledge = []
      Evolution =
        { IsRunning = false
          Generation = 0
          TasksCompleted = 0
          CurrentTask = None
          Log = [] }
      GoT =
        { Input = "What is 2+2? Show your work step by step."
          IsRunning = false
          Progress = []
          Result = None
          NodesExplored = 0
          Pattern = Reflection
          SelectedBranchingFactor = 3
          SelectedMaxDepth = 3 }
      Tools = []
      ToolTestResults = Map.empty
      CognitiveState = None
      KnowledgeFilter = ""
      Error = None }


// ============================================================================
// Tool Conversion (Tool -> OpenAI/Ollama format)
// ============================================================================

/// Convert a Tars Tool to the OpenAI/Ollama tool definition format
let toolToDefinition (tool: Tool) : obj =
    // Create a simple JSON Schema for the tool
    // Most TARS tools take a single string argument
    {| ``type`` = "function"
       ``function`` =
        {| name = tool.Name
           description = tool.Description
           parameters =
            {| ``type`` = "object"
               properties =
                Map.ofList
                    [ ("input",
                       {| ``type`` = "string"
                          description = "Input argument for the tool" |}) ]
               required = [| "input" |] |} |} |}
    :> obj

// ============================================================================
// Update
// ============================================================================

let createLlmRequest (messages: ChatMessage list) (toolRegistry: IToolRegistry) =
    let mapRole r =
        match r with
        | "user" -> Role.User
        | "assistant" -> Role.Assistant
        | _ -> Role.User

    let llmMessages =
        messages
        |> List.map (fun m ->
            { Role = mapRole m.Role
              Content = m.Content })

    // Get key tools only - sending 100+ tools causes timeouts!
    let keyToolNames =
        [ "search_web"
          "http_get"
          "fetch_wikipedia"
          "read_file"
          "list_dir"
          "git_status"
          "analyze_file_complexity"
          "lookup_docs"
          "chain_of_thought" ]

    let tools = toolRegistry.GetAll()

    let keyTools =
        tools
        |> List.filter (fun t -> keyToolNames |> List.exists (fun k -> t.Name.Contains(k)))
        |> List.truncate 10

    printfn $"🔧 CREATE_LLM_REQUEST: Using {keyTools.Length} key tools (of {tools.Length} total)"

    let nativeTools = keyTools |> List.map toolToDefinition

    // Build concise system prompt - DON'T list all 100+ tools!
    let systemPrompt =
        """You are TARS, an advanced AI assistant with tool access.

KEY CAPABILITIES:
- search_web: Search the internet for information
- fetch_wikipedia: Get Wikipedia articles
- http_get: Fetch content from URLs
- read_file: Read local files
- lookup_docs: Look up documentation

RULES:
1. For questions you can answer directly (math, facts, etc.), just answer - no tools needed
2. To actually perform a web search, use: TOOL_CALL: search_web ARGS: {"input": "your query"}
3. After getting tool results, summarize them clearly
4. Be helpful and concise.

Example tool usage:
TOOL_CALL: search_web
ARGS: {"input": "F# programming tutorials 2024"}"""

    { ModelHint = Some "smart"
      Model = None
      SystemPrompt = Some systemPrompt
      MaxTokens = Some 1024
      Temperature = Some 0.5
      Stop = []
      Messages = llmMessages
      Tools = nativeTools // Only key tools, not 100+
      ToolChoice = None
      ResponseFormat = Some ResponseFormat.Text
      Stream = false
      JsonMode = false
      Seed = None }

let executeToolCall (toolRegistry: IToolRegistry) (response: string) (userMessage: string) =
    printfn $"🔧 executeToolCall: Checking response (len={response.Length})..."
    // Check if response contains a tool call
    if response.Contains("TOOL_CALL:") && response.Contains("ARGS:") then
        printfn $"🔧 Found TOOL_CALL and ARGS markers in response"
        // Handle format: multi-line or single line
        // "TOOL_CALL: tool_name\nARGS: {...}" or "TOOL_CALL: tool_name ARGS: {...}"
        let toolCallRegex =
            System.Text.RegularExpressions.Regex(
                @"TOOL_CALL:\s*(\S+)[\s\S]*?ARGS:\s*(\{[^}]*\})",
                System.Text.RegularExpressions.RegexOptions.Singleline
            )

        printfn $"🔧 About to run regex..."
        let match' = toolCallRegex.Match(response)
        printfn $"🔧 Regex match success: {match'.Success}"

        if match'.Success then
            let rawToolName = match'.Groups.[1].Value.Trim()
            let argsJson = match'.Groups.[2].Value.Trim()
            printfn $"🔧 TOOL CALL DETECTED: {rawToolName}"
            printfn $"🔧 ARGS JSON: {argsJson}"

            // Smart routing: If user asked to "search" but LLM used http_get, route to search_web
            let toolName =
                let userLower = userMessage.ToLowerInvariant()

                let isSearchRequest =
                    userLower.Contains("search")
                    || userLower.Contains("find")
                    || userLower.Contains("look up")
                    || userLower.Contains("lookup")

                if rawToolName = "http_get" && isSearchRequest then
                    "search_web"
                else
                    rawToolName

            // Parse JSON args and extract the input value
            let extractedInput =
                try
                    let doc = System.Text.Json.JsonDocument.Parse(argsJson)
                    let root = doc.RootElement

                    try
                        root.GetProperty("input").GetString()
                    with _ ->
                        try
                            root.GetProperty("query").GetString()
                        with _ ->
                            try
                                root.GetProperty("url").GetString()
                            with _ ->
                                argsJson
                with _ ->
                    argsJson

            // If we switched to search_web, use user message as query
            let finalInput =
                if rawToolName = "http_get" && toolName = "search_web" then
                    userMessage
                        .Replace("search", "")
                        .Replace("the web for", "")
                        .Replace("Search", "")
                        .Replace("\"", "")
                        .Trim()
                else
                    extractedInput

            printfn $"🔧 FINAL INPUT: {finalInput}"
            printfn $"🔧 LOOKING UP TOOL: {toolName}"

            match toolRegistry.Get(toolName) with
            | Some tool ->
                printfn $"🔧 EXECUTING TOOL: {toolName}"

                let result =
                    try
                        let res = tool.Execute finalInput |> Async.RunSynchronously

                        match res with
                        | Result.Ok r ->
                            let truncated =
                                if r.Length > 2000 then
                                    r.Substring(0, 2000) + "... [truncated]"
                                else
                                    r

                            printfn $"🔧 TOOL SUCCESS: {truncated.Substring(0, min 100 truncated.Length)}..."
                            $"Tool '{toolName}' returned:\n{truncated}"
                        | Result.Error e ->
                            printfn $"🔧 TOOL ERROR: {e}"
                            $"Tool '{toolName}' error: {e}"
                    with ex ->
                        printfn $"🔧 TOOL EXCEPTION: {ex.Message}"
                        $"Tool execution failed: {ex.Message}"

                Some(result, tool)
            | None ->
                printfn $"🔧 TOOL NOT FOUND: {toolName}"

                let fakeTool =
                    Tool.InternalCreateMinimal(
                        toolName,
                        "Not found",
                        fun _ -> async { return Result.Error "Not found" }
                    )

                Some($"Tool '{toolName}' not found.", fakeTool)
        else
            printfn $"🔧 REGEX DID NOT MATCH"
            None
    else
        None

/// Strip any TOOL_CALL markers from response text before showing to user
let stripToolCallMarkers (text: string) =
    if text.Contains("TOOL_CALL:") then
        // Use regex to remove TOOL_CALL: ... ARGS: {...} patterns
        let pattern = @"TOOL_CALL:\s*\S+[\s\S]*?ARGS:\s*\{[^}]*\}"
        let cleaned = System.Text.RegularExpressions.Regex.Replace(text, pattern, "").Trim()
        // Also clean up any leftover "Let me try..." or similar phrases followed by nothing
        let cleaned2 =
            cleaned
            |> fun s -> System.Text.RegularExpressions.Regex.Replace(s, @":\s*$", ".")
            |> fun s -> s.Trim()

        if String.IsNullOrWhiteSpace(cleaned2) then
            "I'm processing your request..."
        else
            cleaned2
    else
        text

let callLlm (llm: ILlmService) (toolRegistry: IToolRegistry) (messages: ChatMessage list) =
    task {
        printfn "📡 CALL_LLM: Starting..."
        let mutable currentMessages = messages
        let mutable iteration = 0
        let maxIterations = 3
        let mutable finalResponse = ""
        let mutable keepGoing = true

        // Get the last user message for smart routing (stays constant)
        let lastUserMessage =
            messages
            |> List.filter (fun m -> m.Role = "user")
            |> List.tryLast
            |> Option.map (fun m -> m.Content)
            |> Option.defaultValue ""

        while keepGoing && iteration < maxIterations do
            iteration <- iteration + 1
            printfn $"📡 CALL_LLM: Iteration {iteration}/{maxIterations}"

            let req = createLlmRequest currentMessages toolRegistry
            let! response = llm.CompleteAsync req
            printfn $"📡 CALL_LLM: Got response: {response.Text.Substring(0, min 100 response.Text.Length)}..."

            // Check for tool call and execute
            let toolResult = executeToolCall toolRegistry response.Text lastUserMessage
            printfn $"📡 CALL_LLM: Tool result = {toolResult.IsSome}"

            match toolResult with
            | Some(result, tool) ->
                printfn $"📡 CALL_LLM: Tool returned: {result.Substring(0, min 100 result.Length)}..."
                // Add assistant message and tool result, continue loop
                currentMessages <-
                    currentMessages
                    @ [ { MsgId = Guid.NewGuid()
                          Role = "assistant"
                          Content = response.Text
                          Timestamp = DateTime.Now
                          ToolCall = Some tool.Name
                          WoTMetadata =
                            match tool.ThingDescription with
                            | Some td -> td |> Map.toSeq |> Seq.map (fun (k, v) -> k, string v) |> Map.ofSeq |> Some
                            | None -> None }
                        { MsgId = Guid.NewGuid()
                          Role = "user"
                          Content = result
                          Timestamp = DateTime.Now
                          ToolCall = None
                          WoTMetadata = None } ]
            | None ->
                // No tool call, we're done
                printfn "📡 CALL_LLM: No tool call, done"
                finalResponse <- response.Text
                keepGoing <- false

        if String.IsNullOrEmpty(finalResponse) then
            // Reached max iterations, return last response (cleaned)
            printfn "📡 CALL_LLM: Max iterations reached"
            let req = createLlmRequest currentMessages toolRegistry
            let! lastResponse = llm.CompleteAsync req
            return stripToolCallMarkers lastResponse.Text
        else
            return stripToolCallMarkers finalResponse
    }

let update
    (llm: ILlmService)
    (registry: IAgentRegistry)
    (graph: IGraphService)
    (toolRegistry: IToolRegistry)
    message
    model
    =
    match message with
    | NavigateTo page ->
        let cmd =
            match page with
            | Agents -> Cmd.ofMsg LoadAgents
            | Knowledge -> Cmd.ofMsg LoadKnowledge
            | Tools -> Cmd.ofMsg LoadTools
            | _ -> Cmd.none

        { model with CurrentPage = page }, cmd
    | LoadAgents ->
        let cmd = Cmd.OfAsync.perform (fun () -> registry.GetAllAgents()) () AgentsLoaded
        model, cmd
    | AgentsLoaded agents -> { model with Agents = agents }, Cmd.none
    | LoadKnowledge ->
        let cmd = Cmd.OfTask.perform (fun () -> graph.QueryAsync "*") () KnowledgeLoaded
        model, cmd
    | KnowledgeLoaded facts -> { model with Knowledge = facts }, Cmd.none
    | LoadTools ->
        let cmd = Cmd.OfFunc.either (fun () -> toolRegistry.GetAll()) () ToolsLoaded Error
        model, cmd
    | ToolsLoaded tools -> { model with Tools = tools }, Cmd.none
    | ScanCodebase ->
        let scanTask () =
            task {
                let srcDir = System.IO.Path.Combine(System.Environment.CurrentDirectory, "src")

                if System.IO.Directory.Exists(srcDir) then
                    do! AstIngestor.ingestDirectory graph srcDir |> Async.StartAsTask

                    let files =
                        System.IO.Directory.GetFiles(srcDir, "*.fs", System.IO.SearchOption.AllDirectories)

                    return files.Length
                else
                    return 0
            }

        { model with
            IsScanning = true
            ScanResult = None },
        Cmd.OfTask.perform scanTask () ScanCodebaseComplete

    | ScanCodebaseComplete count ->
        { model with
            IsScanning = false
            ScanResult = Some $"Scanned {count} files" },
        Cmd.ofMsg LoadKnowledge
    | SetKnowledgeFilter filter -> { model with KnowledgeFilter = filter }, Cmd.none
    | TestTool name ->
        let cmd =
            Cmd.OfTask.either
                (fun () ->
                    task {
                        match toolRegistry.Get(name) with
                        | Some tool ->
                            // Add timeout to prevent hanging on tools that depend on external services
                            let timeoutMs = 10000 // 10 second timeout
                            let executionTask = tool.Execute "test input" |> Async.StartAsTask
                            let timeoutTask = Task.Delay(timeoutMs)

                            let! completedTask = Task.WhenAny(executionTask, timeoutTask)

                            if completedTask = timeoutTask then
                                return
                                    $"⏱️ Timeout: Tool '{name}' did not respond within 10 seconds. It may require external services (Docker, npm, etc.)."
                            else
                                let! result = executionTask

                                return
                                    match result with
                                    | Result.Ok r -> r
                                    | Result.Error e -> $"Error: {e}"
                        | None -> return "Tool not found"
                    })
                ()
                (fun res -> TestToolResult(name, res))
                Error

        { model with
            ToolTestResults = model.ToolTestResults.Add(name, "Testing...") },
        cmd
    | TestToolResult(name, result) ->
        { model with
            ToolTestResults = model.ToolTestResults.Add(name, result) },
        Cmd.none
    | EventReceived evt ->
        let events = evt :: model.Events |> List.truncate 100
        { model with Events = events }, Cmd.none
    | ClearEvents -> { model with Events = [] }, Cmd.none
    | SetChatInput input -> { model with ChatInput = input }, Cmd.none
    | SendChatMessage when not (String.IsNullOrWhiteSpace model.ChatInput) ->
        let userMsg =
            { MsgId = Guid.NewGuid()
              Role = "user"
              Content = model.ChatInput
              Timestamp = DateTime.Now
              ToolCall = None
              WoTMetadata = None }

        let cmd =
            Cmd.OfTask.either
                (fun () -> callLlm llm toolRegistry (model.ChatMessages @ [ userMsg ]))
                ()
                ReceiveChatResponse
                Error

        { model with
            ChatMessages = model.ChatMessages @ [ userMsg ]
            ChatInput = ""
            IsLoading = true },
        cmd
    | SendChatMessage -> model, Cmd.none
    | ReceiveChatResponse response ->
        let assistantMsg =
            { MsgId = Guid.NewGuid()
              Role = "assistant"
              Content = response
              Timestamp = DateTime.Now
              ToolCall = None
              WoTMetadata = None }

        { model with
            ChatMessages = model.ChatMessages @ [ assistantMsg ]
            IsLoading = false },
        Cmd.ofMsg RefreshCognitiveState
    | Error ex ->
        { model with
            Error = Some ex.Message
            IsLoading = false },
        Cmd.none
    | StartEvolution ->
        // Start evolution - in real impl would call Evolution engine
        let evo =
            { model.Evolution with
                IsRunning = true
                Generation = 0
                TasksCompleted = 0
                Log = [ "Starting evolution..." ] }

        { model with Evolution = evo }, Cmd.none
    | EvolutionProgress(gen, tasks, task) ->
        let log =
            $"Gen {gen}: {task}" :: model.Evolution.Log
            |> List.take (min 50 (model.Evolution.Log.Length + 1))

        let evo =
            { model.Evolution with
                Generation = gen
                TasksCompleted = tasks
                CurrentTask = Some task
                Log = log }

        { model with Evolution = evo }, Cmd.none
    | EvolutionComplete(gens, tasks) ->
        let log =
            $"✅ Evolution complete: {gens} generations, {tasks} tasks"
            :: model.Evolution.Log

        let evo =
            { model.Evolution with
                IsRunning = false
                Generation = gens
                TasksCompleted = tasks
                CurrentTask = None
                Log = log }

        { model with Evolution = evo }, Cmd.none
    // GoT/WoT Messages
    | SetGoTInput input ->
        let got = { model.GoT with Input = input }
        { model with GoT = got }, Cmd.none
    | StartGoT ->
        let got =
            { model.GoT with
                IsRunning = true
                Progress = [ $"[Pattern: {model.GoT.Pattern}] Initializing reasoning engine..." ]
                Result = None
                NodesExplored = 0 }

        let runGoT (pattern: AgenticWorkflowPattern) (input: string) =
            async {
                do! Async.Sleep 1000

                match pattern with
                | Reflection -> return "Decomposition -> Generation -> Critique -> Revision -> Final Answer"
                | ToolUse -> return "Capability Selection -> API Call -> Context Integration -> Answer"
                | ReAct -> return "Thought 1 -> Act 1 -> Obs 1 -> Thought 2 -> Act 2 -> Obs 2 -> Answer"
                | Planning -> return "Decomposition -> Dependency Mapping -> Sequential Execution -> Synthesis"
                | MultiAgent -> return "Orchestrator -> Researcher -> Critic -> Writer -> Final Review"
            }

        let cmd =
            Cmd.OfAsync.perform (fun () -> runGoT model.GoT.Pattern model.GoT.Input) () (fun res ->
                let finalMsg =
                    $"""**Pattern: {model.GoT.Pattern}**
                    
Successful execution for input: "{model.GoT.Input}"

**Execution Trace:**
{res}

**Final Result:**
{model.GoT.Input.ToUpper()} (Processed via {model.GoT.Pattern})"""

                GoTComplete finalMsg)

        { model with GoT = got }, cmd
    | GoTProgress msg ->
        let got =
            { model.GoT with
                Progress = model.GoT.Progress @ [ msg ] }

        { model with GoT = got }, Cmd.none
    | GoTComplete result ->
        let got =
            { model.GoT with
                IsRunning = false
                Result = Some result }

        { model with GoT = got }, Cmd.none
    | RefreshCognitiveState ->
        let cmd =
            Cmd.OfAsync.perform (fun () -> CognitiveAnalyzer(registry).Analyze()) () CognitiveStateUpdated

        model, cmd
    | CognitiveStateUpdated state ->
        { model with
            CognitiveState = Some state },
        Cmd.none
    | SelectPattern p ->
        let got = { model.GoT with Pattern = p }
        { model with GoT = got }, Cmd.none

// ============================================================================
// View Helpers

// ============================================================================

let navLink (page: Page) (label: string) (currentPage: Page) dispatch =
    let isActive = page = currentPage

    div {
        attr.classes
            [ "nav-link"
              if isActive then "active" else "" ]

        on.click (fun _ -> dispatch (NavigateTo page))
        text label
    }

let performativeColor (p: Performative) =
    match p with
    | Performative.Request -> "#3498db"
    | Performative.Inform -> "#2ecc71"
    | Performative.Query -> "#9b59b6"
    | Performative.Propose -> "#f39c12"
    | Performative.Refuse -> "#e74c3c"
    | Performative.Failure -> "#c0392b"
    | Performative.NotUnderstood -> "#95a5a6"
    | Performative.Event -> "#1abc9c"

let endpointToString endpoint =
    match endpoint with
    | MessageEndpoint.System -> "System"
    | MessageEndpoint.User -> "User"
    | MessageEndpoint.Agent(AgentId id) -> id.ToString()
    | MessageEndpoint.Alias name -> name

let entityToHtml (entity: TarsEntity) =
    match entity with
    | TarsEntity.CodePatternE p ->
        span {
            attr.classes [ "entity-tag"; "pattern" ]
            text p.Name
        }
    | TarsEntity.AgentBeliefE b ->
        span {
            attr.classes [ "entity-tag"; "belief" ]
            text (b.Statement.Substring(0, min 20 b.Statement.Length) + "...")
        }
    | TarsEntity.GrammarRuleE g ->
        span {
            attr.classes [ "entity-tag"; "grammar" ]
            text g.Name
        }
    | TarsEntity.CodeModuleE m ->
        span {
            attr.classes [ "entity-tag"; "module" ]
            text (System.IO.Path.GetFileName m.Path)
        }
    | TarsEntity.AnomalyE a ->
        span {
            attr.classes [ "entity-tag"; "anomaly" ]
            text (a.Type.ToString())
        }
    | TarsEntity.ConceptE c ->
        span {
            attr.classes [ "entity-tag"; "concept" ]
            text c.Name
        }
    | TarsEntity.EpisodeE e ->
        span {
            attr.classes [ "entity-tag"; "episode" ]
            text (Episode.typeTag e)
        }
    | TarsEntity.FileE p ->
        span {
            attr.classes [ "entity-tag"; "file" ]
            text (System.IO.Path.GetFileName p)
        }
    | TarsEntity.FunctionE n ->
        span {
            attr.classes [ "entity-tag"; "func" ]
            text n
        }

let factView (fact: TarsFact) =
    let (s, rel, t) =
        match fact with
        | TarsFact.Implements(s, t, _) -> (s, "Implements", Some t)
        | TarsFact.DependsOn(s, t, _) -> (s, "DependsOn", Some t)
        | TarsFact.Contradicts(s, t, _) -> (s, "Contradicts", Some t)
        | TarsFact.EvolvedFrom(s, t, _) -> (s, "EvolvedFrom", Some t)
        | TarsFact.BelongsTo(e, c) -> (e, "BelongsTo " + c, None)
        | TarsFact.SimilarTo(s, t, _) -> (s, "SimilarTo", Some t)
        | TarsFact.DerivedFrom(s, t) -> (s, "DerivedFrom", Some t)
        | TarsFact.Contains(s, t) -> (s, "Contains", Some t)

    div {
        attr.classes [ "fact-card" ]

        div {
            attr.classes [ "fact-subject" ]
            entityToHtml s
        }

        div {
            attr.classes [ "fact-relation" ]
            text rel
        }

        div {
            attr.classes [ "fact-target" ]

            match t with
            | Some target -> entityToHtml target
            | None -> empty ()
        }
    }

let cognitiveStats (state: CognitiveState option) =
    div {
        attr.classes [ "cognitive-stats-panel" ]

        match state with
        | Some s ->
            div {
                attr.classes [ "stat-item" ]

                span {
                    attr.classes [ "stat-label" ]
                    text "Mode: "
                }

                span {
                    attr.classes [ "stat-value"; s.Mode.ToString().ToLower() ]
                    text (s.Mode.ToString())
                }
            }

            div {
                attr.classes [ "stat-item" ]

                span {
                    attr.classes [ "stat-label" ]
                    text "Grounding: "
                }

                span {
                    attr.classes [ "stat-value" ]
                    text (sprintf "%.1f%%" (s.GroundingFidelity * 100.0))
                }
            }

            div {
                attr.classes [ "stat-item" ]

                span {
                    attr.classes [ "stat-label" ]
                    text "Reasoning: "
                }

                span {
                    attr.classes [ "stat-value" ]
                    text (sprintf "%.2f" s.BranchingFactor)
                }
            }
        | None -> text "Initializing cognitive engine..."
    }

let wotMetadataView (toolName: string option) (metadata: Map<string, string> option) =
    match toolName, metadata with
    | Some name, Some m when not m.IsEmpty ->
        div {
            attr.classes [ "wot-metadata" ]

            div {
                attr.classes [ "wot-header" ]

                span {
                    attr.classes [ "wot-icon" ]
                    text "🔗"
                }

                h4 { text $"WoT: {name}" }
            }

            ul {
                forEach (m |> Map.toSeq) (fun (k, v) ->
                    li {
                        span {
                            attr.classes [ "wot-key" ]
                            text k
                        }

                        text ": "

                        span {
                            attr.classes [ "wot-value" ]
                            text v
                        }
                    })
            }
        }
    | Some name, _ ->
        div {
            attr.classes [ "wot-metadata"; "minimal" ]
            text $"Tool used: {name} (Not WoT-grounded)"
        }
    | _ -> empty ()

// ============================================================================
// Chat Page
// ============================================================================

let chatBubble (msg: ChatMessage) =
    let isUser = msg.Role = "user"

    div {
        attr.classes
            [ "chat-message"
              if isUser then "user" else "assistant" ]

        div {
            attr.classes [ "message-avatar" ]
            text (if isUser then "👤" else "🤖")
        }

        div {
            attr.classes [ "message-content" ]

            div {
                attr.classes [ "message-bubble" ]
                text msg.Content
                wotMetadataView msg.ToolCall msg.WoTMetadata
            }

            div {
                attr.classes [ "message-time" ]
                text (msg.Timestamp.ToString("h:mm tt"))
            }
        }
    }

let chatPage model dispatch =
    div {
        attr.classes [ "page"; "chat-page" ]

        // Chat header
        div {
            attr.classes [ "chat-header" ]

            div {
                attr.classes [ "chat-title" ]

                span {
                    attr.classes [ "robot-icon" ]
                    text "🤖"
                }

                h1 { text "TARS Chat" }
            }

            div {
                attr.classes [ "chat-status" ]

                span {
                    attr.classes
                        [ "status-dot"
                          if model.IsLoading then "loading" else "online" ]
                }

                text (if model.IsLoading then "Thinking..." else "Online")
            }

            cognitiveStats model.CognitiveState
        }

        // Messages container
        div {
            attr.classes [ "chat-messages" ]

            forEach model.ChatMessages <| chatBubble

            if model.IsLoading then
                div {
                    attr.classes [ "chat-message"; "assistant"; "typing" ]

                    div {
                        attr.classes [ "message-avatar" ]
                        text "🤖"
                    }

                    div {
                        attr.classes [ "message-content" ]

                        div {
                            attr.classes [ "message-bubble"; "typing-indicator" ]
                            span { }
                            span { }
                            span { }
                        }
                    }
                }
        }

        // Input area
        div {
            attr.classes [ "chat-input-area" ]

            div {
                attr.classes [ "chat-input-container" ]

                textarea {
                    attr.classes [ "chat-input" ]
                    attr.placeholder "Type your message..."
                    attr.rows 1
                    bind.input.string model.ChatInput (fun v -> dispatch (SetChatInput v))

                    on.keydown (fun e ->
                        if e.Key = "Enter" && not e.ShiftKey then
                            dispatch SendChatMessage)
                }

                button {
                    attr.``type`` "button"

                    attr.classes
                        [ "send-button"
                          if String.IsNullOrWhiteSpace model.ChatInput then
                              "disabled"
                          else
                              "" ]

                    on.click (fun _ -> dispatch SendChatMessage)
                    text "Send"
                }
            }

            div {
                attr.classes [ "input-hint" ]
                text "Press Enter to send, Shift+Enter for new line"
            }
        }
    }

// ============================================================================
// Dashboard Page
// ============================================================================

let dashboardPage model dispatch =
    div {
        attr.classes [ "page"; "dashboard" ]

        div {
            attr.classes [ "page-header" ]
            h1 { text "TARS Dashboard" }

            button {
                attr.classes [ "btn"; "btn-secondary" ]
                on.click (fun _ -> dispatch ClearEvents)
                text "Clear"
            }
        }

        div {
            attr.classes [ "event-feed" ]

            if List.isEmpty model.Events then
                div {
                    attr.classes [ "empty-state" ]
                    text "No events yet. Start an agent to see activity."
                }
            else
                forEach model.Events
                <| fun evt ->
                    div {
                        attr.classes [ "event-card" ]
                        attr.style $"border-left: 4px solid {performativeColor evt.Performative}"

                        div {
                            attr.classes [ "event-header" ]

                            span {
                                attr.classes [ "performative" ]
                                text (evt.Performative.ToString())
                            }

                            span {
                                attr.classes [ "event-id" ]
                                text (evt.Id.ToString().Substring(0, 8))
                            }
                        }

                        div {
                            attr.classes [ "event-routing" ]
                            span { text (endpointToString evt.Sender) }

                            span {
                                attr.classes [ "arrow" ]
                                text " → "
                            }

                            span { text (evt.Receiver |> Option.map endpointToString |> Option.defaultValue "ALL") }
                        }

                        div {
                            attr.classes [ "event-content" ]

                            text (
                                match evt.Content with
                                | :? string as s -> s.Substring(0, min 200 s.Length)
                                | x -> x.GetType().Name
                            )
                        }
                    }
        }
    }

let agentsPage model dispatch =
    div {
        attr.classes [ "page"; "agents" ]
        h1 { text "Registered Agents" }

        if List.isEmpty model.Agents then
            div {
                attr.classes [ "placeholder" ]
                text "No agents found."
            }
        else
            div {
                attr.classes [ "agents-grid" ]

                forEach model.Agents
                <| fun agent ->
                    div {
                        attr.classes [ "agent-card" ]
                        h3 { text (agent.Id.ToString()) }

                        div {
                            attr.classes [ "capabilities" ]

                            forEach agent.Capabilities
                            <| fun cap ->
                                span {
                                    attr.classes [ "capability-badge" ]
                                    text (cap.Kind.ToString())
                                }
                        }
                    }
            }
    }

let knowledgePage model dispatch =
    div {
        attr.classes [ "page"; "knowledge" ]
        h1 { text "Knowledge Graph" }

        // Scan codebase action bar
        div {
            attr.classes [ "action-bar" ]

            div {
                attr.classes [ "search-box" ]

                input {
                    attr.classes [ "search-input" ]
                    attr.placeholder "Filter knowledge..."
                    bind.input.string model.KnowledgeFilter (fun v -> dispatch (SetKnowledgeFilter v))
                }
            }

            button {
                attr.classes
                    [ "btn"
                      "btn-primary"
                      if model.IsScanning then "loading" else "" ]

                on.click (fun _ -> dispatch ScanCodebase)
                attr.disabled model.IsScanning

                if model.IsScanning then
                    text "🔍 Scanning..."
                else
                    text "🔍 Scan Codebase"
            }

            match model.ScanResult with
            | Some result ->
                span {
                    attr.classes [ "scan-result" ]
                    text result
                }
            | None -> ()
        }

        if List.isEmpty model.Knowledge then
            div {
                attr.classes [ "placeholder" ]

                div {
                    attr.style "text-align: center; padding: 2rem;"

                    h2 { text "📚 Knowledge Graph" }

                    p {
                        attr.style "margin: 1rem 0;"
                        text "Click 'Scan Codebase' to ingest source files into the semantic graph."
                    }

                    div {
                        attr.style
                            "margin-top: 2rem; padding: 1rem; background: rgba(0,255,136,0.1); border-radius: 8px; border: 1px solid rgba(0,255,136,0.3);"

                        h3 {
                            attr.style "color: #00ff88; margin-bottom: 0.5rem;"
                            text "🧠 NEW: Phase 9 Knowledge Ledger"
                        }

                        p {
                            attr.style "color: #888; font-size: 0.9rem;"
                            text "Use the CLI for event-sourced beliefs with PostgreSQL:"
                        }

                        code {
                            attr.style
                                "display: block; background: #1a1a2e; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; font-family: monospace;"

                            text "tars know status --pg"
                        }

                        code {
                            attr.style
                                "display: block; background: #1a1a2e; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; font-family: monospace;"

                            text "tars know assert 'TARS' is-a 'Agent' --pg"
                        }

                        code {
                            attr.style
                                "display: block; background: #1a1a2e; padding: 0.5rem; border-radius: 4px; margin: 0.5rem 0; font-family: monospace;"

                            text "tars know fetch 'Artificial Intelligence' --pg"
                        }
                    }
                }
            }
        else
            let filteredKnowledge =
                if String.IsNullOrWhiteSpace model.KnowledgeFilter then
                    model.Knowledge
                else
                    model.Knowledge
                    |> List.filter (fun f ->
                        f.ToString().Contains(model.KnowledgeFilter, StringComparison.OrdinalIgnoreCase))

            div {
                attr.classes [ "knowledge-grid" ]
                forEach filteredKnowledge factView
            }
    }

// ============================================================================
// Evolution Page
// ============================================================================

let evolutionPage model dispatch =
    div {
        attr.classes [ "page"; "evolution" ]

        // Header
        div {
            attr.classes [ "page-header" ]
            h1 { text "🧬 Evolution Engine" }

            div {
                attr.classes [ "header-actions" ]

                button {
                    attr.classes
                        [ "btn"
                          "btn-primary"
                          if model.Evolution.IsRunning then "loading" else "" ]

                    on.click (fun _ -> dispatch StartEvolution)
                    attr.disabled model.Evolution.IsRunning

                    if model.Evolution.IsRunning then
                        text "⏳ Running..."
                    else
                        text "▶️ Start Evolution"
                }
            }
        }

        // Stats cards
        div {
            attr.classes [ "stats-grid" ]

            div {
                attr.classes [ "stat-card" ]

                div {
                    attr.classes [ "stat-value" ]
                    text (string model.Evolution.Generation)
                }

                div {
                    attr.classes [ "stat-label" ]
                    text "Generation"
                }
            }

            div {
                attr.classes [ "stat-card" ]

                div {
                    attr.classes [ "stat-value" ]
                    text (string model.Evolution.TasksCompleted)
                }

                div {
                    attr.classes [ "stat-label" ]
                    text "Tasks Completed"
                }
            }

            div {
                attr.classes
                    [ "stat-card"
                      if model.Evolution.IsRunning then "active" else "" ]

                div {
                    attr.classes [ "stat-value" ]
                    text (if model.Evolution.IsRunning then "🟢" else "⚪")
                }

                div {
                    attr.classes [ "stat-label" ]
                    text "Status"
                }
            }
        }

        // Current task
        match model.Evolution.CurrentTask with
        | Some task ->
            div {
                attr.classes [ "current-task" ]
                h3 { text "📋 Current Task" }

                div {
                    attr.classes [ "task-content" ]
                    text task
                }
            }
        | None -> ()

        // Activity log
        div {
            attr.classes [ "activity-log" ]
            h3 { text "📜 Activity Log" }

            div {
                attr.classes [ "log-entries" ]

                if List.isEmpty model.Evolution.Log then
                    div {
                        attr.classes [ "empty-state" ]
                        text "No evolution activity yet. Click 'Start Evolution' to begin."
                    }
                else
                    forEach model.Evolution.Log
                    <| fun entry ->
                        div {
                            attr.classes [ "log-entry" ]
                            text entry
                        }
            }
        }
    }

// ============================================================================
// Reasoning (GoT/WoT) Demo Page
// ============================================================================

let reasoningPage model dispatch =
    div {
        attr.classes [ "page"; "reasoning-page" ]

        div {
            attr.classes [ "page-header" ]
            h1 { text "🧩 Graph-of-Thoughts & Workflow-of-Thought" }

            span {
                attr.classes [ "badge" ]
                text "Demo"
            }
        }

        // Concepts Section: The Five Essential Agentic Workflow Patterns
        div {
            attr.classes [ "concepts-section" ]
            attr.style "margin-bottom: 2rem;"

            h2 {
                attr.style
                    "color: #fff; margin-bottom: 1rem; font-size: 1.5rem; display: flex; align-items: center; gap: 0.5rem;"

                text "🧩 Agentic Workflow Patterns"
            }

            div {
                attr.style "display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;"

                let patterns =
                    [ (Reflection, "Reflection", "Self-critique and iterative refinement.", "#00ff88", "💭")
                      (ToolUse, "Tool Use", "Dynamic selection of APIs and RAG.", "#4488ff", "🔧")
                      (ReAct, "ReAct", "Interleaved reasoning and action loops.", "#ff8844", "🧠")
                      (Planning, "Planning", "Strategic decomposition of complex goals.", "#aa44ff", "📅")
                      (MultiAgent, "MultiAgent", "Teams of specialized agents collaborating.", "#ff44aa", "🤖") ]

                for (pType, name, desc, color, icon) in patterns do
                    div {
                        let active = model.GoT.Pattern = pType

                        let baseStyle =
                            $"""background: {color}11; padding: 1rem; border-radius: 12px; border: 1px solid {if active then color else color + "33"}; cursor: pointer; transition: all 0.2s; position: relative; overflow: hidden;"""

                        let activeStyle =
                            if active then
                                "transform: translateY(-4px); box-shadow: 0 4px 12px rgba(0,0,0,0.5);"
                            else
                                ""

                        attr.style (baseStyle + " " + activeStyle)

                        on.click (fun _ -> dispatch (SelectPattern pType))

                        div {
                            attr.style $"font-size: 1.2rem; margin-bottom: 0.5rem;"
                            text icon
                        }

                        h4 {
                            attr.style $"color: {color}; margin-bottom: 0.25rem; font-size: 0.95rem; font-weight: 700;"
                            text name
                        }

                        p {
                            attr.style "color: #888; font-size: 0.8rem; line-height: 1.3;"
                            text desc
                        }
                    }
            }
        }

        // Interactive Demo Section
        div {
            attr.style "background: #1a1a2e; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;"

            h3 {
                attr.style "color: #fff; margin-bottom: 1rem;"
                text "🚀 Try Graph-of-Thoughts"
            }

            div {
                attr.style "display: flex; gap: 1rem; margin-bottom: 1rem;"

                textarea {
                    attr.style
                        "flex: 1; background: #0d0d1a; border: 1px solid #333; border-radius: 8px; padding: 1rem; color: #fff; font-family: inherit; resize: vertical; min-height: 100px; outline: none;"

                    attr.placeholder
                        "Enter a problem to solve... (e.g. Compare the historical impact of atonal music and jazz)"

                    attr.value model.GoT.Input
                    on.input (fun e -> dispatch (SetGoTInput(string e.Value)))
                }
            }

            div {
                attr.style "display: flex; gap: 1rem; align-items: center;"

                button {
                    attr.classes
                        [ "btn"
                          "btn-primary"
                          if model.GoT.IsRunning then "loading" else "" ]

                    attr.style "padding: 0.75rem 1.5rem; font-size: 1rem;"
                    on.click (fun _ -> dispatch StartGoT)
                    attr.disabled model.GoT.IsRunning

                    if model.GoT.IsRunning then
                        text "🔄 Reasoning..."
                    else
                        text "▶️ Start GoT Reasoning"
                }

                span {
                    attr.style "color: #888; font-size: 0.9rem;"
                    text $"Branching: {model.GoT.SelectedBranchingFactor} | Depth: {model.GoT.SelectedMaxDepth}"
                }
            }

            // Progress Log
            if not (List.isEmpty model.GoT.Progress) then
                div {
                    attr.style
                        "margin-top: 1rem; padding: 1rem; background: #0d0d1a; border-radius: 8px; max-height: 150px; overflow-y: auto;"

                    forEach model.GoT.Progress
                    <| fun entry ->
                        div {
                            attr.style "font-family: monospace; font-size: 0.85rem; color: #00ff88; padding: 2px 0;"
                            text entry
                        }
                }

            // Result Display
            match model.GoT.Result with
            | Some result ->
                div {
                    attr.style
                        "margin-top: 1rem; padding: 1rem; background: linear-gradient(135deg, rgba(0, 255, 136, 0.05), rgba(0, 136, 255, 0.05)); border-radius: 8px; border: 1px solid rgba(0, 255, 136, 0.3);"

                    h4 {
                        attr.style "color: #00ff88; margin-bottom: 0.5rem;"
                        text "✅ Result"
                    }

                    pre {
                        attr.style "color: #ddd; font-size: 0.9rem; white-space: pre-wrap; margin: 0;"
                        text result
                    }
                }
            | None -> ()
        }

        // Node Types Reference
        div {
            attr.style "margin-bottom: 2rem;"

            h3 {
                attr.style "color: #fff; margin-bottom: 1rem;"
                text "📦 WoT Node Types"
            }

            div {
                attr.style "display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem;"

                // Node type badges
                for (icon, name, desc) in
                    [ ("💭", "ThoughtNode", "Reasoning step")
                      ("🔧", "ToolNode", "API/RAG call")
                      ("📋", "PolicyNode", "Compliance check")
                      ("👤", "RoleNode", "Human review")
                      ("🧠", "MemoryNode", "Past context")
                      ("✅", "VerifierNode", "Validation")
                      ("🔍", "CritiqueNode", "Evaluation")
                      ("🔀", "AggregationNode", "Merge") ] do
                    div {
                        attr.style "background: #1a1a2e; padding: 0.75rem; border-radius: 8px; text-align: center;"

                        div {
                            attr.style "font-size: 1.5rem; margin-bottom: 0.25rem;"
                            text icon
                        }

                        div {
                            attr.style "color: #fff; font-size: 0.8rem; font-weight: 600;"
                            text name
                        }

                        div {
                            attr.style "color: #888; font-size: 0.7rem;"
                            text desc
                        }
                    }
            }
        }

        // Edge Types Reference
        div {
            h3 {
                attr.style "color: #fff; margin-bottom: 1rem;"
                text "🔗 Edge Types (Relationships)"
            }

            div {
                attr.style "display: flex; flex-wrap: wrap; gap: 0.5rem;"

                for (color, name) in
                    [ ("#00ff88", "Supports")
                      ("#ff4444", "Contradicts")
                      ("#4488ff", "DependsOn")
                      ("#ffaa00", "Refines")
                      ("#aa44ff", "Merges")
                      ("#ff8844", "Critiques")
                      ("#44ff88", "Validates")
                      ("#ff44aa", "Escalates") ] do
                    span {
                        attr.style
                            $"background: {color}22; color: {color}; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.8rem; border: 1px solid {color}44;"

                        text name
                    }
            }
        }

        // CLI Command Reference
        div {
            attr.style "margin-top: 2rem; padding: 1rem; background: #1a1a2e; border-radius: 8px;"

            h4 {
                attr.style "color: #888; margin-bottom: 0.5rem;"
                text "💻 CLI Command"
            }

            code {
                attr.style
                    "display: block; background: #0d0d1a; padding: 0.75rem; border-radius: 4px; font-family: monospace; color: #00ff88;"

                text "tars agent got \"Your problem here\""
            }
        }
    }

// ============================================================================
// Main View
// ============================================================================

let view model dispatch =
    div {
        attr.classes [ "app" ]

        // Sidebar navigation
        nav {
            attr.classes [ "sidebar" ]

            div {
                attr.classes [ "logo" ]
                text "TARS v2"
            }

            div {
                attr.classes [ "nav-links" ]
                navLink Chat "💬 Chat" model.CurrentPage dispatch
                navLink Reasoning "🧩 Reasoning" model.CurrentPage dispatch
                navLink Dashboard "📊 Dashboard" model.CurrentPage dispatch
                navLink Evolution "🧬 Evolution" model.CurrentPage dispatch
                navLink Tools "🛠️ Tools" model.CurrentPage dispatch
                navLink Agents "🤖 Agents" model.CurrentPage dispatch
                navLink Knowledge "🧠 Knowledge" model.CurrentPage dispatch
            }
        }


        // Main content
        main {
            attr.classes [ "content" ]

            cond model.Error
            <| function
                | Some err ->
                    div {
                        attr.classes [ "error-banner" ]
                        text $"Error: {err}"
                    }
                | None -> empty ()

            cond model.CurrentPage
            <| function
                | Chat -> chatPage model dispatch
                | Reasoning -> reasoningPage model dispatch
                | Dashboard -> dashboardPage model dispatch
                | Evolution -> evolutionPage model dispatch
                | Agents -> agentsPage model dispatch
                | Knowledge -> knowledgePage model dispatch
                | Tools ->

                    div {
                        attr.classes [ "page"; "tools-page" ]

                        div {
                            attr.classes [ "page-header" ]
                            h1 { text "🛠️ Tool Registry" }

                            span {
                                attr.classes [ "badge" ]
                                text $"{model.Tools.Length} Tools Active"
                            }
                        }

                        if List.isEmpty model.Tools then
                            div {
                                attr.classes [ "placeholder" ]
                                text "Loading tools..."
                            }
                        else
                            div {
                                attr.classes [ "tools-grid" ]

                                forEach (model.Tools |> List.sortBy (fun t -> t.Name))
                                <| fun tool ->
                                    div {
                                        attr.classes [ "tool-card" ]

                                        div {
                                            attr.classes [ "tool-card-header" ]
                                            h3 { text tool.Name }

                                            span {
                                                attr.classes [ "version-badge" ]
                                                text tool.Version
                                            }
                                        }

                                        div {
                                            attr.classes [ "tool-description" ]
                                            text tool.Description
                                        }

                                        match model.ToolTestResults.TryFind tool.Name with
                                        | Some result ->
                                            div {
                                                attr.classes [ "tool-test-result" ]
                                                pre { text result }
                                            }
                                        | None -> ()

                                        div {
                                            attr.classes [ "tool-footer" ]

                                            button {
                                                attr.classes [ "btn"; "btn-sm"; "btn-outline" ]
                                                on.click (fun _ -> dispatch (TestTool tool.Name))
                                                text "⚡ Test"
                                            }

                                            span {
                                                attr.classes [ "tool-date" ]
                                                text $"""Created: {tool.CreatedAt.ToString("yyyy-MM-dd")}"""
                                            }

                                            div {
                                                attr.classes [ "tool-status" ]
                                                span { attr.classes [ "status-indicator"; "active" ] }
                                                text "Ready"
                                            }
                                        }
                                    }
                            }
                    }
        }
    }

// ============================================================================
// Program
// ============================================================================

type TarsApp() =
    inherit ProgramComponent<Model, Message>()

    [<Inject>]
    member val Llm = Unchecked.defaultof<ILlmService> with get, set

    [<Inject>]
    member val AgentRegistry = Unchecked.defaultof<IAgentRegistry> with get, set

    [<Inject>]
    member val GraphService = Unchecked.defaultof<IGraphService> with get, set

    [<Inject>]
    member val EventBus = Unchecked.defaultof<IEventBus> with get, set

    [<Inject>]
    member val ToolRegistry = Unchecked.defaultof<IToolRegistry> with get, set

    override this.Program =
        Program.mkProgram
            (fun _ -> initModel, Cmd.none)
            (update this.Llm this.AgentRegistry this.GraphService this.ToolRegistry)
            view

    override this.OnInitializedAsync() =
        let baseInit = base.OnInitializedAsync()

        task {
            do! baseInit
            // Subscribe to Event Stream
            // Note: We don't dispose this subscription manually as Blazor Server handles component lifecycle mostly,
            // but ideally we should implement IDisposable.
            // For now, we just wire it up.
            let sub =
                this.EventBus.Subscribe(
                    "System.Diagnostics",
                    fun msg ->
                        this.Dispatch(EventReceived msg)
                        Task.CompletedTask
                )

            return ()
        }
