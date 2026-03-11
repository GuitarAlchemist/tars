module Tars.Interface.Cli.Commands.Agent

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open Tars.Core
open Tars.Cortex
open Tars.Cortex.Patterns
open Tars.Cortex.WoTTypes
open Tars.Llm
open Tars.Tools
open Tars.Interface.Cli.Commands.AgentHelpers


/// Run the ReAct pattern
let runReact (config: Microsoft.Extensions.Configuration.IConfiguration) (options: AgentOptions) (goal: string) =
    task {
        printfn "🤖 TARS Agent - ReAct Mode"
        printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printfn $"Goal: %s{goal}"
        printfn $"Max Steps: %d{options.MaxSteps}"
        printfn ""

        match createLlmService config with
        | Result.Error msg ->
            printfn $"❌ %s{msg}"
            return 1
        | Result.Ok(llm, _model) ->
            let llmWithEvidence, evidenceHandle = attachEvidence "react" llm options

            let! evidence =
                match evidenceHandle with
                | Some handle ->
                    task {
                        let! value = handle
                        return Some value
                    }
                | None -> Task.FromResult None

            // Create tool registry with standard tools
            let toolRegistry = ToolRegistry()

            // Register some basic tools
            let echoTool: Tool =
                { Name = "echo"
                  Description = "Echoes the input back. Use for testing."
                  Version = "1.0.0"
                  ParentVersion = None
                  CreatedAt = DateTime.UtcNow
                  Execute = fun input -> async { return Result.Ok $"Echo: %s{input}" } }

            let timeTool: Tool =
                { Name = "current_time"
                  Description = "Returns the current date and time."
                  Version = "1.0.0"
                  ParentVersion = None
                  CreatedAt = DateTime.UtcNow
                  Execute = fun _ -> async { return Result.Ok(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")) } }

            let mathTool: Tool =
                { Name = "calculate"
                  Description =
                    "Evaluates a simple math expression (add, subtract, multiply, divide). Input: '5 + 3' or '10 * 2'"
                  Version = "1.0.0"
                  ParentVersion = None
                  CreatedAt = DateTime.UtcNow
                  Execute =
                    fun input ->
                        async {
                            try
                                // Simple expression parser
                                let parts = input.Trim().Split([| ' ' |], StringSplitOptions.RemoveEmptyEntries)

                                if parts.Length = 3 then
                                    let a = float parts.[0]
                                    let op = parts.[1]
                                    let b = float parts.[2]

                                    let result =
                                        match op with
                                        | "+" -> a + b
                                        | "-" -> a - b
                                        | "*" -> a * b
                                        | "/" -> a / b
                                        | _ -> nan // Unknown operator

                                    if Double.IsNaN(result) then
                                        return Result.Error $"Unknown operator: %s{op}"
                                    else
                                        return Result.Ok $"%g{result}"
                                else
                                    return Result.Error "Invalid expression. Use format: '5 + 3'"
                            with ex ->
                                return Result.Error ex.Message
                        } }

            toolRegistry.Register(echoTool)
            toolRegistry.Register(timeTool)
            toolRegistry.Register(mathTool)

            let tools = toolRegistry :> IToolRegistry

            // Create agent context with logging
            let logger msg =
                if options.Verbose then
                    printfn $"  [LOG] %s{msg}"

            let ctx = createAgentContext logger llmWithEvidence None

            printfn "📋 Available Tools:"

            for tool in tools.GetAll() do
                printfn $"   • %s{tool.Name}: %s{tool.Description}"

            printfn ""
            printfn "🔄 Starting ReAct loop..."
            printfn ""

            // Run ReAct pattern
            let workflow = reAct llmWithEvidence tools options.MaxSteps goal
            let! result = workflow ctx |> Async.StartAsTask

            printfn ""
            printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            match evidence with
            | Some(recorder, path) ->
                do! recorder.SaveToFileAsync(path) |> Async.StartAsTask
                printfn $"📒 Evidence saved: %s{path}"
            | None -> ()

            match result with
            | Success answer ->
                printfn "✅ Success!"
                printfn ""
                printfn $"Answer: %s{answer}"
                return 0
            | PartialSuccess(answer, warnings) ->
                printfn "⚠️ Partial Success (with warnings)"
                printfn ""
                printfn $"Answer: %s{answer}"
                printfn ""
                printfn "Warnings:"

                for w in warnings do
                    printfn $"  - %A{w}"

                return 0
            | Failure errors ->
                printfn "❌ Failed"
                printfn ""
                printfn "Errors:"

                for e in errors do
                    printfn $"  - %A{e}"

                return 1
    }

/// Run Chain of Thought pattern
let runChainOfThought
    (config: Microsoft.Extensions.Configuration.IConfiguration)
    (options: AgentOptions)
    (input: string)
    =
    task {
        printfn "🧠 TARS Agent - Chain of Thought Mode"
        printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printfn $"Input: %s{input}"
        printfn ""

        match createLlmService config with
        | Result.Error msg ->
            printfn $"❌ %s{msg}"
            return 1
        | Result.Ok(llm, _) ->
            let llmWithEvidence, evidenceHandle = attachEvidence "cot" llm options

            let! evidence =
                match evidenceHandle with
                | Some handle ->
                    task {
                        let! value = handle
                        return Some value
                    }
                | None -> Task.FromResult None

            let logger msg =
                if options.Verbose then
                    printfn $"  [LOG] %s{msg}"

            let ctx = createAgentContext logger llmWithEvidence None

            // Define reasoning steps
            let analyze =
                llmStep llmWithEvidence "Analyze the problem and identify key aspects."

            let reason = llmStep llmWithEvidence "Reason about solutions step by step."

            let summarize =
                llmStep llmWithEvidence "Summarize your analysis and provide a final answer."

            printfn "🔄 Running Chain of Thought..."
            printfn "  Step 1: Analyze"
            printfn "  Step 2: Reason"
            printfn "  Step 3: Summarize"
            printfn ""

            let workflow = chainOfThought [ analyze; reason; summarize ] input
            let! result = workflow ctx |> Async.StartAsTask

            printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            match evidence with
            | Some(recorder, path) ->
                do! recorder.SaveToFileAsync(path) |> Async.StartAsTask
                printfn $"📒 Evidence saved: %s{path}"
            | None -> ()

            match result with
            | Success answer ->
                printfn "✅ Success!"
                printfn ""
                printfn $"%s{answer}"
                return 0
            | PartialSuccess(answer, _) ->
                printfn "⚠️ Partial Success"
                printfn ""
                printfn $"%s{answer}"
                return 0
            | Failure errors ->
                printfn "❌ Failed"

                for e in errors do
                    printfn $"  - %A{e}"

                return 1
    }

/// Run Graph of Thoughts pattern
let private runReasoningEngine
    (config: Microsoft.Extensions.Configuration.IConfiguration)
    (options: AgentOptions)
    (patternName: string)
    (goal: string)
    (workflowFactory: ILlmService -> GoTConfig -> string -> AgentWorkflow<string>)
    =
    task {
        printfn $"🌐 TARS Agent - {patternName} Mode"
        printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printfn $"Goal: %s{goal}"
        printfn ""

        match createLlmService config with
        | Result.Error msg ->
            printfn $"❌ %s{msg}"
            return 1
        | Result.Ok(llm, _) ->
            try
                let llmWithEvidence, evidenceHandle = attachEvidence patternName llm options

                let! evidence =
                    match evidenceHandle with
                    | Some handle ->
                        task {
                            let! value = handle
                            return Some value
                        }
                    | None -> Task.FromResult None

                let logger msg =
                    if options.Verbose then
                        printfn $"  [LOG] %s{msg}"
                    else if
                        msg.Contains("[GoT] Phase")
                        || msg.Contains("[GoT] Final")
                        || msg.Contains("[GoT] Generated")
                        || msg.Contains("[GoT] Scored")
                        || msg.Contains("[ToT] Step")
                        || msg.Contains("[ToT] Selected")
                        || msg.Contains("[ToT] Proposed")
                        || msg.Contains("[ToT] Valued")
                    then
                        printfn $"  %s{msg}"

                // Minimal agent context for reasoning (doesn't need tools or heavy infra)
                let agent: Agent =
                    { Id = AgentId(Guid.NewGuid())
                      Name = $"TARS {patternName} Agent"
                      Version = "2.0.0"
                      ParentVersion = None
                      CreatedAt = DateTime.UtcNow
                      Model = "default"
                      SystemPrompt = "You are TARS, an autonomous reasoning agent."
                      Tools = []
                      Capabilities = []
                      State = AgentState.Idle
                      Memory = []
                      Fitness = 0.5
                      Drives = { Accuracy = 0.5; Speed = 0.5; Creativity = 0.5; Safety = 0.5 }
                      Constitution = AgentConstitution.Create(AgentId(Guid.NewGuid()), GeneralReasoning) }

                let mockRegistry =
                    { new IAgentRegistry with
                        member _.GetAgent(_) = async { return None }
                        member _.FindAgents(_) = async { return [] }
                        member _.GetAllAgents() = async { return [] } }

                let mockExecutor =
                    { new IAgentExecutor with
                        member _.Execute(_, _) =
                            async { return Success "Not implemented" } }

                let ctx: AgentContext =
                    { Self = agent
                      Registry = mockRegistry
                      Executor = mockExecutor
                      Logger = logger
                      Budget = None
                      Epistemic = None
                      SemanticMemory = None
                      SymbolicReflector = None
                      KnowledgeGraph = None
                      CapabilityStore = None
                      Audit = None
                      CancellationToken = CancellationToken.None }

                let gotConfig: GoTConfig =
                    { BranchingFactor = 3
                      MaxDepth = 3
                      TopK = 2
                      ScoreThreshold = 0.1
                      MinConfidence = 0.4
                      DiversityThreshold = 0.85
                      DiversityPenalty = 0.25
                      Constraints = []
                      EnableCritique = false
                      EnablePolicyChecks = false
                      EnableMemoryRecall = false
                      TrackEdges = false }


                printfn "📊 Configuration:"
                printfn $"   • Branching Factor: %d{gotConfig.BranchingFactor}"
                printfn $"   • Max Depth: %d{gotConfig.MaxDepth}"
                printfn $"   • Top-K: %d{gotConfig.TopK}"
                printfn $"   • Min Confidence: %.2f{gotConfig.MinConfidence}"
                printfn ""
                printfn $"🔄 Starting {patternName} reasoning..."
                printfn ""

                let workflow = workflowFactory llmWithEvidence gotConfig goal
                let! result = workflow ctx |> Async.StartAsTask

                printfn ""
                printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

                match evidence with
                | Some(recorder, path) ->
                    do! recorder.SaveToFileAsync(path) |> Async.StartAsTask
                    printfn $"📒 Evidence saved: %s{path}"
                | None -> ()

                match result with
                | Success answer ->
                    printfn "✅ Success!"
                    printfn ""
                    printfn $"%s{answer}"
                    return 0
                | PartialSuccess(answer, warnings) ->
                    printfn "⚠️ Partial Success"
                    printfn ""
                    printfn $"%s{answer}"

                    if options.Verbose then
                        printfn ""
                        printfn "Warnings:"

                        for w in warnings do
                            printfn $"  - %A{w}"

                    return 0
                | Failure errors ->
                    printfn "❌ Failed"

                    for e in errors do
                        printfn $"  - %A{e}"

                    return 1
            with ex ->
                printfn $"❌ Exception occurred: %s{ex.Message}"
                if ex.InnerException <> null then printfn $"   Inner: %s{ex.InnerException.Message}"

                let errorFile =
                    Path.Combine(
                        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                        $"tars_{patternName.ToLower()}_error.txt"
                    )

                let errorMsg =
                    sprintf
                        "Exception: %s\nInner: %s\nStack:\n%s"
                        ex.Message
                        (if ex.InnerException <> null then
                             ex.InnerException.Message
                         else
                             "None")
                        (ex.StackTrace |> Option.ofObj |> Option.defaultValue "No stack")

                File.WriteAllText(errorFile, errorMsg)
                printfn $"Error written to: %s{errorFile}"
                return 1
    }

let runGraphOfThoughts
    (config: Microsoft.Extensions.Configuration.IConfiguration)
    (options: AgentOptions)
    (goal: string)
    =
    runReasoningEngine config options "Graph-of-Thoughts" goal graphOfThoughts

let runTreeOfThoughts
    (config: Microsoft.Extensions.Configuration.IConfiguration)
    (options: AgentOptions)
    (goal: string)
    =
    runReasoningEngine config options "Tree-of-Thoughts" goal treeOfThoughts

let runWorkflowOfThoughts
    (config: Microsoft.Extensions.Configuration.IConfiguration)
    (options: AgentOptions)
    (goal: string)
    =
    task {
        printfn "🌐 TARS Agent - Workflow-of-Thoughts Mode"
        printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printfn $"Goal: %s{goal}"
        printfn ""

        match createLlmService config with
        | Result.Error msg ->
            printfn $"❌ %s{msg}"
            return 1
        | Result.Ok(llm, _) ->
            let llmWithEvidence, evidenceHandle = attachEvidence "wot" llm options

            let! evidence =
                match evidenceHandle with
                | Some handle ->
                    task {
                        let! value = handle
                        return Some value
                    }
                | None -> Task.FromResult None

            let logger msg =
                if options.Verbose then
                    printfn $"  [LOG] %s{msg}"
                else if msg.Contains("[WoT]") then
                    printfn $"  %s{msg}"

            let toolRegistry = ToolRegistry()
            toolRegistry.RegisterAssembly(typeof<ToolRegistry>.Assembly)

            let wotToolNames =
                match Environment.GetEnvironmentVariable("TARS_WOT_TOOLS") with
                | null
                | "" ->
                    [ "list_files"
                      "read_code"
                      "search_code"
                      "count_lines"
                      "find_todos"
                      "git_status"
                      "git_diff"
                      "summarize"
                      "analyze_code" ]
                | value when value.Trim().Equals("all", StringComparison.OrdinalIgnoreCase) ->
                    toolRegistry.GetAll() |> List.map (fun t -> t.Name)
                | value ->
                    value.Split([| ','; ';' |], StringSplitOptions.RemoveEmptyEntries)
                    |> Array.map (fun v -> v.Trim())
                    |> Array.toList

            let wotTools = wotToolNames |> List.choose toolRegistry.Get

            let ctx =
                let baseCtx = createAgentContext logger llm None

                { baseCtx with
                    Self = { baseCtx.Self with Tools = wotTools } }

            let baseConfig =
                { BranchingFactor = 3
                  MaxDepth = 3
                  TopK = 2
                  ScoreThreshold = 0.3
                  MinConfidence = 0.4
                  DiversityThreshold = 0.85
                  DiversityPenalty = 0.25
                  Constraints = []
                  EnableCritique = true
                  EnablePolicyChecks = true
                  EnableMemoryRecall = true
                  TrackEdges = true }

            let wotConfig: WoTConfig =
                { BaseConfig = baseConfig
                  RequiredPolicies = []
                  AvailableTools = wotTools |> List.map (fun t -> t.Name)
                  RoleAssignments = Map.empty
                  MemoryNamespace = None
                  MaxEscalations = 1
                  TimeoutMs = Some 300000 }

            printfn "📊 Configuration:"
            printfn $"   • Branching Factor: %d{baseConfig.BranchingFactor}"
            printfn $"   • Max Depth: %d{baseConfig.MaxDepth}"
            printfn $"   • Top-K: %d{baseConfig.TopK}"
            printfn $"   • Policy Checks: %b{baseConfig.EnablePolicyChecks}"
            printfn $"   • Tools: %d{wotTools.Length}"
            printfn ""
            printfn "🔄 Starting Workflow-of-Thoughts reasoning..."
            printfn ""

            let workflow = workflowOfThought llmWithEvidence wotConfig goal
            let! result = workflow ctx |> Async.StartAsTask

            printfn ""
            printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            match evidence with
            | Some(recorder, path) ->
                do! recorder.SaveToFileAsync(path) |> Async.StartAsTask
                printfn $"📒 Evidence saved: %s{path}"
            | None -> ()

            match result with
            | Success answer ->
                printfn "✅ Success!"
                printfn ""
                printfn $"%s{answer}"
                return 0
            | PartialSuccess(answer, warnings) ->
                printfn "⚠️ Partial Success"
                printfn ""
                printfn $"%s{answer}"

                if options.Verbose then
                    printfn ""
                    printfn "Warnings:"

                    for w in warnings do
                        printfn $"  - %A{w}"

                return 0
            | Failure errors ->
                printfn "❌ Failed"

                for e in errors do
                    printfn $"  - %A{e}"

                return 1
    }

/// Run via MAF: TarsWoTAgent + AgentOrchestrator + ToolAwareChatClient
let runMaf
    (config: Microsoft.Extensions.Configuration.IConfiguration)
    (options: AgentOptions)
    (goal: string)
    =
    task {
        printfn "🤖 TARS Agent - MAF Mode (Microsoft Agent Framework)"
        printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printfn $"Goal: %s{goal}"
        printfn ""

        match createLlmService config with
        | Result.Error msg ->
            printfn $"❌ %s{msg}"
            return 1
        | Result.Ok(llm, modelName) ->
            try
                let llmWithEvidence, evidenceHandle = attachEvidence "maf" llm options

                let! evidence =
                    match evidenceHandle with
                    | Some handle ->
                        task {
                            let! value = handle
                            return Some value
                        }
                    | None -> Task.FromResult None

                // Create tool registry
                let toolRegistry = ToolRegistry()
                toolRegistry.RegisterAssembly(typeof<ToolRegistry>.Assembly)
                let tools = toolRegistry :> IToolRegistry

                let logger msg =
                    if options.Verbose then
                        printfn $"  [LOG] %s{msg}"
                    else if msg.Contains("[WoT]") then
                        printfn $"  %s{msg}"

                // Create WoT executor
                let executor = WoTExecutor.createExecutor llmWithEvidence tools

                // Create pattern compiler
                let compiler = PatternCompiler.DefaultPatternCompiler() :> IPatternCompiler

                // Create history-aware pattern selector (boosted by golden traces)
                let selector = PatternSelector.HistoryAwareSelector() :> IPatternSelector

                // Create agent context
                let agentCtx = createAgentContext logger llmWithEvidence None

                // Create the MAF agent
                let agent = TarsWoTAgent(executor, compiler, selector, agentCtx)

                // Create orchestrator and register
                let orchestrator = AgentOrchestrator()
                orchestrator.Register(
                    agent,
                    [ "reasoning"; "analysis"; "coding"; "planning"; "workflow" ],
                    priority = 10)

                printfn $"📋 Model: %s{modelName}"
                printfn $"📋 Tools: %d{tools.GetAll().Length} registered"
                printfn $"📋 Agent: %s{agent.Name} (MAF)"
                printfn ""

                // Build MAF tool-aware pipeline
                let aiTools = MafToolAdapter.toAITools toolRegistry
                printfn $"🔧 MAF Tools: %d{aiTools.Length} adapted"
                printfn ""
                printfn "🔄 Executing via MAF orchestrator..."
                printfn ""

                let! result = orchestrator.Execute(goal, CancellationToken.None)

                printfn ""
                printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                printfn $"⏱️  Duration: %d{result.DurationMs}ms"
                printfn $"🤖 Agent: %s{result.AgentName}"

                match evidence with
                | Some(recorder, path) ->
                    do! recorder.SaveToFileAsync(path) |> Async.StartAsTask
                    printfn $"📒 Evidence saved: %s{path}"
                | None -> ()

                if result.Success then
                    printfn "✅ Success!"
                    printfn ""
                    printfn $"%s{result.Response}"
                    return 0
                else
                    printfn "❌ Failed"
                    printfn ""
                    printfn $"%s{result.Response}"
                    return 1
            with ex ->
                printfn $"❌ Exception: %s{ex.Message}"
                if options.Verbose && ex.InnerException <> null then
                    printfn $"   Inner: %s{ex.InnerException.Message}"
                return 1
    }

/// Run a sequential multi-agent pipeline
let runPipeline
    (config: Microsoft.Extensions.Configuration.IConfiguration)
    (options: AgentOptions)
    (goal: string)
    =
    task {
        printfn "🔗 TARS Agent - Pipeline Mode (Analyzer -> Coder -> Reviewer)"
        printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printfn $"Goal: %s{goal}"
        printfn ""

        match createLlmService config with
        | Result.Error msg ->
            printfn $"❌ %s{msg}"
            return 1
        | Result.Ok(llm, _modelName) ->
            try
                let llmWithEvidence, evidenceHandle = attachEvidence "pipeline" llm options

                let! evidence =
                    match evidenceHandle with
                    | Some handle ->
                        task {
                            let! value = handle
                            return Some value
                        }
                    | None -> Task.FromResult None

                let toolRegistry = ToolRegistry()
                toolRegistry.RegisterAssembly(typeof<ToolRegistry>.Assembly)
                let tools = toolRegistry :> IToolRegistry

                let logger msg =
                    if options.Verbose then
                        printfn $"  [LOG] %s{msg}"

                let executor = WoTExecutor.createExecutor llmWithEvidence tools
                let compiler = PatternCompiler.DefaultPatternCompiler() :> IPatternCompiler

                let cotSelector =
                    { new IPatternSelector with
                        member _.Recommend(_goal, _state) = ChainOfThought
                        member _.Score(_goal) = [ ChainOfThought, 1.0 ] |> Map.ofList }

                let reactSelector =
                    { new IPatternSelector with
                        member _.Recommend(_goal, _state) = ReAct
                        member _.Score(_goal) = [ ReAct, 1.0 ] |> Map.ofList }

                let analyzerCtx = createAgentContext logger llmWithEvidence None
                let coderCtx = createAgentContext logger llmWithEvidence None
                let reviewerCtx = createAgentContext logger llmWithEvidence None

                let analyzer = TarsWoTAgent(executor, compiler, cotSelector, analyzerCtx, name = "Analyzer", description = "Analyzes problems and identifies patterns")
                let coder = TarsWoTAgent(executor, compiler, reactSelector, coderCtx, name = "Coder", description = "Implements solutions using ReAct")
                let reviewer = TarsWoTAgent(executor, compiler, cotSelector, reviewerCtx, name = "Reviewer", description = "Reviews and critiques solutions")

                let orchestrator = AgentOrchestrator()
                orchestrator.Register(analyzer, [ "analysis"; "patterns"; "understanding" ], priority = 10)
                orchestrator.Register(coder, [ "coding"; "implementation"; "tools" ], priority = 10)
                orchestrator.Register(reviewer, [ "review"; "critique"; "quality" ], priority = 10)

                printfn "🔄 Running pipeline: Analyzer -> Coder -> Reviewer"
                printfn ""

                let! results = orchestrator.Pipeline(["Analyzer"; "Coder"; "Reviewer"], goal)

                printfn ""
                printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

                match evidence with
                | Some(recorder, path) ->
                    do! recorder.SaveToFileAsync(path) |> Async.StartAsTask
                    printfn $"📒 Evidence saved: %s{path}"
                | None -> ()

                for r in results do
                    printfn $"🤖 Agent: %s{r.AgentName} | Duration: %d{r.DurationMs}ms | Success: %b{r.Success}"
                    printfn $"   %s{r.Response.[..min 200 (r.Response.Length - 1)]}"
                    printfn ""

                let allSuccess = results |> List.forall (fun r -> r.Success)
                if allSuccess then
                    printfn "✅ Pipeline completed successfully!"
                    return 0
                else
                    printfn "❌ Pipeline had failures"
                    return 1
            with ex ->
                printfn $"❌ Exception: %s{ex.Message}"
                if options.Verbose && ex.InnerException <> null then
                    printfn $"   Inner: %s{ex.InnerException.Message}"
                return 1
    }

/// Run agents in parallel (fan-out) on the same goal
let runFanOut
    (config: Microsoft.Extensions.Configuration.IConfiguration)
    (options: AgentOptions)
    (goal: string)
    =
    task {
        printfn "🌀 TARS Agent - Fan-Out Mode (Analyzer | Coder | Reviewer)"
        printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printfn $"Goal: %s{goal}"
        printfn ""

        match createLlmService config with
        | Result.Error msg ->
            printfn $"❌ %s{msg}"
            return 1
        | Result.Ok(llm, _modelName) ->
            try
                let llmWithEvidence, evidenceHandle = attachEvidence "fanout" llm options

                let! evidence =
                    match evidenceHandle with
                    | Some handle ->
                        task {
                            let! value = handle
                            return Some value
                        }
                    | None -> Task.FromResult None

                let toolRegistry = ToolRegistry()
                toolRegistry.RegisterAssembly(typeof<ToolRegistry>.Assembly)
                let tools = toolRegistry :> IToolRegistry

                let logger msg =
                    if options.Verbose then
                        printfn $"  [LOG] %s{msg}"

                let executor = WoTExecutor.createExecutor llmWithEvidence tools
                let compiler = PatternCompiler.DefaultPatternCompiler() :> IPatternCompiler

                let cotSelector =
                    { new IPatternSelector with
                        member _.Recommend(_goal, _state) = ChainOfThought
                        member _.Score(_goal) = [ ChainOfThought, 1.0 ] |> Map.ofList }

                let reactSelector =
                    { new IPatternSelector with
                        member _.Recommend(_goal, _state) = ReAct
                        member _.Score(_goal) = [ ReAct, 1.0 ] |> Map.ofList }

                let analyzerCtx = createAgentContext logger llmWithEvidence None
                let coderCtx = createAgentContext logger llmWithEvidence None
                let reviewerCtx = createAgentContext logger llmWithEvidence None

                let analyzer = TarsWoTAgent(executor, compiler, cotSelector, analyzerCtx, name = "Analyzer", description = "Analyzes problems and identifies patterns")
                let coder = TarsWoTAgent(executor, compiler, reactSelector, coderCtx, name = "Coder", description = "Implements solutions using ReAct")
                let reviewer = TarsWoTAgent(executor, compiler, cotSelector, reviewerCtx, name = "Reviewer", description = "Reviews and critiques solutions")

                let orchestrator = AgentOrchestrator()
                orchestrator.Register(analyzer, [ "analysis"; "patterns"; "understanding" ], priority = 10)
                orchestrator.Register(coder, [ "coding"; "implementation"; "tools" ], priority = 10)
                orchestrator.Register(reviewer, [ "review"; "critique"; "quality" ], priority = 10)

                printfn "🔄 Running fan-out: Analyzer | Coder | Reviewer (parallel)"
                printfn ""

                let! results = orchestrator.FanOut(["Analyzer"; "Coder"; "Reviewer"], goal)

                printfn ""
                printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

                match evidence with
                | Some(recorder, path) ->
                    do! recorder.SaveToFileAsync(path) |> Async.StartAsTask
                    printfn $"📒 Evidence saved: %s{path}"
                | None -> ()

                for r in results do
                    printfn $"🤖 Agent: %s{r.AgentName} | Duration: %d{r.DurationMs}ms | Success: %b{r.Success}"
                    printfn $"   %s{r.Response.[..min 200 (r.Response.Length - 1)]}"
                    printfn ""

                let allSuccess = results |> List.forall (fun r -> r.Success)
                if allSuccess then
                    printfn "✅ Fan-out completed successfully!"
                    return 0
                else
                    printfn "❌ Fan-out had failures"
                    return 1
            with ex ->
                printfn $"❌ Exception: %s{ex.Message}"
                if options.Verbose && ex.InnerException <> null then
                    printfn $"   Inner: %s{ex.InnerException.Message}"
                return 1
    }

/// Main entry point for agent command
let run
    (config: Microsoft.Extensions.Configuration.IConfiguration)
    (subCommand: string)
    (args: string list)
    (options: AgentOptions)
    =
    match subCommand with
    | "react" when args.Length > 0 ->
        let goal = String.Join(" ", args)
        runReact config options goal
    | "cot" when args.Length > 0 ->
        let input = String.Join(" ", args)
        runChainOfThought config options input
    | "got" when args.Length > 0 ->
        let goal = String.Join(" ", args)
        runGraphOfThoughts config options goal
    | "tot" when args.Length > 0 ->
        let goal = String.Join(" ", args)
        runTreeOfThoughts config options goal
    | "wot" when args.Length > 0 ->
        let goal = String.Join(" ", args)
        runWorkflowOfThoughts config options goal
    | "run" when args.Length > 0 ->
        let goal = String.Join(" ", args)
        runMaf config options goal
    | "pipeline" when args.Length > 0 ->
        let goal = String.Join(" ", args)
        runPipeline config options goal
    | "fanout" when args.Length > 0 ->
        let goal = String.Join(" ", args)
        runFanOut config options goal
    | "help"
    | _ ->
        printfn "TARS Agent Commands"
        printfn "━━━━━━━━━━━━━━━━━━━━"
        printfn ""
        printfn "  tars agent run <goal>       Run via MAF (orchestrated WoT + tools)"
        printfn "  tars agent pipeline <goal>  Run sequential pipeline (Analyzer -> Coder -> Reviewer)"
        printfn "  tars agent fanout <goal>    Run parallel fan-out (Analyzer | Coder | Reviewer)"
        printfn "  tars agent react <goal>     Run ReAct (Reason-Act-Observe) loop"
        printfn "  tars agent cot <input>      Run Chain of Thought reasoning"
        printfn "  tars agent got <goal>       Run Graph of Thoughts reasoning"
        printfn "  tars agent tot <goal>       Run Tree of Thoughts reasoning"
        printfn "  tars agent wot <goal>       Run Workflow of Thoughts reasoning"
        printfn ""
        printfn "Options:"
        printfn "  --max-steps N               Maximum reasoning steps (default: 10)"
        printfn "  --verbose                   Show detailed logs"
        printfn "  --model <name>              Use specific model"
        printfn "  --evidence <file|dir>       Write LLM request/response trace to JSON"
        printfn ""
        printfn "Examples:"
        printfn "  tars agent run \"Analyze this codebase and suggest improvements\""
        printfn "  tars agent pipeline \"Design a caching layer for the API\""
        printfn "  tars agent fanout \"Review this module for bugs and improvements\""
        printfn "  tars agent react \"What is 5 + 7?\""
        printfn "  tars agent cot \"Explain quantum computing\" --verbose"
        printfn "  tars agent got \"Design a REST API for a todo app\""
        printfn "  tars agent tot \"Solve the light bulb logic puzzle\""
        System.Threading.Tasks.Task.FromResult(0)
