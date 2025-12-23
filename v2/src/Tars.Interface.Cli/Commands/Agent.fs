module Tars.Interface.Cli.Commands.Agent

open System
open System.IO
open System.Net.Http
open System.Threading
open Tars.Core
open Tars.Cortex
open Tars.Cortex.Patterns
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Tools
open Tars.Kernel

/// Options for the agent command
type AgentOptions =
    { MaxSteps: int
      Verbose: bool
      Model: string option }

let defaultOptions =
    { MaxSteps = 10
      Verbose = false
      Model = None }

/// Create an LLM service with the given configuration
let private createLlmService (config: Microsoft.Extensions.Configuration.IConfiguration) =
    let ollamaUrl = config["OLLAMA_BASE_URL"] |> Option.ofObj
    let defaultModel = config["DEFAULT_OLLAMA_MODEL"] |> Option.ofObj

    match ollamaUrl, defaultModel with
    | None, _ -> Result.Error "Missing OLLAMA_BASE_URL (set via user secrets or env)."
    | _, None -> Result.Error "Missing DEFAULT_OLLAMA_MODEL (set via user secrets or env)."
    | Some url, Some model ->
        let routingCfg: RoutingConfig =
            { OllamaBaseUri = Uri(url)
              VllmBaseUri = Uri("http://localhost:8000/")
              OpenAIBaseUri = Uri("https://api.openai.com/")
              GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
              AnthropicBaseUri = Uri("https://api.anthropic.com/")
              DefaultOllamaModel = model
              DefaultVllmModel = model
              DefaultOpenAIModel = "gpt-4o"
              DefaultGoogleGeminiModel = "gemini-pro"
              DefaultAnthropicModel = "claude-3-opus-20240229"
              DefaultEmbeddingModel = "nomic-embed-text"
              OllamaKey = None
              VllmKey = None
              OpenAIKey = None
              GoogleGeminiKey = None
              AnthropicKey = None
              DockerModelRunnerBaseUri = None
              LlamaCppBaseUri = None
              DefaultDockerModelRunnerModel = None
              DefaultLlamaCppModel = None
              DockerModelRunnerKey = None
              LlamaCppKey = None }

        let svcCfg: LlmServiceConfig = { Routing = routingCfg }
        let httpClient = new HttpClient()
        httpClient.Timeout <- TimeSpan.FromSeconds(300.0)
        let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService
        Result.Ok(llmService, model)

/// Create an AgentContext for running patterns
let private createAgentContext (logger: string -> unit) (llm: ILlmService) =
    let agent: Agent =
        { Id = AgentId(Guid.NewGuid())
          Name = "TARS Agent"
          Version = "2.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = "default"
          SystemPrompt = "You are TARS, an autonomous reasoning agent."
          Tools = []
          Capabilities =
            [ { Kind = CapabilityKind.Reasoning
                Description = "General Reasoning"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.6
                Reputation = Some 0.5 }
              { Kind = CapabilityKind.TaskExecution
                Description = "Task Execution"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.6
                Reputation = Some 0.5 } ]
          State = AgentState.Idle
          Memory = [] }

    let mockRegistry =
        { new IAgentRegistry with
            member _.GetAgent(_) = async { return None }
            member _.FindAgents(_) = async { return [] }
            member _.GetAllAgents() = async { return [] } }

    let mockExecutor =
        { new IAgentExecutor with
            member _.Execute(_, _) =
                async { return Success "Not implemented" } }

    // Initialize semantic memory + knowledge graph for richer context
    let storageRoot =
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".tars",
            "knowledge",
            "semantic_memory"
        )

    let embedder (text: string) = llm.EmbedAsync text |> Async.AwaitTask
    let kernel = KernelBootstrap.createKernel storageRoot embedder llm

    let graphPath =
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".tars",
            "knowledge",
            "graph"
        )

    let graph = InternalGraphService(graphPath) :> IGraphService

    let capabilityStore =
        let tarsHome =
            Environment.GetEnvironmentVariable("TARS_HOME")
            |> Option.ofObj
            |> Option.defaultValue (Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars"))

        let capDir = Path.Combine(tarsHome, "capability_store")
        Directory.CreateDirectory capDir |> ignore
        let vectorPath = Path.Combine(capDir, "capabilities.sqlite")
        let vectorStore = SqliteVectorStore(vectorPath) :> IVectorStore
        CapabilityStore(vectorStore, llm) :> ICapabilityStore

    { Self = agent
      Registry = mockRegistry
      Executor = mockExecutor
      Logger = logger
      Budget = None
      Epistemic = None
      SemanticMemory = Some kernel.SemanticMemory
      KnowledgeGraph = Some graph
      CapabilityStore = Some capabilityStore
      CancellationToken = CancellationToken.None }

/// Run the ReAct pattern
let runReact (config: Microsoft.Extensions.Configuration.IConfiguration) (options: AgentOptions) (goal: string) =
    task {
        printfn "🤖 TARS Agent - ReAct Mode"
        printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        printfn "Goal: %s" goal
        printfn "Max Steps: %d" options.MaxSteps
        printfn ""

        match createLlmService config with
        | Result.Error msg ->
            printfn "❌ %s" msg
            return 1
        | Result.Ok(llm, _model) ->
            // Create tool registry with standard tools
            let toolRegistry = ToolRegistry()

            // Register some basic tools
            let echoTool: Tool =
                { Name = "echo"
                  Description = "Echoes the input back. Use for testing."
                  Version = "1.0.0"
                  ParentVersion = None
                  CreatedAt = DateTime.UtcNow
                  Execute = fun input -> async { return Result.Ok(sprintf "Echo: %s" input) } }

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
                                        return Result.Error(sprintf "Unknown operator: %s" op)
                                    else
                                        return Result.Ok(sprintf "%g" result)
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
                    printfn "  [LOG] %s" msg

            let ctx = createAgentContext logger llm

            printfn "📋 Available Tools:"

            for tool in tools.GetAll() do
                printfn "   • %s: %s" tool.Name tool.Description

            printfn ""
            printfn "🔄 Starting ReAct loop..."
            printfn ""

            // Run ReAct pattern
            let workflow = reAct llm tools options.MaxSteps goal
            let! result = workflow ctx |> Async.StartAsTask

            printfn ""
            printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            match result with
            | Success answer ->
                printfn "✅ Success!"
                printfn ""
                printfn "Answer: %s" answer
                return 0
            | PartialSuccess(answer, warnings) ->
                printfn "⚠️ Partial Success (with warnings)"
                printfn ""
                printfn "Answer: %s" answer
                printfn ""
                printfn "Warnings:"

                for w in warnings do
                    printfn "  - %A" w

                return 0
            | Failure errors ->
                printfn "❌ Failed"
                printfn ""
                printfn "Errors:"

                for e in errors do
                    printfn "  - %A" e

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
        printfn "Input: %s" input
        printfn ""

        match createLlmService config with
        | Result.Error msg ->
            printfn "❌ %s" msg
            return 1
        | Result.Ok(llm, _) ->
            let logger msg =
                if options.Verbose then
                    printfn "  [LOG] %s" msg

            let ctx = createAgentContext logger llm

            // Define reasoning steps
            let analyze = llmStep llm "Analyze the problem and identify key aspects."
            let reason = llmStep llm "Reason about solutions step by step."
            let summarize = llmStep llm "Summarize your analysis and provide a final answer."

            printfn "🔄 Running Chain of Thought..."
            printfn "  Step 1: Analyze"
            printfn "  Step 2: Reason"
            printfn "  Step 3: Summarize"
            printfn ""

            let workflow = chainOfThought [ analyze; reason; summarize ] input
            let! result = workflow ctx |> Async.StartAsTask

            printfn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

            match result with
            | Success answer ->
                printfn "✅ Success!"
                printfn ""
                printfn "%s" answer
                return 0
            | PartialSuccess(answer, _) ->
                printfn "⚠️ Partial Success"
                printfn ""
                printfn "%s" answer
                return 0
            | Failure errors ->
                printfn "❌ Failed"

                for e in errors do
                    printfn "  - %A" e

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
    | "help"
    | _ ->
        printfn "TARS Agent Commands"
        printfn "━━━━━━━━━━━━━━━━━━━━"
        printfn ""
        printfn "  tars agent react <goal>     Run ReAct (Reason-Act-Observe) loop"
        printfn "  tars agent cot <input>      Run Chain of Thought reasoning"
        printfn ""
        printfn "Options:"
        printfn "  --max-steps N               Maximum reasoning steps (default: 10)"
        printfn "  --verbose                   Show detailed logs"
        printfn "  --model <name>              Use specific model"
        printfn ""
        printfn "Examples:"
        printfn "  tars agent react \"What is 5 + 7?\""
        printfn "  tars agent cot \"Explain quantum computing\" --verbose"
        System.Threading.Tasks.Task.FromResult(0)
