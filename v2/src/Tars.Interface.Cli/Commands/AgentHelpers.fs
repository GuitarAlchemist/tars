module Tars.Interface.Cli.Commands.AgentHelpers

open System
open System.IO
open System.Net.Http
open System.Threading
open Tars.Core
open Tars.Cortex
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Kernel

/// Options for the agent command
type AgentOptions =
    { MaxSteps: int
      Verbose: bool
      Model: string option
      EvidencePath: string option }

let defaultOptions =
    { MaxSteps = 10
      Verbose = false
      Model = None
      EvidencePath = None }

let ensureEvidencePath (path: string) (label: string) =
    let finalPath =
        if Path.HasExtension(path) then
            path
        else
            let fileName =
                $"tars-{label}-{DateTime.UtcNow:yyyyMMddHHmmss}.json"

            Path.Combine(path, fileName)

    let directory = Path.GetDirectoryName(finalPath)
    if not (String.IsNullOrWhiteSpace(directory)) then
        Directory.CreateDirectory(directory) |> ignore

    finalPath

let attachEvidence (label: string) (llm: ILlmService) (options: AgentOptions) =
    match options.EvidencePath with
    | None -> llm, None
    | Some path ->
        let traceRecorder = TraceRecorder()
        let evidencePath = ensureEvidencePath path label
        let traced = TracingLlmService(llm, traceRecorder) :> ILlmService
        let traceHandle =
            task {
                let! _ = (traceRecorder :> ITraceRecorder).StartTraceAsync() |> Async.StartAsTask
                return traceRecorder, evidencePath
            }

        traced, Some traceHandle

/// Create an LLM service with the given configuration
let createLlmService (config: Microsoft.Extensions.Configuration.IConfiguration) =
    // Check for llama.cpp first (preferred for local)
    let llamaCppUrl =
        config["Llm:LlamaCppUrl"]
        |> Option.ofObj
        |> Option.orElse (config["LlamaCppUrl"] |> Option.ofObj)
        |> Option.orElse (
            // Also check for OLLAMA_BASE_URL pointing to llama.cpp port
            config["OLLAMA_BASE_URL"]
            |> Option.ofObj
            |> Option.filter (fun url -> url.Contains("8080"))
        )

    let ollamaUrl =
        config["Llm:BaseUrl"]
        |> Option.ofObj
        |> Option.orElse (config["OLLAMA_BASE_URL"] |> Option.ofObj)

    let provider =
        config["Llm:Provider"]
        |> Option.ofObj
        |> Option.map (fun s -> s.ToLowerInvariant())

    let llamaSharpModelPath =
        config["Llm:LlamaSharpModelPath"]
        |> Option.ofObj

    let defaultModel =
        config["Llm:Model"]
        |> Option.ofObj
        |> Option.orElse (config["DEFAULT_OLLAMA_MODEL"] |> Option.ofObj)
        |> Option.orElse (Some "magistral")

    let contextWindow =
        config["Llm:ContextWindow"]
        |> Option.ofObj
        |> Option.bind (fun s ->
            match Int32.TryParse(s) with
            | true, v -> Some v
            | _ -> None
        )

    let temperature =
        config["Llm:Temperature"]
        |> Option.ofObj
        |> Option.bind (fun s ->
            match Double.TryParse(s) with
            | true, v -> Some v
            | _ -> None
        )

    // Determine which backend to use based on provider if available
    let serviceResult =
        match provider with
        | Some "llamasharp" when llamaSharpModelPath.IsSome ->
            let routingCfg: RoutingConfig =
                { RoutingConfig.Default with
                    DefaultOllamaModel = defaultModel |> Option.defaultValue "magistral"
                    LlamaSharpModelPath = llamaSharpModelPath
                    DefaultContextWindow = contextWindow
                    DefaultTemperature = temperature }

            let svcCfg: LlmServiceConfig = { Routing = routingCfg }
            let httpClient = new HttpClient()
            httpClient.Timeout <- TimeSpan.FromSeconds(300.0)
            let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService
            Result.Ok(llmService, "llama-sharp")

        | Some "llamacpp" when llamaCppUrl.IsSome ->
            let model = defaultModel |> Option.defaultValue "magistral"
            let routingCfg: RoutingConfig =
                { RoutingConfig.Default with
                    DefaultOllamaModel = model
                    DefaultLlamaCppModel = Some model
                    LlamaCppBaseUri = Some(Uri(llamaCppUrl.Value))
                    DefaultContextWindow = contextWindow
                    DefaultTemperature = temperature }

            let svcCfg: LlmServiceConfig = { Routing = routingCfg }
            let httpClient = new HttpClient()
            httpClient.Timeout <- TimeSpan.FromSeconds(300.0)
            let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService
            Result.Ok(llmService, model)

        | Some "ollama" when ollamaUrl.IsSome ->
            let model = defaultModel |> Option.defaultValue "magistral"
            let routingCfg: RoutingConfig =
                { RoutingConfig.Default with
                    OllamaBaseUri = Uri(ollamaUrl.Value)
                    DefaultOllamaModel = model
                    DefaultContextWindow = contextWindow
                    DefaultTemperature = temperature }

            let svcCfg: LlmServiceConfig = { Routing = routingCfg }
            let httpClient = new HttpClient()
            httpClient.Timeout <- TimeSpan.FromSeconds(300.0)
            let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService
            Result.Ok(llmService, model)

        | _ ->
            // Fallback to legacy priority matching
            match llamaSharpModelPath, llamaCppUrl, ollamaUrl, defaultModel with
            | Some modelPath, _, _, _ ->
                let routingCfg: RoutingConfig =
                    { RoutingConfig.Default with
                        DefaultOllamaModel = defaultModel |> Option.defaultValue "magistral"
                        LlamaSharpModelPath = Some modelPath
                        DefaultContextWindow = contextWindow
                        DefaultTemperature = temperature }

                let svcCfg: LlmServiceConfig = { Routing = routingCfg }
                let httpClient = new HttpClient()
                httpClient.Timeout <- TimeSpan.FromSeconds(300.0)
                let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService
                Result.Ok(llmService, "llama-sharp")

            | _, Some llamaUrl, _, Some model ->
                let routingCfg: RoutingConfig =
                    { RoutingConfig.Default with
                        DefaultOllamaModel = model
                        DefaultLlamaCppModel = Some model
                        LlamaCppBaseUri = Some(Uri(llamaUrl))
                        DefaultContextWindow = contextWindow
                        DefaultTemperature = temperature }

                let svcCfg: LlmServiceConfig = { Routing = routingCfg }
                let httpClient = new HttpClient()
                httpClient.Timeout <- TimeSpan.FromSeconds(300.0)
                let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService
                Result.Ok(llmService, model)

            | _, None, Some url, Some model ->
                let routingCfg: RoutingConfig =
                    { RoutingConfig.Default with
                        OllamaBaseUri = Uri(url)
                        DefaultOllamaModel = model
                        DefaultContextWindow = contextWindow
                        DefaultTemperature = temperature }

                let svcCfg: LlmServiceConfig = { Routing = routingCfg }
                let httpClient = new HttpClient()
                httpClient.Timeout <- TimeSpan.FromSeconds(300.0)
                let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmService
                Result.Ok(llmService, model)

            | _, _, _, _ -> Result.Error "Missing LLM configuration. Set Provider, LlamaSharpModelPath, LlamaCppUrl, or OLLAMA_BASE_URL."

    serviceResult

let createAgentContext (logger: string -> unit) (llm: ILlmService) (audit: ReasoningAudit option) =
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
        Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "knowledge", "graph")

    let graph = InternalGraphService(graphPath) :> IGraphService

    let capabilityStore =
        let tarsHome =
            Environment.GetEnvironmentVariable("TARS_HOME")
            |> Option.ofObj
            |> Option.defaultValue (
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars")
            )

        let capDir = Path.Combine(tarsHome, "capability_store")
        Directory.CreateDirectory(capDir) |> ignore
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
      SymbolicReflector = None
      CapabilityStore = Some capabilityStore
      Audit = audit
      CancellationToken = CancellationToken.None }
