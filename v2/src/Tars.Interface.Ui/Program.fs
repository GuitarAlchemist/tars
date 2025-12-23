module Tars.Interface.Ui.Program

open System
open Microsoft.AspNetCore.Builder
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Hosting
open Microsoft.AspNetCore.Hosting
open Bolero.Server
open Bolero.Html
open Bolero.Server.Html
open Serilog
open Tars.Core
open Tars.Kernel
open Tars.Interface.Ui.Main
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Llm.Routing
open Tars.Cortex
open Tars.Tools
open Tars.Tools.Standard

/// Generate the host page with Bolero HTML functions
let hostPage =
    doctypeHtml {
        head {
            meta { attr.charset "utf-8" }

            meta {
                attr.name "viewport"
                attr.content "width=device-width, initial-scale=1.0"
            }

            title { "TARS v2 - Dashboard" }
            ``base`` { attr.href "/" }

            link {
                attr.rel "stylesheet"
                attr.href "css/app.css"
            }
        }

        body {
            div {
                attr.id "main"
                comp<TarsApp>
            }

            boleroScript
        }
    }

[<EntryPoint>]
let main args =
    // Configure Serilog
    let logger =
        LoggerConfiguration().MinimumLevel.Debug().WriteTo.Console().CreateLogger()

    Log.Logger <- logger

    let builder = WebApplication.CreateBuilder(args)

    // Force static web assets to resolve blazor.server.js
    builder.WebHost.UseStaticWebAssets() |> ignore

    // Add services for server-side Bolero
    builder.Services.AddControllersWithViews() |> ignore
    builder.Services.AddRazorPages() |> ignore
    builder.Services.AddServerSideBlazor() |> ignore
    builder.Services.AddBoleroHost(server = true, prerendered = true) |> ignore

    // Register EventBus as singleton
    let eventBus = EventBus(logger :> Serilog.ILogger) :> IEventBus
    builder.Services.AddSingleton<IEventBus>(eventBus) |> ignore

    // Initialize Tools
    let toolRegistry = ToolRegistry()
    toolRegistry.RegisterAssembly(typeof<ToolRegistry>.Assembly)
    let availableTools = toolRegistry.GetAll()

    // Register ToolRegistry for chat to use
    builder.Services.AddSingleton<IToolRegistry>(toolRegistry :> IToolRegistry)
    |> ignore

    // Register LLM Services
    builder.Services.AddHttpClient() |> ignore

    let routingConfig =
        { OllamaBaseUri = Uri "http://localhost:11434"
          VllmBaseUri = Uri "http://localhost:8000"
          OpenAIBaseUri = Uri "https://api.openai.com/v1"
          GoogleGeminiBaseUri = Uri "https://generativelanguage.googleapis.com"
          AnthropicBaseUri = Uri "https://api.anthropic.com"
          DockerModelRunnerBaseUri = Some(Uri "http://localhost:12434/v1")
          LlamaCppBaseUri = Some(Uri "http://localhost:8080")
          // Recommended thinking models: qwen3:14b, deepseek-r1:14b, magistral
          DefaultOllamaModel = "qwen3:14b"
          DefaultVllmModel = "facebook/opt-125m"
          DefaultOpenAIModel = "gpt-4"
          DefaultGoogleGeminiModel = "gemini-pro"
          DefaultAnthropicModel = "claude-3-opus-20240229"
          DefaultDockerModelRunnerModel = Some "gpt-3.5-turbo"
          // llama.cpp running at http://localhost:8080 with Qwen3-8B
          DefaultLlamaCppModel = Some "Qwen3-8B-Q4_K_M.gguf"
          DefaultEmbeddingModel = "nomic-embed-text"
          OllamaKey = None
          VllmKey = None
          OpenAIKey = Option.ofObj (Environment.GetEnvironmentVariable("OPENAI_API_KEY"))
          GoogleGeminiKey = Option.ofObj (Environment.GetEnvironmentVariable("GOOGLE_API_KEY"))
          AnthropicKey = Option.ofObj (Environment.GetEnvironmentVariable("ANTHROPIC_API_KEY"))
          DockerModelRunnerKey = None
          LlamaCppKey = None }

    let llmConfig = { Routing = routingConfig }
    builder.Services.AddSingleton(llmConfig) |> ignore
    builder.Services.AddHttpClient<ILlmService, DefaultLlmService>() |> ignore

    // Register Agent Registry
    let registry = AgentRegistry()

    // Seed System Agent
    let systemAgent =
        { Id = AgentId(Guid.NewGuid())
          Name = "TARS System"
          Version = "2.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = "llama3.2"
          SystemPrompt = "You are TARS."
          Tools = availableTools
          Capabilities =
            [ { Kind = CapabilityKind.Reasoning
                Description = "Core Reasoning"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.65
                Reputation = Some 0.5 }
              { Kind = CapabilityKind.Planning
                Description = "Task Planning"
                InputSchema = None
                OutputSchema = None
                Confidence = Some 0.62
                Reputation = Some 0.5 } ]
          State = AgentState.Idle
          Memory = [] }

    registry.Register(systemAgent)

    builder.Services.AddSingleton<IAgentRegistry>(registry) |> ignore
    Tars.Tools.Standard.AgentTools.setRegistry (registry)

    // Register Vector Store
    let pgConn =
        "Host=localhost;Port=5432;Database=tars_memory;Username=postgres;Password=tars_password"

    builder.Services.AddSingleton<IVectorStore>(fun _ -> PostgresVectorStore(pgConn) :> IVectorStore)
    |> ignore

    // Capability store backed by the vector store for semantic routing
    builder.Services.AddSingleton<ICapabilityStore>(fun sp ->
        let vectorStore = sp.GetRequiredService<IVectorStore>()
        let llm = sp.GetRequiredService<ILlmService>()
        CapabilityStore(vectorStore, llm) :> ICapabilityStore)
    |> ignore

    // Register Knowledge Graph
    let dataDir = System.IO.Path.Combine(AppContext.BaseDirectory, "data")

    builder.Services.AddSingleton<IGraphService>(fun _ ->
        let service = InternalGraphService(dataDir)
        // Seed initial fact to ensure not empty
        let tars =
            ConceptE
                { Name = "TARS"
                  Description = "Autonomous Reasoning System"
                  RelatedConcepts = [] }

        let ai =
            ConceptE
                { Name = "Artificial Intelligence"
                  Description = "Machine Intelligence"
                  RelatedConcepts = [] }

        let fact = DerivedFrom(tars, ai)
        (service :> IGraphService).AddFactAsync(fact).GetAwaiter().GetResult() |> ignore
        service :> IGraphService)
    |> ignore

    let app = builder.Build()

    // Register system agent capabilities into capability store for warm start
    let capStoreOpt = app.Services.GetService<ICapabilityStore>()

    capStoreOpt
    |> Option.ofObj
    |> Option.iter (fun store ->
        match store with
        | :? CapabilityStore as concrete ->
            for cap in systemAgent.Capabilities do
                concrete.RegisterAsync(systemAgent.Id, cap).GetAwaiter().GetResult()
        | _ -> ())

    // Configure pipeline
    app.UseStaticFiles() |> ignore
    app.UseRouting() |> ignore
    app.UseAntiforgery() |> ignore

    // Map endpoints
    app.MapRazorPages() |> ignore
    app.MapControllers() |> ignore
    app.MapBlazorHub() |> ignore
    app.MapFallbackToBolero(hostPage) |> ignore

    Log.Information("Starting TARS UI on http://localhost:5000")
    app.Run()
    0
