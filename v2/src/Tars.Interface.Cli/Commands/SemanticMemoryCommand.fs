module Tars.Interface.Cli.Commands.SemanticMemoryCommand

open System
open System.IO
open System.Threading.Tasks
open Tars.Core
open Tars.Kernel
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Llm.Routing
open Tars.Cortex
open Microsoft.Extensions.Configuration
open Spectre.Console

let private createEmbedder (config: IConfiguration) : Embedder =
    // Initialize LLM service for embeddings
    let ollamaUrl = config["OLLAMA_BASE_URL"] |> Option.ofObj |> Option.defaultValue "http://localhost:11434"
    let model = config["DEFAULT_OLLAMA_MODEL"] |> Option.ofObj |> Option.defaultValue "nomic-embed-text"
    
    let routingCfg = {
        OllamaBaseUri = Uri(ollamaUrl)
        VllmBaseUri = Uri("http://localhost:8000")
        OpenAIBaseUri = Uri("https://api.openai.com")
        GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com")
        AnthropicBaseUri = Uri("https://api.anthropic.com")
        DefaultOllamaModel = model
        DefaultVllmModel = model
        DefaultOpenAIModel = "text-embedding-3-small"
        DefaultGoogleGeminiModel = "embedding-001"
        DefaultAnthropicModel = "claude-3-opus-20240229"
        DefaultEmbeddingModel = model
    }
    
    let svcCfg = { Routing = routingCfg }
    let httpClient = new System.Net.Http.HttpClient()
    let llmService = DefaultLlmService(httpClient, svcCfg) :> ILlmServiceFunctional
    
    fun text -> async {
        let! result = llmService.EmbedAsync text
        match result with
        | Result.Ok embedding -> return embedding
        | Result.Error err -> 
            printfn "Embedding error: %A" err
            return Array.empty<float32>
    }

let run (config: IConfiguration) (args: string array) =
    task {
        let storageRoot = Path.Combine(Environment.CurrentDirectory, "knowledge", "semantic_memory")
        let embedder = createEmbedder config
        let kernel = KernelBootstrap.createKernel storageRoot embedder
        
        match args with
        | [| "query"; text |] ->
            let query = {
                TaskId = ""
                TaskKind = ""
                TextContext = text
                Tags = []
            }
            let! results = kernel.SemanticMemory.Retrieve query
            printfn "Found %d results:" results.Length
            for schema in results do
                printfn "- [%s] %s" schema.Id (schema.Logical |> Option.map (fun l -> l.ProblemSummary) |> Option.defaultValue "No summary")
            return 0
            
        | [| "grow" |] ->
            // Dummy grow
            let! id = kernel.SemanticMemory.Grow(obj(), obj())
            printfn "Created memory schema: %s" id
            return 0
            
        | [| "refine" |] ->
            do! kernel.SemanticMemory.Refine()
            printfn "Refinement complete."
            return 0
            
        | [| "demo-perceptual" |] ->
            AnsiConsole.MarkupLine("[bold cyan]Ingesting source code from 'src'...[/]")
            let kg = KnowledgeGraph()
            let srcDir = Path.Combine(Environment.CurrentDirectory, "src")
            if Directory.Exists srcDir then
                do! AstIngestor.ingestDirectory kg srcDir |> Async.StartAsTask
                let cs = AstIngestor.extractCodeStructure kg
                
                let table = Table()
                table.AddColumn("Category") |> ignore
                table.AddColumn("Count") |> ignore
                table.AddRow("Modules", sprintf "%d" cs.Modules.Length) |> ignore
                table.AddRow("Types", sprintf "%d" cs.Types.Length) |> ignore
                table.AddRow("Functions", sprintf "%d" cs.Functions.Length) |> ignore
                AnsiConsole.Write(table)
                
                let trace : MemoryTrace = {
                    TaskId = "perceptual-demo"
                    Variables = Map [ "code_structure", box cs ]
                    StepOutputs = Map.empty
                }
                
                AnsiConsole.MarkupLine("[bold yellow]Growing Semantic Memory...[/]")
                let! id = kernel.SemanticMemory.Grow(trace, obj())
                AnsiConsole.MarkupLine(sprintf "[green]Created Memory Record: %s[/]" id)
                
                // Show sample
                let tree = Tree("Code Structure (Sample)")
                let modNode = tree.AddNode("Modules")
                cs.Modules |> List.truncate 5 |> List.iter (fun m -> modNode.AddNode(m) |> ignore)
                if cs.Modules.Length > 5 then modNode.AddNode("...") |> ignore
                
                let typeNode = tree.AddNode("Types")
                cs.Types |> List.truncate 5 |> List.iter (fun t -> typeNode.AddNode(t) |> ignore)
                if cs.Types.Length > 5 then typeNode.AddNode("...") |> ignore
                
                AnsiConsole.Write(tree)
                
                return 0
            else
                AnsiConsole.MarkupLine("[red]Source directory not found.[/]")
                return 1

        | _ ->
            printfn "Usage: tars smem [query <text> | grow | refine | demo-perceptual]"
            return 1
    }
