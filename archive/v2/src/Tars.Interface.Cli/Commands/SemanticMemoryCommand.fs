module Tars.Interface.Cli.Commands.SemanticMemoryCommand

open System
open System.IO
open Serilog
open Tars.Core
open Tars.Kernel
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Cortex
open Microsoft.Extensions.Configuration
open Tars.Interface.Cli
open Spectre.Console

let private createLlmService (_config: IConfiguration) : ILlmService * ILlmServiceFunctional =
    let service = LlmFactory.create Log.Logger
    (service, service :?> ILlmServiceFunctional)

let private createEmbedder (llmService: ILlmServiceFunctional) : Embedder =
    fun text ->
        async {
            let! result = llmService.EmbedAsync text

            match result with
            | Result.Ok embedding -> return embedding
            | Result.Error err ->
                printfn $"Embedding error: %s{LlmError.toMessage err}"

                match err with
                | UnknownError ex -> printfn $"Exception: {ex}"
                | _ -> ()

                return Array.empty<float32>
        }

let run (config: IConfiguration) (args: string array) =
    task {
        let storageRoot =
            Path.Combine(Environment.CurrentDirectory, ".tars", "knowledge", "semantic_memory")

        let llmService, llmServiceFunctional = createLlmService config
        let embedder = createEmbedder llmServiceFunctional
        let kernel = KernelBootstrap.createKernel storageRoot embedder llmService

        match args with
        | [| "query"; text |] ->
            let query =
                { TaskId = ""
                  TaskKind = ""
                  TextContext = text
                  Tags = [] }

            let! results = kernel.SemanticMemory.Retrieve query
            printfn $"Found %d{results.Length} results:"

            for schema: MemorySchema in results do
                printfn
                    "- [%s] %s"
                    schema.Id
                    (schema.Logical
                     |> Option.map (fun l -> l.ProblemSummary)
                     |> Option.defaultValue "No summary")

            return 0

        | [| "grow" |] ->
            // Dummy grow
            let! id = kernel.SemanticMemory.Grow(obj (), obj ())
            printfn $"Created memory schema: %s{id}"
            return 0

        | [| "refine" |] ->
            do! kernel.SemanticMemory.Refine()
            printfn "Refinement complete."
            return 0

        // | [| "demo-perceptual" |] ->
        //     AnsiConsole.MarkupLine("[bold cyan]Ingesting source code from 'src'...[/]")
        //     let kg = Tars.Core.LegacyKnowledgeGraph.TemporalGraph()
        //     let srcDir = Path.Combine(Environment.CurrentDirectory, "src")

        //     if Directory.Exists srcDir then
        //         do! AstIngestor.ingestDirectory kg srcDir |> Async.StartAsTask
        //         let cs = AstIngestor.extractCodeStructure kg

        //         let table = Table()
        //         table.AddColumn("Category") |> ignore
        //         table.AddColumn("Count") |> ignore
        //         table.AddRow("Modules", sprintf "%d" cs.Modules.Length) |> ignore
        //         table.AddRow("Types", sprintf "%d" cs.Types.Length) |> ignore
        //         table.AddRow("Functions", sprintf "%d" cs.Functions.Length) |> ignore
        //         AnsiConsole.Write(table)

        //         let trace: MemoryTrace =
        //             { TaskId = "perceptual-demo"
        //               Variables = Map [ "code_structure", box cs ]
        //               StepOutputs = Map.empty }

        //         AnsiConsole.MarkupLine("[bold yellow]Growing Semantic Memory...[/]")
        //         let! id = kernel.SemanticMemory.Grow(trace, obj ())
        //         AnsiConsole.MarkupLine(sprintf "[green]Created Memory Record: %s[/]" id)

        //         // Show sample
        //         let tree = Tree("Code Structure (Sample)")
        //         let modNode = tree.AddNode("Modules")

        //         cs.Modules
        //         |> List.truncate 5
        //         |> List.iter (fun m -> modNode.AddNode(m) |> ignore)

        //         if cs.Modules.Length > 5 then
        //             modNode.AddNode("...") |> ignore

        //         let typeNode = tree.AddNode("Types")

        //         cs.Types
        //         |> List.truncate 5
        //         |> List.iter (fun t -> typeNode.AddNode(t) |> ignore)

        //         if cs.Types.Length > 5 then
        //             typeNode.AddNode("...") |> ignore

        //         AnsiConsole.Write(tree)

        //         return 0
        //     else
        //         AnsiConsole.MarkupLine("[red]Source directory not found.[/]")
        //         return 1

        | [| "demo-chunking" |] ->
            AnsiConsole.MarkupLine("[bold cyan]Running Chunking Strategy Comparison...[/]")

            // Sample text (a mix of code and text to challenge the chunker)
            let sampleText =
                "# Semantic Memory in TARS v2\n\n"
                + "Semantic memory allows the agent to learn from past experiences. It consists of two main components:\n"
                + "1. Logical Memory: Abstracted rules and strategies.\n"
                + "2. Perceptual Memory: Raw data and observations.\n\n"
                + "## Implementation Details\n\n"
                + "The system uses vector embeddings to store and retrieve memories. Here is a sample configuration:\n\n"
                + "```fsharp\n"
                + "type RagConfig = {\n"
                + "    CollectionName: string\n"
                + "    TopK: int\n"
                + "    MinScore: float32\n"
                + "}\n"
                + "```\n\n"
                + "When a new task is executed, the agent queries the semantic memory to find relevant past experiences.\n"
                + "This process involves:\n"
                + "- Computing the embedding of the current task.\n"
                + "- Searching the vector store for similar entries.\n"
                + "- Retrieving the associated strategies.\n\n"
                + "The goal is to improve performance over time by avoiding repeated mistakes."

            AnsiConsole.MarkupLine("[bold]Sample Text:[/]")
            AnsiConsole.WriteLine(sampleText.Trim())
            AnsiConsole.WriteLine()

            // 1. Fixed Size Chunking
            AnsiConsole.MarkupLine("[bold yellow]Strategy: Fixed Size (100 chars)[/]")

            let fixedConfig =
                { Chunking.defaultConfig with
                    Strategy = Chunking.FixedSize
                    ChunkSize = 100
                    ChunkOverlap = 0 }

            let fixedChunks = Chunking.chunk fixedConfig "demo" sampleText

            let fixedTable = Table()
            fixedTable.AddColumn("ID") |> ignore
            fixedTable.AddColumn("Content") |> ignore

            for c in fixedChunks do
                let content = c.Content.Replace("\n", "\\n")
                fixedTable.AddRow(c.Id, Markup.Escape(content)) |> ignore

            AnsiConsole.Write(fixedTable)
            AnsiConsole.WriteLine()

            // 2. Semantic Chunking
            AnsiConsole.MarkupLine("[bold green]Strategy: Semantic (Embedding-based)[/]")

            // Adapter for the embedder to match the signature expected by semanticChunkAsync
            let embedderTask (text: string) =
                task {
                    let! res = llmService.EmbedAsync text
                    return res
                }

            let semanticConfig =
                { Chunking.defaultConfig with
                    Strategy = Chunking.Semantic
                    ChunkSize = 200
                    MinChunkSize = 20 }

            let! semanticChunks = Chunking.chunkAsync semanticConfig "demo" sampleText (Some embedderTask) None

            let semanticTable = Table()
            semanticTable.AddColumn("ID") |> ignore
            semanticTable.AddColumn("Content") |> ignore

            for c in semanticChunks do
                let content = c.Content.Replace("\n", "\\n")
                semanticTable.AddRow(c.Id, Markup.Escape(content)) |> ignore

            AnsiConsole.Write(semanticTable)

            // 3. Agentic Chunking
            AnsiConsole.MarkupLine("[bold magenta]Strategy: Agentic (LLM-based)[/]")

            let completerTask (prompt: string) =
                task {
                    try
                        let req =
                            { ModelHint = Some "fast"
                              Model = None
                              SystemPrompt = None
                              MaxTokens = Some 1000
                              Temperature = Some 0.0
                              Stop = []
                              Messages = [ { Role = Role.User; Content = prompt } ]
                              Tools = []
                              ToolChoice = None
                              ResponseFormat = None
                              Stream = false
                              JsonMode = false
                              Seed = None
                              ContextWindow = None }


                        let! res = llmService.CompleteAsync req
                        return res.Text
                    with _ ->
                        // Fallback mock for demo purposes if LLM is offline
                        return
                            """### SECTION: Introduction
# Semantic Memory in TARS v2

Semantic memory allows the agent to learn from past experiences. It consists of two main components:
1. Logical Memory: Abstracted rules and strategies.
2. Perceptual Memory: Raw data and observations.
### END SECTION
### SECTION: Implementation
## Implementation Details

The system uses vector embeddings to store and retrieve memories. Here is a sample configuration:

```fsharp
type RagConfig = {
    CollectionName: string
    TopK: int
    MinScore: float32
}
```
### END SECTION
### SECTION: Process & Goal
When a new task is executed, the agent queries the semantic memory to find relevant past experiences.
This process involves:
- Computing the embedding of the current task.
- Searching the vector store for similar entries.
- Retrieving the associated strategies.

The goal is to improve performance over time by avoiding repeated mistakes.
### END SECTION"""
                }

            let agenticConfig =
                { Chunking.defaultConfig with
                    Strategy = Chunking.Agentic }

            let! agenticChunks = Chunking.chunkAsync agenticConfig "demo" sampleText None (Some completerTask)

            let agenticTable = Table()
            agenticTable.AddColumn("ID") |> ignore
            agenticTable.AddColumn("Content") |> ignore

            for c in agenticChunks do
                let content = c.Content.Replace("\n", "\\n")
                agenticTable.AddRow(c.Id, Markup.Escape(content)) |> ignore

            AnsiConsole.Write(agenticTable)

            // 4. AST Chunking
            AnsiConsole.MarkupLine("[bold blue]Strategy: AST (F# Parser)[/]")

            let fsharpSample =
                """
module Tars.Demo

type DemoConfig = {
    Id: string
    Value: int
}

let processConfig (config: DemoConfig) =
    printfn "Processing %s" config.Id
    config.Value * 2

type Processor() =
    member this.Run() =
        let config = { Id = "test"; Value = 10 }
        processConfig config
"""

            AnsiConsole.MarkupLine("[bold]Sample F# Code:[/]")
            AnsiConsole.WriteLine(fsharpSample.Trim())

            let astConfig =
                { Chunking.defaultConfig with
                    Strategy = Chunking.Ast
                    MinChunkSize = 10 }

            let! astChunks = Chunking.chunkAsync astConfig "demo_ast" fsharpSample None None

            let astTable = Table()
            astTable.AddColumn("ID") |> ignore
            astTable.AddColumn("Content") |> ignore

            for c in astChunks do
                let content = c.Content.Replace("\n", "\\n")
                astTable.AddRow(c.Id, Markup.Escape(content)) |> ignore

            AnsiConsole.Write(astTable)

            // Save to file for documentation
            let outputPath =
                Path.Combine(Environment.CurrentDirectory, "chunking_demo_output.txt")

            use writer = new StreamWriter(outputPath)
            writer.WriteLine("=== Chunking Strategy Comparison ===")
            writer.WriteLine("\n--- Sample Text ---\n")
            writer.WriteLine(sampleText.Trim())

            writer.WriteLine("\n--- Fixed Size Chunking ---")

            for c in fixedChunks do
                let content = c.Content.Replace("\n", "\\n")
                writer.WriteLine($"[{c.Id}] {content}")

            writer.WriteLine("\n--- Semantic Chunking ---")

            for c in semanticChunks do
                let content = c.Content.Replace("\n", "\\n")
                writer.WriteLine($"[{c.Id}] {content}")

            writer.WriteLine("\n--- Agentic Chunking ---")

            for c in agenticChunks do
                let content = c.Content.Replace("\n", "\\n")
                writer.WriteLine($"[{c.Id}] {content}")

            writer.WriteLine("\n--- AST Chunking (F# Sample) ---")
            writer.WriteLine(fsharpSample.Trim())
            writer.WriteLine("")

            for c in astChunks do
                let content = c.Content.Replace("\n", "\\n")
                writer.WriteLine($"[{c.Id}] {content}")

            AnsiConsole.MarkupLine($"\n[bold]Output saved to:[/] {outputPath}")
            return 0

        | [| "demo-compression" |] ->
            AnsiConsole.MarkupLine("[bold cyan]Running Context Compression Demo...[/]")

            let sampleText =
                """
                The TARS v2 system is designed to be a self-improving agentic framework. 
                The TARS v2 system is designed to be a self-improving agentic framework.
                It uses a micro-kernel architecture to ensure modularity and extensibility.
                It uses a micro-kernel architecture to ensure modularity and extensibility.
                The core components include the Cortex, the Evolution Engine, and the Semantic Memory.
                The core components include the Cortex, the Evolution Engine, and the Semantic Memory.
                Semantic Memory allows the agent to learn from past experiences by storing and retrieving memory schemas.
                Semantic Memory allows the agent to learn from past experiences by storing and retrieving memory schemas.
                This redundancy is intentional to test the compression algorithm.
                This redundancy is intentional to test the compression algorithm.
                """

            AnsiConsole.MarkupLine("[bold]Original Text:[/]")
            AnsiConsole.WriteLine(sampleText.Trim())
            AnsiConsole.WriteLine()

            let entropyMonitor = EntropyMonitor()
            let compressor = ContextCompressor(llmService, entropyMonitor)

            try
                // Test AutoCompress
                AnsiConsole.MarkupLine("[bold yellow]Running Auto-Compression...[/]")
                let! compressed = compressor.AutoCompress(sampleText)

                AnsiConsole.MarkupLine($"[bold]Compressed Text ({compressed.Length} chars):[/]")
                AnsiConsole.WriteLine(compressed)
                AnsiConsole.WriteLine()

                // Test Explicit Strategies
                let strategies = [ Summarization; KeyPointExtraction; RemoveRedundancy ]

                for strategy in strategies do
                    AnsiConsole.MarkupLine($"[bold green]Strategy: {strategy}[/]")
                    let! result = compressor.CompressWithMetrics(sampleText, strategy)

                    let grid = Grid()
                    grid.AddColumn() |> ignore
                    grid.AddColumn() |> ignore

                    grid.AddRow([| "Original Length"; $"%d{result.OriginalLength}" |])
                    |> ignore

                    grid.AddRow([| "Compressed Length"; $"%d{result.CompressedLength}" |])
                    |> ignore

                    grid.AddRow([| "Ratio"; $"%.2f{result.CompressionRatio}" |]) |> ignore
                    grid.AddRow([| "Entropy"; $"%.4f{result.Entropy}" |]) |> ignore

                    AnsiConsole.Write(grid)
                    AnsiConsole.WriteLine("--- Output ---")
                    AnsiConsole.WriteLine(result.CompressedText)
                    AnsiConsole.WriteLine()

                return 0
            with ex ->
                AnsiConsole.MarkupLine($"[bold red]Error running compression demo:[/] {ex.Message}")

                if ex.InnerException <> null then
                    AnsiConsole.MarkupLine($"[red]Inner Error:[/] {ex.InnerException.Message}")

                return 1

        | _ ->
            printfn
                "Usage: tars smem [query <text> | grow | refine | demo-perceptual | demo-chunking | demo-compression]"

            return 1
    }
