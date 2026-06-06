module Tars.Interface.Cli.Commands.Run

open System
open System.IO
open System.Text.Json
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Kernel
open Tars.Llm
open Tars.Metascript
open Tars.Metascript.Domain
open Tars.Metascript.Engine
open Tars.Metascript.Config
open Tars.Cortex
open Tars.Interface.Cli

let execute (logger: ILogger) (scriptPath: string) =
    task {
        logger.Information("Executing Metascript: {ScriptPath}", scriptPath)

        if not (File.Exists scriptPath) then
            logger.Error("Script file not found: {ScriptPath}", scriptPath)
            return 1
        else
            try
                let json = File.ReadAllText(scriptPath)
                let options = JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                let workflow = JsonSerializer.Deserialize<Workflow>(json, options)

                // Initialize Kernel & LLM
                let llmService = LlmFactory.create logger

                // Initialize Tools
                let tools = Tars.Tools.ToolRegistry()
                tools.RegisterAssembly(typeof<Tars.Tools.ToolRegistry>.Assembly)

                let metaBudget =
                    BudgetGovernor(
                        { Budget.Infinite with
                            MaxTokens = Some 50000<token>
                            MaxCalls = Some 200<requests> }
                    )

                // Initialize Kernel
                let storageRoot =
                    Path.Combine(Environment.CurrentDirectory, ".tars", "knowledge", "semantic_memory")

                let embedder (text: string) =
                    async {
                        match! (llmService :?> ILlmServiceFunctional).EmbedAsync text with
                        | Result.Ok embedding -> return embedding
                        | Result.Error _ -> return Array.empty<float32>
                    }


                let kernel = KernelBootstrap.createKernel storageRoot embedder llmService

                let metaCtx: MetascriptContext =
                    { Llm = llmService
                      Tools = tools
                      Budget = Some metaBudget
                      VectorStore = None
                      KnowledgeGraph = None
                      SemanticMemory = Some kernel.SemanticMemory
                      EpisodeService = None
                      RagConfig = RagConfig.Default
                      MacroRegistry = None
                      MetascriptRegistry = None }

                // Ingest Code Structure
                let! codeStructureInput =
                    task {
                        let kg = Tars.Core.LegacyKnowledgeGraph.TemporalGraph()
                        let srcDir = Path.Combine(Environment.CurrentDirectory, "src")

                        if Directory.Exists srcDir then
                            do! AstIngestor.ingestDirectory kg srcDir |> Async.StartAsTask
                            let cs = AstIngestor.extractCodeStructure kg
                            return Map [ "code_structure", box cs ]
                        else
                            return Map.empty
                    }

                // Execute
                let! finalState = Engine.run metaCtx workflow codeStructureInput

                printfn "\nWorkflow Completed."
                printfn "Final Outputs:"

                for kvp in finalState.StepOutputs do
                    printfn "Step %s:" kvp.Key

                    for outKvp in kvp.Value do
                        printfn "  %s: %A" outKvp.Key outKvp.Value

                printfn "\nExecution Trace:"

                for trace in finalState.ExecutionTrace do
                    printfn "- %s (%O ms) outputs=%d" trace.StepId trace.Duration.TotalMilliseconds trace.Outputs.Count

                return 0
            with ex ->
                logger.Error(ex, "Failed to execute script")
                return 1
    }
