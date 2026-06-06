namespace Tars.Interface.Cli.Commands

open System
open System.IO
open Serilog
open Tars.Llm
open Tars.Metascript
open Tars.Metascript.V1
open Tars.Metascript.V1Executor
open Tars.Metascript.Domain
open Tars.Metascript.Config
open Tars.Evolution.Reflection
open Tars.Evolution.Optimizer
open Tars.Interface.Cli

open Tars.Tools

module RunCommand =

    let run (logger: ILogger) (workflowPath: string) (shouldOptimize: bool) =
        task {
            try
                logger.Information("Loading script from {Path}...", workflowPath)

                if not (File.Exists workflowPath) then
                    logger.Error("File not found: {Path}", workflowPath)
                    return 1
                else
                    let ext = Path.GetExtension(workflowPath).ToLowerInvariant()

                    if ext = ".tars" || ext = ".trsx" then
                        let text = File.ReadAllText(workflowPath)

                        let metascript =
                            V1Parser.parseMetascript
                                text
                                (Path.GetFileNameWithoutExtension(workflowPath))
                                (Some workflowPath)

                        logger.Information(
                            "Metascript '{Name}' loaded with {Count} blocks.",
                            metascript.Name,
                            metascript.Blocks.Length
                        )

                        let llm = LlmFactory.create logger

                        let handlers: IBlockHandler list =
                            [ TextBlockHandler() :> IBlockHandler
                              { new IBlockHandler with
                                  member _.BlockType = MetascriptBlockType.Meta
                                  member _.Priority = 0
                                  member _.CanHandle _ = true

                                  member _.ExecuteBlockAsync(block, context) =
                                      task {
                                          return
                                              { Block = block
                                                Status = MetascriptExecutionStatus.Success
                                                Output = ""
                                                Error = None
                                                ReturnValue = None
                                                Variables = context.Variables
                                                ExecutionTimeMs = 0.0 }
                                      } }
                              CommandBlockHandler() :> IBlockHandler
                              new FSharpBlockHandler() :> IBlockHandler
                              QueryBlockHandler(llm) :> IBlockHandler ]

                        let executor = MetascriptExecutor(handlers)
                        let! result = (executor :> IMetascriptExecutor).ExecuteAsync(metascript)

                        if result.Status = MetascriptExecutionStatus.Success then
                            logger.Information("Final Output: {Output}", result.Output)
                            return 0
                        else
                            logger.Error(
                                "Metascript failed: {Error}",
                                result.Error |> Option.defaultValue "Unknown error"
                            )

                            return 1
                    else
                        let json = File.ReadAllText(workflowPath)
                        let parseResult = Parser.parseJson json

                        match parseResult with
                        | Parser.ValidationError errs ->
                            logger.Error("Workflow validation failed:")

                            for e in errs do
                                logger.Error("- {Error}", e)

                            return 1
                        | Parser.ParseError(l, c, msg) ->
                            logger.Error("JSON parse error at line {Line}, col {Col}: {Message}", l, c, msg)
                            return 1
                        | Parser.Success workflow ->
                            logger.Information("Workflow '{Name}' loaded successfully.", workflow.Name)

                            let llm = LlmFactory.create logger
                            let tools = ToolRegistry()

                            let ctx: MetascriptContext =
                                { Llm = llm
                                  Tools = tools
                                  Budget = None
                                  VectorStore = None
                                  KnowledgeGraph = None
                                  SemanticMemory = None
                                  EpisodeService = None
                                  RagConfig = RagConfig.Default
                                  MacroRegistry = None
                                  MetascriptRegistry = None }

                            let inputs = Map.empty
                            let mutable executionSuccess = false
                            let mutable execTrace = []
                            let mutable finalOutputs = ""

                            try
                                let! resultState = Engine.run ctx workflow inputs
                                executionSuccess <- true
                                execTrace <- resultState.ExecutionTrace |> List.rev

                                let outs =
                                    resultState.StepOutputs
                                    |> Map.toList
                                    |> List.map (fun (k, v) -> $"%s{k}: %A{v}")
                                    |> String.concat ", "

                                finalOutputs <- outs
                                logger.Information("Workflow completed. Outputs: {Outputs}", outs)
                            with ex ->
                                logger.Error(ex, "Workflow execution failed")
                                executionSuccess <- false
                                finalOutputs <- ex.Message

                            if shouldOptimize then
                                let reflectionAgent = LlmReflectionAgent(llm) :> IReflectionAgent

                                let traceItems =
                                    execTrace
                                    |> List.map (fun t ->
                                        { Step = t.StepId
                                          Input = "N/A"
                                          Output = $"%A{t.Outputs}"
                                          DurationMs = int64 t.Duration.TotalMilliseconds })

                                let! feedback =
                                    reflectionAgent.ReflectAsync(workflow.Description, finalOutputs, traceItems)

                                if feedback.Suggestion.IsSome then
                                    let optimizer = LlmWorkflowOptimizer(llm) :> IWorkflowOptimizer
                                    let! newWorkflow = optimizer.OptimizeAsync(workflow, feedback)

                                    match newWorkflow with
                                    | Some nw ->
                                        let newPath = Path.ChangeExtension(workflowPath, null) + "_optimized.json"
                                        let jsonOpts = System.Text.Json.JsonSerializerOptions(WriteIndented = true)
                                        let newJson = System.Text.Json.JsonSerializer.Serialize(nw, jsonOpts)
                                        File.WriteAllText(newPath, newJson)
                                        logger.Information("✅ Optimized workflow saved to: {Path}", newPath)
                                    | None -> logger.Error("❌ Optimization failed to generate valid workflow.")

                            return if executionSuccess then 0 else 1
            with ex ->
                logger.Error(ex, "Unexpected error")
                return 1
        }
