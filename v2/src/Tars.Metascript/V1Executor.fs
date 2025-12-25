namespace Tars.Metascript

open System
open System.IO
open System.Collections.Generic
open System.Threading.Tasks
open System.Diagnostics
open System.Text
open Tars.Metascript.V1
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Kernel
open FSharp.Compiler.Interactive.Shell

module V1Executor =

    let resolveVariables (text: string) (variables: Map<string, MetascriptVariable>) =
        let mutable result = text

        for kvp in variables do
            let placeholder = sprintf "${%s}" kvp.Key
            result <- result.Replace(placeholder, string kvp.Value.Value)

        result

    type MetascriptExecutor(handlers: IBlockHandler list) =

        let handlerMap =
            handlers
            |> List.sortByDescending (fun h -> h.Priority)
            |> List.groupBy (fun h -> h.BlockType)
            |> Map.ofList

        interface IMetascriptExecutor with
            member this.ExecuteBlockAsync(block, context) =
                task {
                    match Map.tryFind block.Type handlerMap with
                    | Some hs ->
                        let handler = hs |> List.tryFind (fun h -> h.CanHandle block)

                        match handler with
                        | Some h -> return! h.ExecuteBlockAsync(block, context)
                        | None ->
                            return
                                { Block = block
                                  Status = Failure
                                  Output = ""
                                  Error = Some "No handler found for block criteria"
                                  ReturnValue = None
                                  Variables = context.Variables
                                  ExecutionTimeMs = 0.0 }
                    | None ->
                        return
                            { Block = block
                              Status = Failure
                              Output = ""
                              Error = Some $"No handlers registered for block type {block.Type}"
                              ReturnValue = None
                              Variables = context.Variables
                              ExecutionTimeMs = 0.0 }
                }

            member this.ExecuteAsync(metascript, context) =
                task {
                    let mutable currentContext =
                        match context with
                        | Some c -> c
                        | None ->
                            { WorkingDirectory = Environment.CurrentDirectory
                              Variables = Map.empty
                              CurrentMetascript = Some metascript
                              CurrentBlock = None }

                    let mutable results = []
                    let mutable overallStatus = Success
                    let sw = Stopwatch.StartNew()

                    for block in metascript.Blocks do
                        if overallStatus <> Failure then
                            let blockContext =
                                { currentContext with
                                    CurrentBlock = Some block }

                            let! result = (this :> IMetascriptExecutor).ExecuteBlockAsync(block, blockContext)
                            results <- result :: results

                            currentContext <-
                                { currentContext with
                                    Variables = result.Variables }

                            if result.Status = Failure then
                                overallStatus <- Failure

                    sw.Stop()

                    return
                        { Metascript = metascript
                          BlockResults = List.rev results
                          Status = overallStatus
                          Output = String.Join(Environment.NewLine, results |> List.map (fun r -> r.Output))
                          Error =
                            results
                            |> List.tryFind (fun r -> r.Error.IsSome)
                            |> Option.bind (fun r -> r.Error)
                          ExecutionTimeMs = sw.Elapsed.TotalMilliseconds
                          ReturnValue = results |> List.tryLast |> Option.bind (fun r -> r.ReturnValue)
                          Variables = currentContext.Variables
                          Context = Some currentContext
                          Metadata = Map.empty }
                }

    type CommandBlockHandler() =
        let getShellInfo (command: string) =
            if Environment.OSVersion.Platform = PlatformID.Win32NT then
                ProcessStartInfo("cmd.exe", $"/c {command}")
            else
                let escapedCmd = command.Replace("\"", "\\\"")
                ProcessStartInfo("/bin/sh", $"""-c "{escapedCmd}" """)

        interface IBlockHandler with
            member _.BlockType = MetascriptBlockType.Command
            member _.Priority = 0
            member _.CanHandle _ = true

            member _.ExecuteBlockAsync(block, context) =
                task {
                    let sw = Stopwatch.StartNew()

                    try
                        let processInfo = getShellInfo block.Content
                        processInfo.RedirectStandardOutput <- true
                        processInfo.RedirectStandardError <- true
                        processInfo.UseShellExecute <- false
                        processInfo.CreateNoWindow <- true
                        processInfo.WorkingDirectory <- context.WorkingDirectory

                        use p = Process.Start(processInfo)
                        let output = p.StandardOutput.ReadToEnd()
                        let error = p.StandardError.ReadToEnd()
                        p.WaitForExit()
                        sw.Stop()

                        let status = if p.ExitCode = 0 then Success else Failure

                        return
                            { Block = block
                              Status = status
                              Output = output
                              Error = if String.IsNullOrEmpty(error) then None else Some error
                              ReturnValue = Some p.ExitCode
                              Variables = context.Variables
                              ExecutionTimeMs = sw.Elapsed.TotalMilliseconds }
                    with ex ->
                        sw.Stop()

                        return
                            { Block = block
                              Status = Failure
                              Output = ""
                              Error = Some ex.Message
                              ReturnValue = None
                              Variables = context.Variables
                              ExecutionTimeMs = sw.Elapsed.TotalMilliseconds }
                }

    type TextBlockHandler() =
        interface IBlockHandler with
            member _.BlockType = MetascriptBlockType.Text
            member _.Priority = 0
            member _.CanHandle _ = true

            member _.ExecuteBlockAsync(block, context) =
                task {
                    return
                        { Block = block
                          Status = Success
                          Output = block.Content
                          Error = None
                          ReturnValue = Some block.Content
                          Variables = context.Variables
                          ExecutionTimeMs = 0.0 }
                }

    type QueryBlockHandler(llm: ILlmService) =
        interface IBlockHandler with
            member _.BlockType = MetascriptBlockType.Query
            member _.Priority = 0
            member _.CanHandle _ = true

            member _.ExecuteBlockAsync(block, context) =
                task {
                    let sw = Stopwatch.StartNew()
                    let prompt = resolveVariables block.Content context.Variables

                    try
                        let! response =
                            llm.CompleteAsync(
                                { Messages = [ { Role = Role.User; Content = prompt } ]
                                  Temperature = None
                                  MaxTokens = None
                                  Stop = []
                                  ModelHint = None
                                  Model = None
                                  SystemPrompt = None
                                  Tools = []
                                  ToolChoice = None
                                  ResponseFormat = None
                                  Stream = false
                                  JsonMode = false
                                  Seed = None
                                  ContextWindow = None }
                            )

                        sw.Stop()

                        let resultText = response.Text

                        // Handle grammar parameter if present
                        let grammarParam = block.Parameters |> List.tryFind (fun p -> p.Name = "grammar")

                        let (output, status, error) =
                            match grammarParam with
                            | Some p ->
                                // Integration with Tars.Cortex.Grammar
                                // For now, we just check if it parses with the default grammar if "Goal" is requested
                                if p.Value = "Goal" then
                                    try
                                        let goals = Tars.Cortex.Grammar.Parser.parse resultText
                                        (resultText, Success, None)
                                    with ex ->
                                        (resultText, Failure, Some $"Grammar validation failed: {ex.Message}")
                                else
                                    (resultText, Success, None)
                            | None -> (resultText, Success, None)

                        // If block has an output parameter, save to variables
                        let outputParam = block.Parameters |> List.tryFind (fun p -> p.Name = "output")

                        let updatedVariables =
                            match outputParam with
                            | Some p ->
                                context.Variables.Add(
                                    p.Value,
                                    { Name = p.Value
                                      Value = output
                                      Type = typeof<string>
                                      Metadata = Map.empty }
                                )
                            | None -> context.Variables

                        return
                            { Block = block
                              Status = status
                              Output = output
                              Error = error
                              ReturnValue = Some output
                              Variables = updatedVariables
                              ExecutionTimeMs = sw.Elapsed.TotalMilliseconds }
                    with ex ->
                        sw.Stop()

                        return
                            { Block = block
                              Status = Failure
                              Output = ""
                              Error = Some ex.Message
                              ReturnValue = None
                              Variables = context.Variables
                              ExecutionTimeMs = sw.Elapsed.TotalMilliseconds }
                }

    type FSharpBlockHandler() =

        let sbOut = new StringBuilder()
        let sbErr = new StringBuilder()
        let inStream = new StringReader("")
        let outStream = new StringWriter(sbOut)
        let errStream = new StringWriter(sbErr)

        let fsiConfig = FsiEvaluationSession.GetDefaultConfiguration()
        let argv = [| "fsi.exe"; "--noninteractive"; "--quiet" |]

        let fsiSession =
            FsiEvaluationSession.Create(fsiConfig, argv, inStream, outStream, errStream)

        interface IBlockHandler with
            member _.BlockType = MetascriptBlockType.FSharp
            member _.Priority = 0
            member _.CanHandle _ = true

            member _.ExecuteBlockAsync(block, context) =
                task {
                    let sw = Stopwatch.StartNew()
                    sbOut.Clear() |> ignore
                    sbErr.Clear() |> ignore

                    // Inject variables into FSI session
                    for kvp in context.Variables do
                        // This is a bit simplified, but we can try to inject simple types
                        if kvp.Value.Type = typeof<string> then
                            fsiSession.EvalInteraction(sprintf "let %s = \"%s\"" kvp.Key (string kvp.Value.Value))
                        elif kvp.Value.Type = typeof<int> || kvp.Value.Type = typeof<float> then
                            fsiSession.EvalInteraction(sprintf "let %s = %s" kvp.Key (string kvp.Value.Value))

                    try
                        let result, warnings = fsiSession.EvalInteractionNonThrowing(block.Content)
                        sw.Stop()

                        let output = sbOut.ToString()
                        let error = sbErr.ToString()

                        match result with
                        | Choice1Of2 _ ->
                            // If block has an output parameter, try to evaluate it as an expression to get the value
                            let mutable updatedVariables = context.Variables
                            let outputParam = block.Parameters |> List.tryFind (fun p -> p.Name = "output")

                            match outputParam with
                            | Some p ->
                                try
                                    let evalResult, _ = fsiSession.EvalExpressionNonThrowing(p.Value)

                                    match evalResult with
                                    | Choice1Of2(Some value) ->
                                        let var: MetascriptVariable =
                                            { Name = p.Value
                                              Value = value.ReflectionValue
                                              Type = value.ReflectionType
                                              Metadata = Map.empty }

                                        updatedVariables <- updatedVariables.Add(p.Value, var)
                                    | _ -> ()
                                with _ ->
                                    ()
                            | None -> ()

                            return
                                { Block = block
                                  Status = V1.Success
                                  Output = output
                                  Error = if String.IsNullOrEmpty(error) then None else Some error
                                  ReturnValue = Some output
                                  Variables = updatedVariables
                                  ExecutionTimeMs = sw.Elapsed.TotalMilliseconds }
                        | Choice2Of2 ex ->
                            return
                                { Block = block
                                  Status = Failure
                                  Output = output
                                  Error = Some(ex.ToString() + "\n" + error)
                                  ReturnValue = None
                                  Variables = context.Variables
                                  ExecutionTimeMs = sw.Elapsed.TotalMilliseconds }
                    with ex ->
                        sw.Stop()

                        return
                            { Block = block
                              Status = Failure
                              Output = sbOut.ToString()
                              Error = Some(ex.Message + "\n" + sbErr.ToString())
                              ReturnValue = None
                              Variables = context.Variables
                              ExecutionTimeMs = sw.Elapsed.TotalMilliseconds }
                }

        interface IDisposable with
            member _.Dispose() =
                inStream.Dispose()
                outStream.Dispose()
                errStream.Dispose()
