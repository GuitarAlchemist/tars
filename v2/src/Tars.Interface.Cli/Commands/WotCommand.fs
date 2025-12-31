namespace Tars.Interface.Cli.Commands

open System
open System.Threading.Tasks
open Spectre.Console
open Serilog
open Tars.Core
open Tars.Core.WorkflowOfThought
open Tars.DSL.Wot
open Tars.Interface.Cli
open Tars.Interface.Cli.Reasoning
open Tars.Tools

module WotCommand =

    // Invoker implementation
    type private CliToolInvoker(registry: ToolRegistry) =
        interface IToolInvoker with
            member _.Invoke(toolName: string, args: Map<string, string>) =
                async {
                    match registry.Get(toolName) with
                    | None -> return Result.Error $"Tool '{toolName}' not found in registry."
                    | Some tool ->
                        try
                            let json = System.Text.Json.JsonSerializer.Serialize(args)
                            let! res = tool.Execute json
                            match res with
                            | Result.Ok (output: string) -> return Result.Ok (output :> obj)
                            | Result.Error (err: string) -> return Result.Error err
                        with ex ->
                            return Result.Error $"Tool execution failed: {ex.Message}"
                }
    
    /// Options for run command
    type RunOptions =
        { Mode: ReasonStepMode
          Model: string option
          ModelHint: string option
          Temperature: float option
          MaxTokens: int option
          Deterministic: bool
          Seed: int option
          ReplayRunId: string option }
        
        static member Default =
            { Mode = ReasonStepMode.Stub
              Model = None
              ModelHint = None
              Temperature = None
              MaxTokens = None
              Deterministic = false
              Seed = None
              ReplayRunId = None }

    type WotAction =
        | RunFile of path: string * options: RunOptions
        | Diff of runA: string * runB: string
        | CiCheck of workflow: string * baseline: string
        | Help

    let parseArgs (args: string list) : WotAction =
        let rec parseOptions (acc: RunOptions) (rest: string list) =
            match rest with
            | [] -> acc
            | "--reason" :: "llm" :: tail -> parseOptions { acc with Mode = ReasonStepMode.Llm } tail
            | "--reason" :: "stub" :: tail -> parseOptions { acc with Mode = ReasonStepMode.Stub } tail
            | "--reason" :: "replay" :: tail -> parseOptions { acc with Mode = ReasonStepMode.Replay } tail
            | "--replay-run" :: runId :: tail -> parseOptions { acc with ReplayRunId = Some runId } tail
            | "--model" :: name :: tail -> parseOptions { acc with Model = Some name } tail
            | "--model-hint" :: hint :: tail -> parseOptions { acc with ModelHint = Some hint } tail
            | "--temp" :: t :: tail -> 
                match Double.TryParse(t) with
                | true, v -> parseOptions { acc with Temperature = Some v } tail
                | _ -> parseOptions acc tail
            | "--max-tokens" :: n :: tail ->
                match Int32.TryParse(n) with
                | true, v -> parseOptions { acc with MaxTokens = Some v } tail
                | _ -> parseOptions acc tail
            | "--deterministic" :: tail -> parseOptions { acc with Deterministic = true; Temperature = Some 0.0 } tail
            | "--seed" :: s :: tail ->
                match Int32.TryParse(s) with
                | true, v -> parseOptions { acc with Seed = Some v } tail
                | _ -> parseOptions acc tail
            | _ :: tail -> parseOptions acc tail

        match args with
        | "run" :: path :: tail -> 
            let opts = parseOptions RunOptions.Default tail
            RunFile (path, opts)
        | "diff" :: runA :: runB :: _ -> Diff (runA, runB)
        | "ci-check" :: flow :: baseLine :: _ -> CiCheck (flow, baseLine)
        | ["help"] -> Help
        | _ -> Help

    // Helper for running logic (Shared between RunFile and CiCheck)
    let private runWorkflow (path: string) (mode: ReasonStepMode) : Async<Result<CanonicalGolden * string, string>> =
        async {
            // Reusing logic from execute, but focused on getting Golden object
             if not (System.IO.File.Exists path) then return Result.Error $"File not found: {path}"
             else
                match Tars.DSL.Wot.WotParser.parseFile path with
                | Result.Error errs -> return Result.Error "Parse Error" // Simplified for re-use
                | Result.Ok dslParams ->
                    match Tars.DSL.Wot.WotCompiler.compileWorkflowToPlanParsed dslParams with
                    | Result.Error errs -> return Result.Error "Compile Error"
                    | Result.Ok plan ->
                         let tools = ToolRegistry()
                         tools.RegisterAssembly(typeof<Tars.Tools.TarsToolAttribute>.Assembly)
                         if tools.Get("dotnet_build").IsNone then
                             tools.Register(Tool.InternalCreateMinimal("dotnet_build", "Fake build", fun _ -> async { return Result.Ok "Build successful" }))
                         if tools.Get("check_environment").IsNone then
                             tools.Register(Tool.InternalCreateMinimal("check_environment", "Fake check", fun _ -> async { return Result.Ok "Environment ready" }))

                         let invoker = CliToolInvoker(tools)
                         let inputs = plan.Inputs |> Map.toList |> List.map (fun (k, v) -> k, v) |> Map.ofList
                         let execPolicy : WotExecutor.ExecutionPolicy = { AllowedTools = plan.Policy.AllowedTools; MaxToolCalls = plan.Policy.MaxToolCalls }
                         let reasoner = 
                             if mode = ReasonStepMode.Llm then failwith "LLM not configured for runWorkflow helper" 
                             else { new IReasoner with member _.Reason(_,_,_,_) = async { return Result.Ok "<cli-stub-reasoner>" } }
                         
                         let! result = 
                             async {
                                 try return! WotExecutor.executePlanV0 invoker reasoner mode execPolicy inputs plan.Steps
                                 with ex -> return Result.Error (ex.Message, [])
                             }
                         
                         match result with
                         | Result.Error (e, traces) -> return Result.Error e
                         | Result.Ok (ctx, verifyC, traces) ->
                             let passed = verifyC |> Option.map (fun v -> v.Passed)
                             let toolCalls = traces |> List.filter (fun t -> t.Kind = "tool") |> List.length
                             
                             let golden = 
                                 { SchemaVersion = "wot.golden.v1"
                                   Steps = traces |> List.map TraceEvent.toCanonical
                                   Summary = {| ToolCalls = toolCalls
                                                VerifyPassed = passed
                                                FirstError = None
                                                OutputKeys = (ctx.Vars |> Map.toList |> List.map fst)
                                                Mode = mode.ToString() |} }
                             return Result.Ok (golden, path)
        }

    let execute (args: string list) : Task<int> =
        async {
            match parseArgs args with
            | Help ->
                AnsiConsole.MarkupLine("[bold]TARS WoT Runner (v0.7)[/]")
                AnsiConsole.MarkupLine("Usage:")
                AnsiConsole.MarkupLine("  tars wot run <file.wot.trsx> [[OPTIONS]]")
                AnsiConsole.MarkupLine("")
                AnsiConsole.MarkupLine("[dim]Options:[/]")
                AnsiConsole.MarkupLine("  --reason stub|llm|replay  Reasoning mode (default: stub)")
                AnsiConsole.MarkupLine("  --replay-run <runId>      Run ID to replay (for --reason replay)")
                AnsiConsole.MarkupLine("  --model <name>            Specific model to use")
                AnsiConsole.MarkupLine("  --model-hint <hint>       Model routing hint (e.g. reasoning, code)")
                AnsiConsole.MarkupLine("  --temp <float>            Temperature (0.0-2.0)")
                AnsiConsole.MarkupLine("  --max-tokens <n>          Maximum tokens to generate")
                AnsiConsole.MarkupLine("  --deterministic           Force temp=0, seed=42")
                AnsiConsole.MarkupLine("  --seed <n>                Random seed for reproducibility")
                AnsiConsole.MarkupLine("")
                AnsiConsole.MarkupLine("[dim]Other Commands:[/]")
                AnsiConsole.MarkupLine("  tars wot diff <runA> <runB>")
                AnsiConsole.MarkupLine("  tars wot ci-check <workflow> <baseline>")
                return 0

            | CiCheck (flow, baseline) ->
                if not (System.IO.File.Exists baseline) then
                    AnsiConsole.MarkupLine($"[red]Baseline not found: {baseline}[/]")
                    return 1
                else
                    // Run "Stub" mode for CI check usually
                    match! runWorkflow flow ReasonStepMode.Stub with
                    | Result.Error e -> 
                        AnsiConsole.MarkupLine($"[red]Run Failed:[/] {e}")
                        return 1
                    | Result.Ok (currentGolden, _) ->
                         let options = System.Text.Json.JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                         options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                         try
                             let baselineGolden = System.Text.Json.JsonSerializer.Deserialize<CanonicalGolden>(System.IO.File.ReadAllText baseline, options)
                             let diff = GoldenDiff.compute baselineGolden currentGolden
                             if diff.HasChanges then
                                 AnsiConsole.MarkupLine("[bold red]CI Check Failed: Regression detected[/]")
                                 AnsiConsole.MarkupLine("Differences found between baseline and current run.")
                                 return 1
                             else
                                 AnsiConsole.MarkupLine("[bold green]CI Check Passed: No regression.[/]")
                                 return 0
                         with ex ->
                             AnsiConsole.MarkupLine($"[red]Error loading baseline: {ex.Message}[/]")
                             return 1

            | Diff (runA, runB) ->
                let resolvePath (p: string) =
                    if System.IO.File.Exists p then p
                    else
                        let runPath = System.IO.Path.Combine(".wot", "runs", p, "golden.json")
                        if System.IO.File.Exists runPath then runPath else p

                let pathA = resolvePath runA
                let pathB = resolvePath runB

                if not (System.IO.File.Exists pathA) then
                    AnsiConsole.MarkupLine($"[red]File not found: {pathA}[/]")
                    return 1
                elif not (System.IO.File.Exists pathB) then
                    AnsiConsole.MarkupLine($"[red]File not found: {pathB}[/]")
                    return 1
                else
                    let options = System.Text.Json.JsonSerializerOptions(PropertyNameCaseInsensitive = true)
                    options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                    
                    try
                        let g1 = System.Text.Json.JsonSerializer.Deserialize<CanonicalGolden>(System.IO.File.ReadAllText pathA, options)
                        let g2 = System.Text.Json.JsonSerializer.Deserialize<CanonicalGolden>(System.IO.File.ReadAllText pathB, options)
                        
                        let diff = GoldenDiff.compute g1 g2
                        
                        if diff.HasChanges then
                            AnsiConsole.MarkupLine("[bold yellow]Differences Found:[/]")
                            
                            if not diff.SummaryChanges.IsEmpty then
                                AnsiConsole.MarkupLine("[bold]Summary Changes:[/]")
                                for kvp in diff.SummaryChanges do
                                    let oldV, newV = kvp.Value
                                    AnsiConsole.MarkupLine($"  {kvp.Key}: [red]{Markup.Escape(oldV)}[/] -> [green]{Markup.Escape(newV)}[/]")
                            
                            if not diff.StepChanges.IsEmpty then
                                AnsiConsole.MarkupLine("[bold]Step Changes:[/]")
                                for kvp in diff.StepChanges do
                                    let sId = kvp.Key
                                    match kvp.Value with
                                    | MissingInNew -> AnsiConsole.MarkupLine($"  Step [bold]{sId}[/]: [red]Missing in B[/]")
                                    | ExtraInNew -> AnsiConsole.MarkupLine($"  Step [bold]{sId}[/]: [green]Added in B[/]")
                                    | Changed changes ->
                                        AnsiConsole.MarkupLine($"  Step [bold]{sId}[/]:")
                                        for ch in changes do
                                            let fName = ch.Key
                                            let oVal, nVal = ch.Value
                                            AnsiConsole.MarkupLine($"    {fName}: [red]{Markup.Escape(oVal)}[/] -> [green]{Markup.Escape(nVal)}[/]")
                            return 1
                        else
                            AnsiConsole.MarkupLine("[bold green]No logical differences found.[/]")
                            return 0
                    with ex ->
                        AnsiConsole.MarkupLine($"[red]Error deserializing or comparing: {ex.Message}[/]")
                        return 1

            | RunFile (path, opts) ->
                if not (System.IO.File.Exists path) then
                     AnsiConsole.MarkupLine($"[red]File not found: {path}[/]")
                     return 1
                else
                     AnsiConsole.MarkupLine($"[bold blue]Running WoT Workflow:[/] {path} (Mode: {opts.Mode})")
                     let startTime = DateTime.UtcNow
                     let runId = startTime.ToString("yyyyMMdd-HHmmss")
                     let runDir = System.IO.Path.Combine(".wot", "runs", runId)
                     
                     // Create run directory early for journaling
                     System.IO.Directory.CreateDirectory(runDir) |> ignore
                     
                     match Tars.DSL.Wot.WotParser.parseFile path with
                     | Result.Error (errs: Tars.DSL.Wot.ParseError list) -> 
                         AnsiConsole.MarkupLine("[red]Parse Errors:[/]")
                         for e in errs do
                             AnsiConsole.MarkupLine($"  Line {e.Line}: {Markup.Escape(e.Message)}")
                         return 1
                     | Result.Ok (dslParams: Tars.DSL.Wot.DslWorkflow) ->
                         
                         match Tars.DSL.Wot.WotCompiler.compileWorkflowToPlanParsed dslParams with
                         | Result.Error (errs: Tars.DSL.Wot.CompileError list) -> 
                             AnsiConsole.MarkupLine("[red]Compilation Errors:[/]")
                             for e in errs do
                                 AnsiConsole.MarkupLine($"  {e.ToString()}")
                             return 1
                         | Result.Ok (plan: Tars.DSL.Wot.Plan<Tars.DSL.Wot.Parsed>) -> 
                             AnsiConsole.MarkupLine($"[green]Parsed & Compiled OK.[/] Goal: [bold]{Markup.Escape(plan.Goal)}[/]")
                             
                             let tools = ToolRegistry()
                             tools.RegisterAssembly(typeof<Tars.Tools.TarsToolAttribute>.Assembly)
                             if tools.Get("dotnet_build").IsNone then
                                 tools.Register(Tool.InternalCreateMinimal("dotnet_build", "Fake build", fun _ -> async { return Result.Ok "Build successful" }))
                             if tools.Get("check_environment").IsNone then
                                 tools.Register(Tool.InternalCreateMinimal("check_environment", "Fake check", fun _ -> async { return Result.Ok "Environment ready" }))

                             let invoker = CliToolInvoker(tools)
                             
                             AnsiConsole.MarkupLine("[bold]Executing Steps...[/]")
                             
                             let inputs = plan.Inputs |> Map.toList |> List.map (fun (k, v) -> k, v) |> Map.ofList
                             let execPolicy : WotExecutor.ExecutionPolicy = { AllowedTools = plan.Policy.AllowedTools; MaxToolCalls = plan.Policy.MaxToolCalls }

                             // Wire up reasoner based on mode
                             let reasoner : IReasoner = 
                                 match opts.Mode with
                                 | ReasonStepMode.Llm ->
                                     let logger = Log.Logger
                                     let llm = LlmFactory.create logger
                                     let reasonerSettings : ReasonerSettings = 
                                         { Model = opts.Model
                                           ModelHint = opts.ModelHint |> Option.orElse (Some "reasoning")
                                           Temperature = opts.Temperature
                                           MaxTokens = opts.MaxTokens
                                           Deterministic = opts.Deterministic
                                           Seed = opts.Seed }
                                     let modelStr = opts.Model |> Option.defaultValue "<default>"
                                     let tempStr = opts.Temperature |> Option.map string |> Option.defaultValue "<default>"
                                     AnsiConsole.MarkupLine($"[dim]Using LLM reasoner (model={modelStr}, temp={tempStr}, deterministic={opts.Deterministic})[/]")
                                     CliReasoner(llm, runDir, reasonerSettings, logger) :> IReasoner
                                 | ReasonStepMode.Replay ->
                                     match opts.ReplayRunId with
                                     | None ->
                                         failwith "--reason replay requires --replay-run <runId>"
                                     | Some replayRunId ->
                                         let replayRunDir = System.IO.Path.Combine(".wot", "runs", replayRunId)
                                         if not (System.IO.Directory.Exists replayRunDir) then
                                             failwith $"Replay run directory not found: {replayRunDir}"
                                         AnsiConsole.MarkupLine($"[dim]Using Replay reasoner (replaying from run {replayRunId})[/]")
                                         ReplayReasoner(replayRunDir, Log.Logger) :> IReasoner
                                 | ReasonStepMode.Stub ->
                                     { new IReasoner with 
                                         member _.Reason(_, _, _, _) = async { return Result.Ok "<cli-stub-reasoner>" } }
                             
                             let! result = 
                                 async {
                                     try
                                         return! WotExecutor.executePlanV0 invoker reasoner opts.Mode execPolicy inputs plan.Steps
                                     with ex ->
                                         return Result.Error (ex.Message, [])
                                 }
                             
                             let endTime = DateTime.UtcNow
                             let duration = int64 (endTime - startTime).TotalMilliseconds
                             
                             try
                                 System.IO.Directory.CreateDirectory(runDir) |> ignore
                                 let options = System.Text.Json.JsonSerializerOptions(WriteIndented = true)
                                 options.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
                                 
                                 let writeJson filename (obj: obj) =
                                     let path = System.IO.Path.Combine(runDir, filename)
                                     let json = System.Text.Json.JsonSerializer.Serialize(obj, options)
                                     System.IO.File.WriteAllText(path, json)
                                     AnsiConsole.MarkupLine($"[dim]Artifact saved: {path}[/]")

                                 writeJson "plan.json" plan
                                 
                                 let summary (traces: TraceEvent list) (passed: bool option) (err: string option) (outKeys: string list) (toolCalls: int option) =
                                     {| 
                                        RunId = runId
                                        DurationMs = duration
                                        ToolCalls = toolCalls |> Option.defaultValue (traces |> List.filter (fun t -> t.Kind = "tool") |> List.length)
                                        VerifyPassed = passed
                                        FirstError = err
                                        Outputs = outKeys
                                        Mode = opts.Mode.ToString()
                                        Reasoner = {|
                                            Model = opts.Model
                                            ModelHint = opts.ModelHint
                                            Temperature = opts.Temperature
                                            MaxTokens = opts.MaxTokens
                                            Deterministic = opts.Deterministic
                                            Seed = opts.Seed
                                        |}
                                     |}

                                 match result with
                                 | Result.Error (e, traces) ->
                                     writeJson "trace.json" traces
                                     writeJson "run_summary.json" (summary traces None (Some e) [] None)
                                     AnsiConsole.MarkupLine($"[red]Execution Failed:[/] {Markup.Escape(e)}")
                                     return 1
                                 | Result.Ok (ctx, verifyC, traces) ->
                                     writeJson "trace.json" traces
                                     writeJson "outputs.json" ctx.Vars
                                     
                                     let passed = verifyC |> Option.map (fun v -> v.Passed)
                                     let toolCalls = traces |> List.filter (fun t -> t.Kind = "tool") |> List.length
                                     let sumObj = summary traces passed None (ctx.Vars |> Map.toList |> List.map fst) (Some toolCalls)
                                     writeJson "run_summary.json" sumObj

                                     let golden = 
                                         { SchemaVersion = "wot.golden.v1"
                                           Steps = traces |> List.map TraceEvent.toCanonical
                                           Summary = {| ToolCalls = sumObj.ToolCalls
                                                        VerifyPassed = sumObj.VerifyPassed
                                                        FirstError = sumObj.FirstError
                                                        OutputKeys = sumObj.Outputs
                                                        Mode = sumObj.Mode |} }
                                     writeJson "golden.json" golden
                                     
                                     AnsiConsole.MarkupLine("\n[bold green]Execution Complete[/]")
                                     
                                     if not ctx.Vars.IsEmpty then
                                         AnsiConsole.MarkupLine("[dim]Outputs:[/]")
                                         for kvp in ctx.Vars do
                                             AnsiConsole.MarkupLine($"  {kvp.Key} = [cyan]{Markup.Escape(kvp.Value.ToString())}[/]")
                                     
                                     match verifyC with
                                     | None -> ()
                                     | Some v ->
                                         if v.Passed then
                                             AnsiConsole.MarkupLine("\n[bold green]Verification PASSED[/]")
                                         else
                                             AnsiConsole.MarkupLine("\n[bold red]Verification FAILED[/]")
                                             for e in v.Errors do
                                                 AnsiConsole.MarkupLine($"  - {Markup.Escape(e)}")
                                     
                                     return 0
                             with ex ->
                                 AnsiConsole.MarkupLine($"[red]Failed to save artifacts: {ex.Message}[/]")
                                 return 1
        } |> Async.StartAsTask
