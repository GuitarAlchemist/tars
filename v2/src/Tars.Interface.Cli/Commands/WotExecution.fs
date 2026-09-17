namespace Tars.Interface.Cli.Commands

open System
open Serilog
open Spectre.Console
open Tars.Core
open Tars.Core.WorkflowOfThought
open Tars.DSL.Wot
open Tars.Evolution
open Tars.Interface.Cli
open Tars.Interface.Cli.Reasoning
open Tars.Tools
open Tars.Cortex.WoTTypes

/// Running a compiled `.wot.trsx` plan (#44): the V0 executor and the bridge to the
/// Cortex executor, shared by every `tars wot` subcommand instead of rebuilt in each.
module WotExecution =

    // =========================================================================
    // Bridge: DSL Plan -> Cortex WoTPlan
    // =========================================================================

    /// Convert a compiled DSL Step into a Cortex WoTNode.
    let private stepToWoTNode (step: Step) : WoTNode =
        let kind, payload =
            match step.Action with
            | StepAction.Reason op ->
                let prompt =
                    match op with
                    | ReasonOperation.Plan goal -> $"Plan: {goal}"
                    | ReasonOperation.Generate topic -> $"Generate: {topic}"
                    | ReasonOperation.Explain topic -> $"Explain: {topic}"
                    | ReasonOperation.Critique(NodeId target) -> $"Critique the output of step '{target}'."
                    | ReasonOperation.Synthesize sources ->
                        let ids = sources |> List.map (fun (NodeId s) -> s) |> String.concat ", "
                        $"Synthesize the outputs of: {ids}"
                    | ReasonOperation.Rewrite(NodeId target, instruction) -> $"Rewrite output of '{target}': {instruction}"
                    | ReasonOperation.Aggregate sources ->
                        let ids = sources |> List.map (fun (NodeId s) -> s) |> String.concat ", "
                        $"Aggregate: {ids}"
                    | ReasonOperation.Refine(NodeId target) -> $"Refine the output of step '{target}'."
                    | ReasonOperation.Contradict(NodeId target) -> $"Generate a counterargument to the output of step '{target}'."
                    | ReasonOperation.Distill(NodeId target) -> $"Distill the key points from step '{target}'."
                    | ReasonOperation.Backtrack(NodeId target) -> $"Reconsider and revise the approach taken in step '{target}'."
                    | ReasonOperation.Score(NodeId target) -> $"Score the quality of step '{target}' output."
                    | ReasonOperation.VerifyConsensus _ -> "Verify consensus among outputs."
                WoTNodeKind.Reason, ({ ReasonPayload.Prompt = prompt; Hint = None } :> obj)

            | StepAction.Work workOp ->
                match workOp with
                | WorkOperation.ToolCall(toolName, args) ->
                    WoTNodeKind.Tool, ({ ToolPayload.Tool = toolName; Args = args } :> obj)
                | WorkOperation.Verify checks ->
                    let invariants =
                        checks |> List.mapi (fun i check ->
                            let op =
                                match check with
                                | WotCheck.NonEmpty _ -> VerificationOp.CustomOp "non_empty"
                                | WotCheck.Contains(_, needle) -> VerificationOp.Contains needle
                                | WotCheck.RegexMatch(_, pattern) -> VerificationOp.Regex pattern
                                | WotCheck.SchemaMatch(_, schema) -> VerificationOp.Schema schema
                                | WotCheck.ToolResult(tool, args, _) ->
                                    VerificationOp.ToolCheck(tool, args |> Map.map (fun _ v -> v :> obj))
                                | WotCheck.Threshold(metric, _, value) ->
                                    VerificationOp.CustomOp $"threshold:{metric}:{value}"
                            { WoTInvariant.Name = $"check_{i}"; Op = op; Weight = 1.0 })
                    WoTNodeKind.Validate, ({ ValidatePayload.Invariants = invariants } :> obj)
                | _ ->
                    // For other work operations (Redact, Persist, Fetch, Transform),
                    // map to a Reason node with description
                    WoTNodeKind.Reason, ({ ReasonPayload.Prompt = $"Execute work operation: {workOp}"; Hint = None } :> obj)

        let extraMeta =
            step.Metadata
            |> Map.fold (fun acc k v ->
                match v with
                | MStr s -> acc |> Map.add k s
                | _ -> acc) Map.empty

        { WoTNode.Id = step.Id
          Kind = kind
          Payload = payload
          Metadata =
            { Label = Some step.Id
              Tags = []
              Extra = extraMeta } }

    /// Convert DSL Plan edges (implicit from step ordering) and metadata into WoTEdges.
    let private stepsToWoTEdges (steps: Step list) : WoTEdge list =
        steps
        |> List.pairwise
        |> List.map (fun (a, b) ->
            { WoTEdge.From = a.Id
              To = b.Id
              Label = None
              Confidence = None })

    /// Bridge a compiled DSL Plan to a Cortex WoTPlan.
    let toCortexPlan (plan: Plan<Parsed>) : WoTPlan =
        let nodes = plan.Steps |> List.map stepToWoTNode
        let edges = stepsToWoTEdges plan.Steps
        let entryNode = plan.Steps |> List.tryHead |> Option.map (fun s -> s.Id) |> Option.defaultValue ""
        { WoTPlan.Id = Guid.NewGuid()
          Nodes = nodes
          Edges = edges
          EntryNode = entryNode
          Metadata =
            { Kind = WorkflowOfThought
              SourceGoal = plan.Goal
              CompiledAt = DateTime.UtcNow
              EstimatedTokens = None
              EstimatedSteps = Some plan.Steps.Length }
          Policy = plan.Policy.AllowedTools |> Set.toList }

    /// Tool registry for V0 runs: every [TarsTool], plus fakes for the build and
    /// environment tools the demo workflows call when the real ones are absent.
    let v0Tools () : IToolRegistry =
        let tools = ToolRegistry()
        tools.RegisterAssembly(typeof<Tars.Tools.TarsToolAttribute>.Assembly)

        if tools.Get("dotnet_build").IsNone then
            tools.Register(
                Tool.InternalCreateMinimal(
                    "dotnet_build",
                    "Fake build",
                    fun _ -> async { return Result.Ok "Build successful" }
                )
            )

        if tools.Get("check_environment").IsNone then
            tools.Register(
                Tool.InternalCreateMinimal(
                    "check_environment",
                    "Fake check",
                    fun _ -> async { return Result.Ok "Environment ready" }
                )
            )

        tools :> IToolRegistry

    /// Execute a plan with the V0 executor under the plan's own tool policy.
    /// An exception from the executor becomes an error with no traces.
    let runV0
        (tools: IToolRegistry)
        (reasoner: IReasoner)
        (reflector: ISymbolicReflector option)
        (constitution: AgentConstitution option)
        (mode: ReasonStepMode)
        (plan: Plan<Parsed>)
        =
        let policy: WotExecutor.ExecutionPolicy =
            { AllowedTools = plan.Policy.AllowedTools
              MaxToolCalls = plan.Policy.MaxToolCalls }

        async {
            try
                return!
                    WotExecutor.executePlanV0
                        (ToolInvoker.create tools)
                        reasoner
                        None
                        reflector
                        constitution
                        mode
                        policy
                        plan.Inputs
                        plan.Steps
                        (SymbolicMemorySink())
            with ex ->
                return Result.Error(ex.Message, [])
        }

    /// The golden record of a completed V0 run, which `wot ci-check` and `wot diff` compare.
    let toGolden (mode: ReasonStepMode) (ctx: ExecContext) (verify: VerifyResult option) (traces: TraceEvent list) =
        { SchemaVersion = "wot.golden.v1"
          Steps = traces |> List.map TraceEvent.toCanonical
          Summary =
            { ToolCalls = traces |> List.filter (fun t -> t.Kind = "tool") |> List.length
              VerifyPassed = verify |> Option.map (fun v -> v.Passed)
              FirstError = None
              OutputKeys = ctx.Vars |> Map.toList |> List.map fst
              Mode = mode.ToString()
              PassRate = None
              EstimatedCost = 0m
              DiffCount = 0
              TotalTokens = 0 } }

    /// Options for run command
    type RunOptions =
        { Mode: ReasonStepMode
          Model: string option
          ModelHint: string option
          Temperature: float option
          MaxTokens: int option
          Deterministic: bool
          Seed: int option
          ReplayRunId: string option
          UseCortex: bool }

        static member Default =
            { Mode = ReasonStepMode.Stub
              Model = None
              ModelHint = None
              Temperature = None
              MaxTokens = None
              Deterministic = false
              Seed = None
              ReplayRunId = None
              UseCortex = false }

    /// One way to run a compiled workflow. Each executor owns its own reporting and
    /// journaling and returns the process exit code, so `wot run` only has to pick one.
    type IWorkflowExecutor =
        abstract Execute: Plan<Parsed> -> Async<int>

    /// The default executor: `executePlanV0` plus the run journal under `.wot/runs/<runId>`.
    type V0Executor(opts: RunOptions) =

        interface IWorkflowExecutor with
            member _.Execute(plan: Plan<Parsed>) : Async<int> =
                async {
                    let startTime = DateTime.UtcNow
                    let runId = startTime.ToString("yyyyMMdd-HHmmss")
                    let runDir = System.IO.Path.Combine(".wot", "runs", runId)
                    System.IO.Directory.CreateDirectory(runDir) |> ignore

                    AnsiConsole.MarkupLine("[bold]Executing Steps...[/]")

                    // Wire up reasoner based on mode
                    let reasoner: IReasoner =
                        match opts.Mode with
                        | ReasonStepMode.Llm ->
                            let logger = Log.Logger
                            let llm = LlmFactory.create logger

                            let reasonerSettings: ReasonerSettings =
                                { Model = opts.Model
                                  ModelHint = opts.ModelHint |> Option.orElse (Some "thought")
                                  Temperature = opts.Temperature
                                  MaxTokens = opts.MaxTokens
                                  Deterministic = opts.Deterministic
                                  Seed = opts.Seed
                                  ContextWindow = None
                                  AgentHint = None
                                  GrammarConstraint = None }

                            let modelStr = opts.Model |> Option.defaultValue "<default>"

                            let tempStr =
                                opts.Temperature |> Option.map string |> Option.defaultValue "<default>"

                            AnsiConsole.MarkupLine(
                                $"[dim]Using LLM reasoner (model={modelStr}, temp={tempStr}, deterministic={opts.Deterministic})[/]"
                            )

                            CliReasoner(llm, runDir, reasonerSettings, logger) :> IReasoner
                        | ReasonStepMode.Replay ->
                            match opts.ReplayRunId with
                            | None -> failwith "--reason replay requires --replay-run <runId>"
                            | Some replayRunId ->
                                let replayRunDir = System.IO.Path.Combine(".wot", "runs", replayRunId)

                                if not (System.IO.Directory.Exists replayRunDir) then
                                    failwith $"Replay run directory not found: {replayRunDir}"

                                AnsiConsole.MarkupLine(
                                    $"[dim]Using Replay reasoner (replaying from run {replayRunId})[/]"
                                )

                                ReplayReasoner(replayRunDir, Log.Logger) :> IReasoner
                        | ReasonStepMode.Stub ->
                            { new IReasoner with
                                member _.Reason(_, _, _, _, _) =
                                    async {
                                        return
                                            Result.Ok
                                                { Content = "<cli-stub-reasoner>"
                                                  Usage = None }
                                    } }

                    let! result =
                        runV0 (v0Tools ()) reasoner None None opts.Mode plan

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

                        let summary
                            (traces: TraceEvent list)
                            (passed: bool option)
                            (err: string option)
                            (outKeys: string list)
                            (toolCalls: int option)
                            =
                            {| RunId = runId
                               DurationMs = duration
                               ToolCalls =
                                toolCalls
                                |> Option.defaultValue (
                                    traces |> List.filter (fun t -> t.Kind = "tool") |> List.length
                                )
                               VerifyPassed = passed
                               FirstError = err
                               Outputs = outKeys
                               Mode = opts.Mode.ToString()
                               Reasoner =
                                {| Model = opts.Model
                                   ModelHint = opts.ModelHint
                                   Temperature = opts.Temperature
                                   MaxTokens = opts.MaxTokens
                                   Deterministic = opts.Deterministic
                                   Seed = opts.Seed |} |}

                        match result with
                        | Result.Error(e, traces) ->
                            writeJson "trace.json" traces
                            writeJson "run_summary.json" (summary traces None (Some e) [] None)
                            AnsiConsole.MarkupLine($"[red]Execution Failed:[/] {Markup.Escape(e)}")
                            return 1
                        | Result.Ok(ctx, verifyC, traces) ->
                            writeJson "trace.json" traces
                            writeJson "outputs.json" ctx.Vars

                            let passed = verifyC |> Option.map (fun v -> v.Passed)
                            let toolCalls = traces |> List.filter (fun t -> t.Kind = "tool") |> List.length

                            // Phase 15.6 Pattern Compilation Trigger
                            // Treat None as success if there are traces (implicit success)
                            let isImplicitSuccess = passed |> Option.defaultValue true

                            if isImplicitSuccess && opts.Mode = ReasonStepMode.Llm then
                                AnsiConsole.MarkupLine(
                                    "[bold blue]🧠 Compiling Pattern from successful run...[/]"
                                )

                                let llm = LlmFactory.create Log.Logger
                                let canonicalTraces = traces |> List.map TraceEvent.toCanonical

                                try
                                    let! patternRes =
                                        TraceCompiler.compileFromTrace
                                            llm
                                            (Guid.NewGuid())
                                            canonicalTraces
                                            plan.Goal

                                    match patternRes with
                                    | Result.Ok p ->
                                        AnsiConsole.MarkupLine($"[green]Pattern Compiled:[/] {p.Name}")

                                        if not (System.IO.Directory.Exists(".tars/patterns")) then
                                            System.IO.Directory.CreateDirectory(".tars/patterns") |> ignore

                                        let pJson = System.Text.Json.JsonSerializer.Serialize(p, options)
                                        System.IO.File.WriteAllText($".tars/patterns/{p.Name}.json", pJson)
                                    | Result.Error e ->
                                        AnsiConsole.MarkupLine($"[yellow]Pattern Compile Failed:[/] {e}")
                                with ex ->
                                    AnsiConsole.MarkupLine($"[yellow]Pattern Compile Error:[/] {ex.Message}")

                            let sumObj =
                                summary
                                    traces
                                    passed
                                    None
                                    (ctx.Vars |> Map.toList |> List.map fst)
                                    (Some toolCalls)

                            writeJson "run_summary.json" sumObj
                            writeJson "golden.json" (toGolden opts.Mode ctx verifyC traces)

                            AnsiConsole.MarkupLine("\n[bold green]Execution Complete[/]")

                            if not ctx.Vars.IsEmpty then
                                AnsiConsole.MarkupLine("[dim]Outputs:[/]")

                                for kvp in ctx.Vars do
                                    AnsiConsole.MarkupLine(
                                        $"  {kvp.Key} = [cyan]{Markup.Escape(kvp.Value.ToString())}[/]"
                                    )

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
                }

    /// `--cortex`: bridges the plan to a Cortex WoTPlan and runs the Cortex executor.
    type CortexExecutor(llm: Tars.Llm.ILlmService, toolRegistry: ToolRegistry) =

        interface IWorkflowExecutor with
            member _.Execute(plan: Plan<Parsed>) : Async<int> =
                AnsiConsole.MarkupLine("[bold]Executing via Cortex WoT Executor...[/]")

                async {
                    let wotPlan = toCortexPlan plan
                    let cortexToolRegistry = toolRegistry :> IToolRegistry

                    let executor =
                        Tars.Cortex.WoTExecutor.DefaultWoTExecutor(llm, cortexToolRegistry)
                        :> IWoTExecutor

                    let agentCtx = AgentHelpers.createAgentContext (fun msg -> AnsiConsole.MarkupLine($"[dim]{Markup.Escape(msg)}[/]")) llm None

                    let mutable stepCount = 0

                    let onProgress (step: WoTTraceStep) =
                        stepCount <- stepCount + 1
                        let statusStr =
                            match step.Status with
                            | Completed(_, ms) -> $"[green]OK[/] ({ms}ms)"
                            | Failed(err, ms) -> $"[red]FAIL[/] ({ms}ms): {Markup.Escape(err)}"
                            | Skipped reason -> $"[yellow]SKIP[/]: {Markup.Escape(reason)}"
                            | Pending -> "[dim]pending[/]"
                            | Running -> "[blue]running[/]"
                        let outputPreview =
                            match step.Output with
                            | Some o when o.Length > 120 -> Markup.Escape(o.Substring(0, 120)) + "..."
                            | Some o -> Markup.Escape(o)
                            | None -> "[dim]<none>[/]"
                        AnsiConsole.MarkupLine($"  [{stepCount}] [bold]{Markup.Escape(step.NodeId)}[/] ({step.NodeType}) {statusStr}")
                        AnsiConsole.MarkupLine($"      Output: {outputPreview}")

                    AnsiConsole.MarkupLine($"[bold blue]Cortex WoT Executor[/] - {wotPlan.Nodes.Length} nodes")
                    AnsiConsole.MarkupLine("")

                    let! result = executor.ExecuteWithProgress(wotPlan, agentCtx, onProgress)

                    AnsiConsole.MarkupLine("")

                    // Print summary
                    if result.Success then
                        AnsiConsole.MarkupLine("[bold green]Execution Succeeded[/]")
                    else
                        AnsiConsole.MarkupLine("[bold red]Execution Failed[/]")

                    AnsiConsole.MarkupLine($"  Steps: {result.Metrics.TotalSteps} total, {result.Metrics.SuccessfulSteps} succeeded, {result.Metrics.FailedSteps} failed")
                    AnsiConsole.MarkupLine($"  Duration: {result.Metrics.TotalDurationMs}ms")
                    AnsiConsole.MarkupLine($"  Tokens: {result.Metrics.TotalTokens}")

                    if not result.ToolsUsed.IsEmpty then
                        let toolList = String.Join(", ", result.ToolsUsed)
                        AnsiConsole.MarkupLine($"  Tools Used: {toolList}")

                    if not result.Warnings.IsEmpty then
                        AnsiConsole.MarkupLine("[yellow]Warnings:[/]")
                        for w in result.Warnings do
                            AnsiConsole.MarkupLine($"  - {Markup.Escape(w)}")

                    if not result.Errors.IsEmpty then
                        AnsiConsole.MarkupLine("[red]Errors:[/]")
                        for e in result.Errors do
                            AnsiConsole.MarkupLine($"  - {Markup.Escape(e)}")

                    match result.CognitiveStateAfter with
                    | Some state ->
                        AnsiConsole.MarkupLine($"  Cognitive State: {state.Mode} (Entropy: {state.Entropy:F2}, Eigenvalue: {state.Eigenvalue:F2})")
                    | None -> ()

                    // Print final output
                    if not (String.IsNullOrWhiteSpace result.Output) then
                        AnsiConsole.MarkupLine("")
                        AnsiConsole.MarkupLine("[bold]Final Output:[/]")
                        let preview =
                            if result.Output.Length > 2000 then result.Output.Substring(0, 2000) + "..."
                            else result.Output
                        AnsiConsole.MarkupLine(Markup.Escape(preview))

                    return if result.Success then 0 else 1
                }
