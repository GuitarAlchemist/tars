namespace Tars.Interface.Cli.Commands

open System
open Tars.Core
open Tars.Core.WorkflowOfThought
open Tars.DSL.Wot
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
