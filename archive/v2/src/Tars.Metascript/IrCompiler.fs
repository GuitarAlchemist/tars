namespace Tars.Metascript

open System
open Tars.Core.HybridBrain
open Tars.Core.HybridBrain
open Tars.Core.WorkflowOfThought
open Tars.Metascript.Domain

module IrCompiler =

    /// Compiles a Metascript Step into a Typed IR Step
    let compileStep (idMap: Map<string, int>) (step: WorkflowStep) (index: int) : Tars.Core.HybridBrain.Step =
        let getParam (key: string) =
            step.Params |> Option.bind (fun p -> p.TryFind key)

        let parseStepRefs (value: string) =
            value.Split([| ',' |], StringSplitOptions.RemoveEmptyEntries)
            |> Array.map (fun s -> s.Trim())
            |> Array.choose (fun id -> idMap.TryFind id)
            |> Array.toList

        let parseStepList (keys: string list) =
            keys
            |> List.tryPick getParam
            |> Option.map parseStepRefs
            |> Option.defaultValue []

        let parseBool (key: string) =
            match getParam key with
            | Some v when v.Equals("true", StringComparison.OrdinalIgnoreCase) -> true
            | Some v when v.Equals("yes", StringComparison.OrdinalIgnoreCase) -> true
            | _ -> false

        let parseInt (key: string) (fallback: int) =
            match getParam key with
            | Some v ->
                match Int32.TryParse(v) with
                | true, parsed -> parsed
                | _ -> fallback
            | None -> fallback

        let action =
            match step.Type.ToLowerInvariant() with
            | "tool" ->
                match step.Tool with
                | Some toolName ->
                    UseTool(
                        Tars.Core.HybridBrain.Tool.Custom(
                            toolName,
                            step.Params |> Option.defaultValue Map.empty |> Map.map (fun k v -> v :> obj)
                        )
                    )
                | None -> NoOp
            | "agent" ->
                let prompt = step.Instruction |> Option.defaultValue ""
                let model = step.Params |> Option.bind (fun p -> p.TryFind "model")
                UseTool(LlmCall(prompt, model))
            | "map" ->
                let steps = parseStepList [ "steps"; "body"; "targets" ]

                if not steps.IsEmpty then
                    if parseBool "parallel" || getParam "mode" = Some "parallel" then
                        Action.Parallel(steps |> List.map (fun s -> [ s ]))
                    else
                        Action.Sequence steps
                else
                    UseTool(
                        Tars.Core.HybridBrain.Tool.Custom(
                            "map",
                            step.Params |> Option.defaultValue Map.empty |> Map.map (fun k v -> v :> obj)
                        )
                    )
            | "loop" ->
                let steps = parseStepList [ "steps"; "body" ]
                let maxIterations = parseInt "maxIterations" 1
                let condition = step.Instruction |> Option.defaultValue "true"

                if not steps.IsEmpty then
                    Action.Loop(condition, steps, maxIterations)
                else
                    UseTool(
                        Tars.Core.HybridBrain.Tool.Custom(
                            "loop",
                            step.Params |> Option.defaultValue Map.empty |> Map.map (fun k v -> v :> obj)
                        )
                    )
            | "decision" ->
                let thenSteps = parseStepList [ "then"; "thenSteps" ]
                let elseSteps = parseStepList [ "else"; "elseSteps" ]
                let condition = step.Instruction |> Option.defaultValue "true"

                if not thenSteps.IsEmpty || not elseSteps.IsEmpty then
                    Action.Branch(condition, thenSteps, elseSteps)
                else
                    UseTool(
                        Tars.Core.HybridBrain.Tool.Custom(
                            "decision",
                            step.Params |> Option.defaultValue Map.empty |> Map.map (fun k v -> v :> obj)
                        )
                    )
            | _ -> NoOp

        { Tars.Core.HybridBrain.Step.Id = index + 1
          Name = step.Id
          Description = step.Instruction |> Option.defaultValue step.Id
          Action = action
          Preconditions = []
          Postconditions = []
          EvidenceRequired = false
          Timeout = None
          RetryCount = 3 }

    /// Compiles a Metascript Workflow into a Typed IR Plan (Draft)
    let compile (workflow: Workflow) : Plan<Draft> =
        let idMap = workflow.Steps |> List.mapi (fun i s -> s.Id, i + 1) |> Map.ofList

        let steps = workflow.Steps |> List.mapi (fun i s -> compileStep idMap s i)

        let compiledGoal: Tars.Core.HybridBrain.Goal =
            { Id = Guid.NewGuid()
              Description = workflow.Name
              SuccessCriteria = []
              Priority = 1
              Deadline = None
              Motivations = [] }

        let draft: Plan<Draft> =
            { Id = Guid.NewGuid()
              Goal = compiledGoal
              Description = workflow.Description
              Steps = steps
              Assumptions = []
              Unknowns = []
              RequiresSources = []
              Budget =
                { MaxTokens = 10000
                  MaxTime = TimeSpan.FromMinutes(10.0)
                  MaxCost = 1.0m
                  MaxMemory = 100L * 1024L * 1024L
                  MaxApiCalls = 50 }
              Policy =
                { AllowedTools = None
                  ForbiddenTools = []
                  AllowedDomains = None
                  ForbiddenDomains = []
                  RequireEvidence = false
                  MaxConfidenceWithoutEvidence = 0.5
                  SandboxOnly = false }
              CreatedAt = DateTimeOffset.UtcNow
              CreatedBy = "MetascriptCompiler"
              Version = 1
              ParentVersion = None }

        draft

    /// Compiles a WorkflowGraph (from .trsx) into a Typed IR Plan (Draft)
    let compileFromGraph (graph: WorkflowGraph) : Plan<Draft> =
        // 1. Topological Sort (naive - assume linear or simple dependencies for now)
        // For a full compiler, we'd do a real topological sort.
        // Here we just traverse from EntryPoint if linear, or list all nodes if disconnected.

        // Simple strategy: Sort by dependencies
        let sortedNodes =
            let visited = System.Collections.Generic.HashSet<NodeId>()
            let result = System.Collections.Generic.List<NodeId>()

            // Dependency Map: Node -> List of Dependencies
            let dependencies =
                graph.Edges
                |> List.choose (fun (src, edge, target) -> if edge = DependsOn then Some(src, target) else None)
                |> List.groupBy fst
                |> Map.ofList
                |> Map.map (fun _ list -> list |> List.map snd)

            let rec visit (id: NodeId) =
                if not (visited.Contains id) then
                    visited.Add id |> ignore
                    // Visit dependencies first
                    match Map.tryFind id dependencies with
                    | Some deps -> deps |> List.iter visit
                    | None -> ()

                    result.Add id

            graph.Nodes |> Map.iter (fun id _ -> visit id)

            result |> Seq.toList

        let idMap = sortedNodes |> List.mapi (fun i guid -> guid, i + 1) |> Map.ofList

        let compileNode (id: NodeId) (node: WotNode) (index: int) : Tars.Core.HybridBrain.Step =
            let name =
                match node with
                | Reason r -> r.Name
                | Work w -> w.Name

            let action =
                match node with
                | Reason r ->
                    match r.Operation with
                    | ReasonOperation.Plan goal ->
                        // Plan is usually the start. In IR, maybe just a comment or NoOp if implied?
                        // Or a call to LLM to Generate Plan?
                        // For .trsx, the "Plan" node usually *contains* the thought trace.
                        UseTool(LlmCall($"Plan: {goal}", r.Model))
                    | ReasonOperation.Critique target -> UseTool(LlmCall($"Critique target {target}", r.Model))
                    | ReasonOperation.Synthesize sources -> UseTool(LlmCall($"Synthesize sources", r.Model))
                    | ReasonOperation.Explain topic -> UseTool(LlmCall($"Explain {topic}", r.Model))
                    | ReasonOperation.Rewrite(target, instr) -> UseTool(LlmCall($"Rewrite {target}: {instr}", r.Model))

                | Work w ->
                    match w.Operation with
                    | WorkOperation.ToolCall(tool, args) ->
                        match tool.ToLower() with
                        | "extract_function" ->
                            let name =
                                args |> Map.tryFind "name" |> Option.map string |> Option.defaultValue "helper"

                            let start =
                                args
                                |> Map.tryFind "startLine"
                                |> Option.map (fun v ->
                                    try
                                        Convert.ToInt32(v)
                                    with _ ->
                                        0)
                                |> Option.defaultValue 0

                            let stop =
                                args
                                |> Map.tryFind "endLine"
                                |> Option.map (fun v ->
                                    try
                                        Convert.ToInt32(v)
                                    with _ ->
                                        0)
                                |> Option.defaultValue 0

                            ExtractFunction(name, start, stop)
                        | "add_documentation" ->
                            let text =
                                args |> Map.tryFind "docText" |> Option.map string |> Option.defaultValue ""

                            let line =
                                args
                                |> Map.tryFind "lineNumber"
                                |> Option.map (fun v ->
                                    try
                                        Convert.ToInt32(v)
                                    with _ ->
                                        0)
                                |> Option.defaultValue 0

                            AddDocumentation(text, line)
                        | "remove_lines"
                        | "remove_dead_code" ->
                            let start =
                                args
                                |> Map.tryFind "startLine"
                                |> Option.map (fun v ->
                                    try
                                        Convert.ToInt32(v)
                                    with _ ->
                                        0)
                                |> Option.defaultValue 0

                            let stop =
                                args
                                |> Map.tryFind "endLine"
                                |> Option.map (fun v ->
                                    try
                                        Convert.ToInt32(v)
                                    with _ ->
                                        0)
                                |> Option.defaultValue 0

                            RemoveDeadCode(start, stop)
                        | "rename_symbol" ->
                            let old = args |> Map.tryFind "old" |> Option.map string |> Option.defaultValue ""

                            let newName =
                                args |> Map.tryFind "new" |> Option.map string |> Option.defaultValue ""

                            RenameSymbol(old, newName)
                        | _ -> UseTool(Tars.Core.HybridBrain.Tool.Custom(tool, args))
                    | WorkOperation.Verify(check, expected) ->
                        // Map to AssertBelief or Custom Tool
                        UseTool(
                            Tars.Core.HybridBrain.Tool.Custom(
                                "verify",
                                Map [ ("check", box check); ("expected", box expected) ]
                            )
                        )
                    | WorkOperation.Redact patterns -> NoOp // Or specialized Action
                    | WorkOperation.Persist loc -> NoOp
                    | WorkOperation.Fetch src -> UseTool(WebSearch src)
                    | WorkOperation.Transform fn -> NoOp

            { Tars.Core.HybridBrain.Step.Id = index + 1
              Name = name
              Description = name
              Action = action
              Preconditions = []
              Postconditions = []
              EvidenceRequired = false
              Timeout = None
              RetryCount = 3 }

        let steps =
            graph.Nodes
            |> Map.toList
            |> List.map (fun (id, node) -> compileNode id node (idMap.[id] - 1)) // Use idMap for order?
            // Actually, let's just ignore topological sort for the demo and trust the Map order or implement a simple linear sort
            |> List.sortBy (fun s -> s.Id)

        // Fixup: The Map iteration order is non-deterministic or guid-based.
        // We *must* sort steps by dependency to make a valid plan.
        // Users of .trsx likely write in order.
        // Let's rely on the Edge "DependsOn" to re-order steps in a valid sequence if possible.
        // For this iteration, we return the steps in an arbitrary order but with valid IDs.
        // The Validator might complain about order if we had strict data dependencies, but TypedIR passes IDs.

        let compiledGoal: Tars.Core.HybridBrain.Goal =
            { Id = Guid.NewGuid()
              Description = graph.Name
              SuccessCriteria = []
              Priority = 1
              Deadline = None
              Motivations = [] }

        { Id = Guid.NewGuid()
          Goal = compiledGoal
          Description = graph.Description
          Steps = steps
          Assumptions = []
          Unknowns = []
          RequiresSources = []
          Budget = Budget.default'
          Policy =
            { AllowedTools = None
              ForbiddenTools = []
              AllowedDomains = None
              ForbiddenDomains = []
              RequireEvidence = false
              MaxConfidenceWithoutEvidence = 0.5
              SandboxOnly = false }
          CreatedAt = DateTimeOffset.UtcNow
          CreatedBy = "TrsxCompiler"
          Version = 1
          ParentVersion = None }
