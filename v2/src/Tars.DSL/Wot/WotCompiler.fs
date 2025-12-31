namespace Tars.DSL.Wot

open System
open Tars.Core.WorkflowOfThought

// Placeholder for Plan type - adapt to your actual IR
type Plan<'T> = {
    Goal: string
    Steps: Step list
    Policy: DslPolicy
    Inputs: DslInputs
    Version: int
}
and Parsed = Parsed 
// Removed local definitions of Step and Action to use Core ones

// ----- Errors -----
type CompileError =
  | InvalidGraph of string
  | InvalidNode of nodeId: string * message: string

// ----- Helpers -----
module private Graph =

  let inDegree (edges: (DslId*DslId) list) (n: DslId) =
    edges |> List.filter (fun (_,dst) -> dst = n) |> List.length

  let outDegree (edges: (DslId*DslId) list) (n: DslId) =
    edges |> List.filter (fun (src,_) -> src = n) |> List.length

  let nextOf (edges: (DslId*DslId) list) (n: DslId) : DslId option =
    edges |> List.tryPick (fun (src,dst) -> if src = n then Some dst else None)

  let validateChain (nodes: DslNode list) (edges: (DslId*DslId) list) : Result<DslId, CompileError> =
    if edges.Length <> nodes.Length - 1 then
      Error (InvalidGraph $"Expected edges = nodes - 1, got edges={edges.Length} nodes={nodes.Length}")
    else
      let starts =
        nodes
        |> List.filter (fun n -> inDegree edges n.Id = 0)
      match starts with
      | [start] -> Ok start.Id
      | [] -> Error (InvalidGraph "No start node (in-degree 0) found.")
      | _ -> Error (InvalidGraph "Multiple start nodes found (graph is not a single chain).")

  let orderChain (nodes: DslNode list) (edges: (DslId*DslId) list) (start: DslId) : Result<DslNode list, CompileError> =
    let nodeById = nodes |> List.map (fun n -> n.Id, n) |> Map.ofList

    let rec loop (current: DslId) (visited: Set<DslId>) (acc: DslNode list) =
      if visited.Contains current then
        Error (InvalidGraph $"Cycle detected at node {current}.")
      else
        let n =
          match nodeById.TryFind current with
          | Some x -> x
          | None -> failwith "Impossible: start/next not in nodeById"

        match nextOf edges current with
        | None ->
            // end
            Ok (List.rev (n :: acc))
        | Some nxt ->
            loop nxt (visited.Add current) (n :: acc)

    loop start Set.empty []

// ----- Compiler -----
module WotCompiler =

  let private compileWorkNodeToStep (node: DslNode) : Result<Step, CompileError> =
    match node.Tool, node.Checks with
    | Some toolName, _ ->
        let args = defaultArg node.Args Map.empty
        // Mapping generic map to ToolArgs if needed
        let action = StepAction.Work (ToolCall (toolName, args))
        Ok { Id = node.Id; Inputs = node.Inputs; Outputs = node.Outputs; Action = action }
        
    | None, checks when not (List.isEmpty checks) ->
        // Checks are already WotCheck (shared type or mapped)
        let action = StepAction.Work (Verify checks)
        Ok { Id = node.Id; Inputs = node.Inputs; Outputs = node.Outputs; Action = action }
        
    | _ ->
        Error (InvalidNode (node.Id, "Work node must have either tool or checks."))

  let private compileReasonNodeToStep (node: DslNode) : Result<Step, CompileError> =
    // Mapping Reason node to ReasonOperation
    // This requires adapting the DSL fields to the Core ReasonOperation union cases
    // For v0, let's map generic "Reason" to a "Plan" or "Explain" if possible, 
    // or assume a generic reason step.
    
    // Since ReasonOperation is a DU, we need to pick one.
    // Based on node.Goal, we could map to Plan goal.
    // Or based on inputs/outputs.
    
    let op = 
        match node.Goal with
        | Some g -> Plan g
        | None -> Explain (node.Name) // Fallback
        
    let action = StepAction.Reason op
    Ok { Id = node.Id; Inputs = node.Inputs; Outputs = node.Outputs; Action = action }

  let private compileNodeToStep (node: DslNode) : Result<Step, CompileError> =
    match node.Kind with
    | NodeKind.Work -> compileWorkNodeToStep node
    | NodeKind.Reason -> compileReasonNodeToStep node

  let compileWorkflowToPlanParsed (wf: DslWorkflow) : Result<Plan<Parsed>, CompileError list> =
    // 1) validate + order
    match Graph.validateChain wf.Nodes wf.Edges with
    | Error e -> Error [e]
    | Ok start ->
        match Graph.orderChain wf.Nodes wf.Edges start with
        | Error e -> Error [e]
        | Ok orderedNodes ->
            // 2) compile steps
            let stepResults = orderedNodes |> List.map compileNodeToStep

            let errors =
              stepResults
              |> List.choose (function Error e -> Some e | _ -> None)

            if errors.Length > 0 then
              Error errors
            else
              let steps =
                stepResults
                |> List.choose (function Ok s -> Some s | _ -> None)

              // 3) build Plan<Parsed>
              let planParsed : Plan<Parsed> =
                { Goal = wf.Name
                  Steps = steps
                  Policy = wf.Policy
                  Inputs = wf.Inputs
                  Version = 1 }

              Ok planParsed
