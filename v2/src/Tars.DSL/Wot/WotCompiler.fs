namespace Tars.DSL.Wot

open Tars.Core.WorkflowOfThought

// Placeholder for Plan type - adapt to your actual IR
type Plan<'T> =
    { Goal: string
      Steps: Step list
      Policy: DslPolicy
      Inputs: DslInputs
      Version: int }

and Parsed = Parsed
// Removed local definitions of Step and Action to use Core ones

// ----- Helper to extract output names -----
module private OutputHelper =
    let toOutputName (output: NodeOutput) : string =
        match output with
        | SimpleOutput name -> name
        | StructuredOutput(name, _) -> name

    let toOutputNames (outputs: NodeOutput list) : string list = outputs |> List.map toOutputName

    let toAgentStr (a: NodeAgent) : string option =
        match a with
        | ByRole r -> Some r
        | ById r -> Some r
        | Default -> None

// ----- Errors -----

type CompileError =
    | InvalidGraph of string
    | InvalidNode of nodeId: string * message: string

// ----- Helpers -----
module private Graph =

    let inDegree (edges: (DslId * DslId) list) (n: DslId) =
        edges |> List.filter (fun (_, dst) -> dst = n) |> List.length

    let outDegree (edges: (DslId * DslId) list) (n: DslId) =
        edges |> List.filter (fun (src, _) -> src = n) |> List.length

    let nextOf (edges: (DslId * DslId) list) (n: DslId) : DslId option =
        edges |> List.tryPick (fun (src, dst) -> if src = n then Some dst else None)

    let validateChain (nodes: DslNode list) (edges: (DslId * DslId) list) : Result<DslId, CompileError> =
        if edges.Length <> nodes.Length - 1 then
            Error(InvalidGraph $"Expected edges = nodes - 1, got edges={edges.Length} nodes={nodes.Length}")
        else
            let starts = nodes |> List.filter (fun n -> inDegree edges n.Id = 0)

            match starts with
            | [ start ] -> Ok start.Id
            | [] -> Error(InvalidGraph "No start node (in-degree 0) found.")
            | _ -> Error(InvalidGraph "Multiple start nodes found (graph is not a single chain).")

    let orderChain
        (nodes: DslNode list)
        (edges: (DslId * DslId) list)
        (start: DslId)
        : Result<DslNode list, CompileError> =
        let nodeById = nodes |> List.map (fun n -> n.Id, n) |> Map.ofList

        let rec loop (current: DslId) (visited: Set<DslId>) (acc: DslNode list) =
            if visited.Contains current then
                Error(InvalidGraph $"Cycle detected at node {current}.")
            else
                let n =
                    match nodeById.TryFind current with
                    | Some x -> x
                    | None -> failwith "Impossible: start/next not in nodeById"

                match nextOf edges current with
                | None ->
                    // end
                    Ok(List.rev (n :: acc))
                | Some nxt -> loop nxt (visited.Add current) (n :: acc)

        loop start Set.empty []

    /// Topological sort using Kahn's algorithm. Works for DAGs including parallel fan-out/fan-in.
    let topoSort (nodes: DslNode list) (edges: (DslId * DslId) list) : Result<DslNode list, CompileError> =
        let nodeById = nodes |> List.map (fun n -> n.Id, n) |> Map.ofList
        let mutable inDeg = nodes |> List.map (fun n -> n.Id, inDegree edges n.Id) |> Map.ofList
        let mutable queue = nodes |> List.filter (fun n -> inDeg.[n.Id] = 0) |> List.map (fun n -> n.Id)
        let mutable result = []
        let mutable visited = 0

        while not (List.isEmpty queue) do
            let current = List.head queue
            queue <- List.tail queue
            result <- current :: result
            visited <- visited + 1
            // Decrease in-degree for successors
            for (src, dst) in edges do
                if src = current then
                    inDeg <- inDeg.Add(dst, inDeg.[dst] - 1)
                    if inDeg.[dst] = 0 then
                        queue <- queue @ [ dst ]

        if visited <> nodes.Length then
            Error(InvalidGraph "Cycle detected in workflow graph.")
        else
            let ordered = result |> List.rev |> List.choose (fun id -> nodeById.TryFind id)
            Ok ordered

    /// Validate a DAG (allows parallel structure, not just strict chains).
    /// Returns the list of start nodes (in-degree 0).
    let validateDAG (nodes: DslNode list) (edges: (DslId * DslId) list) : Result<DslId list, CompileError> =
        let starts = nodes |> List.filter (fun n -> inDegree edges n.Id = 0)
        match starts with
        | [] -> Error(InvalidGraph "No start node (in-degree 0) found.")
        | starts ->
            // Verify it's a valid DAG via topo sort
            match topoSort nodes edges with
            | Error e -> Error e
            | Ok _ -> Ok (starts |> List.map (fun n -> n.Id))

// ----- Compiler -----
module WotCompiler =

    /// Attach condition to step metadata when present on the DSL node.
    let private addConditionMetadata (node: DslNode) (meta: Meta) : Meta =
        match node.Condition with
        | Some cond -> meta.Add("condition", MStr cond)
        | None -> meta

    /// Attach parallel group membership to step metadata if applicable.
    let private addParallelMetadata (parallelGroups: ParallelGroup list) (nodeId: DslId) (meta: Meta) : Meta =
        let group = parallelGroups |> List.tryFind (fun g -> g.NodeIds |> List.contains nodeId)
        match group with
        | Some g -> meta.Add("parallel_group", MStr g.GroupId)
        | None -> meta

    /// Resolve ${variable} references in a goal string using workflow inputs.
    /// At compile time we only have workflow-level inputs (not runtime vars),
    /// so this does a best-effort substitution. Variables not found in inputs
    /// are left as-is for runtime resolution by the executor.
    let private resolveGoalVariables (inputs: DslInputs) (goal: string) : string =
        let ctx : ExecContext = { Inputs = inputs; Vars = Map.empty }
        match VariableResolution.resolveString ctx goal with
        | Ok resolved -> resolved
        | Error _ -> goal // Leave unresolved for runtime

    let private compileWorkNodeToStep (parallelGroups: ParallelGroup list) (node: DslNode) : Result<Step, CompileError> =
        match node.Tool, node.Checks with
        | Some toolName, _ ->
            let args = defaultArg node.Args Map.empty
            let action = StepAction.Work(ToolCall(toolName, args))
            let meta = node.Metadata |> addConditionMetadata node |> addParallelMetadata parallelGroups node.Id

            Ok
                { Id = node.Id
                  Inputs = node.Inputs
                  Outputs = OutputHelper.toOutputNames node.Outputs
                  Action = action
                  Agent = OutputHelper.toAgentStr node.Agent
                  Metadata = meta }

        | None, checks when not (List.isEmpty checks) ->
            let action = StepAction.Work(Verify checks)
            let meta = node.Metadata |> addConditionMetadata node |> addParallelMetadata parallelGroups node.Id

            Ok
                { Id = node.Id
                  Inputs = node.Inputs
                  Outputs = OutputHelper.toOutputNames node.Outputs
                  Action = action
                  Agent = OutputHelper.toAgentStr node.Agent
                  Metadata = meta }

        | _ -> Error(InvalidNode(node.Id, "Work node must have either tool or checks."))

    let private compileReasonNodeToStep (inputs: DslInputs) (parallelGroups: ParallelGroup list) (node: DslNode) : Result<Step, CompileError> =
        // Resolve variables in goal at compile time (best-effort with inputs)
        let resolvedGoal = node.Goal |> Option.map (resolveGoalVariables inputs)

        let op =
            match node.Transformation with
            | Some trans ->
                match trans with
                | GoTTransformation.Generate -> Generate(resolvedGoal |> Option.defaultValue node.Name)
                | GoTTransformation.Aggregate -> Aggregate(node.Inputs |> List.map NodeId)
                | GoTTransformation.Refine -> Refine(NodeId(node.Inputs |> List.tryHead |> Option.defaultValue ""))
                | GoTTransformation.Contradict ->
                    Contradict(NodeId(node.Inputs |> List.tryHead |> Option.defaultValue ""))
                | GoTTransformation.Distill -> Distill(NodeId(node.Inputs |> List.tryHead |> Option.defaultValue ""))
                | GoTTransformation.Backtrack ->
                    Backtrack(NodeId(node.Inputs |> List.tryHead |> Option.defaultValue ""))
                | GoTTransformation.Score -> Score(NodeId(node.Inputs |> List.tryHead |> Option.defaultValue ""))
            | None ->
                match resolvedGoal with
                | Some g -> Plan g
                | None -> Explain(node.Name)

        let action = StepAction.Reason op
        let meta = node.Metadata |> addConditionMetadata node |> addParallelMetadata parallelGroups node.Id

        Ok
            { Id = node.Id
              Inputs = node.Inputs
              Outputs = OutputHelper.toOutputNames node.Outputs
              Action = action
              Agent = OutputHelper.toAgentStr node.Agent
              Metadata = meta }

    let private compileNodeToStep (inputs: DslInputs) (parallelGroups: ParallelGroup list) (node: DslNode) : Result<Step, CompileError> =
        match node.Kind with
        | NodeKind.Work -> compileWorkNodeToStep parallelGroups node
        | NodeKind.Reason -> compileReasonNodeToStep inputs parallelGroups node

    /// Expand parallel groups into fan-out/fan-in edges.
    /// For each parallel group, find its predecessor and successor in the
    /// existing edge list, remove the edge between them, and add edges from
    /// predecessor -> each parallel node and each parallel node -> successor.
    let expandParallelEdges (nodes: DslNode list) (edges: (DslId * DslId) list) (groups: ParallelGroup list) : (DslId * DslId) list =
        if groups.IsEmpty then edges
        else
            let parallelNodeSet =
                groups |> List.collect (fun g -> g.NodeIds) |> Set.ofList

            // For each group, determine predecessor and successor from the
            // edge list. A parallel group's nodes have no inter-edges in the
            // explicit edge list. The predecessor is a node that has an edge
            // pointing to any parallel node in the group. The successor is a
            // node pointed to from any parallel node.
            let mutable expandedEdges = edges

            for group in groups do
                let groupSet = Set.ofList group.NodeIds

                // Find edges coming INTO the group from outside
                let incomingEdges =
                    expandedEdges
                    |> List.filter (fun (src, dst) -> not (groupSet.Contains src) && groupSet.Contains dst)

                // Find edges going OUT of the group to outside
                let outgoingEdges =
                    expandedEdges
                    |> List.filter (fun (src, dst) -> groupSet.Contains src && not (groupSet.Contains dst))

                // Find edges between group members (remove them - they're parallel)
                let intraGroupEdges =
                    expandedEdges
                    |> List.filter (fun (src, dst) -> groupSet.Contains src && groupSet.Contains dst)

                // Remove intra-group edges
                expandedEdges <- expandedEdges |> List.filter (fun e -> not (List.contains e intraGroupEdges))

                // If no incoming/outgoing edges exist yet (group was declared
                // via PARALLEL block without explicit edges), build fan-out/fan-in
                // from predecessor/successor.
                if incomingEdges.IsEmpty && outgoingEdges.IsEmpty then
                    // Find the node that should precede the group and follow it
                    // by looking at the overall node order (index-based).
                    let nodeIds = nodes |> List.map (fun n -> n.Id)
                    let firstGroupIdx =
                        group.NodeIds
                        |> List.choose (fun gid -> nodeIds |> List.tryFindIndex (fun nid -> nid = gid))
                        |> (fun idxs -> if idxs.IsEmpty then -1 else List.min idxs)
                    let lastGroupIdx =
                        group.NodeIds
                        |> List.choose (fun gid -> nodeIds |> List.tryFindIndex (fun nid -> nid = gid))
                        |> (fun idxs -> if idxs.IsEmpty then -1 else List.max idxs)

                    let predecessorId =
                        if firstGroupIdx > 0 then Some nodeIds.[firstGroupIdx - 1] else None
                    let successorId =
                        if lastGroupIdx >= 0 && lastGroupIdx < nodeIds.Length - 1 then
                            Some nodeIds.[lastGroupIdx + 1]
                        else None

                    // Remove any existing direct edge from predecessor -> successor
                    match predecessorId, successorId with
                    | Some pred, Some succ ->
                        expandedEdges <- expandedEdges |> List.filter (fun (s, d) -> not (s = pred && d = succ))
                    | _ -> ()

                    // Add fan-out edges: predecessor -> each parallel node
                    match predecessorId with
                    | Some pred ->
                        for nodeId in group.NodeIds do
                            if not (expandedEdges |> List.exists (fun (s, d) -> s = pred && d = nodeId)) then
                                expandedEdges <- expandedEdges @ [ (pred, nodeId) ]
                    | None -> ()

                    // Add fan-in edges: each parallel node -> successor
                    match successorId with
                    | Some succ ->
                        for nodeId in group.NodeIds do
                            if not (expandedEdges |> List.exists (fun (s, d) -> s = nodeId && d = succ)) then
                                expandedEdges <- expandedEdges @ [ (nodeId, succ) ]
                    | None -> ()

            expandedEdges

    let compileWorkflowToPlanParsed (wf: DslWorkflow) : Result<Plan<Parsed>, CompileError list> =
        // Expand parallel groups into proper DAG edges
        let expandedEdges = expandParallelEdges wf.Nodes wf.Edges wf.ParallelGroups
        let hasParallelGroups = not wf.ParallelGroups.IsEmpty

        // For workflows with parallel groups, use DAG validation + topo sort.
        // For simple chains, use the original strict chain validation for backward compat.
        if hasParallelGroups then
            match Graph.validateDAG wf.Nodes expandedEdges with
            | Error e -> Error [ e ]
            | Ok _ ->
                match Graph.topoSort wf.Nodes expandedEdges with
                | Error e -> Error [ e ]
                | Ok orderedNodes ->
                    let stepResults = orderedNodes |> List.map (compileNodeToStep wf.Inputs wf.ParallelGroups)
                    let errors = stepResults |> List.choose (function Error e -> Some e | _ -> None)
                    if errors.Length > 0 then Error errors
                    else
                        let steps = stepResults |> List.choose (function Ok s -> Some s | _ -> None)
                        Ok { Goal = wf.Name; Steps = steps; Policy = wf.Policy; Inputs = wf.Inputs; Version = 1 }
        else
            // Original chain validation for backward compatibility
            match Graph.validateChain wf.Nodes wf.Edges with
            | Error e -> Error [ e ]
            | Ok start ->
                match Graph.orderChain wf.Nodes wf.Edges start with
                | Error e -> Error [ e ]
                | Ok orderedNodes ->
                    let stepResults = orderedNodes |> List.map (compileNodeToStep wf.Inputs wf.ParallelGroups)
                    let errors = stepResults |> List.choose (function Error e -> Some e | _ -> None)
                    if errors.Length > 0 then Error errors
                    else
                        let steps = stepResults |> List.choose (function Ok s -> Some s | _ -> None)
                        Ok { Goal = wf.Name; Steps = steps; Policy = wf.Policy; Inputs = wf.Inputs; Version = 1 }
