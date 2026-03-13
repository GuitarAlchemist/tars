namespace Tars.Evolution

/// MCTS state implementation for WoT DSL derivation search.
/// Explores the space of valid WoT workflow graphs to find high-reward
/// derivations (measured by structural quality, check coverage, connectivity).
///
/// This does NOT execute workflows against an LLM — it uses structural
/// heuristics to evaluate graph quality, making MCTS fast enough for
/// real-time derivation search.
module WotMctsState =

    open System
    open Tars.DSL.Wot
    open MctsTypes

    // =========================================================================
    // Actions in the WoT derivation space
    // =========================================================================

    /// Actions the MCTS can take to build a WoT workflow graph
    type WotAction =
        | AddNode of DslNode
        | AddEdge of DslEdge
        | SetTransformation of nodeId: DslId * GoTTransformation
        | Complete
    with
        override this.ToString() =
            match this with
            | AddNode n -> $"AddNode({n.Id})"
            | AddEdge e -> $"AddEdge({e.From}->{e.To})"
            | SetTransformation (id, t) -> $"SetTransform({id},{t})"
            | Complete -> "Complete"

    // =========================================================================
    // Derivation state
    // =========================================================================

    /// The state of a WoT workflow being derived
    type WotDerivationState = {
        /// Nodes added so far
        Nodes: DslNode list
        /// Edges added so far
        Edges: DslEdge list
        /// Pool of node templates available to add
        AvailableTemplates: DslNode list
        /// Workflow metadata
        Meta: DslMeta
        /// Maximum nodes allowed
        MaxNodes: int
        /// Whether derivation is complete
        IsComplete: bool
    }

    // =========================================================================
    // Structural reward heuristics
    // =========================================================================

    /// Compute a [0.0, 1.0] reward for a derivation state.
    /// Uses structural quality heuristics (no LLM needed).
    let computeReward (state: WotDerivationState) : float =
        if state.Nodes.IsEmpty then 0.0
        else
            let nodeCount = float state.Nodes.Length
            let edgeCount = float state.Edges.Length

            // 1. Connectivity: edges / (nodes - 1) — perfect chain = 1.0
            let connectivityScore =
                if nodeCount <= 1.0 then 0.5
                else min 1.0 (edgeCount / (nodeCount - 1.0))

            // 2. Check coverage: fraction of nodes that have WotChecks
            let nodesWithChecks =
                state.Nodes |> List.filter (fun n -> not n.Checks.IsEmpty) |> List.length
            let checkScore = float nodesWithChecks / nodeCount

            // 3. Tool coverage: fraction of Work nodes that have tools assigned
            let workNodes = state.Nodes |> List.filter (fun n -> n.Kind = Work)
            let toolScore =
                if workNodes.IsEmpty then 0.5
                else
                    let withTools = workNodes |> List.filter (fun n -> n.Tool.IsSome) |> List.length
                    float withTools / float workNodes.Length

            // 4. Mix diversity: bonus for having both Reason and Work nodes
            let hasReason = state.Nodes |> List.exists (fun n -> n.Kind = Reason)
            let hasWork = state.Nodes |> List.exists (fun n -> n.Kind = Work)
            let diversityScore = if hasReason && hasWork then 1.0 else 0.5

            // 5. DAG validity: penalize if edges create potential cycles
            let edgeTargets = state.Edges |> List.map (fun e -> e.To) |> Set.ofList
            let edgeSources = state.Edges |> List.map (fun e -> e.From) |> Set.ofList
            let hasRoot = state.Nodes |> List.exists (fun n -> not (edgeTargets.Contains n.Id))
            let dagScore = if hasRoot || state.Edges.IsEmpty then 1.0 else 0.3

            // 6. Size score: prefer non-trivial graphs (2-MaxNodes), bell curve around MaxNodes/2
            let idealSize = float state.MaxNodes / 2.0
            let sizeScore = 1.0 - (abs (nodeCount - idealSize) / idealSize) |> max 0.0

            // Weighted combination
            connectivityScore * 0.25
            + checkScore * 0.20
            + toolScore * 0.15
            + diversityScore * 0.15
            + dagScore * 0.15
            + sizeScore * 0.10

    // =========================================================================
    // Legal actions
    // =========================================================================

    /// Compute legal actions from current derivation state
    let legalActions (state: WotDerivationState) : WotAction list =
        if state.IsComplete then []
        else
            let actions = ResizeArray<WotAction>()

            // Can add nodes if under limit and templates available
            if state.Nodes.Length < state.MaxNodes then
                for template in state.AvailableTemplates do
                    // Don't add duplicate IDs
                    if not (state.Nodes |> List.exists (fun n -> n.Id = template.Id)) then
                        actions.Add(AddNode template)

            // Can add edges between existing nodes (no self-loops, no duplicates)
            for src in state.Nodes do
                for dst in state.Nodes do
                    if src.Id <> dst.Id then
                        let edgeExists =
                            state.Edges |> List.exists (fun e -> e.From = src.Id && e.To = dst.Id)
                        if not edgeExists then
                            actions.Add(AddEdge { From = src.Id; To = dst.Id; Relation = EdgeDependsOn })

            // Can set transformations on nodes without one
            for node in state.Nodes do
                if node.Transformation.IsNone && node.Kind = Reason then
                    for t in [ Generate; Aggregate; Refine; Score ] do
                        actions.Add(SetTransformation (node.Id, t))

            // Can complete if we have at least 2 nodes and 1 edge
            if state.Nodes.Length >= 2 && not state.Edges.IsEmpty then
                actions.Add(Complete)

            actions |> Seq.toList

    // =========================================================================
    // Action application
    // =========================================================================

    /// Apply an action to produce a new derivation state
    let applyAction (state: WotDerivationState) (action: WotAction) : WotDerivationState =
        match action with
        | AddNode node ->
            { state with
                Nodes = state.Nodes @ [ node ]
                AvailableTemplates = state.AvailableTemplates |> List.filter (fun t -> t.Id <> node.Id) }
        | AddEdge edge ->
            { state with Edges = state.Edges @ [ edge ] }
        | SetTransformation (nodeId, transformation) ->
            { state with
                Nodes = state.Nodes |> List.map (fun n ->
                    if n.Id = nodeId then { n with Transformation = Some transformation }
                    else n) }
        | Complete ->
            { state with IsComplete = true }

    // =========================================================================
    // IMctsState implementation
    // =========================================================================

    /// Create an IMctsState adapter for WoT derivation search
    let rec createMctsState (state: WotDerivationState) : IMctsState<WotAction> =
        { new IMctsState<WotAction> with
            member _.LegalActions() = legalActions state
            member _.Apply(action) = createMctsState (applyAction state action)
            member _.IsTerminal = state.IsComplete || legalActions state |> List.isEmpty
            member _.Reward() = computeReward state }

    // =========================================================================
    // Convenience: search for best workflow derivation
    // =========================================================================

    /// Search for the best WoT workflow derivation from a pool of node templates.
    let searchDerivation
        (config: MctsConfig)
        (meta: DslMeta)
        (templates: DslNode list)
        (maxNodes: int)
        : MctsResult<WotAction> =
        let initialState = {
            Nodes = []
            Edges = []
            AvailableTemplates = templates
            Meta = meta
            MaxNodes = maxNodes
            IsComplete = false
        }
        MctsSolver.search config (createMctsState initialState)
