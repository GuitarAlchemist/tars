namespace Tars.Evolution

/// Pure F# Monte Carlo Tree Search implementation.
/// UCB1 selection, random rollout, full backpropagation.
/// Serves as fallback when ix's Rust MCTS is unavailable.
module MctsSolver =

    open System
    open MctsTypes

    // =========================================================================
    // UCB1 Selection
    // =========================================================================

    /// UCB1 score: exploitation + exploration
    let ucb1 (explorationConstant: float) (parent: MctsNode<'Action>) (child: MctsNode<'Action>) : float =
        if child.Visits = 0 then
            Double.MaxValue // always explore unvisited
        else
            let exploitation = child.TotalReward / float child.Visits
            let exploration = explorationConstant * sqrt (log (float parent.Visits) / float child.Visits)
            exploitation + exploration

    /// Select the best child by UCB1
    let selectChild (config: MctsConfig) (node: MctsNode<'Action>) : MctsNode<'Action> =
        node.Children
        |> List.maxBy (ucb1 config.ExplorationConstant node)

    // =========================================================================
    // Tree Policy: Select → Expand
    // =========================================================================

    /// Walk down the tree using UCB1 until we find an expandable or terminal node
    let rec private treePolicy (config: MctsConfig) (node: MctsNode<'Action>) : MctsNode<'Action> =
        if node.State.IsTerminal then node
        elif not node.UntriedActions.IsEmpty then
            // Expand: pick first untried action
            expand node
        else
            // Select best child and recurse
            let best = selectChild config node
            treePolicy config best

    /// Expand a node by trying one untried action
    and expand (node: MctsNode<'Action>) : MctsNode<'Action> =
        match node.UntriedActions with
        | [] -> node
        | action :: rest ->
            node.UntriedActions <- rest
            let newState = node.State.Apply action
            let child = createNode newState (Some action) (Some node)
            node.Children <- child :: node.Children
            child

    // =========================================================================
    // Default Policy: Random Rollout
    // =========================================================================

    /// Random rollout from a state to terminal or max depth
    let rollout (maxDepth: int) (state: IMctsState<'Action>) (rng: Random) : float =
        let mutable current = state
        let mutable depth = 0
        while not current.IsTerminal && depth < maxDepth do
            let actions = current.LegalActions()
            if actions.IsEmpty then
                depth <- maxDepth // force exit
            else
                let action = actions.[rng.Next(actions.Length)]
                current <- current.Apply action
                depth <- depth + 1
        current.Reward()

    // =========================================================================
    // Backpropagation
    // =========================================================================

    /// Propagate reward up from leaf to root
    let backpropagate (leaf: MctsNode<'Action>) (reward: float) : unit =
        let mutable node = Some leaf
        while node.IsSome do
            let n = node.Value
            n.Visits <- n.Visits + 1
            n.TotalReward <- n.TotalReward + reward
            node <- n.Parent

    // =========================================================================
    // Main Search
    // =========================================================================

    /// Run MCTS search and return the best action sequence
    let search (config: MctsConfig) (initialState: IMctsState<'Action>) : MctsResult<'Action> =
        let rng =
            match config.Seed with
            | Some s -> Random(s)
            | None -> Random()

        let root = createNode initialState None None
        let stopwatch = Diagnostics.Stopwatch.StartNew()
        let mutable iterations = 0

        for _ in 1 .. config.MaxIterations do
            // Check time budget
            match config.TimeBudget with
            | Some budget when stopwatch.Elapsed > budget -> ()
            | _ ->
                // Select + Expand
                let leaf = treePolicy config root
                // Rollout
                let reward = rollout config.MaxRolloutDepth leaf.State rng
                // Backpropagate
                backpropagate leaf reward
                iterations <- iterations + 1

        // Extract best action from root (most visited child)
        let bestChild =
            root.Children
            |> List.sortByDescending (fun c -> c.Visits)
            |> List.tryHead

        // Build best action sequence by following most-visited path
        let rec bestPath (node: MctsNode<'Action>) : 'Action list =
            match node.Children with
            | [] -> []
            | children ->
                let best = children |> List.maxBy (fun c -> c.Visits)
                match best.Action with
                | Some a -> a :: bestPath best
                | None -> bestPath best

        let avgReward =
            if root.Visits > 0 then root.TotalReward / float root.Visits
            else 0.0

        { BestActions = bestPath root
          BestRootAction = bestChild |> Option.bind (fun c -> c.Action)
          Iterations = iterations
          AverageReward = avgReward
          RootVisits = root.Visits }

    /// Search and return just the best first action with its stats
    let searchBest (config: MctsConfig) (initialState: IMctsState<'Action>) : ('Action * int * float) option =
        let result = search config initialState
        result.BestRootAction
        |> Option.map (fun a -> (a, result.Iterations, result.AverageReward))
