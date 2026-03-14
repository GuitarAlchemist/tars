namespace Tars.Evolution

open System

/// Generic MCTS types for tree search over grammar derivation spaces.
/// Follows the same pattern as MachinBridge: pure F# implementation
/// with optional bridge to MachinDeOuf's Rust MCTS when available.
module MctsTypes =

    /// Interface for states in the MCTS search tree.
    /// Implement this for any domain (WoT graphs, grammar derivations, etc).
    type IMctsState<'Action when 'Action : equality> =
        /// All valid actions from this state
        abstract LegalActions: unit -> 'Action list
        /// Apply an action to produce a new state
        abstract Apply: 'Action -> IMctsState<'Action>
        /// Whether this state is terminal (no more expansions possible)
        abstract IsTerminal: bool
        /// Reward signal for this terminal state [0.0, 1.0]
        abstract Reward: unit -> float

    /// A node in the MCTS search tree (mutable for backpropagation efficiency)
    type MctsNode<'Action when 'Action : equality> = {
        State: IMctsState<'Action>
        /// Action that led to this node (None for root)
        Action: 'Action option
        /// Parent node reference (None for root)
        Parent: MctsNode<'Action> option
        /// Expanded children
        mutable Children: MctsNode<'Action> list
        /// Visit count for UCB1
        mutable Visits: int
        /// Accumulated reward for UCB1
        mutable TotalReward: float
        /// Actions not yet expanded
        mutable UntriedActions: 'Action list
    }

    /// MCTS search configuration
    type MctsConfig = {
        /// Maximum search iterations
        MaxIterations: int
        /// UCB1 exploration constant (sqrt(2) is standard)
        ExplorationConstant: float
        /// Maximum depth for random rollouts
        MaxRolloutDepth: int
        /// Optional time budget (stops early if exceeded)
        TimeBudget: TimeSpan option
        /// Random seed for reproducibility
        Seed: int option
    }

    let defaultMctsConfig = {
        MaxIterations = 1000
        ExplorationConstant = sqrt 2.0
        MaxRolloutDepth = 50
        TimeBudget = None
        Seed = Some 42
    }

    /// Result of an MCTS search
    type MctsResult<'Action> = {
        /// Best action sequence found
        BestActions: 'Action list
        /// Best action from root
        BestRootAction: 'Action option
        /// Total iterations performed
        Iterations: int
        /// Average reward across all rollouts
        AverageReward: float
        /// Root visit count
        RootVisits: int
    }

    /// Create a fresh tree node from a state
    let createNode (state: IMctsState<'Action>) (action: 'Action option) (parent: MctsNode<'Action> option) : MctsNode<'Action> =
        { State = state
          Action = action
          Parent = parent
          Children = []
          Visits = 0
          TotalReward = 0.0
          UntriedActions = state.LegalActions() }
