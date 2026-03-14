/// Neuro-Symbolic Integration for Evolution Engine
/// Wires symbolic constraints into agent selection and mutation evaluation
module Tars.Evolution.NeurosymbolicIntegration

open System
open Tars.Core
open Tars.Symbolic

/// Track agent performance across evolution iterations
type EvolutionPerformanceTracker() =
    let mutable performanceHistory =
        Map.empty<AgentId, NeuralSymbolicFeedback.AgentPerformance>

    let mutable contradictionPatterns = []
    let mutable lowScoringPatterns = []
    let random = Random()

    /// Update agent performance
    member _.RecordAgentPerformance(agentId: AgentId, success: bool, score: float) =
        let current = Map.tryFind agentId performanceHistory
        let updated = NeuralSymbolicFeedback.updatePerformance agentId success score current
        performanceHistory <- Map.add agentId updated performanceHistory

    /// Get performance history
    member _.GetPerformanceHistory() = performanceHistory

    /// Record contradiction pattern
    member _.RecordContradiction(pattern: string) =
        if not (List.contains pattern contradictionPatterns) then
            contradictionPatterns <- pattern :: (contradictionPatterns |> List.truncate 10)

    /// Record low-scoring pattern
    member _.RecordLowScoringPattern(pattern: string, score: float) =
        if score < 0.5 && not (List.contains pattern lowScoringPatterns) then
            lowScoringPatterns <- pattern :: (lowScoringPatterns |> List.truncate 10)

    /// Get known contradictions
    member _.GetContradictions() = contradictionPatterns

    /// Get low-scoring patterns
    member _.GetLowScoringPatterns() = lowScoringPatterns

    /// Select agent with stability biasing
    member _.SelectWeightedAgent(agents: Agent list, baseWeights: Map<AgentId, float>) =
        let weightedAgents =
            NeuralSymbolicFeedback.biasAgentSelection agents performanceHistory baseWeights

        NeuralSymbolicFeedback.selectAgent weightedAgents random

    /// Shape prompt with symbolic warnings
    member _.ShapePrompt(basePrompt: string) =
        NeuralSymbolicFeedback.shapePrompt basePrompt contradictionPatterns lowScoringPatterns

    /// Get feedback metrics
    member _.GetMetrics(promptsShapedCount: int, mutationsFiltered: int, mutationsAccepted: int) =
        NeuralSymbolicFeedback.calculateMetrics
            performanceHistory
            promptsShapedCount
            mutationsFiltered
            mutationsAccepted
            contradictionPatterns

/// Score a code mutation using constraints
let scoreMutation (code: string) : float =
    // Create context for scoring
    let context = Map.empty<string, obj>

    // Define invariants for code quality
    let invariants =
        [ StandardInvariants.complexityLimit 15.0
          StandardInvariants.parseableGrammar "F#" code ]

    // Score using weighted average
    ConstraintScoring.scoreInvariants invariants context ConstraintScoring.CombinationStrategy.AverageScore

/// Evaluate if a mutation should be accepted
let shouldAcceptMutation (code: string) (minScore: float) : bool * float =
    let score = scoreMutation code
    (score >= minScore, score)

/// Score a belief for consistency
let scoreBeliefConsistency (newBelief: string) (existingBeliefs: string list) : float =
    ConstraintScoring.scoreBeliefConsistency newBelief existingBeliefs

/// Filter mutations by constraint scores
let filterMutationsByScore (mutations: string list) (minScore: float) : string list =
    NeuralSymbolicFeedback.filterMutations mutations scoreMutation minScore

/// Create standard invariants for agent behavior
let createAgentInvariants () : SymbolicInvariant list =
    [ StandardInvariants.tokenBudget 50000
      StandardInvariants.memoryLimit 512
      StandardInvariants.complexityLimit 10.0 ]

/// Score agent action against standard invariants
let scoreAgentAction (action: string) : float =
    let invariants = createAgentInvariants ()
    let context = Map.empty<string, obj>

    ConstraintScoring.scoreInvariants invariants context ConstraintScoring.CombinationStrategy.AverageScore

/// Helper to log constraint metrics
let logConstraintMetrics (metrics: NeuralSymbolicFeedback.FeedbackMetrics) =
    let report = NeuralSymbolicFeedback.printMetrics metrics
    printfn $"%s{report}"

/// Integration configuration
type NeuroSymbolicConfig =
    { EnableFeedbackLoop: bool
      MinMutationScore: float
      MinBeliefScore: float
      EnablePromptShaping: bool
      EnableAgentBiasing: bool
      LogMetrics: bool }

/// Default configuration
let defaultConfig =
    { EnableFeedbackLoop = true
      MinMutationScore = 0.5
      MinBeliefScore = 0.7
      EnablePromptShaping = true
      EnableAgentBiasing = true
      LogMetrics = true }

/// Evolution run with neuro-symbolic integration
type NeuroSymbolicEvolution(config: NeuroSymbolicConfig) =
    let tracker = EvolutionPerformanceTracker()
    let mutable promptsShapedCount = 0
    let mutable mutationsFiltered = 0
    let mutable mutationsAccepted = 0

    /// Select agent for next iteration
    member _.SelectAgent(agents: Agent list, baseWeights: Map<AgentId, float>) =
        if config.EnableAgentBiasing then
            tracker.SelectWeightedAgent(agents, baseWeights)
        else if
            // Fallback to random selection
            agents.IsEmpty
        then
            None
        else
            Some agents.[Random().Next(agents.Length)]

    /// Prepare prompt with symbolic warnings
    member _.PreparePrompt(basePrompt: string) =
        if config.EnablePromptShaping then
            promptsShapedCount <- promptsShapedCount + 1
            tracker.ShapePrompt(basePrompt)
        else
            basePrompt

    /// Evaluate mutation
    member _.EvaluateMutation(code: string) =
        let accepted, score = shouldAcceptMutation code config.MinMutationScore

        if accepted then
            mutationsAccepted <- mutationsAccepted + 1
        else
            mutationsFiltered <- mutationsFiltered + 1
            tracker.RecordLowScoringPattern(code.Substring(0, min 100 code.Length), score)

        (accepted, score)

    /// Record agent result
    member _.RecordAgentResult(agentId: AgentId, success: bool, output: string) =
        let score = scoreAgentAction output
        tracker.RecordAgentPerformance(agentId, success, score)

        // Check for contradictions (simple heuristic)
        if output.Contains("NOT") && output.Contains("contradiction") then
            tracker.RecordContradiction(output.Substring(0, min 100 output.Length))

    /// Get current metrics
    member _.GetMetrics() =
        tracker.GetMetrics(promptsShapedCount, mutationsFiltered, mutationsAccepted)

    /// Log metrics if enabled
    member this.LogMetrics() =
        if config.LogMetrics then
            let metrics = this.GetMetrics()
            logConstraintMetrics metrics

    /// Get performance tracker (for testing/inspection)
    member _.Tracker = tracker
