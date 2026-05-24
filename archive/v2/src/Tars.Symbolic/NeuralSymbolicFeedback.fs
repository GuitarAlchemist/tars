namespace Tars.Symbolic

open System
open Tars.Core

/// Neural-Symbolic Feedback Loop
/// This is the KEY INNOVATION: Symbolic scores shape neural behavior
module NeuralSymbolicFeedback =

    /// Agent performance history for stability tracking
    type AgentPerformance =
        { AgentId: AgentId
          SuccessfulActions: int
          FailedActions: int
          AverageConstraintScore: float
          LastUpdated: DateTime }

    /// Calculate success rate for an agent
    let successRate (perf: AgentPerformance) : float =
        let total = perf.SuccessfulActions + perf.FailedActions

        if total = 0 then
            0.5 // Neutral for new agents
        else
            float perf.SuccessfulActions / float total

    /// Calculate stability-weighted priority for agent selection
    let calculatePriority (perf: AgentPerformance) (baseWeight: float) : float =
        let successWeight = successRate perf
        let constraintWeight = perf.AverageConstraintScore

        // Weighted combination: 40% success rate, 40% constraint score, 20% base weight
        (successWeight * 0.4) + (constraintWeight * 0.4) + (baseWeight * 0.2)

    /// Bias agent selection based on historical stability
    /// Returns weighted list where higher-performing agents have higher probability
    let biasAgentSelection
        (agents: Agent list)
        (performanceHistory: Map<AgentId, AgentPerformance>)
        (baseWeights: Map<AgentId, float>)
        : (Agent * float) list =

        agents
        |> List.map (fun agent ->
            let perf =
                performanceHistory
                |> Map.tryFind agent.Id
                |> Option.defaultValue
                    { AgentId = agent.Id
                      SuccessfulActions = 0
                      FailedActions = 0
                      AverageConstraintScore = 0.5
                      LastUpdated = DateTime.UtcNow }

            let baseWeight = Map.tryFind agent.Id baseWeights |> Option.defaultValue 1.0
            let priority = calculatePriority perf baseWeight

            (agent, priority))

    /// Select agent using weighted random selection
    /// Higher weights = higher probability of selection
    let selectAgent (weightedAgents: (Agent * float) list) (random: Random) : Agent option =
        if weightedAgents.IsEmpty then
            None
        else
            let totalWeight = weightedAgents |> List.sumBy snd
            let randomValue = random.NextDouble() * totalWeight

            let rec findAgent cumulative remaining =
                match remaining with
                | [] -> weightedAgents |> List.last |> fst |> Some
                | (agent, weight) :: rest ->
                    let newCumulative = cumulative + weight

                    if randomValue <= newCumulative then
                        Some agent
                    else
                        findAgent newCumulative rest

            findAgent 0.0 weightedAgents

    /// Shape prompt to avoid known contradictions
    /// Adds warnings about patterns that led to constraint violations
    let shapePrompt (basePrompt: string) (knownContradictions: string list) (lowScoringPatterns: string list) : string =

        if knownContradictions.IsEmpty && lowScoringPatterns.IsEmpty then
            basePrompt
        else
            let warnings = System.Text.StringBuilder()
            warnings.AppendLine(basePrompt) |> ignore
            warnings.AppendLine() |> ignore
            warnings.AppendLine("⚠️  SYMBOLIC WARNINGS:") |> ignore

            if not knownContradictions.IsEmpty then
                warnings.AppendLine("Known contradiction patterns to avoid:") |> ignore

                knownContradictions
                |> List.iter (fun c -> warnings.AppendLine($"  - {c}") |> ignore)

            if not lowScoringPatterns.IsEmpty then
                warnings.AppendLine("Patterns that scored poorly in the past:") |> ignore

                lowScoringPatterns
                |> List.iter (fun p -> warnings.AppendLine($"  - {p}") |> ignore)

            warnings.ToString()

    /// Filter mutations that violate invariants
    /// Returns only mutations that satisfy minimum constraint score
    let filterMutations (mutations: 'T list) (scoringFunc: 'T -> float) (minScore: float) : 'T list =

        mutations
        |> List.map (fun m -> (m, scoringFunc m))
        |> List.filter (fun (_, score) -> score >= minScore)
        |> List.sortByDescending snd // Best scores first
        |> List.map fst

    /// Update agent performance after action
    let updatePerformance
        (agentId: AgentId)
        (success: bool)
        (constraintScore: float)
        (currentPerf: AgentPerformance option)
        : AgentPerformance =

        match currentPerf with
        | None ->
            { AgentId = agentId
              SuccessfulActions = if success then 1 else 0
              FailedActions = if success then 0 else 1
              AverageConstraintScore = constraintScore
              LastUpdated = DateTime.UtcNow }
        | Some perf ->
            let newSuccessful =
                if success then
                    perf.SuccessfulActions + 1
                else
                    perf.SuccessfulActions

            let newFailed =
                if success then
                    perf.FailedActions
                else
                    perf.FailedActions + 1

            let totalActions = newSuccessful + newFailed

            // Running average of constraint scores
            let newAvgScore =
                (perf.AverageConstraintScore * float (totalActions - 1) + constraintScore)
                / float totalActions

            { AgentId = agentId
              SuccessfulActions = newSuccessful
              FailedActions = newFailed
              AverageConstraintScore = newAvgScore
              LastUpdated = DateTime.UtcNow }

    /// Track belief stability impact of agents
    /// Returns agents sorted by their impact on belief stability
    let selectForStability (agents: Agent list) (performanceHistory: Map<AgentId, AgentPerformance>) : Agent list =

        agents
        |> List.map (fun agent ->
            let perf = Map.tryFind agent.Id performanceHistory

            let score =
                perf
                |> Option.map (fun p -> p.AverageConstraintScore)
                |> Option.defaultValue 0.5

            (agent, score))
        |> List.sortByDescending snd
        |> List.map fst

    /// Extract contradiction patterns from failed actions
    let extractContradictionPatterns (failedBeliefs: string list) : string list =
        failedBeliefs
        |> List.filter (fun b -> b.Contains("NOT") || b.Contains("contradiction"))
        |> List.distinct
        |> List.truncate 5 // Top 5 patterns

    /// Extract low-scoring patterns from action history
    let extractLowScoringPatterns (actionHistory: (string * float) list) (threshold: float) : string list =

        actionHistory
        |> List.filter (fun (_, score) -> score < threshold)
        |> List.map fst
        |> List.distinct
        |> List.truncate 5 // Top 5 patterns

    /// Recommend actions based on constraint scores
    type ActionRecommendation =
        | StronglyRecommend of score: float * reason: string
        | Recommend of score: float * reason: string
        | Neutral of score: float
        | Discourage of score: float * reason: string
        | StronglyDiscourage of score: float * reason: string

    /// Classify action based on constraint score
    let classifyAction (score: float) (reasons: string list) : ActionRecommendation =
        let reason =
            if reasons.IsEmpty then
                "No specific reason"
            else
                String.concat "; " reasons

        match score with
        | s when s >= 0.9 -> StronglyRecommend(s, reason)
        | s when s >= 0.7 -> Recommend(s, reason)
        | s when s >= 0.5 -> Neutral(s)
        | s when s >= 0.3 -> Discourage(s, reason)
        | s -> StronglyDiscourage(s, reason)

    /// Calculate confidence in recommendation
    let recommendationConfidence (recommendation: ActionRecommendation) : float =
        match recommendation with
        | StronglyRecommend(score, _) -> score
        | Recommend(score, _) -> score
        | Neutral(score) -> 0.5
        | Discourage(score, _) -> 1.0 - score
        | StronglyDiscourage(score, _) -> 1.0 - score

    /// Feedback loop metrics for monitoring
    type FeedbackMetrics =
        { TotalAgents: int
          AgentsWithGoodStability: int
          AverageConstraintScore: float
          PromptsShapedCount: int
          MutationsFiltered: int
          MutationsAccepted: int
          ContradictionPatternsFound: int }

    /// Calculate feedback loop effectiveness metrics
    let calculateMetrics
        (performanceHistory: Map<AgentId, AgentPerformance>)
        (promptsShapedCount: int)
        (mutationsFiltered: int)
        (mutationsAccepted: int)
        (contradictionPatterns: string list)
        : FeedbackMetrics =

        let agentsWithGoodStability =
            performanceHistory
            |> Map.toList
            |> List.filter (fun (_, perf) -> perf.AverageConstraintScore >= 0.7)
            |> List.length

        let avgScore =
            if performanceHistory.IsEmpty then
                0.0
            else
                performanceHistory
                |> Map.toList
                |> List.map (fun (_, perf) -> perf.AverageConstraintScore)
                |> List.average

        { TotalAgents = performanceHistory.Count
          AgentsWithGoodStability = agentsWithGoodStability
          AverageConstraintScore = avgScore
          PromptsShapedCount = promptsShapedCount
          MutationsFiltered = mutationsFiltered
          MutationsAccepted = mutationsAccepted
          ContradictionPatternsFound = contradictionPatterns.Length }

    /// Pretty-print feedback metrics
    let printMetrics (metrics: FeedbackMetrics) : string =
        let sb = System.Text.StringBuilder()
        sb.AppendLine("🔄 Neural-Symbolic Feedback Loop Metrics") |> ignore
        sb.AppendLine(String.replicate 50 "=") |> ignore
        sb.AppendLine($"Total Agents: {metrics.TotalAgents}") |> ignore

        sb.AppendLine($"Agents with Good Stability (≥0.7): {metrics.AgentsWithGoodStability}")
        |> ignore

        sb.AppendLine($"Average Constraint Score: {metrics.AverageConstraintScore:F3}")
        |> ignore

        sb.AppendLine($"Prompts Shaped: {metrics.PromptsShapedCount}") |> ignore
        sb.AppendLine($"Mutations Filtered: {metrics.MutationsFiltered}") |> ignore
        sb.AppendLine($"Mutations Accepted: {metrics.MutationsAccepted}") |> ignore

        sb.AppendLine($"Contradiction Patterns: {metrics.ContradictionPatternsFound}")
        |> ignore

        let filterRate =
            if metrics.MutationsFiltered + metrics.MutationsAccepted = 0 then
                0.0
            else
                float metrics.MutationsFiltered
                / float (metrics.MutationsFiltered + metrics.MutationsAccepted)

        sb.AppendLine($"Filter Rate: {filterRate:P1}") |> ignore

        sb.ToString()
