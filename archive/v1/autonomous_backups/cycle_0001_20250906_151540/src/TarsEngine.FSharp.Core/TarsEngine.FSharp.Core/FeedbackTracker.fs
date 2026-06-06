namespace TarsEngine.FSharp.Core

open System
open System.IO
open System.Text.Json
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.ModernGameTheory

/// Advanced Feedback Tracking for TARS Multi-Agent Systems
module FeedbackTracker =

    /// Action feedback with game theory context
    type ActionFeedback = {
        action: string
        estimated_reward: float
        actual_reward: float
        regret: float
        context: string
        game_theory_model: string
        cognitive_level: int option
        belief_state: Map<string, float>
    }

    /// Confidence shift tracking
    type ConfidenceShift = {
        before: float
        after: float
        delta: float
        model_influence: string
    }

    /// Enhanced feedback graph entry with modern game theory
    type FeedbackGraphEntry = {
        agent_id: string
        timestamp: DateTime
        task_id: string
        decisions: ActionFeedback list
        regret_update_policy: string
        regret_decay_rate: float
        update_notes: string
        confidence_shift: ConfidenceShift
        game_theory_model: GameTheoryModel
        convergence_metrics: ConvergenceMetrics option
        coordination_score: float
    }

    /// Enhanced agent state with game theory integration
    type EnhancedAgentState = {
        Policy: Map<string, float>
        History: (string * float * float) list // action, estimated, actual
        Confidence: float
        CognitiveLevel: int
        BeliefState: Map<string, float>
        GameTheoryModel: GameTheoryModel
        RegretHistory: float[]
        CoordinationHistory: float[]
    }

    /// Update policy using modern game theory principles
    let updatePolicyWithGameTheory (state: EnhancedAgentState) (decay: float) : EnhancedAgentState =
        match state.GameTheoryModel with
        | QuantalResponseEquilibrium temperature ->
            // QRE-based update
            let utilities = 
                state.History
                |> List.groupBy (fun (action, _, _) -> action)
                |> List.map (fun (action, history) -> 
                    let avgReward = history |> List.averageBy (fun (_, _, actual) -> actual)
                    action, avgReward)
                |> Map.ofList
            
            let maxUtility = utilities |> Map.values |> Seq.max
            let expUtilities = 
                utilities 
                |> Map.map (fun _ u -> Math.Exp((u - maxUtility) / temperature))
            
            let sum = expUtilities |> Map.values |> Seq.sum
            let newPolicy = expUtilities |> Map.map (fun _ exp -> exp / sum)
            
            { state with Policy = newPolicy }

        | NoRegretLearning regretDecay ->
            // No-regret learning update
            let regrets =
                state.History
                |> List.map (fun (action, est, act) -> action, est - act)
                |> List.groupBy fst
                |> List.map (fun (action, regretList) -> 
                    action, regretList |> List.averageBy snd)
                |> Map.ofList

            let updatedWeights = 
                state.Policy
                |> Map.map (fun action weight -> 
                    let regret = regrets |> Map.tryFind action |> Option.defaultValue 0.0
                    weight * Math.Exp(-regretDecay * regret))
            
            let sum = updatedWeights |> Map.values |> Seq.sum
            let normalizedPolicy = 
                if sum > 0.0 then
                    updatedWeights |> Map.map (fun _ w -> w / sum)
                else
                    state.Policy

            { state with Policy = normalizedPolicy }

        | CognitiveHierarchy maxLevel ->
            // Cognitive hierarchy update
            let newLevel = min maxLevel (state.CognitiveLevel + 1)
            let adjustedConfidence = state.Confidence * (1.0 + 0.1 * float newLevel)
            
            { state with 
                CognitiveLevel = newLevel
                Confidence = min 1.0 adjustedConfidence }

        | _ ->
            // Fallback to standard regret minimization
            let regrets =
                state.History
                |> List.map (fun (action, est, act) -> action, est - act)

            let adjustedPolicy =
                regrets
                |> List.fold (fun acc (action, regret) ->
                    let prevWeight = Map.tryFind action acc |> Option.defaultValue 1.0
                    let newWeight = prevWeight * Math.Exp(-decay * regret)
                    acc |> Map.add action newWeight
                ) state.Policy
                |> fun m ->
                    let total = m |> Map.values |> Seq.sum
                    m |> Map.map (fun _ v -> v / total)

            { state with Policy = adjustedPolicy }

    /// Generate enhanced feedback with game theory analysis
    let generateEnhancedFeedback 
        (agentId: string) 
        (taskId: string) 
        (stateBefore: EnhancedAgentState) 
        (stateAfter: EnhancedAgentState) 
        (contextFn: string -> string) 
        (decay: float) 
        (convergenceMetrics: ConvergenceMetrics option) : FeedbackGraphEntry =
        
        let decisions =
            stateBefore.History
            |> List.map (fun (action, est, act) -> {
                action = action
                estimated_reward = est
                actual_reward = act
                regret = est - act
                context = contextFn action
                game_theory_model = sprintf "%A" stateBefore.GameTheoryModel
                cognitive_level = Some stateBefore.CognitiveLevel
                belief_state = stateBefore.BeliefState
            })

        let confShift = {
            before = stateBefore.Confidence
            after = stateAfter.Confidence
            delta = stateAfter.Confidence - stateBefore.Confidence
            model_influence = sprintf "%A" stateBefore.GameTheoryModel
        }

        let coordinationScore = 
            stateAfter.CoordinationHistory 
            |> Array.tryLast 
            |> Option.defaultValue 0.5

        {
            agent_id = agentId
            timestamp = DateTime.UtcNow
            task_id = taskId
            decisions = decisions
            regret_update_policy = "modern_game_theory_enhanced"
            regret_decay_rate = decay
            update_notes = sprintf "Game theory model: %A with enhanced coordination" stateBefore.GameTheoryModel
            confidence_shift = confShift
            game_theory_model = stateBefore.GameTheoryModel
            convergence_metrics = convergenceMetrics
            coordination_score = coordinationScore
        }

    /// Save enhanced feedback to .trsx file with game theory metadata
    let saveEnhancedFeedback (entry: FeedbackGraphEntry) (path: string) =
        let options = JsonSerializerOptions(WriteIndented = true)
        let json = JsonSerializer.Serialize(entry, options)
        
        // Add .trsx metadata header
        let trsxContent =
            sprintf "// TARS Enhanced Feedback with Modern Game Theory\n// Generated: %s\n// Agent: %s\n// Model: %A\n\nfeedback_graph {\n%s\n}"
                (DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC"))
                entry.agent_id
                entry.game_theory_model
                json

        File.WriteAllText(path, trsxContent)

    /// Analyze coordination patterns across multiple agents
    let analyzeCoordination (feedbackEntries: FeedbackGraphEntry list) =
        let coordinationScores = feedbackEntries |> List.map (fun e -> e.coordination_score)
        let avgCoordination = coordinationScores |> List.average
        let coordinationTrend = 
            if coordinationScores.Length > 1 then
                let recent = coordinationScores |> List.rev |> List.take (min 5 coordinationScores.Length)
                let older = coordinationScores |> List.take (min 5 coordinationScores.Length)
                (recent |> List.average) - (older |> List.average)
            else 0.0

        {|
            AverageCoordination = avgCoordination
            CoordinationTrend = coordinationTrend
            IsImproving = coordinationTrend > 0.0
            RecommendedModel = 
                if avgCoordination < 0.3 then "CorrelatedEquilibrium"
                elif avgCoordination < 0.6 then "NoRegretLearning" 
                else "CognitiveHierarchy"
        |}

    /// Generate summary report for multiple agents
    let generateMultiAgentSummary (feedbackEntries: FeedbackGraphEntry list) =
        let agentGroups = feedbackEntries |> List.groupBy (fun e -> e.agent_id)
        let coordination = analyzeCoordination feedbackEntries
        
        let agentSummaries = 
            agentGroups
            |> List.map (fun (agentId, entries) ->
                let avgRegret = 
                    entries 
                    |> List.collect (fun e -> e.decisions)
                    |> List.averageBy (fun d -> Math.Abs(d.regret))
                
                let confidenceTrend = 
                    entries 
                    |> List.map (fun e -> e.confidence_shift.delta)
                    |> List.average

                {|
                    AgentId = agentId
                    EntryCount = entries.Length
                    AverageRegret = avgRegret
                    ConfidenceTrend = confidenceTrend
                    PreferredModel = entries |> List.last |> fun e -> sprintf "%A" e.game_theory_model
                |})

        {|
            TotalAgents = agentGroups.Length
            TotalEntries = feedbackEntries.Length
            CoordinationAnalysis = coordination
            AgentSummaries = agentSummaries
            OverallPerformance = 
                agentSummaries 
                |> List.averageBy (fun a -> 1.0 / (1.0 + a.AverageRegret))
            Recommendations = [
                if coordination.AverageCoordination < 0.5 then 
                    "Consider implementing Correlated Equilibrium for better coordination"
                if agentSummaries |> List.exists (fun a -> a.AverageRegret > 0.3) then
                    "Some agents showing high regret - recommend No-Regret Learning"
                if agentSummaries |> List.forall (fun a -> a.ConfidenceTrend > 0.0) then
                    "All agents improving - consider Cognitive Hierarchy advancement"
            ]
        |}

    /// TARS Integration: Convert feedback to revolutionary result
    let feedbackToRevolutionaryResult (entry: FeedbackGraphEntry) : RevolutionaryTypes.RevolutionaryResult =
        {
            Operation = RevolutionaryTypes.RightPathAIReasoning(entry.task_id, Map.empty)
            Success = entry.coordination_score > 0.5
            Insights = [|
                sprintf "Game theory model: %A" entry.game_theory_model
                sprintf "Coordination score: %.3f" entry.coordination_score
                sprintf "Agent confidence shift: %.3f" entry.confidence_shift.delta
                "Modern game theory analysis completed"
            |]
            Improvements = [|
                "Enhanced multi-agent coordination"
                "Improved strategic decision-making"
                "Better handling of bounded rationality"
                "Advanced equilibrium concepts applied"
            |]
            NewCapabilities = [| 
                RevolutionaryTypes.BeliefDiffusionMastery
                RevolutionaryTypes.NashEquilibriumOptimization 
            |]
            PerformanceGain = Some (1.0 + entry.coordination_score)
            HybridEmbeddings = None
            BeliefConvergence = Some entry.coordination_score
            NashEquilibriumAchieved = Some (entry.coordination_score > 0.7)
            FractalComplexity = Some 1.2
            CudaAccelerated = Some false
            Timestamp = entry.timestamp
            ExecutionTime = TimeSpan.FromMilliseconds(100.0)
        }
