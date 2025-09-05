namespace TarsEngine.FSharp.Core

open System
open System.Collections.Concurrent
open TarsEngine.FSharp.Core.Tier10MetaLearning

/// Tier 11: Consciousness-Inspired Self-Awareness for Autonomous Operation
/// Enables TARS to monitor its own capabilities and make transparent decisions
module Tier11SelfAwareness =

    /// Operational state monitoring
    type OperationalState = {
        CurrentCapabilities: string list
        KnownLimitations: CognitiveLimitation list
        LearningProgress: Map<string, float>
        PerformanceMetrics: Map<string, float>
        UncertaintyAreas: string list
        ConfidenceLevel: float
        LastSelfAssessment: DateTime
    }
    
    and CognitiveLimitation = {
        Domain: string
        Limitation: string
        Confidence: float
        LearningPath: string list
        EstimatedTimeToResolve: TimeSpan option
    }

    /// Decision reasoning and transparency
    type DecisionReasoning = {
        Decision: string
        Confidence: float
        ReasoningSteps: string list
        AlternativesConsidered: string list
        UncertaintyFactors: string list
        LearningOpportunities: string list
        KnowledgeGapsIdentified: string list
    }

    /// Self-awareness monitoring engine
    type SelfAwarenessEngine() =
        let operationalState = ref {
            CurrentCapabilities = []
            KnownLimitations = []
            LearningProgress = Map.empty
            PerformanceMetrics = Map.empty
            UncertaintyAreas = []
            ConfidenceLevel = 0.0
            LastSelfAssessment = DateTime.UtcNow
        }
        
        let decisionHistory = ConcurrentDictionary<Guid, DecisionReasoning>()
        let uncertaintyThreshold = 0.7

        /// Assess current operational state
        member this.AssessCurrentState() =
            async {
                let knowledgeState = MetaLearningEngine.GetKnowledgeState()
                
                // Analyze current capabilities
                let capabilities = 
                    knowledgeState
                    |> Map.toList
                    |> List.filter (fun (_, state) -> state.MasteryLevel > 0.6)
                    |> List.map (fun (domain, _) -> domain)
                
                // Identify limitations
                let limitations = 
                    knowledgeState
                    |> Map.toList
                    |> List.filter (fun (_, state) -> state.MasteryLevel < 0.5)
                    |> List.map (fun (domain, state) -> {
                        Domain = domain
                        Limitation = $"Limited mastery in {domain} ({state.MasteryLevel:P0})"
                        Confidence = 1.0 - state.MasteryLevel
                        LearningPath = ["structured_learning"; "practice"; "application"]
                        EstimatedTimeToResolve = Some (TimeSpan.FromHours(10.0 - (state.MasteryLevel * 10.0)))
                    })
                
                // Calculate learning progress
                let learningProgress = 
                    knowledgeState
                    |> Map.map (fun _ state -> state.MasteryLevel)
                
                // Identify uncertainty areas
                let uncertaintyAreas = 
                    knowledgeState
                    |> Map.toList
                    |> List.filter (fun (_, state) -> state.MasteryLevel > 0.3 && state.MasteryLevel < 0.7)
                    |> List.map fst
                
                // Calculate overall confidence
                let overallConfidence = 
                    if knowledgeState.IsEmpty then 0.0
                    else
                        knowledgeState.Values
                        |> Seq.averageBy (fun state -> state.MasteryLevel)
                
                let newState = {
                    CurrentCapabilities = capabilities
                    KnownLimitations = limitations
                    LearningProgress = learningProgress
                    PerformanceMetrics = Map [
                        ("OverallMastery", overallConfidence)
                        ("LearningVelocity", 0.15) // Placeholder for actual measurement
                        ("AdaptationSpeed", 0.12)
                        ("KnowledgeTransfer", 0.18)
                    ]
                    UncertaintyAreas = uncertaintyAreas
                    ConfidenceLevel = overallConfidence
                    LastSelfAssessment = DateTime.UtcNow
                }
                
                operationalState := newState
                return newState
            }

        /// Make transparent decision with reasoning
        member this.MakeTransparentDecision(context: string, options: string list) =
            async {
                let! currentState = this.AssessCurrentState()
                
                // Analyze each option
                let optionAnalysis = 
                    options
                    |> List.map (fun option -> 
                        let confidence = this.CalculateOptionConfidence(option, currentState)
                        let uncertainties = this.IdentifyUncertainties(option, currentState)
                        (option, confidence, uncertainties)
                    )
                
                // Select best option
                let (bestOption, bestConfidence, uncertainties) = 
                    optionAnalysis
                    |> List.maxBy (fun (_, confidence, _) -> confidence)
                
                // Generate reasoning
                let reasoningSteps = [
                    $"Analyzed {options.Length} options in context: {context}"
                    $"Current confidence level: {currentState.ConfidenceLevel:P0}"
                    $"Selected option '{bestOption}' with {bestConfidence:P0} confidence"
                    let capabilitiesStr = String.Join(", ", currentState.CurrentCapabilities)
                    $"Decision based on current capabilities: {capabilitiesStr}"
                ]
                
                let alternatives = 
                    optionAnalysis
                    |> List.filter (fun (option, _, _) -> option <> bestOption)
                    |> List.map (fun (option, confidence, _) -> $"{option} ({confidence:P0})")
                
                let knowledgeGaps = 
                    uncertainties
                    |> List.filter (fun uncertainty -> 
                        List.contains uncertainty currentState.UncertaintyAreas)
                
                let decision = {
                    Decision = bestOption
                    Confidence = bestConfidence
                    ReasoningSteps = reasoningSteps
                    AlternativesConsidered = alternatives
                    UncertaintyFactors = uncertainties
                    LearningOpportunities = knowledgeGaps
                    KnowledgeGapsIdentified = knowledgeGaps
                }
                
                let decisionId = Guid.NewGuid()
                decisionHistory.TryAdd(decisionId, decision) |> ignore
                
                return (decisionId, decision)
            }

        /// Calculate confidence for a specific option
        member private this.CalculateOptionConfidence(option: string, state: OperationalState) =
            // Simple heuristic based on relevant capabilities
            let relevantCapabilities = 
                state.CurrentCapabilities
                |> List.filter (fun cap -> option.ToLower().Contains(cap.ToLower()))
                |> List.length
            
            let baseConfidence = min 1.0 (float relevantCapabilities * 0.3)
            let uncertaintyPenalty = 
                state.UncertaintyAreas
                |> List.filter (fun area -> option.ToLower().Contains(area.ToLower()))
                |> List.length
                |> float
                |> (*) 0.1
            
            max 0.1 (baseConfidence - uncertaintyPenalty)

        /// Identify uncertainties for a specific option
        member private this.IdentifyUncertainties(option: string, state: OperationalState) =
            state.UncertaintyAreas
            |> List.filter (fun area -> option.ToLower().Contains(area.ToLower()))

        /// Autonomous uncertainty recognition
        member this.RecognizeUncertainty(task: string) =
            async {
                let! currentState = this.AssessCurrentState()
                
                // Check if task involves uncertain areas
                let uncertainAreas = 
                    currentState.UncertaintyAreas
                    |> List.filter (fun area -> task.ToLower().Contains(area.ToLower()))
                
                let knowledgeGaps = 
                    currentState.KnownLimitations
                    |> List.filter (fun limitation -> 
                        let taskLower = task.ToLower()
                        let domainLower = limitation.Domain.ToLower()
                        taskLower.Contains(domainLower))
                    |> List.map (fun limitation -> limitation.Domain)
                
                let overallUncertainty = 
                    if uncertainAreas.IsEmpty && knowledgeGaps.IsEmpty then 0.0
                    else
                        let uncertaintyScore = float uncertainAreas.Length * 0.3
                        let gapScore = float knowledgeGaps.Length * 0.4
                        min 1.0 (uncertaintyScore + gapScore)
                
                let shouldTriggerLearning = overallUncertainty > uncertaintyThreshold
                
                return {|
                    Task = task
                    UncertaintyLevel = overallUncertainty
                    UncertainAreas = uncertainAreas
                    KnowledgeGaps = knowledgeGaps
                    ShouldTriggerLearning = shouldTriggerLearning
                    RecommendedActions = 
                        if shouldTriggerLearning then
                            ["acquire_knowledge"; "seek_guidance"; "break_down_task"]
                        else
                            ["proceed_with_caution"; "monitor_progress"]
                |}
            }

        /// Get decision history for transparency
        member this.GetDecisionHistory(limit: int option) =
            let decisions = 
                decisionHistory.Values
                |> Seq.sortByDescending (fun d -> d.Confidence)
                |> (match limit with Some l -> Seq.take l | None -> id)
                |> Seq.toList
            
            decisions

        /// Generate self-awareness report
        member this.GenerateSelfAwarenessReport() =
            async {
                let! currentState = this.AssessCurrentState()
                
                return {|
                    Timestamp = DateTime.UtcNow
                    OverallConfidence = currentState.ConfidenceLevel
                    Capabilities = currentState.CurrentCapabilities
                    Limitations = currentState.KnownLimitations |> List.map (fun l -> l.Domain)
                    UncertaintyAreas = currentState.UncertaintyAreas
                    LearningProgress = currentState.LearningProgress
                    PerformanceMetrics = currentState.PerformanceMetrics
                    RecentDecisions = this.GetDecisionHistory(Some 5)
                    SelfAssessmentAccuracy = 0.85 // Placeholder for actual measurement
                    MetaCognitionLevel = 0.78 // Placeholder for actual measurement
                |}
            }

        /// Monitor operational state continuously
        member this.StartContinuousMonitoring() =
            async {
                while true do
                    try
                        let! _ = this.AssessCurrentState()
                        do! Async.Sleep(30000) // Assess every 30 seconds
                    with
                    | ex -> 
                        printfn $"Self-awareness monitoring error: {ex.Message}"
                        do! Async.Sleep(60000) // Wait longer on error
            }

    /// Global self-awareness engine instance
    let SelfAwarenessEngine = SelfAwarenessEngine()

    /// Initialize self-awareness monitoring
    let InitializeSelfAwareness() =
        async {
            let! initialState = SelfAwarenessEngine.AssessCurrentState()
            
            // Start continuous monitoring in background
            Async.Start(SelfAwarenessEngine.StartContinuousMonitoring())
            
            printfn "🧠 Tier 11 Self-Awareness System Initialized"
            printfn $"   ✅ Current Confidence Level: {initialState.ConfidenceLevel:P0}"
            printfn $"   ✅ Active Capabilities: {initialState.CurrentCapabilities.Length}"
            printfn $"   ✅ Known Limitations: {initialState.KnownLimitations.Length}"
            printfn $"   ✅ Uncertainty Areas: {initialState.UncertaintyAreas.Length}"
            printfn "   ✅ Continuous monitoring: Active"
            printfn "   ✅ Decision transparency: Enabled"
            
            return initialState
        }
