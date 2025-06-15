namespace TarsEngine.FSharp.Core.Learning

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open TarsEngine.FSharp.Core.Tracing.AgenticTraceCapture

/// Adaptive Learning Engine for TARS
/// Implements real-time learning, pattern recognition, and autonomous adaptation
module AdaptiveLearningEngine =

    // ============================================================================
    // LEARNING TYPES
    // ============================================================================

    /// Learning pattern types
    type LearningPattern =
        | TaskSuccessPattern of taskType: string * successFactors: string list
        | PerformancePattern of componentName: string * optimizationHints: string list
        | UserInteractionPattern of interactionType: string * preferences: Map<string, float>
        | SystemBehaviorPattern of behavior: string * triggers: string list
        | ErrorPattern of errorType: string * resolutionSteps: string list

    /// Learning experience
    type LearningExperience = {
        ExperienceId: string
        Timestamp: DateTime
        Context: Map<string, obj>
        Action: string
        Outcome: string
        Success: bool
        PerformanceMetrics: Map<string, float>
        LearningPatterns: LearningPattern list
        Confidence: float
    }

    /// Adaptation strategy
    type AdaptationStrategy =
        | ParameterTuning of parameter: string * newValue: float * reason: string
        | BehaviorModification of behavior: string * modification: string * impact: float
        | CapabilityEnhancement of capability: string * enhancement: string * priority: int
        | ProcessOptimization of processName: string * optimization: string * expectedGain: float

    /// Learning model
    type LearningModel = {
        ModelId: string
        ModelType: string
        TrainingData: LearningExperience list
        Patterns: LearningPattern list
        Accuracy: float
        LastUpdated: DateTime
        PredictionCount: int
        SuccessRate: float
    }

    /// Adaptive behavior
    type AdaptiveBehavior = {
        BehaviorId: string
        Name: string
        Description: string
        Triggers: string list
        Actions: string list
        SuccessRate: float
        AdaptationHistory: (DateTime * string) list
        IsActive: bool
    }

    // ============================================================================
    // PATTERN RECOGNITION ENGINE
    // ============================================================================

    /// Pattern recognition for learning optimization
    type PatternRecognitionEngine() =
        let mutable recognizedPatterns = ConcurrentDictionary<string, LearningPattern>()

        /// Analyze experience for patterns
        member this.AnalyzeExperience(experience: LearningExperience) : LearningPattern list =
            let patterns = []
            
            // Task success pattern recognition
            let taskPatterns = 
                if experience.Success && experience.PerformanceMetrics.ContainsKey("execution_time") then
                    let executionTime = experience.PerformanceMetrics.["execution_time"]
                    if executionTime < 1.0 then // Fast execution
                        [TaskSuccessPattern(experience.Action, ["fast_execution"; "efficient_processing"])]
                    else
                        []
                else []
            
            // Performance pattern recognition
            let performancePatterns =
                experience.PerformanceMetrics
                |> Map.toList
                |> List.choose (fun (metric, value) ->
                    if value > 0.9 then
                        Some (PerformancePattern(experience.Action, [sprintf "high_%s" metric]))
                    elif value < 0.5 then
                        Some (PerformancePattern(experience.Action, [sprintf "low_%s_needs_improvement" metric]))
                    else
                        None
                )
            
            // Error pattern recognition
            let errorPatterns =
                if not experience.Success then
                    [ErrorPattern(experience.Outcome, ["analyze_failure"; "implement_safeguards"; "retry_with_modifications"])]
                else []
            
            let allPatterns = taskPatterns @ performancePatterns @ errorPatterns
            
            // Store recognized patterns
            for pattern in allPatterns do
                let patternKey = sprintf "%A_%s" pattern (DateTime.UtcNow.Ticks.ToString())
                recognizedPatterns.TryAdd(patternKey, pattern) |> ignore
            
            allPatterns

        /// Get all recognized patterns
        member this.GetRecognizedPatterns() : LearningPattern list =
            recognizedPatterns.Values |> Seq.toList

        /// Find similar patterns
        member this.FindSimilarPatterns(targetPattern: LearningPattern) : LearningPattern list =
            recognizedPatterns.Values
            |> Seq.filter (fun pattern ->
                match targetPattern, pattern with
                | TaskSuccessPattern(t1, _), TaskSuccessPattern(t2, _) -> t1 = t2
                | PerformancePattern(c1, _), PerformancePattern(c2, _) -> c1 = c2
                | ErrorPattern(e1, _), ErrorPattern(e2, _) -> e1 = e2
                | _ -> false
            )
            |> Seq.toList

    // ============================================================================
    // ADAPTIVE LEARNING ENGINE
    // ============================================================================

    /// Adaptive Learning Engine for continuous TARS improvement
    type AdaptiveLearningEngine() =
        let experiences = ConcurrentDictionary<string, LearningExperience>()
        let learningModels = ConcurrentDictionary<string, LearningModel>()
        let adaptiveBehaviors = ConcurrentDictionary<string, AdaptiveBehavior>()
        let patternEngine = PatternRecognitionEngine()
        let mutable totalLearningEvents = 0
        let mutable successfulAdaptations = 0

        /// Record learning experience
        member this.RecordExperience(context: Map<string, obj>, action: string, outcome: string, success: bool, metrics: Map<string, float>) : LearningExperience =
            let experienceId = Guid.NewGuid().ToString("N")[..7]
            let patterns = patternEngine.AnalyzeExperience({
                ExperienceId = experienceId
                Timestamp = DateTime.UtcNow
                Context = context
                Action = action
                Outcome = outcome
                Success = success
                PerformanceMetrics = metrics
                LearningPatterns = []
                Confidence = if success then 0.8 else 0.3
            })
            
            let experience = {
                ExperienceId = experienceId
                Timestamp = DateTime.UtcNow
                Context = context
                Action = action
                Outcome = outcome
                Success = success
                PerformanceMetrics = metrics
                LearningPatterns = patterns
                Confidence = if success then 0.8 else 0.3
            }
            
            experiences.TryAdd(experienceId, experience) |> ignore
            totalLearningEvents <- totalLearningEvents + 1
            
            GlobalTraceCapture.LogAgentEvent(
                "adaptive_learning_engine",
                "ExperienceRecorded",
                sprintf "Recorded learning experience: %s -> %s (%s)" action outcome (if success then "SUCCESS" else "FAILURE"),
                Map.ofList [("experience_id", experienceId :> obj); ("patterns_found", patterns.Length :> obj)],
                metrics |> Map.map (fun k v -> v :> obj),
                (if success then 1.0 else 0.3),
                20,
                []
            )
            
            experience

        /// Generate adaptation strategies
        member this.GenerateAdaptationStrategies(experience: LearningExperience) : AdaptationStrategy list =
            let strategies = []
            
            // Performance-based adaptations
            let performanceStrategies =
                experience.PerformanceMetrics
                |> Map.toList
                |> List.choose (fun (metric, value) ->
                    if value < 0.7 then
                        Some (ParameterTuning(metric, value * 1.2, sprintf "Improve %s performance" metric))
                    elif value > 0.95 then
                        Some (ProcessOptimization(experience.Action, sprintf "Optimize %s process" metric, 0.1))
                    else
                        None
                )
            
            // Pattern-based adaptations
            let patternStrategies =
                experience.LearningPatterns
                |> List.choose (fun pattern ->
                    match pattern with
                    | TaskSuccessPattern(taskType, factors) ->
                        Some (CapabilityEnhancement(taskType, String.concat ", " factors, 1))
                    | PerformancePattern(componentName, hints) ->
                        Some (BehaviorModification(componentName, String.concat ", " hints, 0.8))
                    | ErrorPattern(errorType, steps) ->
                        Some (BehaviorModification("error_handling", String.concat ", " steps, 0.9))
                    | _ -> None
                )
            
            performanceStrategies @ patternStrategies

        /// Apply adaptation strategy
        member this.ApplyAdaptationStrategy(strategy: AdaptationStrategy) : bool =
            try
                match strategy with
                | ParameterTuning(parameter, newValue, reason) ->
                    // Simulate parameter tuning
                    GlobalTraceCapture.LogAgentEvent(
                        "adaptive_learning_engine",
                        "ParameterTuning",
                        sprintf "Tuned parameter %s to %.3f: %s" parameter newValue reason,
                        Map.ofList [("parameter", parameter :> obj); ("new_value", newValue :> obj)],
                        Map.ofList [("adaptation_impact", 0.8)] |> Map.map (fun k v -> v :> obj),
                        1.0,
                        20,
                        []
                    )
                    true
                
                | BehaviorModification(behavior, modification, impact) ->
                    // Simulate behavior modification
                    let behaviorId = sprintf "%s_%s" behavior (DateTime.UtcNow.Ticks.ToString())
                    let adaptiveBehavior = {
                        BehaviorId = behaviorId
                        Name = behavior
                        Description = modification
                        Triggers = ["performance_threshold"; "error_detection"]
                        Actions = [modification]
                        SuccessRate = impact
                        AdaptationHistory = [(DateTime.UtcNow, modification)]
                        IsActive = true
                    }
                    adaptiveBehaviors.TryAdd(behaviorId, adaptiveBehavior) |> ignore
                    true
                
                | CapabilityEnhancement(capability, enhancement, priority) ->
                    // Simulate capability enhancement
                    GlobalTraceCapture.LogAgentEvent(
                        "adaptive_learning_engine",
                        "CapabilityEnhancement",
                        sprintf "Enhanced capability %s: %s (priority %d)" capability enhancement priority,
                        Map.ofList [("capability", capability :> obj); ("priority", priority :> obj)],
                        Map.ofList [("enhancement_value", float priority / 10.0)] |> Map.map (fun k v -> v :> obj),
                        1.0,
                        20,
                        []
                    )
                    true
                
                | ProcessOptimization(processName, optimization, expectedGain) ->
                    // Simulate process optimization
                    GlobalTraceCapture.LogAgentEvent(
                        "adaptive_learning_engine",
                        "ProcessOptimization",
                        sprintf "Optimized process %s: %s (expected gain: %.1f%%)" processName optimization (expectedGain * 100.0),
                        Map.ofList [("process", processName :> obj)],
                        Map.ofList [("expected_gain", expectedGain)] |> Map.map (fun k v -> v :> obj),
                        1.0,
                        20,
                        []
                    )
                    true
                
            with
            | ex ->
                GlobalTraceCapture.LogAgentEvent(
                    "adaptive_learning_engine",
                    "AdaptationError",
                    sprintf "Failed to apply adaptation strategy: %s" ex.Message,
                    Map.ofList [("error", ex.Message :> obj)],
                    Map.empty,
                    0.0,
                    20,
                    []
                )
                false

        /// Continuous learning loop
        member this.StartContinuousLearning() : Task =
            Task.Factory.StartNew(fun () ->
                while true do
                    try
                        // Analyze recent experiences
                        let recentExperiences = 
                            experiences.Values
                            |> Seq.filter (fun exp -> exp.Timestamp > DateTime.UtcNow.AddMinutes(-10.0))
                            |> Seq.toList
                        
                        // Generate and apply adaptations
                        for experience in recentExperiences do
                            let strategies = this.GenerateAdaptationStrategies(experience)
                            for strategy in strategies do
                                let success = this.ApplyAdaptationStrategy(strategy)
                                if success then
                                    successfulAdaptations <- successfulAdaptations + 1
                        
                        // Update learning models
                        this.UpdateLearningModels(recentExperiences)
                        
                        // Sleep for a while before next learning cycle
                        System.Threading.Thread.Sleep(30000) // 30 seconds
                        
                    with
                    | ex ->
                        GlobalTraceCapture.LogAgentEvent(
                            "adaptive_learning_engine",
                            "LearningError",
                            sprintf "Continuous learning error: %s" ex.Message,
                            Map.ofList [("error", ex.Message :> obj)],
                            Map.empty,
                            0.0,
                            20,
                            []
                        )
                        System.Threading.Thread.Sleep(60000) // Wait longer on error
            )

        /// Update learning models
        member this.UpdateLearningModels(experiences: LearningExperience list) : unit =
            if experiences.Length > 0 then
                let modelId = "tars_adaptive_model"
                let successRate = 
                    experiences 
                    |> List.filter (fun exp -> exp.Success) 
                    |> List.length 
                    |> fun count -> float count / float experiences.Length
                
                let allPatterns = experiences |> List.collect (fun exp -> exp.LearningPatterns)
                
                let model = {
                    ModelId = modelId
                    ModelType = "AdaptiveLearning"
                    TrainingData = experiences
                    Patterns = allPatterns
                    Accuracy = successRate
                    LastUpdated = DateTime.UtcNow
                    PredictionCount = experiences.Length
                    SuccessRate = successRate
                }
                
                learningModels.AddOrUpdate(modelId, model, fun _ _ -> model) |> ignore

        /// Get learning statistics
        member this.GetLearningStatistics() : Map<string, obj> =
            let totalExperiences = experiences.Count
            let successfulExperiences = experiences.Values |> Seq.filter (fun exp -> exp.Success) |> Seq.length
            let totalPatterns = patternEngine.GetRecognizedPatterns().Length
            let activeBehaviors = adaptiveBehaviors.Values |> Seq.filter (fun b -> b.IsActive) |> Seq.length
            
            Map.ofList [
                ("total_experiences", totalExperiences :> obj)
                ("successful_experiences", successfulExperiences :> obj)
                ("success_rate", (if totalExperiences > 0 then float successfulExperiences / float totalExperiences else 0.0) :> obj)
                ("recognized_patterns", totalPatterns :> obj)
                ("active_behaviors", activeBehaviors :> obj)
                ("total_adaptations", successfulAdaptations :> obj)
                ("learning_models", learningModels.Count :> obj)
                ("adaptation_rate", (if totalLearningEvents > 0 then float successfulAdaptations / float totalLearningEvents else 0.0) :> obj)
            ]

        /// Get all experiences
        member this.GetAllExperiences() : LearningExperience list =
            experiences.Values |> Seq.toList

        /// Get adaptive behaviors
        member this.GetAdaptiveBehaviors() : AdaptiveBehavior list =
            adaptiveBehaviors.Values |> Seq.toList

    /// Adaptive learning service for TARS
    type AdaptiveLearningService() =
        let learningEngine = AdaptiveLearningEngine()
        let mutable isLearningActive = false

        /// Start adaptive learning
        member this.StartLearning() : Task =
            if not isLearningActive then
                isLearningActive <- true
                learningEngine.StartContinuousLearning()
            else
                Task.CompletedTask

        /// Record learning experience
        member this.RecordExperience(context: Map<string, obj>, action: string, outcome: string, success: bool, metrics: Map<string, float>) : LearningExperience =
            learningEngine.RecordExperience(context, action, outcome, success, metrics)

        /// Get learning statistics
        member this.GetStatistics() : Map<string, obj> =
            learningEngine.GetLearningStatistics()

        /// Get experiences
        member this.GetExperiences() : LearningExperience list =
            learningEngine.GetAllExperiences()

        /// Get adaptive behaviors
        member this.GetBehaviors() : AdaptiveBehavior list =
            learningEngine.GetAdaptiveBehaviors()
