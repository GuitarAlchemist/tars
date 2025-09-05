namespace TarsEngine.FSharp.Cli.Core

open System
open System.Collections.Generic
open System.Threading
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem

/// Unified Agent Reasoning Engine - Agent reasoning using unified architecture
module UnifiedAgentReasoningEngine =
    
    /// Agent belief with unified types
    type UnifiedAgentBelief = {
        BeliefId: string
        AgentId: string
        Content: string
        Confidence: float
        Timestamp: DateTime
        Source: string
        Evidence: string list
        Tags: string list
        CorrelationId: string
    }
    
    /// Reasoning rule types
    type UnifiedReasoningRule =
        | Implication of premise: string * conclusion: string * confidence: float
        | Contradiction of belief1: string * belief2: string * severity: float
        | Temporal of condition: string * timeConstraint: TimeSpan * confidence: float
        | ConfidenceThreshold of belief: string * minConfidence: float
        | EvidenceRequirement of belief: string * minEvidenceCount: int
    
    /// Contradiction detection result
    type UnifiedContradictionResult = {
        ContradictionId: string
        ConflictingBeliefs: UnifiedAgentBelief list
        Severity: float
        DetectedAt: DateTime
        Resolution: string option
        ProofId: string option
        CorrelationId: string
    }
    
    /// Reasoning metrics
    type UnifiedReasoningMetrics = {
        AgentId: string
        CycleId: string
        StartTime: DateTime
        EndTime: DateTime
        BeliefsProcessed: int
        RulesApplied: int
        ContradictionsDetected: int
        NewBeliefsGenerated: int
        ConfidenceChanges: int
        ProofId: string option
        CorrelationId: string
    }
    
    /// Agent reasoning context
    type ReasoningContext = {
        ConfigManager: UnifiedConfigurationManager
        ProofGenerator: UnifiedProofGenerator
        Logger: ITarsLogger
        CorrelationId: string
        MaxReasoningDepth: int
        ConfidenceThreshold: float
        ContradictionSeverityThreshold: float
    }
    
    /// Create reasoning context
    let createReasoningContext (logger: ITarsLogger) (configManager: UnifiedConfigurationManager) (proofGenerator: UnifiedProofGenerator) =
        {
            ConfigManager = configManager
            ProofGenerator = proofGenerator
            Logger = logger
            CorrelationId = generateCorrelationId()
            MaxReasoningDepth = ConfigurationExtensions.getInt configManager "tars.reasoning.maxDepth" 10
            ConfidenceThreshold = ConfigurationExtensions.getFloat configManager "tars.reasoning.confidenceThreshold" 0.7
            ContradictionSeverityThreshold = ConfigurationExtensions.getFloat configManager "tars.reasoning.contradictionThreshold" 0.8
        }
    
    /// Apply reasoning rule to beliefs
    let applyReasoningRule (context: ReasoningContext) (rule: UnifiedReasoningRule) (beliefs: UnifiedAgentBelief list) =
        task {
            try
                context.Logger.LogInformation(context.CorrelationId, $"Applying reasoning rule: {rule}")
                
                match rule with
                | Implication (premise, conclusion, confidence) ->
                    let matchingBeliefs = beliefs |> List.filter (fun b -> b.Content.Contains(premise) && b.Confidence >= context.ConfidenceThreshold)
                    let newBeliefs = 
                        matchingBeliefs 
                        |> List.map (fun b -> {
                            BeliefId = generateCorrelationId()
                            AgentId = b.AgentId
                            Content = conclusion
                            Confidence = Math.Min(b.Confidence * confidence, 1.0)
                            Timestamp = DateTime.UtcNow
                            Source = $"Implication from {b.BeliefId}"
                            Evidence = [b.BeliefId]
                            Tags = ["derived"; "implication"] @ b.Tags
                            CorrelationId = context.CorrelationId
                        })
                    return Success (newBeliefs, Map [("ruleType", box "Implication"); ("newBeliefs", box newBeliefs.Length)])
                
                | Contradiction (belief1, belief2, severity) ->
                    let conflicts = 
                        beliefs 
                        |> List.filter (fun b -> b.Content.Contains(belief1))
                        |> List.collect (fun b1 -> 
                            beliefs 
                            |> List.filter (fun b2 -> b2.Content.Contains(belief2) && b2.BeliefId <> b1.BeliefId)
                            |> List.map (fun b2 -> (b1, b2)))
                    
                    let contradictions = 
                        conflicts 
                        |> List.map (fun (b1, b2) -> {
                            ContradictionId = generateCorrelationId()
                            ConflictingBeliefs = [b1; b2]
                            Severity = severity
                            DetectedAt = DateTime.UtcNow
                            Resolution = None
                            ProofId = None
                            CorrelationId = context.CorrelationId
                        })
                    
                    return Success ([], Map [("ruleType", box "Contradiction"); ("contradictions", box contradictions)])
                
                | Temporal (condition, timeConstraint, confidence) ->
                    let recentBeliefs = 
                        beliefs 
                        |> List.filter (fun b -> 
                            b.Content.Contains(condition) && 
                            DateTime.UtcNow - b.Timestamp <= timeConstraint)
                    
                    return Success ([], Map [("ruleType", box "Temporal"); ("recentBeliefs", box recentBeliefs.Length)])
                
                | ConfidenceThreshold (belief, minConfidence) ->
                    let filteredBeliefs = 
                        beliefs 
                        |> List.filter (fun b -> b.Content.Contains(belief) && b.Confidence >= minConfidence)
                    
                    return Success ([], Map [("ruleType", box "ConfidenceThreshold"); ("qualifyingBeliefs", box filteredBeliefs.Length)])
                
                | EvidenceRequirement (belief, minEvidenceCount) ->
                    let evidenceBeliefs = 
                        beliefs 
                        |> List.filter (fun b -> b.Content.Contains(belief) && b.Evidence.Length >= minEvidenceCount)
                    
                    return Success ([], Map [("ruleType", box "EvidenceRequirement"); ("evidenceBeliefs", box evidenceBeliefs.Length)])
            
            with
            | ex ->
                context.Logger.LogError(context.CorrelationId, TarsError.create "ReasoningRuleError" "Failed to apply reasoning rule" (Some ex), ex)
                let error = ExecutionError ($"Reasoning rule application failed: {ex.Message}", Some ex)
                return Failure (error, context.CorrelationId)
        }
    
    /// Detect contradictions in beliefs
    let detectContradictions (context: ReasoningContext) (beliefs: UnifiedAgentBelief list) =
        task {
            try
                context.Logger.LogInformation(context.CorrelationId, $"Detecting contradictions in {beliefs.Length} beliefs")
                
                let contradictions = ResizeArray<UnifiedContradictionResult>()
                
                // Simple contradiction detection based on opposing keywords
                let opposingPairs = [
                    ("true", "false")
                    ("yes", "no")
                    ("exists", "does not exist")
                    ("valid", "invalid")
                    ("correct", "incorrect")
                ]
                
                for (positive, negative) in opposingPairs do
                    let positiveBeliefs = beliefs |> List.filter (fun b -> b.Content.ToLower().Contains(positive))
                    let negativeBeliefs = beliefs |> List.filter (fun b -> b.Content.ToLower().Contains(negative))
                    
                    for posBelief in positiveBeliefs do
                        for negBelief in negativeBeliefs do
                            if posBelief.BeliefId <> negBelief.BeliefId then
                                let severity = Math.Min(posBelief.Confidence + negBelief.Confidence, 1.0)
                                if severity >= context.ContradictionSeverityThreshold then
                                    contradictions.Add({
                                        ContradictionId = generateCorrelationId()
                                        ConflictingBeliefs = [posBelief; negBelief]
                                        Severity = severity
                                        DetectedAt = DateTime.UtcNow
                                        Resolution = None
                                        ProofId = None
                                        CorrelationId = context.CorrelationId
                                    })
                
                // Generate proof for contradiction detection
                let! contradictionProof =
                    ProofExtensions.generateExecutionProof
                        context.ProofGenerator
                        $"ContradictionDetection_{contradictions.Count}"
                        context.CorrelationId
                
                let proofId = match contradictionProof with
                              | Success (proof, _) -> Some proof.ProofId
                              | Failure _ -> None
                
                return Success (contradictions |> Seq.toList, Map [
                    ("contradictionsFound", box contradictions.Count)
                    ("beliefsAnalyzed", box beliefs.Length)
                    ("proofId", box proofId)
                ])
            
            with
            | ex ->
                context.Logger.LogError(context.CorrelationId, TarsError.create "ContradictionDetectionError" "Failed to detect contradictions" (Some ex), ex)
                let error = ExecutionError ($"Contradiction detection failed: {ex.Message}", Some ex)
                return Failure (error, context.CorrelationId)
        }
    
    /// Perform reasoning cycle for agent
    let performReasoningCycle (context: ReasoningContext) (agentId: string) (beliefs: UnifiedAgentBelief list) (rules: UnifiedReasoningRule list) =
        task {
            try
                let startTime = DateTime.UtcNow
                let cycleId = generateCorrelationId()
                
                context.Logger.LogInformation(context.CorrelationId, $"Starting reasoning cycle for agent {agentId}")
                
                // Generate proof for reasoning cycle
                let! cycleProof =
                    ProofExtensions.generateAgentActionProof
                        context.ProofGenerator
                        agentId
                        $"ReasoningCycle_{cycleId}"
                        context.CorrelationId
                
                let mutable processedBeliefs = beliefs
                let mutable newBeliefs = []
                let mutable rulesApplied = 0
                let mutable confidenceChanges = 0
                
                // Apply reasoning rules
                for rule in rules do
                    let! ruleResult = applyReasoningRule context rule processedBeliefs
                    match ruleResult with
                    | Success (derivedBeliefs, metadata) ->
                        newBeliefs <- newBeliefs @ derivedBeliefs
                        processedBeliefs <- processedBeliefs @ derivedBeliefs
                        rulesApplied <- rulesApplied + 1
                    | Failure (error, _) ->
                        context.Logger.LogWarning(context.CorrelationId, $"Rule application failed: {TarsError.toString error}")
                
                // Detect contradictions
                let! contradictionResult = detectContradictions context processedBeliefs
                let contradictionsDetected = 
                    match contradictionResult with
                    | Success (contradictions, _) -> contradictions.Length
                    | Failure _ -> 0
                
                let endTime = DateTime.UtcNow
                
                let proofId = match cycleProof with
                              | Success (proof, _) -> Some proof.ProofId
                              | Failure _ -> None
                
                let metrics = {
                    AgentId = agentId
                    CycleId = cycleId
                    StartTime = startTime
                    EndTime = endTime
                    BeliefsProcessed = beliefs.Length
                    RulesApplied = rulesApplied
                    ContradictionsDetected = contradictionsDetected
                    NewBeliefsGenerated = newBeliefs.Length
                    ConfidenceChanges = confidenceChanges
                    ProofId = proofId
                    CorrelationId = context.CorrelationId
                }
                
                return Success (metrics, Map [
                    ("cycleId", box cycleId)
                    ("duration", box (endTime - startTime).TotalMilliseconds)
                    ("newBeliefs", box newBeliefs)
                ])
            
            with
            | ex ->
                context.Logger.LogError(context.CorrelationId, TarsError.create "ReasoningCycleError" "Reasoning cycle failed" (Some ex), ex)
                let error = ExecutionError ($"Reasoning cycle failed: {ex.Message}", Some ex)
                return Failure (error, context.CorrelationId)
        }
    
    /// Unified Agent Reasoning Engine implementation
    type UnifiedAgentReasoningEngine(logger: ITarsLogger, configManager: UnifiedConfigurationManager, proofGenerator: UnifiedProofGenerator) =
        
        let context = createReasoningContext logger configManager proofGenerator
        let mutable agentBeliefs = Map.empty<string, UnifiedAgentBelief list>
        let mutable reasoningRules = []
        
        /// Add belief for agent
        member this.AddBeliefAsync(agentId: string, belief: UnifiedAgentBelief) : Task<TarsResult<string, TarsError>> =
            task {
                try
                    let currentBeliefs = agentBeliefs.TryFind(agentId) |> Option.defaultValue []
                    let updatedBeliefs = belief :: currentBeliefs
                    agentBeliefs <- agentBeliefs.Add(agentId, updatedBeliefs)
                    
                    // Generate proof for belief addition
                    let! beliefProof =
                        ProofExtensions.generateAgentActionProof
                            context.ProofGenerator
                            agentId
                            $"BeliefAdded_{belief.BeliefId}"
                            context.CorrelationId
                    
                    context.Logger.LogInformation(context.CorrelationId, $"Added belief {belief.BeliefId} for agent {agentId}")
                    
                    return Success (belief.BeliefId, Map [
                        ("agentId", box agentId)
                        ("beliefId", box belief.BeliefId)
                        ("totalBeliefs", box updatedBeliefs.Length)
                    ])
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to add belief: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Get beliefs for agent
        member this.GetBeliefsAsync(agentId: string) : Task<TarsResult<UnifiedAgentBelief list, TarsError>> =
            task {
                try
                    let beliefs = agentBeliefs.TryFind(agentId) |> Option.defaultValue []
                    return Success (beliefs, Map [
                        ("agentId", box agentId)
                        ("beliefCount", box beliefs.Length)
                    ])
                
                with
                | ex ->
                    let error = ExecutionError ($"Failed to get beliefs: {ex.Message}", Some ex)
                    return Failure (error, context.CorrelationId)
            }
        
        /// Add reasoning rule
        member this.AddReasoningRule(rule: UnifiedReasoningRule) : unit =
            reasoningRules <- rule :: reasoningRules
            context.Logger.LogInformation(context.CorrelationId, $"Added reasoning rule: {rule}")
        
        /// Perform reasoning cycle for agent
        member this.PerformReasoningCycleAsync(agentId: string) : Task<TarsResult<UnifiedReasoningMetrics, TarsError>> =
            task {
                let beliefs = agentBeliefs.TryFind(agentId) |> Option.defaultValue []
                let! result = performReasoningCycle context agentId beliefs reasoningRules
                return result
            }
        
        /// Detect contradictions for agent
        member this.DetectContradictionsAsync(agentId: string) : Task<TarsResult<UnifiedContradictionResult list, TarsError>> =
            task {
                let beliefs = agentBeliefs.TryFind(agentId) |> Option.defaultValue []
                let! result = detectContradictions context beliefs
                return result
            }
        
        /// Get reasoning statistics
        member this.GetStatistics() : Map<string, obj> =
            let totalBeliefs = agentBeliefs |> Map.toSeq |> Seq.sumBy (fun (_, beliefs) -> beliefs.Length)
            let agentCount = agentBeliefs.Count
            
            Map [
                ("totalAgents", box agentCount)
                ("totalBeliefs", box totalBeliefs)
                ("totalRules", box reasoningRules.Length)
                ("maxReasoningDepth", box context.MaxReasoningDepth)
                ("confidenceThreshold", box context.ConfidenceThreshold)
                ("contradictionThreshold", box context.ContradictionSeverityThreshold)
                ("correlationId", box context.CorrelationId)
                ("isInitialized", box true)
            ]
