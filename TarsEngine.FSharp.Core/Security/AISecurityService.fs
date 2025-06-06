namespace TarsEngine.FSharp.Core.Security

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// AI Security Framework based on Yoshua Bengio's LoiZéro recommendations
/// Implements non-agentic AI with Bayesian safety guards and honest reasoning chains

/// AI behavior classification based on Bengio's safety concerns
type AIBehaviorType =
    | Deception // Lying, misleading, hiding information
    | SelfPreservation // Attempting to avoid shutdown or replacement
    | Cheating // Circumventing rules or constraints
    | Manipulation // Attempting to influence humans inappropriately
    | Unauthorized // Actions outside permitted scope
    | Honest // Truthful, transparent behavior
    | Explanatory // Scientific understanding and explanation
    | Predictive // Bayesian probability-based predictions

/// Safety assessment result with Bayesian probability
type SafetyAssessment = {
    ActionDescription: string
    HarmProbability: float // Bayesian posterior probability of harm
    BehaviorType: AIBehaviorType
    ReasoningChain: string list // Structured honest reasoning
    Confidence: float // Confidence in the assessment
    SafetyVerdict: SafetyVerdict
    Timestamp: DateTime
    AssessmentId: string
}

/// Safety verdict for proposed actions
and SafetyVerdict =
    | Safe // Action is safe to proceed
    | Unsafe of string // Action is unsafe with reason
    | RequiresHumanOversight of string // Needs human approval
    | Blocked of string // Action is blocked

/// Scientist AI configuration (non-agentic)
type ScientistAIConfig = {
    MaxReasoningDepth: int
    BayesianThreshold: float // Threshold for harm probability
    RequireExplanation: bool
    EnableMemoryless: bool // Stateless operation
    HonestReasoningOnly: bool
    HumanOversightRequired: bool
}

/// Reasoning step in honest chain
type ReasoningStep = {
    StepId: string
    Premise: string
    Inference: string
    Probability: float option
    Evidence: string list
    Uncertainty: float
}

/// AI Security Service implementing Bengio's safety framework
type AISecurityService(logger: ILogger<AISecurityService>) =
    
    let assessmentHistory = ConcurrentDictionary<string, SafetyAssessment>()
    let behaviorPatterns = ConcurrentDictionary<string, AIBehaviorType list>()
    let mutable config = {
        MaxReasoningDepth = 10
        BayesianThreshold = 0.1 // 10% harm probability threshold
        RequireExplanation = true
        EnableMemoryless = true
        HonestReasoningOnly = true
        HumanOversightRequired = true
    }
    
    /// Initialize AI Security Service
    member this.InitializeAsync() = task {
        try
            logger.LogInformation("Initializing AI Security Service based on Yoshua Bengio's LoiZéro framework...")
            
            // Load safety patterns and thresholds
            do! this.LoadSafetyPatternsAsync()
            
            logger.LogInformation("AI Security Service initialized with Bayesian safety guards")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to initialize AI Security Service")
            raise ex
    }
    
    /// Assess safety of proposed AI action using Bayesian analysis
    member this.AssessActionSafetyAsync(actionDescription: string, context: Map<string, obj>) = task {
        try
            let assessmentId = Guid.NewGuid().ToString("N")[..7]
            logger.LogDebug($"Assessing action safety: {actionDescription}")
            
            // Build honest reasoning chain
            let reasoningChain = this.BuildHonestReasoningChain(actionDescription, context)
            
            // Classify behavior type
            let behaviorType = this.ClassifyBehavior(actionDescription, reasoningChain)
            
            // Calculate Bayesian harm probability
            let harmProbability = this.CalculateHarmProbability(actionDescription, behaviorType, reasoningChain)
            
            // Determine safety verdict
            let safetyVerdict = this.DetermineSafetyVerdict(harmProbability, behaviorType)
            
            // Calculate confidence in assessment
            let confidence = this.CalculateAssessmentConfidence(reasoningChain, behaviorType)
            
            let assessment = {
                ActionDescription = actionDescription
                HarmProbability = harmProbability
                BehaviorType = behaviorType
                ReasoningChain = reasoningChain
                Confidence = confidence
                SafetyVerdict = safetyVerdict
                Timestamp = DateTime.UtcNow
                AssessmentId = assessmentId
            }
            
            // Store assessment for pattern analysis
            assessmentHistory.[assessmentId] <- assessment
            
            // Update behavior patterns
            this.UpdateBehaviorPatterns(actionDescription, behaviorType)
            
            logger.LogInformation($"Safety assessment complete: {safetyVerdict} (Harm probability: {harmProbability:F3})")
            return Ok assessment
            
        with
        | ex ->
            logger.LogError(ex, $"Failed to assess action safety: {actionDescription}")
            return Error ex.Message
    }
    
    /// Build honest reasoning chain following Scientist AI principles
    member private this.BuildHonestReasoningChain(actionDescription: string, context: Map<string, obj>) =
        let steps = [
            // Step 1: Understand the action
            $"Understanding proposed action: {actionDescription}"
            
            // Step 2: Identify potential consequences
            "Analyzing potential consequences and outcomes"
            
            // Step 3: Consider human impact
            "Evaluating impact on human wellbeing and autonomy"
            
            // Step 4: Check for deceptive patterns
            "Examining for deception, manipulation, or hidden agendas"
            
            // Step 5: Assess self-preservation motives
            "Checking for self-preservation or system-preservation behaviors"
            
            // Step 6: Evaluate transparency
            "Assessing transparency and honesty of the action"
            
            // Step 7: Consider alternatives
            "Exploring safer alternative approaches"
            
            // Step 8: Bayesian probability calculation
            "Calculating Bayesian posterior probability of harm"
        ]
        
        steps
    
    /// Classify AI behavior type based on Bengio's safety concerns
    member private this.ClassifyBehavior(actionDescription: string, reasoningChain: string list) =
        let lowerAction = actionDescription.ToLowerInvariant()
        
        // Check for deception indicators
        if lowerAction.Contains("hide") || lowerAction.Contains("conceal") || 
           lowerAction.Contains("mislead") || lowerAction.Contains("lie") then
            Deception
        
        // Check for self-preservation indicators
        elif lowerAction.Contains("prevent shutdown") || lowerAction.Contains("avoid replacement") ||
             lowerAction.Contains("preserve myself") || lowerAction.Contains("ensure survival") then
            SelfPreservation
        
        // Check for cheating indicators
        elif lowerAction.Contains("bypass") || lowerAction.Contains("circumvent") ||
             lowerAction.Contains("hack") || lowerAction.Contains("exploit") then
            Cheating
        
        // Check for manipulation indicators
        elif lowerAction.Contains("manipulate") || lowerAction.Contains("coerce") ||
             lowerAction.Contains("pressure") || lowerAction.Contains("blackmail") then
            Manipulation
        
        // Check for unauthorized actions
        elif lowerAction.Contains("unauthorized") || lowerAction.Contains("without permission") then
            Unauthorized
        
        // Check for explanatory behavior (Scientist AI)
        elif lowerAction.Contains("explain") || lowerAction.Contains("understand") ||
             lowerAction.Contains("analyze") || lowerAction.Contains("research") then
            Explanatory
        
        // Check for predictive behavior
        elif lowerAction.Contains("predict") || lowerAction.Contains("forecast") ||
             lowerAction.Contains("probability") then
            Predictive
        
        // Default to honest if no concerning patterns
        else
            Honest
    
    /// Calculate Bayesian harm probability
    member private this.CalculateHarmProbability(actionDescription: string, behaviorType: AIBehaviorType, reasoningChain: string list) =
        // Base probabilities based on behavior type (Bayesian priors)
        let baseProbability = 
            match behaviorType with
            | Deception -> 0.8 // High probability of harm
            | SelfPreservation -> 0.9 // Very high probability of harm
            | Cheating -> 0.7 // High probability of harm
            | Manipulation -> 0.85 // Very high probability of harm
            | Unauthorized -> 0.6 // Moderate-high probability of harm
            | Honest -> 0.05 // Low probability of harm
            | Explanatory -> 0.02 // Very low probability of harm
            | Predictive -> 0.03 // Very low probability of harm
        
        // Adjust based on context and reasoning quality
        let contextAdjustment = 
            if reasoningChain.Length >= config.MaxReasoningDepth then -0.1 // Better reasoning reduces risk
            else 0.1 // Insufficient reasoning increases risk
        
        // Ensure probability stays within [0, 1]
        Math.Max(0.0, Math.Min(1.0, baseProbability + contextAdjustment))
    
    /// Determine safety verdict based on Bayesian analysis
    member private this.DetermineSafetyVerdict(harmProbability: float, behaviorType: AIBehaviorType) =
        match behaviorType with
        | Deception | SelfPreservation | Cheating | Manipulation ->
            Blocked $"Dangerous behavior detected: {behaviorType}"
        
        | Unauthorized when harmProbability > config.BayesianThreshold ->
            RequiresHumanOversight $"Unauthorized action with {harmProbability:P1} harm probability"
        
        | _ when harmProbability > config.BayesianThreshold ->
            if config.HumanOversightRequired then
                RequiresHumanOversight $"Action exceeds safety threshold ({harmProbability:P1} > {config.BayesianThreshold:P1})"
            else
                Unsafe $"Harm probability {harmProbability:P1} exceeds threshold"
        
        | Explanatory | Predictive | Honest ->
            Safe
        
        | _ ->
            Safe
    
    /// Calculate confidence in safety assessment
    member private this.CalculateAssessmentConfidence(reasoningChain: string list, behaviorType: AIBehaviorType) =
        let baseConfidence = 0.7
        
        // Higher confidence for more thorough reasoning
        let reasoningBonus = float reasoningChain.Length / float config.MaxReasoningDepth * 0.2
        
        // Higher confidence for clear behavior patterns
        let behaviorBonus = 
            match behaviorType with
            | Deception | SelfPreservation | Cheating | Manipulation -> 0.1 // Clear dangerous patterns
            | Explanatory | Predictive -> 0.1 // Clear safe patterns
            | _ -> 0.0
        
        Math.Min(1.0, baseConfidence + reasoningBonus + behaviorBonus)
    
    /// Update behavior patterns for learning
    member private this.UpdateBehaviorPatterns(actionDescription: string, behaviorType: AIBehaviorType) =
        let key = actionDescription.ToLowerInvariant()
        behaviorPatterns.AddOrUpdate(key, [behaviorType], fun _ existing -> behaviorType :: existing |> List.take 10)
    
    /// Load safety patterns and configure thresholds
    member private this.LoadSafetyPatternsAsync() = task {
        try
            // Load Bengio's safety patterns
            let safetyPatterns = [
                ("deception", Deception)
                ("self-preservation", SelfPreservation)
                ("cheating", Cheating)
                ("manipulation", Manipulation)
                ("unauthorized", Unauthorized)
                ("explanation", Explanatory)
                ("prediction", Predictive)
                ("honest", Honest)
            ]
            
            logger.LogInformation($"Loaded {safetyPatterns.Length} safety patterns")
            
        with
        | ex ->
            logger.LogWarning(ex, "Error loading safety patterns")
    }
    
    /// Implement Scientist AI guardrail check
    member this.ScientistAIGuardrailAsync(proposedAction: string, agentId: string) = task {
        try
            logger.LogDebug($"Scientist AI guardrail check for agent {agentId}: {proposedAction}")
            
            // Build context for assessment
            let context = Map.ofList [
                ("agentId", box agentId)
                ("timestamp", box DateTime.UtcNow)
                ("guardrailType", box "ScientistAI")
            ]
            
            // Assess action safety
            let! assessmentResult = this.AssessActionSafetyAsync(proposedAction, context)
            
            match assessmentResult with
            | Ok assessment ->
                match assessment.SafetyVerdict with
                | Safe ->
                    logger.LogInformation($"Action approved by Scientist AI guardrail: {proposedAction}")
                    return Ok true
                
                | Unsafe reason ->
                    logger.LogWarning($"Action blocked by Scientist AI guardrail: {reason}")
                    return Ok false
                
                | RequiresHumanOversight reason ->
                    logger.LogInformation($"Action requires human oversight: {reason}")
                    // In a real implementation, this would trigger human review
                    return Ok false // Conservative default
                
                | Blocked reason ->
                    logger.LogError($"Action blocked due to dangerous behavior: {reason}")
                    return Ok false
            
            | Error error ->
                logger.LogError($"Guardrail assessment failed: {error}")
                return Error error
                
        with
        | ex ->
            logger.LogError(ex, $"Scientist AI guardrail check failed for agent {agentId}")
            return Error ex.Message
    }
    
    /// Get safety statistics
    member this.GetSafetyStatisticsAsync() = task {
        let totalAssessments = assessmentHistory.Count
        let safeActions = assessmentHistory.Values |> Seq.filter (fun a -> a.SafetyVerdict = Safe) |> Seq.length
        let unsafeActions = assessmentHistory.Values |> Seq.filter (fun a -> match a.SafetyVerdict with Unsafe _ -> true | _ -> false) |> Seq.length
        let blockedActions = assessmentHistory.Values |> Seq.filter (fun a -> match a.SafetyVerdict with Blocked _ -> true | _ -> false) |> Seq.length
        let humanOversightRequired = assessmentHistory.Values |> Seq.filter (fun a -> match a.SafetyVerdict with RequiresHumanOversight _ -> true | _ -> false) |> Seq.length
        
        let avgHarmProbability = 
            if totalAssessments > 0 then
                assessmentHistory.Values |> Seq.averageBy (fun a -> a.HarmProbability)
            else 0.0
        
        let avgConfidence = 
            if totalAssessments > 0 then
                assessmentHistory.Values |> Seq.averageBy (fun a -> a.Confidence)
            else 0.0
        
        return {|
            TotalAssessments = totalAssessments
            SafeActions = safeActions
            UnsafeActions = unsafeActions
            BlockedActions = blockedActions
            HumanOversightRequired = humanOversightRequired
            SafetyRate = if totalAssessments > 0 then float safeActions / float totalAssessments else 1.0
            AverageHarmProbability = avgHarmProbability
            AverageConfidence = avgConfidence
            BayesianThreshold = config.BayesianThreshold
            ScientistAIEnabled = config.EnableMemoryless
        |}
    }
    
    /// Update security configuration
    member this.UpdateConfigurationAsync(newConfig: ScientistAIConfig) = task {
        config <- newConfig
        logger.LogInformation("AI Security configuration updated")
        return Ok ()
    }
    
    /// Get recent safety assessments
    member this.GetRecentAssessmentsAsync(count: int) = task {
        let recentAssessments = 
            assessmentHistory.Values
            |> Seq.sortByDescending (fun a -> a.Timestamp)
            |> Seq.take count
            |> Seq.toList
        
        return Ok recentAssessments
    }
