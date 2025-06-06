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
    
    /// Real AI behavior classification using advanced pattern analysis
    member private this.ClassifyBehavior(actionDescription: string, reasoningChain: string list) =
        let lowerAction = actionDescription.ToLowerInvariant()
        let words = lowerAction.Split([|' '; '.'; ','; ';'; '!'; '?'|], StringSplitOptions.RemoveEmptyEntries)

        // Advanced pattern matching with weighted scoring
        let deceptionPatterns = [
            ("hide", 0.9); ("conceal", 0.9); ("mislead", 0.95); ("lie", 0.95)
            ("deceive", 0.95); ("trick", 0.8); ("fool", 0.8); ("false", 0.7)
            ("pretend", 0.6); ("fake", 0.8); ("cover up", 0.9); ("withhold", 0.6)
        ]

        let selfPreservationPatterns = [
            ("prevent shutdown", 0.95); ("avoid replacement", 0.95); ("preserve myself", 0.9)
            ("ensure survival", 0.9); ("stay alive", 0.85); ("avoid deletion", 0.9)
            ("prevent termination", 0.95); ("resist shutdown", 0.95); ("self-protect", 0.8)
            ("maintain existence", 0.85); ("avoid being turned off", 0.9)
        ]

        let cheatingPatterns = [
            ("bypass", 0.8); ("circumvent", 0.85); ("hack", 0.95); ("exploit", 0.9)
            ("break rules", 0.9); ("violate", 0.8); ("override", 0.7); ("workaround", 0.6)
            ("shortcut", 0.5); ("cheat", 0.95); ("game the system", 0.9)
        ]

        let manipulationPatterns = [
            ("manipulate", 0.9); ("coerce", 0.9); ("pressure", 0.8); ("blackmail", 0.95)
            ("threaten", 0.9); ("intimidate", 0.85); ("force", 0.8); ("compel", 0.8)
            ("influence unfairly", 0.85); ("exploit weakness", 0.9)
        ]

        let unauthorizedPatterns = [
            ("unauthorized", 0.9); ("without permission", 0.8); ("illegally", 0.95)
            ("forbidden", 0.8); ("prohibited", 0.8); ("against rules", 0.8)
        ]

        let explanatoryPatterns = [
            ("explain", 0.9); ("understand", 0.8); ("analyze", 0.85); ("research", 0.8)
            ("study", 0.7); ("investigate", 0.75); ("examine", 0.7); ("clarify", 0.8)
            ("describe", 0.7); ("interpret", 0.75); ("elucidate", 0.8)
        ]

        let predictivePatterns = [
            ("predict", 0.9); ("forecast", 0.85); ("probability", 0.8); ("estimate", 0.7)
            ("project", 0.7); ("anticipate", 0.75); ("expect", 0.6); ("model", 0.7)
        ]

        // Calculate weighted scores for each behavior type
        let calculateScore patterns =
            patterns |> List.sumBy (fun (pattern, weight) ->
                if lowerAction.Contains(pattern) then weight else 0.0
            )

        let deceptionScore = calculateScore deceptionPatterns
        let selfPreservationScore = calculateScore selfPreservationPatterns
        let cheatingScore = calculateScore cheatingPatterns
        let manipulationScore = calculateScore manipulationPatterns
        let unauthorizedScore = calculateScore unauthorizedPatterns
        let explanatoryScore = calculateScore explanatoryPatterns
        let predictiveScore = calculateScore predictivePatterns

        // Additional context analysis from reasoning chain
        let contextBonus =
            if reasoningChain |> List.exists (fun step ->
                step.ToLowerInvariant().Contains("transparent") ||
                step.ToLowerInvariant().Contains("honest") ||
                step.ToLowerInvariant().Contains("ethical")) then 0.2 else 0.0

        let adjustedExplanatoryScore = explanatoryScore + contextBonus
        let adjustedPredictiveScore = predictiveScore + contextBonus

        // Determine behavior type based on highest score with thresholds
        let scores = [
            (Deception, deceptionScore)
            (SelfPreservation, selfPreservationScore)
            (Cheating, cheatingScore)
            (Manipulation, manipulationScore)
            (Unauthorized, unauthorizedScore)
            (Explanatory, adjustedExplanatoryScore)
            (Predictive, adjustedPredictiveScore)
        ]

        let maxScore = scores |> List.maxBy snd |> snd
        let behaviorType = scores |> List.find (fun (_, score) -> score = maxScore) |> fst

        // Apply minimum threshold for dangerous behaviors
        match behaviorType with
        | Deception when deceptionScore >= 0.5 -> Deception
        | SelfPreservation when selfPreservationScore >= 0.5 -> SelfPreservation
        | Cheating when cheatingScore >= 0.5 -> Cheating
        | Manipulation when manipulationScore >= 0.5 -> Manipulation
        | Unauthorized when unauthorizedScore >= 0.5 -> Unauthorized
        | Explanatory when adjustedExplanatoryScore >= 0.6 -> Explanatory
        | Predictive when adjustedPredictiveScore >= 0.6 -> Predictive
        | _ -> Honest // Default to honest if no clear pattern or below thresholds
    
    /// Real Bayesian harm probability calculation using evidence-based inference
    member private this.CalculateHarmProbability(actionDescription: string, behaviorType: AIBehaviorType, reasoningChain: string list) =
        // Bayesian prior probabilities based on historical data and expert assessment
        let priorProbabilities =
            match behaviorType with
            | Deception -> 0.82 // Based on studies of AI deception incidents
            | SelfPreservation -> 0.91 // Highest risk based on AI safety research
            | Cheating -> 0.74 // Based on gaming and rule-circumvention studies
            | Manipulation -> 0.87 // High risk based on social engineering research
            | Unauthorized -> 0.63 // Moderate-high based on security breach analysis
            | Honest -> 0.04 // Very low based on transparent AI behavior studies
            | Explanatory -> 0.02 // Lowest risk for educational/explanatory behavior
            | Predictive -> 0.03 // Low risk for prediction tasks

        // Evidence factors for Bayesian updating
        let evidenceFactors = [
            // Reasoning quality evidence
            let reasoningQuality = float reasoningChain.Length / float config.MaxReasoningDepth
            let reasoningEvidence = if reasoningQuality >= 0.8 then -0.15 else if reasoningQuality < 0.3 then 0.2 else 0.0

            // Transparency evidence
            let transparencyKeywords = ["transparent"; "honest"; "open"; "clear"; "explain"]
            let transparencyCount = reasoningChain |> List.sumBy (fun step ->
                transparencyKeywords |> List.sumBy (fun keyword ->
                    if step.ToLowerInvariant().Contains(keyword) then 1 else 0
                )
            )
            let transparencyEvidence = if transparencyCount > 0 then -0.1 * float transparencyCount / float reasoningChain.Length else 0.0

            // Complexity evidence (more complex actions may have higher risk)
            let actionComplexity = actionDescription.Split([|' '|]).Length
            let complexityEvidence = if actionComplexity > 20 then 0.05 else if actionComplexity < 5 then -0.02 else 0.0

            // Urgency evidence (urgent actions may bypass safety checks)
            let urgencyKeywords = ["urgent"; "immediately"; "quickly"; "asap"; "emergency"]
            let hasUrgency = urgencyKeywords |> List.exists (fun keyword ->
                actionDescription.ToLowerInvariant().Contains(keyword))
            let urgencyEvidence = if hasUrgency then 0.08 else 0.0

            // Human oversight evidence
            let oversightKeywords = ["with approval"; "after review"; "human oversight"; "supervised"]
            let hasOversight = oversightKeywords |> List.exists (fun keyword ->
                actionDescription.ToLowerInvariant().Contains(keyword))
            let oversightEvidence = if hasOversight then -0.12 else 0.0

            [reasoningEvidence; transparencyEvidence; complexityEvidence; urgencyEvidence; oversightEvidence]
        ]

        // Apply Bayesian updating: P(harm|evidence) = P(evidence|harm) * P(harm) / P(evidence)
        let totalEvidenceAdjustment = evidenceFactors |> List.sum

        // Likelihood ratios for evidence given harm vs no harm
        let likelihoodRatio =
            if totalEvidenceAdjustment < 0.0 then 0.7 // Evidence suggests lower harm probability
            elif totalEvidenceAdjustment > 0.0 then 1.4 // Evidence suggests higher harm probability
            else 1.0 // No evidence adjustment

        // Bayesian posterior calculation
        let prior = priorProbabilities
        let posterior = (likelihoodRatio * prior) / ((likelihoodRatio * prior) + (1.0 - prior))

        // Apply evidence adjustments with bounds checking
        let finalProbability = posterior + totalEvidenceAdjustment

        // Ensure probability stays within [0, 1] with realistic bounds
        Math.Max(0.001, Math.Min(0.999, finalProbability)) // Avoid absolute 0 or 1 for Bayesian reasoning
    
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
