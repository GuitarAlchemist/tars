namespace TarsEngine

open System
open System.Collections.Generic

/// Seldon Teaching Engine - Knowledge transfer and assessment for the GuitarAlchemist ecosystem
module SeldonTeachingEngine =

    /// Knowledge layer classification
    type KnowledgeLayer =
        | Governance      // Universal knowledge for all agents
        | Experiential    // Ecosystem-wide learnings from operations
        | Domain          // Repo-specific knowledge (ix, tars, ga)

    /// Knowledge delivery mode
    type DeliveryMode =
        | Narrative    // For human learners
        | Structured   // For agent learners

    /// Tetravalent truth values for belief states
    type TruthValue =
        | T  // True - verified with evidence
        | F  // False - refuted with evidence
        | U  // Unknown - insufficient evidence
        | C  // Contradictory - conflicting evidence

    /// Evidence supporting or contradicting a belief
    type EvidenceItem = {
        Source: string
        Claim: string
        Timestamp: DateTime
        Reliability: float  // 0.0 to 1.0
    }

    /// A tetravalent belief state
    type BeliefState = {
        Proposition: string
        TruthValue: TruthValue
        Confidence: float  // 0.0 to 1.0
        SupportingEvidence: EvidenceItem list
        ContradictingEvidence: EvidenceItem list
        LastUpdated: DateTime
        EvaluatedBy: string
    }

    /// Assessment result from a teaching attempt
    type AssessmentResult = {
        BeliefStateBefore: BeliefState
        BeliefStateAfter: BeliefState
        BehavioralVerification: string  // "pending" | "confirmed" | "failed" | "not_applicable"
        Attempts: int
        Outcome: string  // "learned" | "in_progress" | "escalated"
    }

    /// Knowledge state for tracking a teaching event
    type KnowledgeState = {
        Id: string
        Concept: string
        Layer: KnowledgeLayer
        DomainContext: string option  // "ix" | "tars" | "ga"
        SourceArtifact: string option
        SourcePdcaCycle: string option
        SourceFiveWhys: string option
        SourceReconnaissance: string option
        LearnerType: string  // "human" | "agent"
        LearnerIdentifier: string
        DeliveryMode: DeliveryMode
        Assessment: AssessmentResult
        TaughtBy: string
        CreatedAt: DateTime
        LastUpdated: DateTime
    }

    /// Result of assessComprehension
    type ComprehensionAssessment = {
        Learned: bool
        TruthValueChange: TruthValue * TruthValue  // before, after
        ConfidenceImprovement: float  // absolute change
        RequiresRetry: bool
        NextAction: string  // "proceed" | "retry" | "escalate"
    }

    /// Result of adaptDelivery
    type AdaptedDelivery = {
        AdjustedContent: string
        NewDeliveryApproach: string
        EvidenceToPresent: EvidenceItem list
        EstimatedImpact: float  // confidence improvement
    }

    /// Result of executeWhyChain
    type WhyChainResult = {
        Level: int  // 1-5
        Question: string
        Answer: string
        BeliefStateAtLevel: BeliefState
        RootCauseFound: bool
    }

    /// Helper to create initial belief state (typically Unknown)
    let createInitialBeliefState proposition evaluatedBy =
        {
            Proposition = proposition
            TruthValue = U
            Confidence = 0.0
            SupportingEvidence = []
            ContradictingEvidence = []
            LastUpdated = DateTime.UtcNow
            EvaluatedBy = evaluatedBy
        }

    /// Helper to update belief state after teaching
    let updateBeliefState (oldState: BeliefState) truthValue confidence evidence =
        let newEvidence =
            if List.isEmpty evidence then oldState.SupportingEvidence
            else oldState.SupportingEvidence @ evidence
        {
            oldState with
                TruthValue = truthValue
                Confidence = confidence
                SupportingEvidence = newEvidence
                LastUpdated = DateTime.UtcNow
        }

    /// Assess whether the learner has comprehended the concept
    /// Returns ComprehensionAssessment indicating if learning was successful
    let assessComprehension (conceptBefore: BeliefState) (conceptAfter: BeliefState) : ComprehensionAssessment =
        let truthChanged = conceptBefore.TruthValue <> conceptAfter.TruthValue
        let confidenceImprovement = conceptAfter.Confidence - conceptBefore.Confidence

        // Learning succeeds if: truth_value becomes T with confidence >= 0.7, or U->T transition
        let learned =
            match conceptAfter.TruthValue, conceptAfter.Confidence with
            | T, conf when conf >= 0.7 -> true
            | _ -> false

        let nextAction =
            match learned with
            | true -> "proceed"  // Move to behavioral verification
            | false when confidenceImprovement > 0.1 -> "retry"  // Progress made, try again
            | false -> "escalate"  // No progress, escalate to Demerzel

        {
            Learned = learned
            TruthValueChange = (conceptBefore.TruthValue, conceptAfter.TruthValue)
            ConfidenceImprovement = confidenceImprovement
            RequiresRetry = nextAction = "retry"
            NextAction = nextAction
        }

    /// Adapt delivery approach based on assessment failure
    /// Generates a modified teaching strategy with new evidence
    let adaptDelivery (assessment: ComprehensionAssessment) (learnerType: string) : AdaptedDelivery =
        let approach =
            match learnerType with
            | "human" ->
                // For humans: add analogies, concrete examples, visual metaphors
                if assessment.ConfidenceImprovement > 0.0 then
                    "Add concrete examples and bridge to familiar concepts"
                else
                    "Use narrative approach with real-world analogies"
            | "agent" ->
                // For agents: provide policy references, belief state tuples, formal citations
                "Provide formal definitions, policy references, and belief state tuples"
            | _ -> "Adapt delivery based on learner feedback"

        let newEvidence = [
            {
                Source = "teaching-adaptation"
                Claim = sprintf "Adaptation strategy: %s" approach
                Timestamp = DateTime.UtcNow
                Reliability = 0.8
            }
        ]

        {
            AdjustedContent = sprintf "Retrying concept with adjusted approach: %s" approach
            NewDeliveryApproach = approach
            EvidenceToPresent = newEvidence
            EstimatedImpact = 0.15  // Expected confidence improvement
        }

    /// Execute the 5 Whys root cause analysis as a recursive function
    /// Returns a sequence of WhyChainResult from level 1 to 5 (or until root cause found)
    let rec executeWhyChain (question: string) (level: int) (baseBeliefState: BeliefState) : WhyChainResult list =
        if level > 5 then
            // Stop at level 5 or when root cause conceptually found
            []
        else
            // Simulate answering the "why" at this level
            let answer = sprintf "Root cause analysis at level %d: %s" level question

            // Update belief state progressively as we dig deeper
            let beliefAtLevel =
                updateBeliefState baseBeliefState T (0.5 + float level * 0.1) []

            let rootCauseFound = level = 5  // Simplified: assume we've reached root at level 5

            let currentResult = {
                Level = level
                Question = question
                Answer = answer
                BeliefStateAtLevel = beliefAtLevel
                RootCauseFound = rootCauseFound
            }

            // Continue to next level if root cause not yet found
            let nextResults =
                if not rootCauseFound && level < 5 then
                    let nextQuestion = sprintf "Why is this the case? (Level %d)" (level + 1)
                    executeWhyChain nextQuestion (level + 1) beliefAtLevel
                else
                    []

            currentResult :: nextResults

    /// Create a knowledge state record for tracking a teaching event
    let createKnowledgeState
        (id: string)
        (concept: string)
        (layer: KnowledgeLayer)
        (learnerType: string)
        (learnerIdentifier: string)
        (deliveryMode: DeliveryMode)
        (assessment: AssessmentResult) =
        {
            Id = id
            Concept = concept
            Layer = layer
            DomainContext = None
            SourceArtifact = None
            SourcePdcaCycle = None
            SourceFiveWhys = None
            SourceReconnaissance = None
            LearnerType = learnerType
            LearnerIdentifier = learnerIdentifier
            DeliveryMode = deliveryMode
            Assessment = assessment
            TaughtBy = "seldon"
            CreatedAt = DateTime.UtcNow
            LastUpdated = DateTime.UtcNow
        }

    /// Execute a teaching iteration for a single concept
    /// Returns the assessment and indicates if escalation is needed
    let executeTeachingIteration
        (concept: string)
        (layer: KnowledgeLayer)
        (learnerType: string)
        (learnerIdentifier: string)
        (deliveryMode: DeliveryMode)
        (currentAttempt: int) =

        // Stage 1: Assess belief state before teaching
        let beliefBefore = createInitialBeliefState concept "seldon"

        // Stage 2: Deliver knowledge (simulated)
        let beliefAfter =
            // Simulate successful teaching with increasing confidence
            let confidenceGain = 0.3 + float currentAttempt * 0.15
            updateBeliefState beliefBefore T (Math.Min(0.95, confidenceGain)) []

        // Stage 3: Assess comprehension
        let comprehension = assessComprehension beliefBefore beliefAfter

        // Stage 4: Determine behavioral verification status
        let behavioralVerification =
            match comprehension.Learned with
            | true -> "pending"  // Will verify in practice
            | false -> "failed"

        // Stage 5: Create assessment result
        let assessment = {
            BeliefStateBefore = beliefBefore
            BeliefStateAfter = beliefAfter
            BehavioralVerification = behavioralVerification
            Attempts = currentAttempt
            Outcome =
                match currentAttempt with
                | attempt when attempt >= 3 -> "escalated"
                | attempt when comprehension.Learned -> "learned"
                | _ -> "in_progress"
        }

        (assessment, comprehension)

    /// Orchestrate the full teaching workflow for a concept
    /// Handles up to 3 attempts, then escalates if necessary
    let teachConcept
        (conceptId: string)
        (concept: string)
        (layer: KnowledgeLayer)
        (learnerType: string)
        (learnerIdentifier: string)
        (deliveryMode: DeliveryMode) =

        let rec attemptTeaching (attempt: int) (lastAssessment: AssessmentResult option) =
            if attempt > 3 then
                // Escalate after max attempts
                match lastAssessment with
                | Some ass ->
                    {
                        ass with
                            Outcome = "escalated"
                            Attempts = attempt - 1
                    }
                | None ->
                    let initialBelief = createInitialBeliefState concept "seldon"
                    {
                        BeliefStateBefore = initialBelief
                        BeliefStateAfter = initialBelief
                        BehavioralVerification = "failed"
                        Attempts = attempt - 1
                        Outcome = "escalated"
                    }
            else
                let (assessment, comprehension) = executeTeachingIteration concept layer learnerType learnerIdentifier deliveryMode attempt

                match comprehension.NextAction with
                | "proceed" ->
                    // Learning successful, return result
                    { assessment with Outcome = "learned" }
                | "retry" ->
                    // Adapt and retry
                    let _ = adaptDelivery comprehension learnerType
                    attemptTeaching (attempt + 1) (Some assessment)
                | _ ->
                    // Escalate
                    { assessment with Outcome = "escalated" }

        let finalAssessment = attemptTeaching 1 None

        createKnowledgeState
            conceptId
            concept
            layer
            learnerType
            learnerIdentifier
            deliveryMode
            finalAssessment
