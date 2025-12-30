namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService

// Note: Using Async internally and converting to Task at boundaries

/// <summary>
/// Self-Improvement module: Learn from successful task completions.
/// Implements Phase 4 of the TARS Evolution plan.
/// </summary>
module SelfImprovement =

    // =========================================================================
    // Types
    // =========================================================================

    /// Result of a learning episode
    type LearningResult =
        { TaskDescription: string
          Solution: string
          ExtractedBelief: EpistemicBelief option
          WasVerified: bool
          Curriculum: string option
          Timestamp: DateTime }

    /// Learning session state
    type LearningSession =
        { CompletedTasks: (string * string) list // (task, solution) pairs
          ExtractedBeliefs: EpistemicBelief list
          CurriculumSuggestions: string list
          BeliefGraph: BeliefGraph
          StartedAt: DateTime }

    /// Configuration for self-improvement
    type SelfImprovementConfig =
        { MinConfidenceForPrinciple: float
          MaxVariantsForVerification: int
          AutoLearnFromSuccess: bool }

    let defaultConfig =
        { MinConfidenceForPrinciple = 0.7
          MaxVariantsForVerification = 3
          AutoLearnFromSuccess = true }

    // =========================================================================
    // Learning Session Management
    // =========================================================================

    /// Create a new learning session
    let createSession () =
        { CompletedTasks = []
          ExtractedBeliefs = []
          CurriculumSuggestions = []
          BeliefGraph = BeliefGraph.empty ()
          StartedAt = DateTime.UtcNow }

    /// Create a session from an existing EpistemicBelief graph
    let createSessionFromGraph (graph: BeliefGraph) =
        { CompletedTasks = []
          ExtractedBeliefs = graph.Beliefs |> Map.values |> Seq.toList
          CurriculumSuggestions = []
          BeliefGraph = graph
          StartedAt = DateTime.UtcNow }

    // =========================================================================
    // Core Learning Functions
    // =========================================================================

    /// Learn from a completed task using the epistemic governor
    let learnFromTaskAsync
        (governor: IEpistemicGovernor)
        (config: SelfImprovementConfig)
        (taskDesc: string)
        (solution: string)
        (session: LearningSession)
        : Async<LearningResult * LearningSession> =
        async {
            // 1. Extract a principle from the task/solution
            let! belief = governor.ExtractPrinciple(taskDesc, solution) |> Async.AwaitTask

            // 2. Generate variants and verify generalization
            let! variants =
                governor.GenerateVariants(taskDesc, config.MaxVariantsForVerification)
                |> Async.AwaitTask

            let! verification =
                if variants.IsEmpty then
                    async {
                        return
                            { IsVerified = true
                              Score = 0.5
                              Feedback = "No variants generated"
                              FailedVariants = [] }
                    }
                else
                    governor.VerifyGeneralization(taskDesc, solution, variants) |> Async.AwaitTask

            // 3. Update EpistemicBelief status based on verification
            let finalBelief =
                if
                    verification.IsVerified
                    && verification.Score >= config.MinConfidenceForPrinciple
                then
                    { belief with
                        Status = EpistemicStatus.VerifiedFact
                        Confidence = verification.Score }
                elif verification.IsVerified then
                    { belief with
                        Status = EpistemicStatus.Hypothesis
                        Confidence = verification.Score }
                else
                    { belief with
                        Status = EpistemicStatus.Fallacy
                        Confidence = verification.Score }

            // 4. Update the EpistemicBelief graph
            let updatedGraph = session.BeliefGraph |> BeliefGraph.addBelief finalBelief

            // 5. Get curriculum suggestion
            let completedTasksList = (taskDesc :: (session.CompletedTasks |> List.map fst))

            let activeBeliefsList =
                (session.ExtractedBeliefs @ [ finalBelief ])
                |> List.filter (fun b -> b.Status <> EpistemicStatus.Fallacy)
                |> List.map (fun b -> b.Statement)

            let! curriculum =
                governor.SuggestCurriculum(completedTasksList, activeBeliefsList, false)
                |> Async.AwaitTask

            // 6. Build the result
            let result =
                { TaskDescription = taskDesc
                  Solution = solution
                  ExtractedBelief = Some finalBelief
                  WasVerified = verification.IsVerified
                  Curriculum =
                    if String.IsNullOrWhiteSpace(curriculum) then
                        None
                    else
                        Some curriculum
                  Timestamp = DateTime.UtcNow }

            let updatedSession =
                { session with
                    CompletedTasks = (taskDesc, solution) :: session.CompletedTasks
                    ExtractedBeliefs = finalBelief :: session.ExtractedBeliefs
                    CurriculumSuggestions =
                        match result.Curriculum with
                        | Some c -> c :: session.CurriculumSuggestions
                        | None -> session.CurriculumSuggestions
                    BeliefGraph = updatedGraph }

            return (result, updatedSession)
        }

    /// Learn from a completed task (Task wrapper)
    let learnFromTask governor config taskDesc solution session =
        learnFromTaskAsync governor config taskDesc solution session
        |> Async.StartAsTask

    /// Batch learn from multiple completed tasks
    let learnFromTasksAsync
        (governor: IEpistemicGovernor)
        (config: SelfImprovementConfig)
        (tasks: (string * string) list)
        (session: LearningSession)
        : Async<LearningResult list * LearningSession> =
        async {
            let mutable currentSession = session
            let results = ResizeArray<LearningResult>()

            for (t, s) in tasks do
                let! (result, newSession) = learnFromTaskAsync governor config t s currentSession
                results.Add(result)
                currentSession <- newSession

            return (results |> Seq.toList, currentSession)
        }

    /// Batch learn (Task wrapper)
    let learnFromTasks governor config tasks session =
        learnFromTasksAsync governor config tasks session |> Async.StartAsTask

    // =========================================================================
    // EpistemicBelief Analysis
    // =========================================================================

    /// Get principles that have been verified across multiple tasks
    let getVerifiedPrinciples (session: LearningSession) =
        session.ExtractedBeliefs
        |> List.filter (fun b ->
            b.Status = EpistemicStatus.VerifiedFact
            || b.Status = EpistemicStatus.UniversalPrinciple)
        |> List.sortByDescending (fun b -> b.Confidence)

    /// Get hypotheses that need more evidence
    let getHypotheses (session: LearningSession) =
        session.ExtractedBeliefs
        |> List.filter (fun b -> b.Status = EpistemicStatus.Hypothesis)
        |> List.sortByDescending (fun b -> b.Confidence)

    /// Get gaps in knowledge (areas where beliefs were refuted)
    let getKnowledgeGaps (session: LearningSession) =
        session.ExtractedBeliefs
        |> List.filter (fun b -> b.Status = EpistemicStatus.Fallacy)
        |> List.map (fun b -> b.Context)
        |> List.distinct

    /// Find contradictions in learned beliefs
    let findContradictions (session: LearningSession) =
        BeliefGraph.findContradictions session.BeliefGraph

    // =========================================================================
    // Curriculum Generation
    // =========================================================================

    /// Get the next suggested learning task
    let getNextCurriculumTask (session: LearningSession) =
        session.CurriculumSuggestions |> List.tryHead

    /// Get all curriculum suggestions
    let getAllCurriculumSuggestions (session: LearningSession) = session.CurriculumSuggestions

    // =========================================================================
    // Session Statistics
    // =========================================================================

    /// Get statistics about the learning session
    let getSessionStats (session: LearningSession) =
        let statusCounts = BeliefGraph.countByStatus session.BeliefGraph

        {| TasksCompleted = session.CompletedTasks.Length
           BeliefsExtracted = session.ExtractedBeliefs.Length
           VerifiedPrinciples =
            statusCounts
            |> Map.tryFind EpistemicStatus.VerifiedFact
            |> Option.defaultValue 0
           Hypotheses = statusCounts |> Map.tryFind EpistemicStatus.Hypothesis |> Option.defaultValue 0
           Refuted = statusCounts |> Map.tryFind EpistemicStatus.Fallacy |> Option.defaultValue 0
           CurriculumSuggestions = session.CurriculumSuggestions.Length
           SessionDuration = DateTime.UtcNow - session.StartedAt |}

    // =========================================================================
    // Integration with AgentWorkflow
    // =========================================================================

    /// Wrap an agent workflow to automatically learn from successful completions
    let withLearning
        (governor: IEpistemicGovernor)
        (config: SelfImprovementConfig)
        (taskDescription: string)
        (workflow: AgentWorkflow<string>)
        (session: LearningSession ref)
        : AgentWorkflow<string * LearningResult option> =
        fun ctx ->
            async {
                let! result = workflow ctx

                match result with
                | Success solution when config.AutoLearnFromSuccess ->
                    ctx.Logger(sprintf "[SelfImprovement] Learning from successful task: %s" taskDescription)

                    let! (learningResult, newSession) =
                        learnFromTask governor config taskDescription solution !session
                        |> Async.AwaitTask

                    session := newSession

                    ctx.Logger(
                        sprintf
                            "[SelfImprovement] Extracted EpistemicBelief: %s (confidence: %.2f)"
                            (learningResult.ExtractedBelief
                             |> Option.map (fun b -> b.Statement)
                             |> Option.defaultValue "none")
                            (learningResult.ExtractedBelief
                             |> Option.map (fun b -> b.Confidence)
                             |> Option.defaultValue 0.0)
                    )

                    return Success(solution, Some learningResult)

                | Success solution -> return Success(solution, None)

                | PartialSuccess(solution, warnings) when config.AutoLearnFromSuccess ->
                    ctx.Logger "[SelfImprovement] Partial success - still learning"

                    let! (learningResult, newSession) =
                        learnFromTask governor config taskDescription solution !session
                        |> Async.AwaitTask

                    session := newSession
                    return PartialSuccess((solution, Some learningResult), warnings)

                | PartialSuccess(solution, warnings) -> return PartialSuccess((solution, None), warnings)

                | Failure errors ->
                    ctx.Logger "[SelfImprovement] Task failed - no learning"
                    return Failure errors
            }

    // =========================================================================
    // Persistence Helpers
    // =========================================================================

    /// Serialize session to JSON for persistence
    let serializeSession (session: LearningSession) =
        // Simple serialization - in production, use proper JSON
        sprintf
            """{"tasksCompleted":%d,"beliefsExtracted":%d,"startedAt":"%s"}"""
            session.CompletedTasks.Length
            session.ExtractedBeliefs.Length
            (session.StartedAt.ToString("o"))

    /// Get a summary of the session for display
    let getSessionSummary (session: LearningSession) =
        let stats = getSessionStats session

        sprintf
            """Learning Session Summary
========================
Tasks Completed: %d
Beliefs Extracted: %d
  - Verified Principles: %d
  - Hypotheses: %d
  - Refuted: %d
Curriculum Suggestions: %d
Session Duration: %s"""
            stats.TasksCompleted
            stats.BeliefsExtracted
            stats.VerifiedPrinciples
            stats.Hypotheses
            stats.Refuted
            stats.CurriculumSuggestions
            (stats.SessionDuration.ToString(@"hh\:mm\:ss"))
