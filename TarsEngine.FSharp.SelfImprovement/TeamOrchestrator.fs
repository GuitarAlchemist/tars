namespace TarsEngine.FSharp.SelfImprovement

open System
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.SelfImprovement.AutonomousSpecHarness
open TarsEngine.FSharp.SelfImprovement.AutonomousNextStepPlanner
open TarsEngine.FSharp.SelfImprovement.ValidatorCoordination
open TarsEngine.FSharp.Core.Services.CrossAgentValidation

module RT = TarsEngine.FSharp.Core.Services.ReasoningTrace

/// Coordinates multi-agent autonomous development cycles.
module TeamOrchestrator =

    type SquadRole =
        | GoalSynthesizer
        | PlannerLead
        | Implementer
        | QaLead
        | SafetyLead
        | MetaCritic

    type SquadVerdict =
        { Role: SquadRole
          Verdict: string
          Confidence: float option
          Notes: string option
          Timestamp: DateTime }

    type TeamCycleSummary =
        { PlannedFeatureId: string option
          PlannedTaskId: string option
          CycleStatus: string
          PlannerScore: float option
          PlannerPriorityWeight: float option
          SquadVerdicts: SquadVerdict list
          Duration: TimeSpan option
          ValidatorFindingCount: int option
          ValidatorDisagreementRatio: float option
          ValidatorDisagreementCount: int option }

    type TeamCycleResult =
        { Summary: TeamCycleSummary
          PlannerRecommendation: PlannerRecommendation option
          EnqueuedRecommendations: PlannerRecommendation list
          Iteration: SpecDrivenIterationResult option }

    type TeamCycleDependencies =
        { PlanNext: ILoggerFactory -> int option -> int option -> Async<PlannerRecommendation list * PlannerRecommendation list>
          RunIteration: ILoggerFactory -> string option -> SpecKitWorkspace.SpecKitHarnessOptions option -> ExecutionHarness.ICommandExecutor option -> Async<SpecDrivenIterationResult option> }

    type TeamCycleOptions =
        { BaseDirectory: string option
          HarnessOptions: SpecKitWorkspace.SpecKitHarnessOptions option
          PlannerTopCandidates: int option
          PlannerRecentMemory: int option
          Executor: ExecutionHarness.ICommandExecutor option }
        static member Default =
            { BaseDirectory = None
              HarnessOptions = None
              PlannerTopCandidates = None
              PlannerRecentMemory = None
              Executor = None }

    let private normalizeScore score =
        1.0 / (1.0 + Math.Exp(-score))

    let private verdict role verdict confidence notes =
        { Role = role
          Verdict = verdict
          Confidence = confidence
          Notes = notes
          Timestamp = DateTime.UtcNow }

    let private buildPlannerVerdict (recommendation: PlannerRecommendation) =
        let selection = recommendation.Selection
        let plannedNotes =
            [ $"Target feature {selection.Feature.Id}"
              $"Task description: {selection.Task.Description}" ]
            @ (if recommendation.Rationale.IsEmpty then [] else recommendation.Rationale)
            |> String.concat "; "

        verdict
            PlannerLead
            "recommended"
            (Some (normalizeScore recommendation.Score))
            (Some plannedNotes)

    let private buildGoalSynthVerdict (enqueued: PlannerRecommendation list) =
        if enqueued.IsEmpty then
            verdict GoalSynthesizer "idle" (Some 0.2) (Some "No new roadmap items enqueued.")
        else
            let formatEntry (item: PlannerRecommendation) =
                let taskId = item.Selection.Task.TaskId |> Option.defaultValue "<unknown>"
                $"{item.Selection.Feature.Id}:{taskId}"
            let notes =
                enqueued
                |> List.map formatEntry
                |> String.concat ", "
            verdict GoalSynthesizer "enqueued" (Some 0.75) (Some $"Queued items -> {notes}")

    let private buildImplementerVerdict (iteration: SpecDrivenIterationResult option) =
        match iteration with
        | None ->
            verdict Implementer "skipped" (Some 0.1) (Some "No iteration executed.")
        | Some iter ->
            let success =
                iter.HarnessReport
                |> Option.exists (fun report ->
                    match report.Outcome with
                    | ExecutionHarness.HarnessOutcome.AllPassed _ -> true
                    | _ -> false)

            let verdictText = if success then "completed" else "blocked"
            let confidence =
                iter.HarnessReport
                |> Option.map (fun report ->
                    let total = report.Commands.Length
                    let succeeded =
                        report.Commands
                        |> List.filter (fun cmd -> cmd.ExitCode = 0)
                        |> List.length
                    if total = 0 then 0.5 else float succeeded / float total)

            let notes =
                iter.HarnessReport
                |> Option.bind (fun report ->
                    match report.Outcome with
                    | ExecutionHarness.HarnessOutcome.AllPassed _ -> Some "All validation commands succeeded."
                    | ExecutionHarness.HarnessOutcome.Failed (_, reason) -> Some($"Failure: {reason}")
                )

            verdict Implementer verdictText confidence notes

    let private buildQaVerdict (iteration: SpecDrivenIterationResult option) =
        match iteration with
        | None -> verdict QaLead "idle" (Some 0.2) (Some "QA not executed.")
        | Some iter ->
            match iter.HarnessReport with
            | None -> verdict QaLead "no-data" (Some 0.3) (Some "Harness report missing.")
            | Some report ->
                let status =
                    match report.Outcome with
                    | ExecutionHarness.HarnessOutcome.AllPassed _ -> "passed"
                    | ExecutionHarness.HarnessOutcome.Failed _ -> "failed"
                let notes =
                    report.Commands
                    |> List.map (fun cmd -> $"{cmd.Command.Name}:{cmd.ExitCode}")
                    |> String.concat "; "
                verdict QaLead status (Some 0.8) (Some notes)

    let private buildSafetyVerdict (iteration: SpecDrivenIterationResult option) =
        match iteration with
        | None -> verdict SafetyLead "idle" (Some 0.2) (Some "No iteration executed.")
        | Some iter ->
            match iter.Consensus with
            | Some (ConsensusPassed agents) ->
                let notes =
                    agents
                    |> List.map (fun agent -> $"{agent.Role}:{agent.Outcome}")
                    |> String.concat "; "
                verdict SafetyLead "approved" (Some 0.9) (Some notes)
            | Some (ConsensusFailed (_, reason)) ->
                verdict SafetyLead "rejected" (Some 0.1) (Some reason)
            | Some (ConsensusNeedsReview (_, reason)) ->
                verdict SafetyLead "needs-review" (Some 0.4) (Some reason)
            | None ->
                verdict SafetyLead "missing" (Some 0.3) (Some "Consensus not produced.")

    let private buildMetaCriticVerdict (iteration: SpecDrivenIterationResult option) =
        match iteration with
        | None -> verdict MetaCritic "idle" (Some 0.2) (Some "Meta-critique not executed.")
        | Some iter ->
            match iter.CriticVerdict with
            | None -> verdict MetaCritic "unknown" (Some 0.3) (Some "Critic verdict unavailable.")
            | Some RT.CriticVerdict.Accept ->
                verdict MetaCritic "accept" (Some 0.9) (Some "Meta critic approved iteration.")
            | Some (RT.CriticVerdict.NeedsReview reason) ->
                verdict MetaCritic "needs-review" (Some 0.4) (Some reason)
            | Some (RT.CriticVerdict.Reject reason) ->
                verdict MetaCritic "reject" (Some 0.15) (Some reason)

    let private computeDuration (iteration: SpecDrivenIterationResult option) =
        iteration
        |> Option.bind (fun iter ->
            iter.HarnessReport
            |> Option.map (fun report -> report.CompletedAt - report.StartedAt))

    let private computeValidatorStats (iteration: SpecDrivenIterationResult option) =
        iteration
        |> Option.bind (fun iter -> iter.ValidatorSnapshot)
        |> Option.map (fun snapshot ->
            let findingCount = snapshot.Findings.Length
            let disagreementCount = snapshot.Disagreements.Length
            let ratio =
                if findingCount = 0 then 0.0
                else float disagreementCount / float findingCount
            findingCount, disagreementCount, ratio)

    let runCycleAsync
        (loggerFactory: ILoggerFactory)
        (dependencies: TeamCycleDependencies)
        (options: TeamCycleOptions) =
        async {
            let logger = loggerFactory.CreateLogger("TeamOrchestrator")

            logger.LogInformation("Starting autonomous team cycle.")

            let! plannerRecommendations, enqueued =
                dependencies.PlanNext loggerFactory options.PlannerTopCandidates options.PlannerRecentMemory

            let plannerRecommendation = plannerRecommendations |> List.tryHead

            let plannerVerdict =
                plannerRecommendation
                |> Option.map buildPlannerVerdict
                |> Option.toList

            let goalSynthVerdict = buildGoalSynthVerdict enqueued

            let! iteration =
                dependencies.RunIteration loggerFactory options.BaseDirectory options.HarnessOptions options.Executor

            let implementerVerdict = buildImplementerVerdict iteration
            let qaVerdict = buildQaVerdict iteration
            let safetyVerdict = buildSafetyVerdict iteration
            let metaVerdict = buildMetaCriticVerdict iteration

            let squadVerdicts =
                [ goalSynthVerdict ]
                @ plannerVerdict
                @ [ implementerVerdict
                    qaVerdict
                    safetyVerdict
                    metaVerdict ]

            let duration = computeDuration iteration
            let validatorStats = computeValidatorStats iteration

            let findingCount, disagreementCount, ratio =
                match validatorStats with
                | Some (findings, disagreements, ratio) -> Some findings, Some disagreements, Some ratio
                | None -> None, None, None

            let cycleStatus =
                match iteration with
                | None -> "no-work"
                | Some iter ->
                    match iter.HarnessReport with
                    | Some { Outcome = ExecutionHarness.HarnessOutcome.AllPassed _ } -> "completed"
                    | Some { Outcome = ExecutionHarness.HarnessOutcome.Failed _ } -> "failed"
                    | None -> "incomplete"

            let summary =
                { PlannedFeatureId = plannerRecommendation |> Option.map (fun recommendation -> recommendation.Selection.Feature.Id)
                  PlannedTaskId = plannerRecommendation |> Option.bind (fun recommendation -> recommendation.Selection.Task.TaskId)
                  CycleStatus = cycleStatus
                  PlannerScore = plannerRecommendation |> Option.map (fun recommendation -> recommendation.Score)
                  PlannerPriorityWeight = plannerRecommendation |> Option.map (fun recommendation -> recommendation.PriorityWeight)
                  SquadVerdicts = squadVerdicts
                  Duration = duration
                  ValidatorFindingCount = findingCount
                  ValidatorDisagreementRatio = ratio
                  ValidatorDisagreementCount = disagreementCount }

            logger.LogInformation("Completed team cycle status={Status} feature={Feature} task={Task}", summary.CycleStatus, summary.PlannedFeatureId |> Option.defaultValue "<none>", summary.PlannedTaskId |> Option.defaultValue "<none>")

            return
                { Summary = summary
                  PlannerRecommendation = plannerRecommendation
                  EnqueuedRecommendations = enqueued
                  Iteration = iteration }
        }



