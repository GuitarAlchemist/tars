namespace TarsEngine.FSharp.SelfImprovement

open System
open System.Collections.Concurrent
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Services.CrossAgentValidation

/// Facilities for coordinating multiple validator agents and detecting conflicts between their verdicts.
module ValidatorCoordination =

    /// Logical target a validator is assessing (spec iteration, benchmark suite, etc.).
    type ValidatorTarget =
        { SpecId: string
          IterationId: Guid option
          Topic: string option }

    /// Primary finding emitted by a validator agent.
    type ValidatorFinding =
        { FindingId: Guid
          AgentId: string
          Role: AgentRole
          Outcome: ValidationOutcome
          Confidence: float option
          Notes: string option
          Target: ValidatorTarget
          RecordedAt: DateTime }

    /// Free-form comment or follow-up that references an existing finding.
    type ValidatorComment =
        { CommentId: Guid
          FindingId: Guid
          AuthorId: string
          AuthorRole: AgentRole
          Message: string
          RecordedAt: DateTime }

    /// Represents a disagreement between validators on the same target.
    type ValidatorDisagreement =
        { Target: ValidatorTarget
          Roles: AgentRole list
          Outcomes: (AgentRole * ValidationOutcome) list
          ConfidenceSpread: float option
          Trigger: string
          LoggedAt: DateTime }

    /// Snapshot of the current coordination state.
    type CoordinationSnapshot =
        { Findings: ValidatorFinding list
          Comments: ValidatorComment list
          Disagreements: ValidatorDisagreement list }

    let private normaliseKey (target: ValidatorTarget) =
        let specKey =
            if String.IsNullOrWhiteSpace(target.SpecId) then
                target.SpecId
            else
                target.SpecId.Trim().ToUpperInvariant()

        let topicKey =
            target.Topic
            |> Option.map (fun topic -> topic.Trim().ToUpperInvariant())

        specKey, target.IterationId, topicKey

    type CoordinationBus(?logger: ILogger) =
        let findings = ConcurrentBag<ValidatorFinding>()
        let comments = ConcurrentBag<ValidatorComment>()
        let logger = defaultArg logger null

        member _.PublishFinding(finding: ValidatorFinding) =
            findings.Add(finding)
            if not (isNull logger) then
                logger.LogInformation(
                    "Validator finding recorded. agent={AgentId} role={Role} outcome={Outcome} spec={SpecId}",
                    finding.AgentId,
                    finding.Role,
                    finding.Outcome,
                    finding.Target.SpecId)

        member _.PublishComment(comment: ValidatorComment) =
            comments.Add(comment)
            if not (isNull logger) then
                logger.LogDebug(
                    "Validator comment recorded. agent={AgentId} finding={FindingId}",
                    comment.AuthorId,
                    comment.FindingId)

        member private _.LatestByRole(targetFindings: ValidatorFinding list) =
            targetFindings
            |> List.groupBy (fun finding -> finding.Role)
            |> List.map (fun (role, grouped) ->
                let latest =
                    grouped
                    |> List.sortByDescending (fun item -> item.RecordedAt)
                    |> List.head
                role, latest)

        member private this.ComputeDisagreements(findingsSnapshot: ValidatorFinding list) =
            findingsSnapshot
            |> List.groupBy (fun finding -> normaliseKey finding.Target)
            |> List.choose (fun (_, grouped) ->
                let canonicalTarget =
                    grouped
                    |> List.maxBy (fun finding -> finding.RecordedAt)
                    |> fun finding -> finding.Target

                let latestByRole = this.LatestByRole grouped
                let outcomes = latestByRole |> List.map (fun (role, finding) -> role, finding.Outcome)

                let hasOutcome predicate =
                    outcomes |> List.exists (fun (_, outcome) -> predicate outcome)

                let hasPass = hasOutcome ((=) ValidationOutcome.Pass)
                let hasFail = hasOutcome ((=) ValidationOutcome.Fail)
                let hasNeedsReview = hasOutcome ((=) ValidationOutcome.NeedsReview)

                let disagreementDetected =
                    (hasPass && hasFail)
                    || (hasFail && hasNeedsReview)
                    || (hasPass && hasNeedsReview)

                if disagreementDetected then
                    let confidences =
                        latestByRole
                        |> List.choose (fun (_, finding) -> finding.Confidence)

                    let spread =
                        match confidences with
                        | [] -> None
                        | values ->
                            let maxValue = values |> List.max
                            let minValue = values |> List.min
                            Some (maxValue - minValue)

                    let trigger =
                        if hasPass && hasFail then
                            "conflicting_pass_fail"
                        elif hasFail && hasNeedsReview then
                            "needs_review_vs_fail"
                        elif hasPass && hasNeedsReview then
                            "needs_review_vs_pass"
                        else
                            "divergent_outcomes"

                    Some
                        { Target = canonicalTarget
                          Roles = latestByRole |> List.map fst
                          Outcomes = outcomes
                          ConfidenceSpread = spread
                          Trigger = trigger
                          LoggedAt = DateTime.UtcNow }
                else
                    None)

        member this.Snapshot() =
            let findingsSnapshot = findings |> Seq.toList
            let commentsSnapshot = comments |> Seq.toList
            let disagreements = this.ComputeDisagreements findingsSnapshot

            { Findings = findingsSnapshot
              Comments = commentsSnapshot
              Disagreements = disagreements }

        member this.GetFindingsForSpec(specId: string) =
            let key =
                if String.IsNullOrWhiteSpace(specId) then
                    specId
                else
                    specId.Trim().ToUpperInvariant()

            findings
            |> Seq.filter (fun finding ->
                let specKey, _, _ = normaliseKey finding.Target
                specKey = key)
            |> Seq.toList

        member _.GetCommentsForFinding(findingId: Guid) =
            comments
            |> Seq.filter (fun comment -> comment.FindingId = findingId)
            |> Seq.toList
