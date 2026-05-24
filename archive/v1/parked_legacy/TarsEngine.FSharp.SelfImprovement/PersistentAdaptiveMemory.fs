namespace TarsEngine.FSharp.SelfImprovement

open System
open System.Collections
open System.Collections.Generic
open System.Globalization
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open TarsEngine.FSharp.Core.Services.CrossAgentValidation
open TarsEngine.FSharp.Core.Services.ReasoningTrace
open ExecutionHarness
open TarsEngine.FSharp.SelfImprovement.CrossAgentFeedback
open TarsEngine.FSharp.SelfImprovement.ValidatorCoordination

/// Persistent storage for adaptive self-improvement signals and policy evolution.
module PersistentAdaptiveMemory =

    [<CLIMutable>]
    type ReasoningEventSnapshot =
        { agentId: string
          step: string
          message: string
          score: float option
          metadata: Dictionary<string, string>
          createdAt: DateTime }

    [<CLIMutable>]
    type ReasoningTraceSnapshot =
        { correlationId: string
          summary: string option
          events: ReasoningEventSnapshot list }

    [<CLIMutable>]
    type AgentResultSnapshot =
        { agentId: string
          role: string
          outcome: string
          confidence: float option
          notes: string option
          producedAt: DateTime }

    [<CLIMutable>]
    type ConsensusSummary =
        { status: string
          message: string option
          agents: AgentResultSnapshot list }

    [<CLIMutable>]
    type CriticContext =
        { source: string option
          threshold: float option
          sampleSize: int option
          indicators: string list }

    [<CLIMutable>]
    type CriticSummary =
        { status: string
          message: string option
          source: string option
          threshold: float option
          sampleSize: int option
          indicators: string list }

    [<CLIMutable>]
    type ConsensusRuleSnapshot =
        { minimumPassCount: int
          requiredRoles: string list
          allowNeedsReview: bool
          minimumConfidence: float option
          maxFailureCount: int option }

    [<CLIMutable>]
    type PolicySnapshot =
        { requireConsensus: bool
          requireCritic: bool
          stopOnFailure: bool
          captureLogs: bool
          patchCommands: int
          validationCommands: int
          benchmarkCommands: int
          hasAgentProvider: bool
          hasTraceProvider: bool
          hasFeedbackSink: bool
          hasReasoningCritic: bool
          consensusRule: ConsensusRuleSnapshot option }

    [<CLIMutable>]
    type PolicyDelta =
        { field: string
          previousValue: string
          currentValue: string
          rationale: string option }

    [<CLIMutable>]
    type AgentFeedbackSnapshot =
        { agentId: string
          role: string
          verdict: string
          detail: string option
          confidence: float option
          notes: string option
          suggestedActions: string list
          recordedAt: DateTime }

    [<CLIMutable>]
    type AgentFeedbackAggregate =
        { role: string
          approve: int
          needsWork: int
          reject: int
          escalate: int
          samples: int
          averageConfidence: float option }

    [<CLIMutable>]
    type ValidatorFindingSnapshot =
        { findingId: Guid
          agentId: string
          role: string
          outcome: string
          confidence: float option
          notes: string option
          specId: string
          iterationId: Guid option
          topic: string option
          recordedAt: DateTime }

    [<CLIMutable>]
    type ValidatorDisagreementSnapshot =
        { specId: string
          iterationId: Guid option
          topic: string option
          roles: string list
          outcomes: string list
          trigger: string
          confidenceSpread: float option
          loggedAt: DateTime }

    [<CLIMutable>]
    type HarnessOutcomeSnapshot =
        { status: string
          failureReason: string option
          failedCommandCount: int option }

    [<CLIMutable>]
    type MemoryEntry =
        { runId: Guid
          specId: string
          specPath: string
          description: string option
          timestamp: DateTime
          consensus: ConsensusSummary option
          critic: CriticSummary option
          reasoning: ReasoningTraceSnapshot list
          policyBefore: PolicySnapshot
          policyAfter: PolicySnapshot
          policyChanges: PolicyDelta list
          inferenceTelemetry: Dictionary<string, obj>
          agentFeedback: AgentFeedbackSnapshot list
          validatorFindings: ValidatorFindingSnapshot list
          validatorDisagreements: ValidatorDisagreementSnapshot list
          harness: HarnessOutcomeSnapshot option }

    let private prefixKey (prefix: string) (key: string) =
        if String.IsNullOrWhiteSpace(prefix) then key else $"{prefix}.{key}"

    let private toInvariantString (value: obj) =
        match value with
        | null -> ""
        | :? string as str -> str
        | :? bool as b -> b.ToString()
        | :? IFormattable as formattable -> formattable.ToString(null, CultureInfo.InvariantCulture)
        | :? JsonElement as json -> json.ToString()
        | _ -> value.ToString()

    let rec private flattenJsonElement (dict: Dictionary<string, obj>) (prefix: string) (json: JsonElement) =
        match json.ValueKind with
        | JsonValueKind.Object ->
            for property in json.EnumerateObject() do
                flattenJsonElement dict (prefixKey prefix property.Name) property.Value
        | JsonValueKind.Array ->
            let items =
                json.EnumerateArray()
                |> Seq.map (fun element ->
                    match element.ValueKind with
                    | JsonValueKind.Object
                    | JsonValueKind.Array -> element.ToString()
                    | JsonValueKind.String -> element.GetString()
                    | JsonValueKind.Number ->
                        let mutable intValue = 0L
                        if element.TryGetInt64(&intValue) then
                            intValue.ToString(CultureInfo.InvariantCulture)
                        else
                            let mutable dbl = 0.0
                            if element.TryGetDouble(&dbl) then
                                dbl.ToString(CultureInfo.InvariantCulture)
                            else
                                element.ToString()
                    | JsonValueKind.True -> "true"
                    | JsonValueKind.False -> "false"
                    | _ -> "")
                |> Seq.filter (fun value -> not (String.IsNullOrWhiteSpace(value)))
                |> Seq.toArray

            if items.Length > 0 then
                dict.[prefix] <- box (String.Join(",", items))
        | JsonValueKind.String ->
            dict.[prefix] <- box (json.GetString())
        | JsonValueKind.Number ->
            let mutable intValue = 0L
            if json.TryGetInt64(&intValue) then
                dict.[prefix] <- box intValue
            else
                let mutable dbl = 0.0
                if json.TryGetDouble(&dbl) then
                    dict.[prefix] <- box dbl
                else
                    dict.[prefix] <- box (json.GetDecimal())
        | JsonValueKind.True
        | JsonValueKind.False ->
            dict.[prefix] <- box (json.GetBoolean())
        | JsonValueKind.Null
        | JsonValueKind.Undefined -> ()
        | _ ->
            dict.[prefix] <- box (json.ToString())

    let rec private flattenTelemetryValue (dict: Dictionary<string, obj>) (prefix: string) (value: obj) =
        let setSimple v = dict.[prefix] <- v

        match value with
        | null -> ()
        | :? JsonElement as json ->
            flattenJsonElement dict prefix json
        | :? Map<string, obj> as map ->
            map |> Map.iter (fun key inner -> flattenTelemetryValue dict (prefixKey prefix key) inner)
        | :? IDictionary as dictionary ->
            dictionary
            |> Seq.cast<DictionaryEntry>
            |> Seq.iter (fun entry ->
                let key = prefixKey prefix (string entry.Key)
                flattenTelemetryValue dict key entry.Value)
        | :? (byte[]) as bytes ->
            if bytes.Length > 0 then
                setSimple (box (Convert.ToBase64String(bytes)))
        | :? Array as array ->
            let items =
                array
                |> Seq.cast<obj>
                |> Seq.map toInvariantString
                |> Seq.filter (fun value -> not (String.IsNullOrWhiteSpace(value)))
                |> Seq.toArray

            if items.Length > 0 then
                setSimple (box (String.Join(",", items)))
        | :? IEnumerable as enumerable when not (value :? string) ->
            let items =
                enumerable
                |> Seq.cast<obj>
                |> Seq.map toInvariantString
                |> Seq.filter (fun value -> not (String.IsNullOrWhiteSpace(value)))
                |> Seq.toArray

            if items.Length > 0 then
                setSimple (box (String.Join(",", items)))
        | :? DateTime as dt ->
            setSimple (box (dt.ToString("o", CultureInfo.InvariantCulture)))
        | :? DateTimeOffset as dto ->
            setSimple (box (dto.ToString("o", CultureInfo.InvariantCulture)))
        | :? TimeSpan as span ->
            setSimple (box span.TotalMilliseconds)
        | :? Guid as guid ->
            setSimple (box (guid.ToString("D")))
        | :? bool
        | :? byte
        | :? sbyte
        | :? int16
        | :? uint16
        | :? int
        | :? uint32
        | :? int64
        | :? uint64
        | :? single
        | :? double
        | :? decimal
        | :? string ->
            setSimple value
        | :? IFormattable as formattable ->
            setSimple (box (formattable.ToString(null, CultureInfo.InvariantCulture)))
        | _ ->
            setSimple (box (value.ToString()))

    let captureTelemetry (data: Map<string, obj>) =
        let telemetry = Dictionary<string, obj>()
        data |> Map.iter (fun key value -> flattenTelemetryValue telemetry key value)
        telemetry

    let flattenEvolutionData (data: Map<string, obj>) =
        captureTelemetry data
        |> Seq.map (fun pair -> pair.Key, pair.Value)
        |> Map.ofSeq

    let private serializerOptions =
        let options = JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = false)
        options.Converters.Add(JsonStringEnumConverter())
        options

    let mutable private lastWritePath: string option = None

    let getLastWritePath () = lastWritePath

    let clearLastWritePath () =
        lastWritePath <- None

    let setLastWritePath value =
        lastWritePath <- value

    /// Compact representation of policy knobs used for self-optimization.
    type PolicyGenome =
        { requireConsensus: bool
          requireCritic: bool
          stopOnFailure: bool
          captureLogs: bool
          minimumPassCount: int option
          allowNeedsReview: bool option
          minimumConfidence: float option
          maxFailureCount: int option }

    /// Aggregated performance data for a specific policy genome.
    type PolicyPerformanceSummary =
        { Genome: PolicyGenome
          Trials: int
          Successes: int
          Failures: int
          CriticRejectRate: float
          CriticNeedsReviewRate: float }

    let private genomeKey (genome: PolicyGenome) =
        [ genome.requireConsensus.ToString()
          genome.requireCritic.ToString()
          genome.stopOnFailure.ToString()
          genome.captureLogs.ToString()
          (genome.minimumPassCount |> Option.map string |> Option.defaultValue "none")
          (genome.allowNeedsReview |> Option.map string |> Option.defaultValue "none")
          (genome.minimumConfidence |> Option.map (fun v -> v.ToString("F2")) |> Option.defaultValue "none")
          (genome.maxFailureCount |> Option.map string |> Option.defaultValue "none") ]
        |> String.concat "|"

    let private snapshotToGenome (snapshot: PolicySnapshot) =
        { requireConsensus = snapshot.requireConsensus
          requireCritic = snapshot.requireCritic
          stopOnFailure = snapshot.stopOnFailure
          captureLogs = snapshot.captureLogs
          minimumPassCount = snapshot.consensusRule |> Option.map (fun rule -> rule.minimumPassCount)
          allowNeedsReview = snapshot.consensusRule |> Option.map (fun rule -> rule.allowNeedsReview)
          minimumConfidence = snapshot.consensusRule |> Option.bind (fun rule -> rule.minimumConfidence)
          maxFailureCount = snapshot.consensusRule |> Option.bind (fun rule -> rule.maxFailureCount) }

    let summarizePolicyOutcomes (entries: MemoryEntry list) =
        let accumulate (state: Map<string, PolicyPerformanceSummary>) (entry: MemoryEntry) =
            let genome = snapshotToGenome entry.policyAfter
            let key = genomeKey genome
            let success =
                match entry.harness with
                | Some outcome when outcome.status = "passed" -> true
                | Some outcome when outcome.status = "failed" -> false
                | _ ->
                    match entry.critic with
                    | Some critic when critic.status = "reject" -> false
                    | _ -> entry.consensus |> Option.exists (fun c -> c.status = "passed")

            let criticReject =
                entry.critic
                |> Option.exists (fun critic -> critic.status = "reject")

            let criticNeedsReview =
                entry.critic
                |> Option.exists (fun critic -> critic.status = "needs_review")

            let update summary =
                let trials = summary.Trials + 1
                let successes = summary.Successes + (if success then 1 else 0)
                let failures = summary.Failures + (if success then 0 else 1)
                let rejectCount =
                    summary.CriticRejectRate * float summary.Trials
                    + (if criticReject then 1.0 else 0.0)
                let needsReviewCount =
                    summary.CriticNeedsReviewRate * float summary.Trials
                    + (if criticNeedsReview then 1.0 else 0.0)

                { summary with
                    Trials = trials
                    Successes = successes
                    Failures = failures
                    CriticRejectRate = rejectCount / float trials
                    CriticNeedsReviewRate = needsReviewCount / float trials }

            let initialSummary =
                { Genome = genome
                  Trials = 0
                  Successes = 0
                  Failures = 0
                  CriticRejectRate = 0.0
                  CriticNeedsReviewRate = 0.0 }

            let updated = update (state |> Map.tryFind key |> Option.defaultValue initialSummary)
            state |> Map.add key updated

        entries |> List.fold accumulate Map.empty |> Map.toList |> List.map snd

    let private ensureDirectory (path: string) =
        let fullPath = Path.GetFullPath(path)
        let directory = Path.GetDirectoryName(fullPath)
        if not (String.IsNullOrWhiteSpace(directory)) then
            Directory.CreateDirectory(directory) |> ignore
        fullPath

    let private stringifyRole = function
        | AgentRole.Custom value -> value
        | role -> role.ToString()

    let private stringifyOutcome = function
        | ValidationOutcome.Pass -> "pass"
        | ValidationOutcome.Fail -> "fail"
        | ValidationOutcome.NeedsReview -> "needs_review"

    let private captureAgent (result: AgentValidationResult) =
        { agentId = result.AgentId
          role = stringifyRole result.Role
          outcome = stringifyOutcome result.Outcome
          confidence = result.Confidence
          notes = result.Notes
          producedAt = result.ProducedAt }

    let captureConsensus (outcome: ConsensusOutcome option) =
        outcome
        |> Option.map (fun outcome ->
            match outcome with
            | ConsensusOutcome.ConsensusPassed agents ->
                { status = "passed"
                  message = None
                  agents = agents |> List.map captureAgent }
            | ConsensusOutcome.ConsensusNeedsReview (agents, message) ->
                { status = "needs_review"
                  message = Some message
                  agents = agents |> List.map captureAgent }
            | ConsensusOutcome.ConsensusFailed (agents, message) ->
                { status = "failed"
                  message = Some message
                  agents = agents |> List.map captureAgent })

    let captureCritic (verdict: CriticVerdict option) (context: CriticContext option) =
        verdict
        |> Option.map (fun verdict ->
            let baseSummary =
                match verdict with
                | CriticVerdict.Accept ->
                    { status = "accept"
                      message = None
                      source = None
                      threshold = None
                      sampleSize = None
                      indicators = [] }
                | CriticVerdict.NeedsReview message ->
                    { status = "needs_review"
                      message = Some message
                      source = None
                      threshold = None
                      sampleSize = None
                      indicators = [] }
                | CriticVerdict.Reject message ->
                    { status = "reject"
                      message = Some message
                      source = None
                      threshold = None
                      sampleSize = None
                      indicators = [] }

            match context with
            | None -> baseSummary
            | Some ctx ->
                { baseSummary with
                    source = ctx.source
                    threshold = ctx.threshold
                    sampleSize = ctx.sampleSize
                    indicators = ctx.indicators })

    let captureAgentFeedback (feedback: CrossAgentFeedback.AgentFeedback list) =
        feedback
        |> List.map (fun (entry: CrossAgentFeedback.AgentFeedback) ->
            { agentId = entry.AgentId
              role = stringifyRole entry.Role
              verdict = CrossAgentFeedback.verdictToString entry.Verdict
              detail = CrossAgentFeedback.verdictDetail entry.Verdict
              confidence = entry.Confidence
              notes = entry.Notes
              suggestedActions = entry.SuggestedActions
              recordedAt = entry.RecordedAt })

    let captureValidatorFindings (findings: ValidatorCoordination.ValidatorFinding list) =
        findings
        |> List.map (fun finding ->
            { findingId = finding.FindingId
              agentId = finding.AgentId
              role = stringifyRole finding.Role
              outcome = stringifyOutcome finding.Outcome
              confidence = finding.Confidence
              notes = finding.Notes
              specId = finding.Target.SpecId
              iterationId = finding.Target.IterationId
              topic = finding.Target.Topic
              recordedAt = finding.RecordedAt })

    let captureValidatorDisagreements (disagreements: ValidatorCoordination.ValidatorDisagreement list) =
        disagreements
        |> List.map (fun disagreement ->
            let outcomes =
                disagreement.Outcomes
                |> List.map (fun (role, outcome) -> $"{stringifyRole role}={stringifyOutcome outcome}")

            { specId = disagreement.Target.SpecId
              iterationId = disagreement.Target.IterationId
              topic = disagreement.Target.Topic
              roles = disagreement.Roles |> List.map stringifyRole
              outcomes = outcomes
              trigger = disagreement.Trigger
              confidenceSpread = disagreement.ConfidenceSpread
              loggedAt = disagreement.LoggedAt })

    type private AgentFeedbackAccumulation =
        { Approve: int
          NeedsWork: int
          Reject: int
          Escalate: int
          Samples: int
          ConfidenceSum: float
          ConfidenceCount: int }

    let summarizeFeedbackByRole (entries: MemoryEntry list) =
        let table = Dictionary<string, AgentFeedbackAccumulation>()

        let determineVerdictCounts verdict state =
            match verdict with
            | "approve" ->
                { state with Approve = state.Approve + 1; Samples = state.Samples + 1 }
            | "needs_work" ->
                { state with NeedsWork = state.NeedsWork + 1; Samples = state.Samples + 1 }
            | "reject" ->
                { state with Reject = state.Reject + 1; Samples = state.Samples + 1 }
            | "escalate" ->
                { state with Escalate = state.Escalate + 1; Samples = state.Samples + 1 }
            | _ ->
                { state with Samples = state.Samples + 1 }

        for entry in entries do
            for feedback in entry.agentFeedback do
                let role = feedback.role
                let verdict = feedback.verdict.ToLowerInvariant()
                let current =
                    match table.TryGetValue(role) with
                    | true, state -> state
                    | _ ->
                        { Approve = 0
                          NeedsWork = 0
                          Reject = 0
                          Escalate = 0
                          Samples = 0
                          ConfidenceSum = 0.0
                          ConfidenceCount = 0 }

                let updated = determineVerdictCounts verdict current
                let updated =
                    match feedback.confidence with
                    | Some confidence ->
                        { updated with
                            ConfidenceSum = updated.ConfidenceSum + confidence
                            ConfidenceCount = updated.ConfidenceCount + 1 }
                    | None -> updated

                table.[role] <- updated

        table
        |> Seq.map (fun kvp ->
            let state = kvp.Value
            let averageConfidence =
                if state.ConfidenceCount = 0 then None
                else Some (state.ConfidenceSum / float state.ConfidenceCount)
            { role = kvp.Key
              approve = state.Approve
              needsWork = state.NeedsWork
              reject = state.Reject
              escalate = state.Escalate
              samples = state.Samples
              averageConfidence = averageConfidence })
        |> Seq.toList


    let captureReasoning (traces: ReasoningTrace list) =
        traces
        |> List.map (fun trace ->
            { correlationId = trace.CorrelationId
              summary = trace.Summary
              events =
                trace.Events
                |> List.map (fun event ->
                    let metadata = Dictionary<string, string>()
                    event.Metadata
                    |> Map.iter (fun key value ->
                        let serialized =
                            match value with
                            | null -> ""
                            | :? string as text -> text
                            | other -> other.ToString()
                        metadata.Add(key, serialized))
                    { agentId = event.AgentId
                      step = event.Step
                      message = event.Message
                      score = event.Score
                      metadata = metadata
                      createdAt = event.CreatedAt }) })

    let expandReasoning (snapshots: ReasoningTraceSnapshot list) =
        snapshots
        |> List.map (fun snapshot ->
            { ReasoningTrace.CorrelationId = snapshot.correlationId
              Summary = snapshot.summary
              Events =
                snapshot.events
                |> List.map (fun evt ->
                    { ReasoningEvent.AgentId = evt.agentId
                      Step = evt.step
                      Message = evt.message
                      Score = evt.score
                      Metadata =
                        evt.metadata
                        |> Seq.map (fun kvp -> kvp.Key, kvp.Value :> obj)
                        |> Map.ofSeq
                      CreatedAt = evt.createdAt }) })

    let captureConsensusRule (rule: ConsensusRule option) =
        rule
        |> Option.map (fun rule ->
            { minimumPassCount = rule.MinimumPassCount
              requiredRoles = rule.RequiredRoles |> List.map stringifyRole
              allowNeedsReview = rule.AllowNeedsReview
              minimumConfidence = rule.MinimumConfidence
              maxFailureCount = rule.MaxFailureCount })

    let diffPolicies (beforeSnapshot: PolicySnapshot) (afterSnapshot: PolicySnapshot) =
        let diffs = ResizeArray<PolicyDelta>()

        let push field previousValue currentValue rationale =
            if previousValue <> currentValue then
                diffs.Add({ field = field; previousValue = previousValue; currentValue = currentValue; rationale = rationale })

        let stringOfBool value = if value then "true" else "false"
        let stringOfOption opt = opt |> Option.map string |> Option.defaultValue "none"

        push "requireConsensus" (stringOfBool beforeSnapshot.requireConsensus) (stringOfBool afterSnapshot.requireConsensus)
            (Some "Adjust consensus enforcement gate based on historical outcomes.")

        push "requireCritic" (stringOfBool beforeSnapshot.requireCritic) (stringOfBool afterSnapshot.requireCritic)
            (Some "Critic approval policy adapted to recent verdict trends.")

        push "stopOnFailure" (stringOfBool beforeSnapshot.stopOnFailure) (stringOfBool afterSnapshot.stopOnFailure) None

        push "captureLogs" (stringOfBool beforeSnapshot.captureLogs) (stringOfBool afterSnapshot.captureLogs) None

        push "patchCommands" (string beforeSnapshot.patchCommands) (string afterSnapshot.patchCommands) None
        push "validationCommands" (string beforeSnapshot.validationCommands) (string afterSnapshot.validationCommands) None
        push "benchmarkCommands" (string beforeSnapshot.benchmarkCommands) (string afterSnapshot.benchmarkCommands) None

        push "hasAgentProvider" (stringOfBool beforeSnapshot.hasAgentProvider) (stringOfBool afterSnapshot.hasAgentProvider) None
        push "hasTraceProvider" (stringOfBool beforeSnapshot.hasTraceProvider) (stringOfBool afterSnapshot.hasTraceProvider) None
        push "hasFeedbackSink" (stringOfBool beforeSnapshot.hasFeedbackSink) (stringOfBool afterSnapshot.hasFeedbackSink) None
        push "hasReasoningCritic" (stringOfBool beforeSnapshot.hasReasoningCritic) (stringOfBool afterSnapshot.hasReasoningCritic) None

        match beforeSnapshot.consensusRule, afterSnapshot.consensusRule with
        | None, Some _ ->
            diffs.Add(
                { field = "consensusRule"
                  previousValue = "disabled"
                  currentValue = "enabled"
                  rationale = Some "Consensus rule introduced to tighten multi-agent validation." })
        | Some _, None ->
            diffs.Add(
                { field = "consensusRule"
                  previousValue = "enabled"
                  currentValue = "disabled"
                  rationale = Some "Consensus rule disabled due to sustained pass streak." })
        | Some beforeRule, Some afterRule ->
            push "consensus.minimumPassCount" (string beforeRule.minimumPassCount) (string afterRule.minimumPassCount) None
            push "consensus.allowNeedsReview" (stringOfBool beforeRule.allowNeedsReview) (stringOfBool afterRule.allowNeedsReview) None
            push "consensus.minimumConfidence" (stringOfOption beforeRule.minimumConfidence) (stringOfOption afterRule.minimumConfidence) None
            push "consensus.maxFailureCount" (stringOfOption beforeRule.maxFailureCount) (stringOfOption afterRule.maxFailureCount) None

            let beforeRoles = beforeRule.requiredRoles |> String.concat ","
            let afterRoles = afterRule.requiredRoles |> String.concat ","
            push "consensus.requiredRoles" beforeRoles afterRoles None
        | None, None -> ()

        diffs |> Seq.toList

    let captureHarness (report: HarnessReport option) =
        report
        |> Option.map (fun report ->
            match report.Outcome with
            | HarnessOutcome.AllPassed _ ->
                { status = "passed"
                  failureReason = None
                  failedCommandCount = Some(report.Commands |> List.filter (fun c -> c.ExitCode <> 0) |> List.length) }
            | HarnessOutcome.Failed (_, reason) ->
                { status = "failed"
                  failureReason = Some reason
                  failedCommandCount = Some(report.Commands |> List.filter (fun c -> c.ExitCode <> 0) |> List.length) })

    let append (storePath: string) (entry: MemoryEntry) =
        if String.IsNullOrWhiteSpace(storePath) then
            ()
        else
            let fullPath = ensureDirectory storePath
            let json = JsonSerializer.Serialize(entry, serializerOptions)
            File.AppendAllText(fullPath, json + Environment.NewLine)
            Console.WriteLine($"[AdaptiveMemory] appended entry to {fullPath}")
            lastWritePath <- Some fullPath

    let loadAll (storePath: string) =
        if String.IsNullOrWhiteSpace(storePath) || not (File.Exists(storePath)) then
            []
        else
            File.ReadLines(storePath)
            |> Seq.choose (fun line ->
                let trimmed = line.Trim()
                if String.IsNullOrWhiteSpace(trimmed) then
                    None
                else
                    try
                        let entry = JsonSerializer.Deserialize<MemoryEntry>(trimmed, serializerOptions)
                        if obj.ReferenceEquals(entry, null) then None else Some entry
                    with
                    | _ -> None)
            |> Seq.toList

    let loadRecent (storePath: string) (limit: int) =
        loadAll storePath
        |> List.rev
        |> fun entries ->
            if limit <= 0 then entries else entries |> List.truncate limit

    let summarizeFeedbackFile (path: string) (limit: int option) =
        if String.IsNullOrWhiteSpace(path) || not (File.Exists(path)) then
            []
        else
            let entries =
                match limit with
                | Some l when l > 0 -> loadRecent path l
                | _ -> loadAll path
            summarizeFeedbackByRole entries


