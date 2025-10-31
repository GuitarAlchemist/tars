namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Net.Http
open System.Text
open Microsoft.Extensions.Logging
open ImprovementTypes
open PatternRecognition
open OllamaCodeAnalyzer
open CodeTransforms
open ExecutionHarness
open SpecKitWorkspace
open SpecKitGoalPlanner
open TarsEngine.FSharp.SelfImprovement.AuggieIntegration
open PersistentAdaptiveMemory
open AgentPolicyTuning
open TarsEngine.FSharp.Core.Services.CrossAgentValidation
open TarsEngine.FSharp.Core.Services.ReasoningTrace
open TarsEngine.FSharp.Core.Services.MetascriptClosureIntegrationService
open TarsEngine.FSharp.SelfImprovement.AutonomousSpecHarness
open TarsEngine.FSharp.SelfImprovement.CrossAgentFeedback
open TarsEngine.FSharp.SelfImprovement.ValidatorCoordination
open TarsEngine.FSharp.SelfImprovement.TeamOrchestrator
open TarsEngine.FSharp.SelfImprovement.GovernanceLedger
open TarsEngine.FSharp.Metascript
open TarsEngine.FSharp.Metascript.BlockHandlers
open TarsEngine.FSharp.Metascript.Services
open ProductBacklogDecomposer

module internal SelfImprovementHelpers =
    let agentRoleToString = function
        | AgentRole.Custom value -> value
        | role -> role.ToString()

    let policySnapshotFromConfig (config: SpecDrivenIterationConfig) =
        { requireConsensus = config.RequireConsensusForExecution
          requireCritic = config.RequireCriticApproval
          stopOnFailure = config.StopOnFailure
          captureLogs = config.CaptureLogs
          patchCommands = config.PatchCommands |> List.length
          validationCommands = config.ValidationCommands |> List.length
          benchmarkCommands = config.BenchmarkCommands |> List.length
          hasAgentProvider = config.AgentResultProvider |> Option.isSome
          hasTraceProvider = config.ReasoningTraceProvider |> Option.isSome
          hasFeedbackSink = config.ReasoningFeedbackSink |> Option.isSome
          hasReasoningCritic = config.ReasoningCritic |> Option.isSome
          consensusRule =
            config.ConsensusRule
            |> Option.map (fun rule ->
                { minimumPassCount = rule.MinimumPassCount
                  requiredRoles = rule.RequiredRoles |> List.map agentRoleToString
                  allowNeedsReview = rule.AllowNeedsReview
                  minimumConfidence = rule.MinimumConfidence
                  maxFailureCount = rule.MaxFailureCount }) }

    let resolveAdaptiveMemoryPath () =
        match Environment.GetEnvironmentVariable("TARS_ADAPTIVE_MEMORY_PATH") with
        | null
        | "" -> Path.Combine(Environment.CurrentDirectory, "output", "adaptive_memory_spec-kit.jsonl")
        | value -> value

    let buildReasoningTraceFromClosure (closureResult: MetascriptClosureResult) =
        let summary =
            if String.IsNullOrWhiteSpace closureResult.OutputSummary then
                "Metascript execution completed."
            else
                closureResult.OutputSummary

        let eventMetadata =
            closureResult.EvolutionData

        let event =
            { AgentId = "metascript:closure"
              Step = "release-notes"
              Message = summary
              Score = Some 0.95
              Metadata = eventMetadata
              CreatedAt = DateTime.UtcNow }

        { CorrelationId = closureResult.CommandId
          Summary = Some summary
          Events = [ event ] }

    let defaultReasoningTraceProvider (closureResult: MetascriptClosureResult) =
        [ buildReasoningTraceFromClosure closureResult ]

    type RemediationArtifact =
        { Path: string
          Actions: (string * string list) list }

    type ReasoningCriticResolution =
        { Critic: ReasoningTrace list -> CriticVerdict
          Source: string
          Model: MetaReasoningCritic.CriticModel option }

    let validationOutcomeToString = function
        | ValidationOutcome.Pass -> "pass"
        | ValidationOutcome.Fail -> "fail"
        | ValidationOutcome.NeedsReview -> "needs_review"

    let criticVerdictToString = function
        | None -> "none"
        | Some CriticVerdict.Accept -> "accept"
        | Some (CriticVerdict.NeedsReview _) -> "needs_review"
        | Some (CriticVerdict.Reject _) -> "reject"

    let buildAgentTrace (result: AgentValidationResult) (summary: string) (stepLabel: string) (metadata: (string * obj) list) : ReasoningTrace =
        let baseMetadata =
            metadata
            |> List.fold (fun acc (key, value) -> acc |> Map.add key value) Map.empty
            |> Map.add "agent.role" (agentRoleToString result.Role |> box)
            |> Map.add "agent.outcome" (validationOutcomeToString result.Outcome |> box)

        let enrichedMetadata =
            baseMetadata
            |> fun meta ->
                match result.Confidence with
                | Some value -> meta |> Map.add "agent.confidence" (box value)
                | None -> meta
            |> fun meta ->
                match result.Notes with
                | Some notes when not (String.IsNullOrWhiteSpace notes) -> meta |> Map.add "agent.notes" (box notes)
                | _ -> meta

        let event : ReasoningEvent =
            { AgentId = result.AgentId
              Step = stepLabel
              Message = result.Notes |> Option.defaultValue summary
              Score = result.Confidence
              Metadata = enrichedMetadata
              CreatedAt = result.ProducedAt }

        { CorrelationId = sprintf "%s:%s" result.AgentId (Guid.NewGuid().ToString("N"))
          Summary = Some summary
          Events = [ event ] }

    let buildCriticContext (resolution: ReasoningCriticResolution) : PersistentAdaptiveMemory.CriticContext =
        let indicators =
            resolution.Model
            |> Option.map (fun model -> model.NegativeIndicators |> List.map fst)
            |> Option.defaultValue []

        { source = Some resolution.Source
          threshold = resolution.Model |> Option.map (fun model -> model.ScoreThreshold)
          sampleSize = resolution.Model |> Option.map (fun model -> model.SampleSize)
          indicators = indicators }

    let overrideAgentResultFromEnv envVar (label: string) (result: AgentValidationResult) =
        let normalize (value: string) = value.Trim().ToLowerInvariant()

        match Environment.GetEnvironmentVariable(envVar) with
        | null
        | "" -> result
        | value ->
            let trimmed = normalize value
            let outcome, note =
                match trimmed with
                | "pass" -> ValidationOutcome.Pass, Some $"Environment override ({label}) forced pass result."
                | "fail" -> ValidationOutcome.Fail, Some $"Environment override ({label}) forced failure."
                | "needsreview"
                | "needs_review" -> ValidationOutcome.NeedsReview, Some $"Environment override ({label}) requested manual review."
                | _ -> result.Outcome, Some $"Environment override ({label}) ignored unknown value '{value}'."

            let confidence =
                match outcome with
                | ValidationOutcome.Pass -> Some 0.9
                | ValidationOutcome.Fail -> Some 0.2
                | ValidationOutcome.NeedsReview -> Some 0.4

            { result with
                Outcome = outcome
                Confidence = confidence
                Notes = note |> Option.orElse result.Notes }

    let tryLoadPreviousCriticContext (adaptivePath: string option) (specId: string) =
        match adaptivePath with
        | None -> None
        | Some path when String.IsNullOrWhiteSpace path -> None
        | Some path ->
            try
                if File.Exists(path) then
                    PersistentAdaptiveMemory.loadRecent path 128
                    |> List.tryFind (fun entry -> entry.specId.Equals(specId, StringComparison.OrdinalIgnoreCase))
                    |> Option.bind (fun entry ->
                        entry.critic
                        |> Option.map (fun critic ->
                            { PersistentAdaptiveMemory.CriticContext.source = critic.source
                              threshold = critic.threshold
                              sampleSize = critic.sampleSize
                              indicators = critic.indicators }))
                else
                    None
            with _ -> None

    let buildRemediationTrace (feedback: AgentFeedback) =
        if feedback.SuggestedActions.IsEmpty then
            None
        else
            let summary =
                sprintf "%s remediation actions" (agentRoleToString feedback.Role)

            let message =
                feedback.SuggestedActions
                |> String.concat "; "

            let metadata =
                Map.empty
                |> Map.add "agent.role" (agentRoleToString feedback.Role |> box)
                |> Map.add "agent.verdict" (feedback.Verdict.ToString() |> box)
                |> Map.add "agent.suggested_actions" (message |> box)

            let event : ReasoningEvent =
                { AgentId = feedback.AgentId
                  Step = "remediation"
                  Message = message
                  Score = feedback.Confidence
                  Metadata = metadata
                  CreatedAt = feedback.RecordedAt }

            Some
                { CorrelationId = sprintf "%s:remediation:%s" feedback.AgentId (Guid.NewGuid().ToString("N"))
                  Summary = Some summary
                  Events = [ event ] }

    let createRemediationArtifact
        (iterationId: Guid)
        (specId: string)
        (harnessStatus: bool option)
        (feedback: AgentFeedback list) =

        let actionable =
            feedback
            |> List.filter (fun fb -> fb.Verdict <> FeedbackVerdict.Approve)

        if actionable.IsEmpty then
            None
        else
            let directory = Path.Combine(Environment.CurrentDirectory, "output", "remediation_tasks")
            Directory.CreateDirectory(directory) |> ignore
            let fileName = sprintf "%s_%s.md" (DateTime.UtcNow.ToString("yyyyMMddHHmmss")) (iterationId.ToString("N"))
            let path = Path.Combine(directory, fileName)

            let builder = StringBuilder()
            builder.AppendLine("# Tier3 Remediation Plan") |> ignore
            builder.AppendLine(sprintf "- Spec ID: %s" specId) |> ignore
            builder.AppendLine(sprintf "- Iteration ID: %s" (iterationId.ToString("D"))) |> ignore
            builder.AppendLine(sprintf "- Harness Passed: %b" (harnessStatus |> Option.defaultValue false)) |> ignore
            builder.AppendLine(sprintf "- Generated: %s" (DateTime.UtcNow.ToString("u"))) |> ignore
            builder.AppendLine() |> ignore
            builder.AppendLine("## Actions") |> ignore

            actionable
            |> List.iter (fun fb ->
                builder.AppendLine(sprintf "### %s (%s)" (agentRoleToString fb.Role) fb.AgentId) |> ignore
                builder.AppendLine(sprintf "Verdict: %s" (fb.Verdict.ToString())) |> ignore
                fb.Notes |> Option.iter (fun note -> builder.AppendLine(sprintf "Notes: %s" note) |> ignore)
                if fb.SuggestedActions.IsEmpty then
                    builder.AppendLine("- No suggested actions provided") |> ignore
                else
                    fb.SuggestedActions
                    |> List.iter (fun action -> builder.AppendLine(sprintf "- %s" action) |> ignore)
                builder.AppendLine() |> ignore)

            File.WriteAllText(path, builder.ToString())
            let actionsSummary =
                actionable
                |> List.map (fun fb ->
                    let role = agentRoleToString fb.Role
                    let actions =
                        if fb.SuggestedActions.IsEmpty then
                            [ "Review remediation plan manually." ]
                        else
                            fb.SuggestedActions
                    role, actions)

            Some { Path = path; Actions = actionsSummary }

    let appendRemediationTask (tasksPath: string option) (specId: string) (iterationId: Guid) (artifact: RemediationArtifact) =
        let relativeArtifactPath =
            try
                Path.GetRelativePath(Environment.CurrentDirectory, artifact.Path)
            with _ ->
                artifact.Path

        let actionsText =
            artifact.Actions
            |> List.collect (fun (role, actions) -> actions |> List.map (fun action -> sprintf "%s: %s" role action))
            |> fun entries ->
                if entries.IsEmpty then "Review remediation plan."
                else String.Join("; ", entries)

        match tasksPath with
        | None -> ()
        | Some path ->
            try
                let taskId = iterationId.ToString("N").Substring(0, 8).ToUpperInvariant()
                let line =
                    sprintf "- [ ] TREM-%s [P0] Auto remediation for %s (iteration %s). Actions: %s. See `%s`"
                        taskId
                        specId
                        (DateTime.UtcNow.ToString("u"))
                        actionsText
                        relativeArtifactPath

                if File.Exists(path) then
                    File.AppendAllText(path, Environment.NewLine + line + Environment.NewLine)
                else
                    let content =
                        StringBuilder()
                            .AppendLine("## Auto-Generated Remediations")
                            .AppendLine()
                            .AppendLine(line)
                            .AppendLine()
                            .ToString()
                    File.WriteAllText(path, content)
            with ex ->
                Console.WriteLine($"[remediation-task] Failed to append remediation task for spec {specId}: {ex.Message}")

        try
            SpecKitGoalPlanner.recordRemediationTicket specId iterationId artifact.Path relativeArtifactPath artifact.Actions
        with ex ->
            Console.WriteLine($"[remediation-task] Failed to register remediation backlog entry for spec {specId}: {ex.Message}")

    let internal mergeMetricMaps (maps: Map<string, obj> list) =
        maps
        |> List.fold (fun state map ->
            map |> Map.fold (fun acc key value -> acc |> Map.add key value) state) Map.empty

    type Tier3MetascriptConfig =
        { Path: string
          AgentId: string
          Role: AgentRole
          AsCritic: bool }

    type Tier3MetascriptOutcome =
        { Result: AgentValidationResult
          Trace: ReasoningTrace
          CriticVerdict: CriticVerdict option
          Metrics: Map<string, obj> }

    let private sanitiseAgentKey (agentId: string) =
        agentId.Replace(":", ".").Replace(" ", "_")

    let runTier3Metascripts (loggerFactory: ILoggerFactory) (baseDirectory: string option) =
        async {
            let baseDir =
                match baseDirectory with
                | Some explicit -> explicit
                | None -> Path.Combine(Environment.CurrentDirectory, ".specify", "meta", "tier3")

            let configs =
                [ { Path = Path.Combine(baseDir, "safety-review.trsx")
                    AgentId = "agent:tier3-safety"
                    Role = AgentRole.SafetyGovernor
                    AsCritic = false }
                  { Path = Path.Combine(baseDir, "performance-review.trsx")
                    AgentId = "agent:tier3-performance"
                    Role = AgentRole.PerformanceBenchmarker
                    AsCritic = false }
                  { Path = Path.Combine(baseDir, "critic-federation.trsx")
                    AgentId = "agent:tier3-critic"
                    Role = AgentRole.MetaCritic
                    AsCritic = true } ]

            let registryLogger = loggerFactory.CreateLogger<BlockHandlerRegistry>()
            let registry = BlockHandlerRegistry(registryLogger)
            registry.RegisterHandler(TextBlockHandler(loggerFactory.CreateLogger<TextBlockHandler>()))
            registry.RegisterHandler(CommandBlockHandler(loggerFactory.CreateLogger<CommandBlockHandler>()))
            registry.RegisterHandler(FSharpBlockHandler(loggerFactory.CreateLogger<FSharpBlockHandler>()))

            let executorLogger = loggerFactory.CreateLogger<MetascriptExecutor>()
            let executor = MetascriptExecutor(executorLogger, registry)
            let serviceLogger = loggerFactory.CreateLogger<MetascriptService>()
            let metascriptService = MetascriptService(serviceLogger, executor)

            let! outcomes =
                configs
                |> List.map (fun cfg ->
                    async {
                        if not (File.Exists(cfg.Path)) then
                            let message = $"Metascript not found: {cfg.Path}"
                            let event : ReasoningEvent =
                                { AgentId = cfg.AgentId
                                  Step = "metascript"
                                  Message = message
                                  Score = Some 0.0
                                  Metadata = Map.empty
                                  CreatedAt = DateTime.UtcNow }

                            let trace : ReasoningTrace =
                                { CorrelationId = Guid.NewGuid().ToString("N")
                                  Summary = Some message
                                  Events = [ event ] }

                            let agentResult =
                                { AgentId = cfg.AgentId
                                  Role = cfg.Role
                                  Outcome = ValidationOutcome.Fail
                                  Confidence = Some 0.0
                                  Notes = Some message
                                  ProducedAt = DateTime.UtcNow }

                            let criticVerdict =
                                if cfg.AsCritic then
                                    Some (CriticVerdict.NeedsReview message)
                                else
                                    None

                            let prefix = $"tier3.{sanitiseAgentKey cfg.AgentId}"
                            let metrics =
                                Map.empty
                                |> Map.add $"{prefix}.status" (box "missing")
                                |> Map.add $"{prefix}.blocks.total" (box 0)
                                |> Map.add $"{prefix}.blocks.success" (box 0)
                                |> Map.add $"{prefix}.blocks.failure" (box 0)
                                |> Map.add $"{prefix}.blocks.partial" (box 0)
                                |> Map.add $"{prefix}.blocks.timeout" (box 0)
                                |> Map.add $"{prefix}.blocks.cancelled" (box 0)
                                |> Map.add $"{prefix}.blocks.not_executed" (box 0)

                            return
                                { Result = agentResult
                                  Trace = trace
                                  CriticVerdict = criticVerdict
                                  Metrics = metrics }
                        else
                            let! execResult =
                                metascriptService.ExecuteMetascriptFileAsync(cfg.Path)
                                |> Async.AwaitTask

                            let status = execResult.Status

                            let (outcome, confidence) =
                                match status with
                                | MetascriptExecutionStatus.Success -> ValidationOutcome.Pass, Some 0.9
                                | MetascriptExecutionStatus.Partial -> ValidationOutcome.NeedsReview, Some 0.5
                                | MetascriptExecutionStatus.NotExecuted -> ValidationOutcome.NeedsReview, Some 0.4
                                | MetascriptExecutionStatus.Timeout -> ValidationOutcome.NeedsReview, Some 0.3
                                | _ -> ValidationOutcome.Fail, Some 0.2

                            let truncate (text: string) =
                                let trimmed = text.Trim()
                                if trimmed.Length > 400 then
                                    trimmed.Substring(0, 400) + "..."
                                else
                                    trimmed

                            let primaryMessage =
                                match execResult.Output |> truncate with
                                | output when not (String.IsNullOrWhiteSpace output) -> output
                                | _ ->
                                    execResult.Error
                                    |> Option.orElseWith (fun () ->
                                        execResult.BlockResults
                                        |> List.tryPick (fun block ->
                                            match block.Error with
                                            | Some err when not (String.IsNullOrWhiteSpace err) ->
                                                Some($"Block {block.Block.Type} error: {err}")
                                            | _ -> None))
                                    |> Option.defaultValue "Metascript executed without producing output."

                            let notes =
                                match status with
                                | MetascriptExecutionStatus.Success -> None
                                | _ ->
                                    let blockNotes =
                                        execResult.BlockResults
                                        |> List.choose (fun block ->
                                            match block.Error with
                                            | Some err when not (String.IsNullOrWhiteSpace err) ->
                                                Some($"[{block.Block.Type}] {err}")
                                            | _ -> None)

                                    match execResult.Error, blockNotes with
                                    | Some err, _ when not (String.IsNullOrWhiteSpace err) -> Some err
                                    | _, head :: tail -> Some(String.Join(Environment.NewLine, head :: tail))
                                    | _ -> Some "Metascript execution encountered issues without specific error details."

                            let summaryEvent : ReasoningEvent =
                                { AgentId = cfg.AgentId
                                  Step = "metascript"
                                  Message = primaryMessage
                                  Score = confidence
                                  Metadata = Map.empty
                                  CreatedAt = DateTime.UtcNow }

                            let blockData =
                                execResult.BlockResults
                                |> List.mapi (fun idx block ->
                                    let stepLabel = block.Block.Type.ToString().ToLowerInvariant()
                                    let message =
                                        if not (String.IsNullOrWhiteSpace block.Output) then
                                            truncate block.Output
                                        else
                                            block.Error |> Option.defaultValue "No output produced."

                                    let score =
                                        match block.Status with
                                        | MetascriptExecutionStatus.Success -> Some 0.85
                                        | MetascriptExecutionStatus.Partial -> Some 0.6
                                        | MetascriptExecutionStatus.NotExecuted -> Some 0.4
                                        | _ -> Some 0.2
                                    { AgentId = cfg.AgentId
                                      Step = $"metascript:{stepLabel}"
                                      Message = message
                                      Score = score
                                      Metadata = Map.empty
                                      CreatedAt = DateTime.UtcNow },
                                    (block.Status, block.ExecutionTimeMs, idx))

                            let blockEvents, blockMetrics = blockData |> List.unzip

                            let blockMetricsMap =
                                let prefix = $"tier3.{sanitiseAgentKey cfg.AgentId}.blocks"
                                let initialise =
                                    Map.empty
                                    |> Map.add $"{prefix}.total" (box execResult.BlockResults.Length)

                                let statusCounts =
                                    blockMetrics
                                    |> List.fold (fun acc (status, _, _) ->
                                        let statusLabel = status.ToString().ToLowerInvariant()
                                        let key = $"{prefix}.{statusLabel}"
                                        let current =
                                            acc
                                            |> Map.tryFind key
                                            |> Option.defaultValue 0
                                        acc |> Map.add key (current + 1)) Map.empty

                                let durations =
                                    blockMetrics
                                    |> List.map (fun (_, duration, _) -> duration)

                                let totalDuration = durations |> List.sum
                                let avgDuration =
                                    if execResult.BlockResults |> List.isEmpty then
                                        0.0
                                    else
                                        totalDuration / float execResult.BlockResults.Length
                                let maxDuration =
                                    match durations with
                                    | [] -> 0.0
                                    | _ -> durations |> List.max

                                let withCounts =
                                    statusCounts
                                    |> Map.fold (fun state key value -> state |> Map.add key (box value)) initialise
                                    |> Map.add $"{prefix}.duration_total_ms" (box totalDuration)
                                    |> Map.add $"{prefix}.duration_avg_ms" (box avgDuration)
                                    |> Map.add $"{prefix}.duration_max_ms" (box maxDuration)

                                withCounts

                            let perBlockDurations =
                                blockMetrics
                                |> List.fold (fun state (_, duration, idx) ->
                                    let durationKey = $"tier3.{sanitiseAgentKey cfg.AgentId}.blocks.duration_ms[{idx}]"
                                    state |> Map.add durationKey (box duration)) Map.empty

                            let prefix = $"tier3.{sanitiseAgentKey cfg.AgentId}"

                            let metrics =
                                mergeMetricMaps
                                    [ blockMetricsMap
                                      perBlockDurations
                                      Map.empty
                                      |> Map.add $"{prefix}.status" (box (status.ToString().ToLowerInvariant()))
                                      |> Map.add $"{prefix}.execution_ms" (box execResult.ExecutionTimeMs)
                                      |> Map.add $"{prefix}.primary_message" (box primaryMessage)
                                      |> Map.add $"{prefix}.confidence" (confidence |> Option.defaultValue 0.0 |> box) ]

                            let trace : ReasoningTrace =
                                { CorrelationId = Guid.NewGuid().ToString("N")
                                  Summary = Some($"Metascript {Path.GetFileName(cfg.Path)} -> {status}")
                                  Events = summaryEvent :: blockEvents }

                            let agentResult =
                                { AgentId = cfg.AgentId
                                  Role = cfg.Role
                                  Outcome = outcome
                                  Confidence = confidence
                                  Notes = notes
                                  ProducedAt = DateTime.UtcNow }

                            let criticVerdict =
                                if cfg.AsCritic then
                                    match status with
                                    | MetascriptExecutionStatus.Success -> Some CriticVerdict.Accept
                                    | MetascriptExecutionStatus.Partial
                                    | MetascriptExecutionStatus.NotExecuted
                                    | MetascriptExecutionStatus.Timeout ->
                                        Some (CriticVerdict.NeedsReview primaryMessage)
                                    | _ ->
                                        let rejectionDetail = notes |> Option.defaultValue primaryMessage
                                        Some (CriticVerdict.Reject rejectionDetail)
                                else
                                    None

                            return
                                { Result = agentResult
                                  Trace = trace
                                  CriticVerdict = criticVerdict
                                  Metrics = metrics }
                    })
                |> Async.Parallel

            let outcomeList = outcomes |> Array.toList
            let aggregatedMetrics =
                outcomeList
                |> List.map (fun outcome -> outcome.Metrics)
                |> mergeMetricMaps

            return outcomeList, aggregatedMetrics
        }

    let private tryGetEnv name =
        match Environment.GetEnvironmentVariable(name) with
        | null
        | "" -> None
        | value -> Some value

    let loadTeamRegistry () =
        let tryLoad path =
            if String.IsNullOrWhiteSpace(path) then
                None
            elif File.Exists(path) then
                TeamConfiguration.loadFromFile path
            elif Directory.Exists(path) then
                TeamConfiguration.loadFromDirectory path
            else
                None

        match tryGetEnv "TARS_TEAM_CONFIG_PATH" |> Option.bind tryLoad with
        | Some registry -> Some registry
        | None ->
            let defaultDir = Path.Combine(Environment.CurrentDirectory, ".specify", "teams")
            TeamConfiguration.loadFromDirectory defaultDir

    let resolvePlannerDispatchCount (explicit: int option) =
        match explicit with
        | Some value -> Some value
        | None ->
            match tryGetEnv "TARS_AUTONOMOUS_DISPATCH_COUNT" with
            | Some text ->
                match Int32.TryParse text with
                | true, parsed when parsed > 0 -> Some parsed
                | _ -> None
            | None -> None

    let resolvePlannerTopCandidates (explicit: int option) =
        match explicit with
        | Some value -> Some value
        | None ->
            match tryGetEnv "TARS_AUTONOMOUS_TOP_CANDIDATES" with
            | Some text ->
                match Int32.TryParse text with
                | true, parsed when parsed > 0 -> Some parsed
                | _ -> None
            | None -> None

    let shouldAutoDispatch () =
        match tryGetEnv "TARS_AUTONOMOUS_DISPATCH" with
        | Some flag when flag.Equals("1", StringComparison.OrdinalIgnoreCase)
                        || flag.Equals("true", StringComparison.OrdinalIgnoreCase) -> true
        | _ -> false

    let shouldAutoApplyPatches () =
        match tryGetEnv "TARS_AUTONOMOUS_PATCH" with
        | Some flag when flag.Equals("0", StringComparison.OrdinalIgnoreCase)
                        || flag.Equals("false", StringComparison.OrdinalIgnoreCase) -> false
        | _ -> true

    let resolveAuggieSettings (overrideSettings: AuggieCliSettings option) =
        match overrideSettings with
        | Some settings -> settings
        | None ->
            let baseSettings = AuggieIntegration.defaultSettings
            let toolPath =
                tryGetEnv "AUGGIE_CLI_PATH"
                |> Option.defaultValue baseSettings.ToolPath
            let workingDir =
                tryGetEnv "AUGGIE_CLI_WORKDIR"
                |> Option.orElse baseSettings.WorkingDirectory
            let extraArgs =
                tryGetEnv "AUGGIE_CLI_ARGS"
                |> Option.map (fun args -> args.Split([| ' '; '\t'; '\r'; '\n' |], StringSplitOptions.RemoveEmptyEntries) |> Array.toList)
                |> Option.defaultValue baseSettings.ExtraArgs
            let timeout =
                match tryGetEnv "AUGGIE_CLI_TIMEOUT_SECONDS" with
                | Some text ->
                    match Double.TryParse text with
                    | true, seconds when seconds > 0.0 -> Some(TimeSpan.FromSeconds seconds)
                    | _ -> baseSettings.Timeout
                | None -> baseSettings.Timeout

            { ToolPath = toolPath
              WorkingDirectory = workingDir
              ExtraArgs = extraArgs
              Timeout = timeout }

    let private computeAverageScore (traces: ReasoningTrace list) =
        traces
        |> List.collect (fun trace -> trace.Events)
        |> List.choose (fun evt -> evt.Score)
        |> function
            | [] -> None
            | scores -> Some(scores |> List.average)

    let private collectRiskIndicators (traces: ReasoningTrace list) =
        let riskTerms =
            [| "error"
               "fail"
               "failure"
               "exception"
               "violation"
               "panic"
               "crash"
               "unsafe"
               "rollback"
               "breach" |]

        traces
        |> List.collect (fun trace -> trace.Events)
        |> List.choose (fun evt ->
            let message = evt.Message.ToLowerInvariant()
            if riskTerms |> Array.exists message.Contains then
                Some(evt.Message.Trim())
            else
                None)
        |> List.distinct

    let private heuristicCritic traces =
        let averageScore = computeAverageScore traces
        let signals = collectRiskIndicators traces

        match averageScore, signals with
        | Some score, [] when score >= 0.88 ->
            CriticVerdict.Accept
        | Some score, _ when score < 0.6 ->
            CriticVerdict.Reject(sprintf "Average reasoning confidence %.2f below minimum threshold 0.60." score)
        | _, signal :: _ ->
            let sample =
                signals
                |> List.truncate 3
                |> String.concat "; "
            CriticVerdict.Reject(sprintf "Detected high-risk reasoning indicators: %s" sample)
        | Some score, _ ->
            CriticVerdict.NeedsReview(sprintf "Average reasoning confidence %.2f insufficient for autonomous approval." score)
        | None, _ ->
            CriticVerdict.NeedsReview "Reasoning trace missing confidence scores; manual review required."

    let resolveReasoningCritic (adaptiveMemoryPath: string option) : ReasoningCriticResolution option =
        let path = adaptiveMemoryPath |> Option.defaultValue (resolveAdaptiveMemoryPath ())

        let entries =
            try
                if File.Exists(path) then
                    PersistentAdaptiveMemory.loadRecent path 256
                else
                    []
            with _ ->
                []

        match entries with
        | [] ->
            Some
                { Critic = heuristicCritic
                  Source = "heuristic:no-history"
                  Model = None }
        | _ ->
            match MetaReasoningCritic.train entries with
            | Some model ->
                Some
                    { Critic = MetaReasoningCritic.buildCritic model
                      Source = sprintf "model:%i" model.SampleSize
                      Model = Some model }
            | None ->
                Some
                    { Critic = heuristicCritic
                      Source = "heuristic:insufficient-history"
                      Model = None }

    let private computeHarnessMetrics (report: HarnessReport option) =
        match report with
        | None -> Map.empty, []
        | Some harness ->
            let addStage (label: string) (specs: CommandSpec list) (lookup: Map<string, string>) =
                specs
                |> List.fold (fun acc (spec: CommandSpec) -> acc |> Map.add spec.Name label) lookup

            let stageLookup =
                Map.empty
                |> addStage "pre-validation" harness.Config.PreValidation
                |> addStage "validation" harness.Config.Validation
                |> addStage "benchmark" harness.Config.Benchmarks

            let snapshots =
                harness.Commands
                |> List.map (fun command ->
                    let stage =
                        stageLookup
                        |> Map.tryFind command.Command.Name
                        |> Option.defaultValue "validation"

                    { CommandSnapshot.Name = command.Command.Name
                      Stage = stage
                      ExitCode = command.ExitCode
                      DurationSeconds = command.Duration.TotalSeconds
                      StartedAt = command.StartedAt
                      CompletedAt = command.CompletedAt })

            let total = snapshots.Length
            let successCount = snapshots |> List.sumBy (fun snapshot -> if snapshot.ExitCode = 0 then 1 else 0)
            let failureCount = total - successCount
            let totalDuration = snapshots |> List.sumBy (fun snapshot -> snapshot.DurationSeconds)
            let avgDuration = if total > 0 then totalDuration / float total else 0.0

            let stageDuration name =
                snapshots
                |> List.filter (fun snapshot -> snapshot.Stage.Equals(name, StringComparison.OrdinalIgnoreCase))
                |> List.sumBy (fun snapshot -> snapshot.DurationSeconds)

            let maxDuration =
                match snapshots with
                | [] -> 0.0
                | _ ->
                    snapshots
                    |> List.maxBy (fun snapshot -> snapshot.DurationSeconds)
                    |> fun snapshot -> snapshot.DurationSeconds

            let metrics =
                Map.empty
                |> Map.add "capability.command_count" (box total)
                |> Map.add "capability.success_count" (box successCount)
                |> Map.add "capability.failure_count" (box failureCount)
                |> Map.add "capability.pass_ratio" (if total > 0 then box (float successCount / float total) else box 0.0)
                |> Map.add "capability.total_duration_seconds" (box totalDuration)
                |> Map.add "capability.avg_duration_seconds" (box avgDuration)
                |> Map.add "capability.max_duration_seconds" (box maxDuration)
                |> Map.add "capability.validation_duration_seconds" (box (stageDuration "validation"))
                |> Map.add "capability.prevalidation_duration_seconds" (box (stageDuration "pre-validation"))
                |> Map.add "capability.benchmark_duration_seconds" (box (stageDuration "benchmark"))

            metrics, snapshots

    let private computeConsensusMetrics (outcome: ConsensusOutcome option) =
        match outcome with
        | None -> Map.empty
        | Some consensus ->
            let agents, status, reason =
                match consensus with
                | ConsensusPassed agents -> agents, "passed", None
                | ConsensusNeedsReview (agents, message) -> agents, "needs_review", Some message
                | ConsensusFailed (agents, message) -> agents, "failed", Some message

            let avgConfidence =
                agents
                |> List.choose (fun agent -> agent.Confidence)
                |> fun confidences ->
                    if confidences.IsEmpty then None else Some(List.average confidences)

            let failureCount =
                agents |> List.sumBy (fun agent -> if agent.Outcome = ValidationOutcome.Fail then 1 else 0)

            let needsReviewCount =
                agents |> List.sumBy (fun agent -> if agent.Outcome = ValidationOutcome.NeedsReview then 1 else 0)

            let baseMetrics =
                Map.empty
                |> Map.add "safety.consensus_status" (box status)
                |> Map.add "safety.consensus_agents" (box agents.Length)
                |> Map.add "safety.consensus_failures" (box failureCount)
                |> Map.add "safety.consensus_needs_review" (box needsReviewCount)

            let metrics =
                match avgConfidence with
                | Some value -> baseMetrics |> Map.add "safety.consensus_avg_confidence" (box value)
                | None -> baseMetrics

            match reason with
            | Some message -> metrics |> Map.add "safety.consensus_reason" (box message)
            | None -> metrics

    let private computeCriticMetrics (verdict: CriticVerdict option) (context: PersistentAdaptiveMemory.CriticContext option) =
        let baseMetrics =
            match verdict with
            | None -> Map.empty
            | Some CriticVerdict.Accept -> Map.ofList [ "safety.critic_status", box "accept" ]
            | Some (CriticVerdict.NeedsReview rationale) ->
                Map.empty
                |> Map.add "safety.critic_status" (box "needs_review")
                |> Map.add "safety.critic_note" (box rationale)
            | Some (CriticVerdict.Reject rationale) ->
                Map.empty
                |> Map.add "safety.critic_status" (box "reject")
                |> Map.add "safety.critic_note" (box rationale)

        match context with
        | None -> baseMetrics
        | Some ctx ->
            baseMetrics
            |> Map.add "safety.critic_source" (ctx.source |> Option.defaultValue "unspecified" |> box)
            |> fun metrics ->
                match ctx.threshold with
                | Some threshold -> metrics |> Map.add "safety.critic_threshold" (box threshold)
                | None -> metrics
            |> fun metrics ->
                match ctx.sampleSize with
                | Some size -> metrics |> Map.add "safety.critic_samples" (box size)
                | None -> metrics
            |> fun metrics ->
                if ctx.indicators.IsEmpty then metrics
                else metrics |> Map.add "safety.critic_indicators" (ctx.indicators |> String.concat ", " |> box)

    let private computeCriticDeltaMetrics (current: PersistentAdaptiveMemory.CriticContext option) (previous: PersistentAdaptiveMemory.CriticContext option) =
        match current, previous with
        | Some curr, Some prev ->
            Map.empty
            |> Map.add "safety.critic_threshold_previous"
                (box (
                    match prev.threshold, curr.threshold with
                    | Some value, _ -> value
                    | None, Some current -> current
                    | _ -> 0.0))
            |> Map.add "safety.critic_threshold_delta"
                (match curr.threshold, prev.threshold with
                 | Some c, Some p -> box (c - p)
                 | _ -> box 0.0)
            |> Map.add "safety.critic_samples_previous" (prev.sampleSize |> Option.defaultValue 0 |> box)
            |> Map.add "safety.critic_samples_delta"
                (match curr.sampleSize, prev.sampleSize with
                 | Some c, Some p -> box (c - p)
                 | Some c, None -> box c
                 | _ -> box 0)
            |> Map.add "safety.critic_source_previous" (prev.source |> Option.defaultValue "unknown" |> box)
            |> Map.add "safety.critic_source_changed"
                (box (curr.source <> prev.source))
        | _ -> Map.empty

    let private computeFeedbackMetrics (feedback: AgentFeedback list) =
        let approve, needsWork, reject, escalate, confidenceSum, confidenceCount =
            feedback
            |> List.fold (fun (approve, needsWork, reject, escalate, sum, count) sample ->
                let sum', count' =
                    match sample.Confidence with
                    | Some value -> sum + value, count + 1
                    | None -> sum, count

                match sample.Verdict with
                | FeedbackVerdict.Approve -> (approve + 1, needsWork, reject, escalate, sum', count')
                | FeedbackVerdict.NeedsWork _ -> (approve, needsWork + 1, reject, escalate, sum', count')
                | FeedbackVerdict.Reject _ -> (approve, needsWork, reject + 1, escalate, sum', count')
                | FeedbackVerdict.Escalate _ -> (approve, needsWork, reject, escalate + 1, sum', count')
            ) (0, 0, 0, 0, 0.0, 0)

        let baseMetrics =
            Map.empty
            |> Map.add "feedback.total" (box feedback.Length)
            |> Map.add "feedback.approve" (box approve)
            |> Map.add "feedback.needs_work" (box needsWork)
            |> Map.add "feedback.reject" (box reject)
            |> Map.add "feedback.escalate" (box escalate)

        if confidenceCount > 0 then
            baseMetrics |> Map.add "feedback.avg_confidence" (box (confidenceSum / float confidenceCount))
        else
            baseMetrics

    let private computeValidatorMetrics (snapshot: CoordinationSnapshot option) =
        match snapshot with
        | None -> Map.empty
        | Some snapshot ->
            let totalFindings = snapshot.Findings.Length
            let passCount =
                snapshot.Findings
                |> List.filter (fun finding -> finding.Outcome = ValidationOutcome.Pass)
                |> List.length
            let failCount =
                snapshot.Findings
                |> List.filter (fun finding -> finding.Outcome = ValidationOutcome.Fail)
                |> List.length
            let needsReviewCount =
                snapshot.Findings
                |> List.filter (fun finding -> finding.Outcome = ValidationOutcome.NeedsReview)
                |> List.length
            let targetCount =
                snapshot.Findings
                |> List.map (fun finding -> finding.Target.SpecId, finding.Target.IterationId, finding.Target.Topic)
                |> List.distinct
                |> List.length
            let disagreementCount = snapshot.Disagreements.Length
            let commentsCount = snapshot.Comments.Length

            let baseMetrics =
                Map.empty
                |> Map.add "validators.findings_total" (box totalFindings)
                |> Map.add "validators.findings_pass" (box passCount)
                |> Map.add "validators.findings_fail" (box failCount)
                |> Map.add "validators.findings_needs_review" (box needsReviewCount)
                |> Map.add "validators.disagreements" (box disagreementCount)
                |> Map.add "validators.comments" (box commentsCount)
                |> Map.add "validators.targets" (box targetCount)

            if targetCount = 0 then
                baseMetrics
            else
                baseMetrics
                |> Map.add "validators.disagreement_ratio" (box (float disagreementCount / float targetCount))

    let private computeAgentRoleMetrics (agents: AgentValidationResult list) =
        let tryRole role = agents |> List.tryFind (fun agent -> agent.Role = role)

        let addRoleMetrics label role acc =
            match tryRole role with
            | None -> acc
            | Some agent ->
                let withOutcome =
                    acc
                    |> Map.add ($"agents.{label}.outcome") (box (validationOutcomeToString agent.Outcome))
                let withConfidence =
                    match agent.Confidence with
                    | Some value -> withOutcome |> Map.add ($"agents.{label}.confidence") (box value)
                    | None -> withOutcome
                match agent.Notes with
                | Some notes when not (String.IsNullOrWhiteSpace notes) ->
                    withConfidence |> Map.add ($"agents.{label}.notes") (box notes)
                | _ -> withConfidence

        Map.empty
        |> addRoleMetrics "safety" AgentRole.SafetyGovernor
        |> addRoleMetrics "performance" AgentRole.PerformanceBenchmarker

    let computeLedgerMetrics
        (result: SpecDrivenIterationResult)
        (feedback: AgentFeedback list)
        (agents: AgentValidationResult list)
        (criticContext: PersistentAdaptiveMemory.CriticContext option)
        (previousCriticContext: PersistentAdaptiveMemory.CriticContext option) =
        let harnessMetrics, commandSnapshots = computeHarnessMetrics result.HarnessReport
        let evolutionMetrics = PersistentAdaptiveMemory.flattenEvolutionData result.ClosureResult.EvolutionData

        let combinedMetrics =
            [ evolutionMetrics
              harnessMetrics
              computeConsensusMetrics result.Consensus
              computeCriticMetrics result.CriticVerdict criticContext
              computeCriticDeltaMetrics criticContext previousCriticContext
              computeAgentRoleMetrics agents
              computeValidatorMetrics result.ValidatorSnapshot
              computeFeedbackMetrics feedback ]
            |> mergeMetricMaps

        combinedMetrics, commandSnapshots

    let createValidationCommand name arguments =
        { Name = name
          Executable = "dotnet"
          Arguments = arguments
          WorkingDirectory = Some Environment.CurrentDirectory
          Timeout = None
          Environment = Map.empty }

/// Interface for self-improvement service
type ISelfImprovementService =
    abstract member AnalyzeFileAsync: filePath: string -> Async<AnalysisResult option>
    abstract member AnalyzeDirectoryAsync: directoryPath: string -> Async<AnalysisResult list>
    abstract member ApplyImprovementsAsync: filePath: string * improvements: ImprovementPattern list -> Async<AppliedImprovement list>
    abstract member GetImprovementHistoryAsync: filePath: string option -> Async<AppliedImprovement list>
    abstract member RunExecutionHarnessAsync: config: HarnessConfig * ?executor: ICommandExecutor -> Async<HarnessReport>
    abstract member GetAgentFeedbackSummaryAsync: memoryPath: string * ?limit:int -> Async<PersistentAdaptiveMemory.AgentFeedbackAggregate list>
    abstract member GetRoleDirectivesAsync: memoryPath: string * ?limit:int -> Async<AgentPolicyTuning.RoleDirective list>
    abstract member DiscoverSpecKitFeaturesAsync: baseDirectory: string option -> Async<SpecKitWorkspace.SpecKitFeature list>
    abstract member DecomposeSpecKitFeatureAsync: featureId: string * ?baseDirectory: string -> Async<ProductBacklogDecomposer.DecompositionResult option>
    abstract member GetSpecKitIterationConfigAsync: featureId: string * ?baseDirectory: string * ?options: SpecKitHarnessOptions -> Async<SpecDrivenIterationConfig option>
    abstract member RunSpecKitIterationAsync: featureId: string * loggerFactory: ILoggerFactory * ?baseDirectory: string * ?options: SpecKitHarnessOptions * ?executor: ICommandExecutor -> Async<SpecDrivenIterationResult option>
    abstract member RunNextSpecKitIterationAsync: loggerFactory: ILoggerFactory * ?baseDirectory: string * ?options: SpecKitHarnessOptions * ?executor: ICommandExecutor -> Async<SpecDrivenIterationResult option>
    abstract member RunTeamCycleAsync: loggerFactory: ILoggerFactory * ?options: TeamOrchestrator.TeamCycleOptions -> Async<TeamOrchestrator.TeamCycleResult>
    abstract member PlanNextSpecKitGoalsAsync: loggerFactory: ILoggerFactory * ?topCandidates:int * ?recentMemory:int -> Async<AutonomousNextStepPlanner.PlannerRecommendation list * AutonomousNextStepPlanner.PlannerRecommendation list>
    abstract member DispatchPlannerRecommendationsAsync: loggerFactory: ILoggerFactory * ?topCandidates:int * ?recentMemory:int * ?dispatchCount:int * ?settings:AuggieCliSettings -> Async<AutonomousNextStepPlanner.PlannerRecommendation list * AutonomousNextStepPlanner.PlannerRecommendation list * AuggieDispatchResult list>
    abstract member EnsureRoadmapSpecAsync: loggerFactory: ILoggerFactory * template: RoadmapMaintenance.RoadmapSpecTemplate * ?tasks: string list * ?allowAuggieFallback: bool -> Async<bool>
    abstract member UpdateRoadmapTaskStatusAsync: loggerFactory: ILoggerFactory * specId: string * taskId: string * completed: bool -> Async<bool>

/// Main self-improvement orchestration service
type SelfImprovementService(
    httpClient: HttpClient,
    logger: ILogger<SelfImprovementService>) =
    
    let ollamaAnalyzer = OllamaCodeAnalyzer(httpClient, logger :> ILogger)
    let mutable improvementHistory: AppliedImprovement list = []
    
    /// Analyze a single file for improvements
    member this.AnalyzeFileAsync(filePath: string) =
        async {
            try
                if not (File.Exists(filePath)) then
                    logger.LogWarning("File not found: {FilePath}", filePath)
                    return None
                else
                    logger.LogInformation("Analyzing file: {FilePath}", filePath)
                    
                    let content = File.ReadAllText(filePath)
                    
                    // First, use pattern recognition
                    let patternResult = PatternRecognition.analyzeFile filePath content
                    
                    // Then, enhance with AI analysis if Ollama is available
                    let! isOllamaAvailable = ollamaAnalyzer.IsAvailableAsync()
                    
                    if isOllamaAvailable then
                        logger.LogInformation("Enhancing analysis with Ollama AI")
                        let! aiResult = ollamaAnalyzer.AnalyzeCodeAsync(filePath, content, "llama3")
                        
                        match aiResult with
                        | Some aiAnalysis ->
                            // Combine pattern recognition with AI analysis
                            let combinedIssues = patternResult.Issues @ aiAnalysis.Issues |> List.distinct
                            let combinedRecommendations = patternResult.Recommendations @ aiAnalysis.Recommendations |> List.distinct
                            let averageScore = (patternResult.OverallScore + aiAnalysis.OverallScore) / 2.0
                            
                            let enhancedResult = {
                                FilePath = filePath
                                Issues = combinedIssues
                                OverallScore = averageScore
                                Recommendations = combinedRecommendations
                                AnalyzedAt = DateTime.UtcNow
                            }
                            
                            logger.LogInformation("Enhanced analysis completed: {IssueCount} issues, score: {Score:F1}", 
                                                combinedIssues.Length, averageScore)
                            return Some enhancedResult
                        | None ->
                            logger.LogInformation("AI analysis failed, using pattern recognition only")
                            return Some patternResult
                    else
                        logger.LogInformation("Ollama not available, using pattern recognition only")
                        return Some patternResult
            with
            | ex ->
                logger.LogError(ex, "Error analyzing file: {FilePath}", filePath)
                return None
        }
    
    /// Analyze all files in a directory
    member this.AnalyzeDirectoryAsync(directoryPath: string) =
        async {
            try
                logger.LogInformation("Analyzing directory: {DirectoryPath}", directoryPath)
                
                let sourceFiles = 
                    Directory.GetFiles(directoryPath, "*", SearchOption.AllDirectories)
                    |> Array.filter (fun file -> 
                        let ext = Path.GetExtension(file).ToLower()
                        ext = ".fs" || ext = ".cs" || ext = ".fsx")
                    |> Array.toList
                
                logger.LogInformation("Found {FileCount} source files to analyze", sourceFiles.Length)
                
                // Analyze files in parallel for throughput
                let! analyzed = 
                    sourceFiles
                    |> List.map (fun filePath -> this.AnalyzeFileAsync(filePath))
                    |> Async.Parallel
                let finalResults = analyzed |> Array.choose id |> Array.toList
                logger.LogInformation("Directory analysis completed: {AnalyzedCount}/{TotalCount} files", 
                                    finalResults.Length, sourceFiles.Length)
                
                return finalResults
                
            with
            | ex ->
                logger.LogError(ex, "Error analyzing directory: {DirectoryPath}", directoryPath)
                return []
        }
    
    /// Apply improvements to a file
    member this.ApplyImprovementsAsync(filePath: string, improvements: ImprovementPattern list) =
        async {
            try
                logger.LogInformation("Applying {ImprovementCount} improvements to: {FilePath}", 
                                    improvements.Length, filePath)
                
                if not (File.Exists(filePath)) then
                    logger.LogWarning("File not found: {FilePath}", filePath)
                    return []
                else
                    let originalContent = File.ReadAllText(filePath)
                    let mutable currentContent = originalContent
                    let appliedImprovements = ResizeArray<AppliedImprovement>()
                    let backupPath = filePath + ".bak"
                    let mutable backupCreated = false

                    let ensureBackup () =
                        if not backupCreated then
                            File.WriteAllText(backupPath, originalContent)
                            logger.LogInformation("Created backup: {BackupPath}", backupPath)
                            backupCreated <- true
                    
                    for improvement in improvements do
                        try
                            match CodeTransforms.apply filePath improvement currentContent with
                            | Applied(updated, records) ->
                                ensureBackup()
                                currentContent <- updated
                                records |> List.iter appliedImprovements.Add
                                logger.LogInformation("Applied improvement: {ImprovementName}", improvement.Name)
                            | NoChange ->
                                logger.LogInformation("No matching code for improvement: {ImprovementName}", improvement.Name)
                            | NotSupported ->
                                logger.LogInformation("Skipping unsupported improvement pattern: {Pattern}", improvement.Name)
                        with
                        | ex ->
                            logger.LogError(ex, "Failed to apply improvement: {ImprovementName}", improvement.Name)
                            let failedImprovement: AppliedImprovement = {
                                FilePath = filePath
                                PatternId = improvement.Name
                                PatternName = improvement.Name
                                LineNumber = None
                                OriginalCode = ""
                                ImprovedCode = ""
                                AppliedAt = DateTime.UtcNow
                            }
                            appliedImprovements.Add(failedImprovement)

                    // Persist updated content if changed
                    if not (obj.ReferenceEquals(currentContent, originalContent)) && currentContent <> originalContent then
                        ensureBackup()
                        File.WriteAllText(filePath, currentContent)
                        logger.LogInformation("Wrote updated content to: {FilePath}", filePath)
                    
                    // Update improvement history
                    let newImprovements = appliedImprovements |> Seq.toList
                    improvementHistory <- improvementHistory @ newImprovements
                    
                    logger.LogInformation("Applied {TotalCount} improvements", newImprovements.Length)
                    
                    return newImprovements
            with
            | ex ->
                logger.LogError(ex, "Error applying improvements to: {FilePath}", filePath)
                return []
        }
    
    /// Get improvement history
    member this.GetImprovementHistoryAsync(filePath: string option) =
        async {
            match filePath with
            | Some path ->
                return improvementHistory |> List.filter (fun i -> i.FilePath = path)
            | None ->
                return improvementHistory
        }

    /// Run an execution harness with the provided configuration.
    member this.RunExecutionHarnessAsync(config: HarnessConfig, ?executor: ICommandExecutor) =
        async {
            let harnessExecutor = executor |> Option.defaultValue (ProcessCommandExecutor(logger :> ILogger) :> ICommandExecutor)
            let harness = AutonomousExecutionHarness(logger :> ILogger, harnessExecutor)
            return! harness.RunAsync(config)
        }

    member _.GetAgentFeedbackSummaryAsync(memoryPath: string, ?limit: int) =
        async {
            return summarizeFeedbackFile memoryPath limit
        }

    member this.GetRoleDirectivesAsync(memoryPath: string, ?limit: int) =
        async {
            let! aggregates = this.GetAgentFeedbackSummaryAsync(memoryPath, ?limit = limit)
            return deriveRoleDirectives aggregates
        }

    member this.DiscoverSpecKitFeaturesAsync(baseDirectory: string option) =
        async {
            let features = SpecKitWorkspace.discoverFeatures baseDirectory
            logger.LogInformation("Discovered {FeatureCount} Spec Kit feature(s)", features.Length)
            return features
        }

    member this.DecomposeSpecKitFeatureAsync(featureId: string, ?baseDirectory: string) =
        async {
            let featureOpt = SpecKitWorkspace.tryGetFeature baseDirectory featureId

            match featureOpt with
            | None ->
                logger.LogWarning("Spec Kit feature not found for backlog decomposition: {FeatureId}", featureId)
                return None
            | Some feature ->
                let teamRegistry = SelfImprovementHelpers.loadTeamRegistry ()
                let decomposition = ProductBacklogDecomposer.decomposeFeatureWithTeams feature teamRegistry
                return Some decomposition
        }

    member this.GetSpecKitIterationConfigAsync(featureId: string, ?baseDirectory: string, ?options: SpecKitHarnessOptions) =
        async {
            let featureOpt = SpecKitWorkspace.tryGetFeature baseDirectory featureId

            match featureOpt with
            | None ->
                logger.LogWarning("Spec Kit feature not found: {FeatureId}", featureId)
                return None
            | Some feature ->
                let config = SpecKitWorkspace.buildIterationConfig feature options
                return Some config
        }

    member this.RunSpecKitIterationAsync
        (
            featureId: string,
            loggerFactory: ILoggerFactory,
            ?baseDirectory: string,
            ?options: SpecKitHarnessOptions,
            ?executor: ICommandExecutor
        ) =
        async {
            let! configOpt = this.GetSpecKitIterationConfigAsync(featureId, ?baseDirectory = baseDirectory, ?options = options)

            match configOpt with
            | None -> return None
            | Some config ->
                let! result =
                    match executor with
                    | Some exec -> runIterationWithExecutor loggerFactory config exec
                    | None -> runIteration loggerFactory config

                return Some result
        }

    member this.RunNextSpecKitIterationAsync
        (
            loggerFactory: ILoggerFactory,
            ?baseDirectory: string,
            ?options: SpecKitHarnessOptions,
            ?executor: ICommandExecutor
        ) =
        async {
            let features = SpecKitWorkspace.discoverFeatures baseDirectory

            match SpecKitWorkspace.selectNextFeature features with
            | None ->
                logger.LogInformation("No pending Spec Kit tasks found. baseDirectory={BaseDirectory}", baseDirectory |> Option.defaultValue "<default>")

                if SelfImprovementHelpers.shouldAutoDispatch () then
                    let resolvedTop =
                        SelfImprovementHelpers.resolvePlannerTopCandidates None
                    let resolvedCount =
                        SelfImprovementHelpers.resolvePlannerDispatchCount None
                    let settings = SelfImprovementHelpers.resolveAuggieSettings None

                    let! _, _, dispatchResults =
                        this.DispatchPlannerRecommendationsAsync(
                            loggerFactory,
                            ?topCandidates = resolvedTop,
                            ?dispatchCount = resolvedCount,
                            ?settings = Some settings)

                    if List.isEmpty dispatchResults then
                        logger.LogInformation("Autonomous dispatcher found no planner recommendations to send.")
                    else
                        let successful = dispatchResults |> List.filter (fun result -> result.Succeeded) |> List.length
                        logger.LogInformation("Autonomous dispatcher completed: {SuccessCount}/{Total} Auggie requests succeeded.", successful, dispatchResults.Length)

                return None
            | Some selection ->
                let taskLabel = selection.Task.TaskId |> Option.defaultValue "(anonymous task)"
                logger.LogInformation("Selected Spec Kit feature {FeatureId} task {TaskId} (priority rank {Priority})", selection.Feature.Id, taskLabel, selection.PriorityRank)

                let defaultConsensusRule =
                    { MinimumPassCount = 3
                      RequiredRoles =
                        [ AgentRole.SafetyGovernor
                          AgentRole.PerformanceBenchmarker
                          AgentRole.SpecGuardian ]
                      AllowNeedsReview = false
                      MinimumConfidence = Some 0.7
                      MaxFailureCount = Some 0 }

                let initialHarnessOptions =
                    match options with
                    | Some opt -> opt
                    | None ->
                        let validationCommand =
                            SelfImprovementHelpers.createValidationCommand
                                "dotnet-test"
                                "test Tars.sln -c Release --no-build"
                        { Commands =
                            { SpecKitWorkspace.defaultHarnessCommands with
                                Validation = [ validationCommand ] }
                          Description = Some selection.Feature.Summary.Title
                          ConsensusRule = None
                          PersistAdaptiveMemory = true
                          EnableAutoPatch = true
                          RequireCriticApproval = false
                          ReasoningCritic = None }

                let initialHarnessOptions =
                    if String.Equals(selection.Feature.Id, "tier2-bootstrap", StringComparison.OrdinalIgnoreCase) then
                        let patchCommand =
                            { Name = "update-release-notes"
                              Executable = "dotnet"
                              Arguments = "fsi scripts/update_release_notes.fsx"
                              WorkingDirectory = Some Environment.CurrentDirectory
                              Timeout = None
                              Environment = Map.empty }
                        { initialHarnessOptions with
                            Commands =
                                { initialHarnessOptions.Commands with
                                    Patch = patchCommand :: initialHarnessOptions.Commands.Patch } }
                    else
                        initialHarnessOptions

                logger.LogInformation(
                    "Initial harness option command counts: patch={PatchCount} validation={ValidationCount} benchmarks={BenchmarkCount}",
                    initialHarnessOptions.Commands.Patch.Length,
                    initialHarnessOptions.Commands.Validation.Length,
                    initialHarnessOptions.Commands.Benchmarks.Length)

                let consensusRuleOption =
                    match initialHarnessOptions.ConsensusRule with
                    | Some rule -> Some rule
                    | None when selection.PriorityRank <= 1 -> Some defaultConsensusRule
                    | _ -> None

                let harnessOptions = { initialHarnessOptions with ConsensusRule = consensusRuleOption }

                let! configOpt =
                    this.GetSpecKitIterationConfigAsync(selection.Feature.Id, ?baseDirectory = baseDirectory, options = harnessOptions)

                match configOpt with
                | None -> return None
                | Some config ->
                    let adaptiveMemoryPath =
                        config.AdaptiveMemoryPath
                        |> Option.defaultValue (SelfImprovementHelpers.resolveAdaptiveMemoryPath ())

                    let criticResolution =
                        match harnessOptions.ReasoningCritic with
                        | Some injected ->
                            Some ({ Critic = injected
                                    Source = "injected"
                                    Model = None } : SelfImprovementHelpers.ReasoningCriticResolution)
                        | None when harnessOptions.RequireCriticApproval ->
                            SelfImprovementHelpers.resolveReasoningCritic (Some adaptiveMemoryPath)
                        | _ -> None

                    let criticContext =
                        criticResolution |> Option.map SelfImprovementHelpers.buildCriticContext

                    let config =
                        { config with
                            RequireConsensusForExecution = consensusRuleOption |> Option.isSome
                            RequireCriticApproval = harnessOptions.RequireCriticApproval
                            ReasoningCritic = criticResolution |> Option.map (fun r -> r.Critic)
                            ReasoningTraceProvider = config.ReasoningTraceProvider |> Option.orElse (Some SelfImprovementHelpers.defaultReasoningTraceProvider)
                            AdaptiveMemoryPath = Some adaptiveMemoryPath
                            AutoApplyPatchArtifacts = harnessOptions.EnableAutoPatch }

                    logger.LogInformation(
                        "Harness command counts: patch={PatchCount} validation={ValidationCount} benchmarks={BenchmarkCount}",
                        config.PatchCommands.Length,
                        config.ValidationCommands.Length,
                        config.BenchmarkCommands.Length)

                    criticResolution
                    |> Option.iter (fun resolution ->
                        logger.LogInformation("Activating reasoning critic source={Source}", resolution.Source))

                    let iterationId = Guid.NewGuid()

                    let! result =
                        match executor with
                        | Some exec -> runIterationWithExecutor loggerFactory config exec
                        | None -> runIteration loggerFactory config

                    let! tier3Outcomes, tier3Metrics =
                        if String.Equals(selection.Feature.Id, "tier3-cross-validation", StringComparison.OrdinalIgnoreCase) then
                            SelfImprovementHelpers.runTier3Metascripts loggerFactory None
                        else
                            async { return [], Map.empty<string, obj> }

                    let tier3ResultMap =
                        tier3Outcomes
                        |> List.map (fun outcome -> outcome.Result.Role, outcome.Result)
                        |> Map.ofList

                    let tier3CriticOverride =
                        tier3Outcomes
                        |> List.choose (fun outcome -> outcome.CriticVerdict)
                        |> List.tryHead

                    let tier3Traces = tier3Outcomes |> List.map (fun outcome -> outcome.Trace)

                    let existingTraces = result.ReasoningTraces |> Option.defaultValue []

                    let baseTraces =
                        if tier3Traces.IsEmpty then existingTraces
                        else existingTraces @ tier3Traces

                    let combinedCriticVerdict =
                        tier3CriticOverride |> Option.orElse result.CriticVerdict

                    let combinedEvolution =
                        SelfImprovementHelpers.mergeMetricMaps [ result.ClosureResult.EvolutionData; tier3Metrics ]

                    let closureWithTier3 =
                        { result.ClosureResult with EvolutionData = combinedEvolution }

                    let result =
                        { result with
                            ClosureResult = closureWithTier3
                            CriticVerdict = combinedCriticVerdict }

                    let previousCriticContext =
                        SelfImprovementHelpers.tryLoadPreviousCriticContext config.AdaptiveMemoryPath selection.Feature.Id

                    let overrideResult role defaultResult =
                        tier3ResultMap |> Map.tryFind role |> Option.defaultValue defaultResult

                    let reasonerResult =
                        { AgentId = "agent:reasoner"
                          Role = AgentRole.Reasoner
                          Outcome = if result.ClosureResult.Success then ValidationOutcome.Pass else ValidationOutcome.Fail
                          Confidence = Some 0.85
                          Notes = Some "Metascript closure analysis"
                          ProducedAt = DateTime.UtcNow }

                    let harnessStatus =
                        result.HarnessReport
                        |> Option.map (fun report ->
                            match report.Outcome with
                            | HarnessOutcome.AllPassed _ -> true
                            | HarnessOutcome.Failed _ -> false)

                    let reviewerOutcome =
                        match harnessStatus with
                        | Some true -> ValidationOutcome.Pass
                        | Some false -> ValidationOutcome.Fail
                        | None -> ValidationOutcome.NeedsReview

                    let reviewerResult =
                        { AgentId = "agent:reviewer"
                          Role = AgentRole.Reviewer
                          Outcome = reviewerOutcome
                          Confidence =
                            match reviewerOutcome with
                            | ValidationOutcome.Pass -> Some 0.9
                            | ValidationOutcome.Fail -> Some 0.3
                            | ValidationOutcome.NeedsReview -> Some 0.5
                          Notes = Some "Harness validation review"
                          ProducedAt = DateTime.UtcNow }

                    let hasBenchmarks = not config.BenchmarkCommands.IsEmpty

                    let performanceOutcome, performanceConfidence, performanceNotes =
                        match harnessStatus, hasBenchmarks with
                        | Some true, true -> ValidationOutcome.Pass, Some 0.8, Some "Benchmarks executed with passing harness."
                        | Some true, false -> ValidationOutcome.Pass, Some 0.72, Some "No performance benchmarks configured; passing with reduced confidence."
                        | Some false, _ -> ValidationOutcome.Fail, Some 0.3, Some "Harness failures block performance confidence."
                        | None, true -> ValidationOutcome.NeedsReview, Some 0.4, Some "Benchmark results unavailable; requires review."
                        | None, false -> ValidationOutcome.NeedsReview, Some 0.4, Some "Performance metrics absent."

                    let performanceResult =
                        { AgentId = "agent:performance"
                          Role = AgentRole.PerformanceBenchmarker
                          Outcome = performanceOutcome
                          Confidence = performanceConfidence
                          Notes = performanceNotes
                          ProducedAt = DateTime.UtcNow }
                        |> overrideResult AgentRole.PerformanceBenchmarker
                        |> SelfImprovementHelpers.overrideAgentResultFromEnv "TARS_FORCE_PERFORMANCE_STATUS" "performance"

                    let safetyOutcome, safetyConfidence, safetyNotes =
                        match harnessStatus, result.CriticVerdict with
                        | Some false, _ -> ValidationOutcome.Fail, Some 0.2, Some "Harness failures detected."
                        | _, Some (CriticVerdict.Reject rationale) ->
                            ValidationOutcome.Fail,
                            Some 0.15,
                            Some($"Critic rejected iteration: {rationale}")
                        | _, Some (CriticVerdict.NeedsReview rationale) ->
                            ValidationOutcome.NeedsReview,
                            Some 0.45,
                            Some($"Critic requested review: {rationale}")
                        | Some true, Some CriticVerdict.Accept ->
                            ValidationOutcome.Pass,
                            Some 0.88,
                            Some "Harness and critic sign-off achieved."
                        | Some true, None ->
                            ValidationOutcome.Pass,
                            Some 0.72,
                            Some "Harness passed without critic objections."
                        | None, _ ->
                            ValidationOutcome.NeedsReview,
                            Some 0.5,
                            Some "Safety status indeterminate; awaiting signals."

                    let safetyResult =
                        { AgentId = "agent:safety-governor"
                          Role = AgentRole.SafetyGovernor
                          Outcome = safetyOutcome
                          Confidence = safetyConfidence
                          Notes = safetyNotes
                          ProducedAt = DateTime.UtcNow }
                        |> overrideResult AgentRole.SafetyGovernor
                        |> SelfImprovementHelpers.overrideAgentResultFromEnv "TARS_FORCE_SAFETY_STATUS" "safety"

                    let hasArtifacts = not (result.ClosureResult.Artifacts |> List.isEmpty)

                    let specGuardianOutcome, specGuardianConfidence, specGuardianNotes =
                        if hasArtifacts then
                            ValidationOutcome.Pass, Some 0.78, Some "Candidate patches generated for spec review."
                        elif result.ClosureResult.Success then
                            ValidationOutcome.NeedsReview, Some 0.5, Some "No artifacts emitted; manual spec verification required."
                        else
                            ValidationOutcome.Fail, Some 0.25, Some "Closure execution failed; spec impact unknown."

                    let guardianResult =
                        { AgentId = "agent:spec-guardian"
                          Role = AgentRole.SpecGuardian
                          Outcome = specGuardianOutcome
                          Confidence = specGuardianConfidence
                          Notes = specGuardianNotes
                          ProducedAt = DateTime.UtcNow }

                    let criticResultOpt =
                        tier3ResultMap |> Map.tryFind AgentRole.MetaCritic

                    let toFeedback (result: AgentValidationResult) : AgentFeedback =
                        let verdict =
                            match result.Outcome with
                            | ValidationOutcome.Pass -> Approve
                            | ValidationOutcome.Fail -> Reject "Validation failure detected."
                            | ValidationOutcome.NeedsReview -> NeedsWork "Outcome requires review."

                        { AgentId = result.AgentId
                          Role = result.Role
                          Verdict = verdict
                          Confidence = result.Confidence
                          Notes = result.Notes
                          SuggestedActions =
                            match verdict with
                            | Reject _ -> [ "Initiate rollback"; "Schedule manual review" ]
                            | NeedsWork _ -> [ "Collect additional diagnostics" ]
                            | _ -> []
                          RecordedAt = DateTime.UtcNow }

                    let agentResults =
                        [ reasonerResult
                          reviewerResult
                          performanceResult
                          safetyResult
                          guardianResult ]
                        |> fun results ->
                            match criticResultOpt with
                            | Some critic -> critic :: results
                            | None -> results

                    let feedback = agentResults |> List.map toFeedback

                    let harnessPassed = harnessStatus |> Option.defaultValue false
                    let criticStatusLabel = SelfImprovementHelpers.criticVerdictToString result.CriticVerdict
                    let criticSourceLabel =
                        criticContext
                        |> Option.bind (fun ctx -> ctx.source)
                        |> Option.defaultValue "unspecified"

                    let roleTraces =
                        [ SelfImprovementHelpers.buildAgentTrace
                              performanceResult
                              "Performance benchmarking assessment"
                              "agent/performance"
                              [ "harness.passed", box harnessPassed
                                "benchmarks.enabled", box hasBenchmarks
                                "benchmarks.count", box config.BenchmarkCommands.Length ]
                          SelfImprovementHelpers.buildAgentTrace
                              safetyResult
                              "Safety governor validation"
                              "agent/safety"
                              [ "harness.passed", box harnessPassed
                                "critic.status", box criticStatusLabel
                                "critic.source", box criticSourceLabel ] ]

                    let remediationTraces =
                        feedback
                        |> List.choose SelfImprovementHelpers.buildRemediationTrace

                    let finalTraces =
                        baseTraces
                        |> fun traces -> if roleTraces.IsEmpty then traces else traces @ roleTraces
                        |> fun traces -> if remediationTraces.IsEmpty then traces else traces @ remediationTraces

                    let result =
                        { result with ReasoningTraces = Some finalTraces }

                    let previousCriticContext =
                        SelfImprovementHelpers.tryLoadPreviousCriticContext config.AdaptiveMemoryPath selection.Feature.Id

                    let consensusRule =
                        consensusRuleOption |> Option.defaultValue defaultConsensusRule

                    let consensusOutcome = evaluate consensusRule agentResults

                    let validatorTarget =
                        { SpecId = selection.Feature.Id
                          IterationId = Some iterationId
                          Topic = Some "spec-kit" }

                    let toValidatorFinding (result: AgentValidationResult) =
                        { FindingId = Guid.NewGuid()
                          AgentId = result.AgentId
                          Role = result.Role
                          Outcome = result.Outcome
                          Confidence = result.Confidence
                          Notes = result.Notes
                          Target = validatorTarget
                          RecordedAt = result.ProducedAt }

                    let coordinationLogger = loggerFactory.CreateLogger("ValidatorCoordination")
                    let coordinationBus = CoordinationBus coordinationLogger
                    let validatorFindings = agentResults |> List.map toValidatorFinding
                    validatorFindings |> List.iter coordinationBus.PublishFinding

                    let initialValidatorSnapshot = coordinationBus.Snapshot()

                    let generatedComments =
                        initialValidatorSnapshot.Disagreements
                        |> List.collect (fun disagreement ->
                            disagreement.Outcomes
                            |> List.filter (fun (_, outcome) ->
                                outcome = ValidationOutcome.Fail || outcome = ValidationOutcome.NeedsReview)
                            |> List.choose (fun (role, outcome) ->
                                validatorFindings
                                |> List.tryFind (fun finding -> finding.Role = role)
                                |> Option.map (fun finding ->
                                    let descriptor =
                                        match outcome with
                                        | ValidationOutcome.Fail -> "failing condition"
                                        | ValidationOutcome.NeedsReview -> "pending review"
                                        | _ -> "note"
                                    { CommentId = Guid.NewGuid()
                                      FindingId = finding.FindingId
                                      AuthorId = "agent:coordination-lead"
                                      AuthorRole = AgentRole.CoordinationLead
                                      Message = $"Coordinator escalation: {role} reported a {descriptor} for {disagreement.Target.SpecId}."
                                      RecordedAt = DateTime.UtcNow })))

                    generatedComments |> List.iter coordinationBus.PublishComment

                    let validatorSnapshot =
                        if generatedComments.IsEmpty then
                            initialValidatorSnapshot
                        else
                            coordinationBus.Snapshot()

                    let feedback = agentResults |> List.map toFeedback

                    let validatorSnapshotOption = Some validatorSnapshot

                    let enrichedResult =
                        { result with
                            Consensus = Some consensusOutcome
                            CrossAgentFeedback = Some feedback
                            ValidatorSnapshot = validatorSnapshotOption }

                    let status =
                        enrichedResult.HarnessReport
                        |> Option.map (fun h ->
                            match h.Outcome with
                            | HarnessOutcome.AllPassed _ -> "passed"
                            | HarnessOutcome.Failed (_, reason) when String.IsNullOrWhiteSpace(reason) -> "failed"
                            | HarnessOutcome.Failed (_, reason) -> $"failed:{reason}")
                        |> Option.defaultValue "unknown"

                    let criticRegression =
                        match criticContext, previousCriticContext with
                        | Some current, Some prev ->
                            match current.threshold, prev.threshold with
                            | Some curr, Some prior when curr + 0.03 < prior -> true
                            | _ -> false
                        | _ -> false

                    if criticRegression then
                        logger.LogWarning(
                            "Critic threshold regressed from {Previous} to {Current}; scheduling remediation goal.",
                            previousCriticContext |> Option.bind (fun ctx -> ctx.threshold) |> Option.defaultValue Double.NaN,
                            criticContext |> Option.bind (fun ctx -> ctx.threshold) |> Option.defaultValue Double.NaN)

                    if harnessOptions.PersistAdaptiveMemory then
                        let snapshot = SelfImprovementHelpers.policySnapshotFromConfig config
                        let entry: PersistentAdaptiveMemory.MemoryEntry =
                            { runId = iterationId
                              specId = selection.Feature.Id
                              specPath = selection.Feature.SpecPath
                              description = config.Description
                              timestamp = DateTime.UtcNow
                              consensus = PersistentAdaptiveMemory.captureConsensus enrichedResult.Consensus
                              critic = PersistentAdaptiveMemory.captureCritic enrichedResult.CriticVerdict criticContext
                              reasoning = enrichedResult.ReasoningTraces |> Option.defaultValue [] |> PersistentAdaptiveMemory.captureReasoning
                              policyBefore = snapshot
                              policyAfter = snapshot
                              policyChanges = []
                              inferenceTelemetry = PersistentAdaptiveMemory.captureTelemetry enrichedResult.ClosureResult.EvolutionData
                              agentFeedback = feedback |> PersistentAdaptiveMemory.captureAgentFeedback
                              validatorFindings =
                                  validatorSnapshotOption
                                  |> Option.map (fun snapshot -> snapshot.Findings)
                                  |> Option.defaultValue []
                                  |> PersistentAdaptiveMemory.captureValidatorFindings
                              validatorDisagreements =
                                  validatorSnapshotOption
                                  |> Option.map (fun snapshot -> snapshot.Disagreements)
                                  |> Option.defaultValue []
                                  |> PersistentAdaptiveMemory.captureValidatorDisagreements
                              harness = PersistentAdaptiveMemory.captureHarness enrichedResult.HarnessReport }

                        PersistentAdaptiveMemory.append adaptiveMemoryPath entry
                        let ledgerLogger = loggerFactory.CreateLogger("GovernanceLedger")
                        let metrics, commandSnapshots =
                            SelfImprovementHelpers.computeLedgerMetrics
                                enrichedResult
                                feedback
                                agentResults
                                criticContext
                                previousCriticContext
                        let remediationArtifacts =
                            SelfImprovementHelpers.createRemediationArtifact iterationId selection.Feature.Id harnessStatus feedback
                            |> Option.map (fun artifact ->
                                SelfImprovementHelpers.appendRemediationTask selection.Feature.TasksPath selection.Feature.Id iterationId artifact
                                artifact.Path)
                            |> Option.toList

                        let artifacts = (remediationArtifacts @ enrichedResult.ClosureResult.Artifacts) |> List.distinct
                        let nextSteps = enrichedResult.ClosureResult.NextSteps |> List.distinct
                        GovernanceLedger.recordIteration ledgerLogger entry status metrics artifacts commandSnapshots nextSteps

                    let criticRegression =
                        match criticContext, previousCriticContext with
                        | Some current, Some prev ->
                            match current.threshold, prev.threshold with
                            | Some curr, Some old when curr + 0.03 < old -> true
                            | _ -> false
                        | _ -> false

                    if criticRegression then
                        logger.LogWarning(
                            "Critic threshold regressed from {Previous} to {Current}; forcing remediation goal.",
                            previousCriticContext |> Option.bind (fun ctx -> ctx.threshold) |> Option.defaultValue Double.NaN,
                            criticContext |> Option.bind (fun ctx -> ctx.threshold) |> Option.defaultValue Double.NaN)

                    let shouldGenerateGoal =
                        status.StartsWith("failed", StringComparison.OrdinalIgnoreCase)
                        || match enrichedResult.Consensus with
                           | Some (ConsensusFailed _)
                           | Some (ConsensusNeedsReview _) -> true
                           | _ -> false
                        || criticRegression
                        || criticRegression

                    if shouldGenerateGoal then
                        SpecKitGoalPlanner.recordGoal selection (Some(Guid.NewGuid()))

                    logger.LogInformation("Spec Kit iteration completed. feature={FeatureId} status={Status}", selection.Feature.Id, status)

                    return Some enrichedResult
        }

    member this.RunTeamCycleAsync(loggerFactory: ILoggerFactory, ?options: TeamOrchestrator.TeamCycleOptions) =
        async {
            let resolved = defaultArg options TeamOrchestrator.TeamCycleOptions.Default
            let dependencies: TeamOrchestrator.TeamCycleDependencies =
                { PlanNext = fun logger top recent -> this.PlanNextSpecKitGoalsAsync(logger, ?topCandidates = top, ?recentMemory = recent)
                  RunIteration = fun logger baseDir harness executor -> this.RunNextSpecKitIterationAsync(logger, ?baseDirectory = baseDir, ?options = harness, ?executor = executor) }
            return! TeamOrchestrator.runCycleAsync loggerFactory dependencies resolved
        }

    member this.PlanNextSpecKitGoalsAsync(loggerFactory: ILoggerFactory, ?topCandidates: int, ?recentMemory: int) =
        async {
            let plannerLogger = loggerFactory.CreateLogger("AutonomousNextStepPlanner")
            let memoryPath = SelfImprovementHelpers.resolveAdaptiveMemoryPath ()

            let baselineConfig = AutonomousNextStepPlanner.defaultConfig
            let plannerConfig =
                { baselineConfig with
                    TopCandidates = defaultArg topCandidates baselineConfig.TopCandidates
                    RecentMemoryLimit = defaultArg recentMemory baselineConfig.RecentMemoryLimit
                    MemoryPath = Some memoryPath }

            return! AutonomousNextStepPlanner.planAndEnqueue plannerLogger (Some plannerConfig)
        }

    member this.DispatchPlannerRecommendationsAsync(loggerFactory: ILoggerFactory, ?topCandidates: int, ?recentMemory: int, ?dispatchCount: int, ?settings: AuggieCliSettings) =
        async {
            let plannerLogger = loggerFactory.CreateLogger("AutonomousNextStepPlanner")
            let memoryPath = SelfImprovementHelpers.resolveAdaptiveMemoryPath ()

            let plannerBaseline = AutonomousNextStepPlanner.defaultConfig
            let resolvedTop =
                SelfImprovementHelpers.resolvePlannerTopCandidates topCandidates
                |> Option.defaultValue plannerBaseline.TopCandidates

            let plannerConfig =
                { plannerBaseline with
                    TopCandidates = resolvedTop
                    RecentMemoryLimit = defaultArg recentMemory plannerBaseline.RecentMemoryLimit
                    MemoryPath = Some memoryPath }

            let! recommendations, enqueued = AutonomousNextStepPlanner.planAndEnqueue plannerLogger (Some plannerConfig)

            let dispatchLimit =
                SelfImprovementHelpers.resolvePlannerDispatchCount dispatchCount
                |> Option.filter (fun value -> value > 0)
                |> Option.defaultValue 1

            let toSend = recommendations |> List.truncate dispatchLimit
            let dispatchLogger = loggerFactory.CreateLogger("AuggieDispatch")
            let cliSettings = SelfImprovementHelpers.resolveAuggieSettings settings

            let buildInstruction (recommendation: AutonomousNextStepPlanner.PlannerRecommendation) =
                let feature = recommendation.Selection.Feature
                let task = recommendation.Selection.Task
                let summary = feature.Summary
                let priorityLabel = task.Priority |> Option.defaultValue "P?"
                let formattedScore = recommendation.Score.ToString("F3")
                let formattedPriorityWeight = recommendation.PriorityWeight.ToString("F2")
                let formattedSimilarity = recommendation.SimilarityScore.ToString("F2")
                let failureSignalValue = recommendation.FailureSignal.ToString()

                let acceptanceBlock =
                    summary.UserStories
                    |> List.collect (fun story -> story.AcceptanceCriteria)
                    |> function
                        | [] -> [ "Execute the associated metascript and ensure autonomous harness validation passes." ]
                        | values -> values
                    |> List.mapi (fun idx item -> $"{idx + 1}. {item}")
                    |> String.concat Environment.NewLine

                let edgeCasesBlock =
                    match summary.EdgeCases with
                    | [] -> "None documented."
                    | cases -> cases |> List.map (fun case -> $"- {case}") |> String.concat Environment.NewLine

                let rationaleBlock =
                    recommendation.Rationale
                    |> List.map (fun reason -> $"- {reason}")
                    |> String.concat Environment.NewLine

                let instructionLines =
                    [ "## Autonomous Improvement Request"
                      $"Spec: {feature.Id}"
                      $"Task: {task.Description}"
                      $"Priority: {priorityLabel}"
                      $"Spec Path: {feature.SpecPath}"
                      $"Recommendation Score: {formattedScore} (priority weight {formattedPriorityWeight}, similarity {formattedSimilarity}, failure signal {failureSignalValue})"
                      ""
                      "### Acceptance Checklist"
                      acceptanceBlock
                      ""
                      "### Edge Cases"
                      edgeCasesBlock
                      ""
                      "### Planner Rationale"
                      rationaleBlock
                      ""
                      "### Action"
                      "Generate the necessary code patches, tests, and validation steps to complete this task. Provide concise instructions or apply patches directly if the environment allows." ]

                instructionLines |> String.concat Environment.NewLine

            let dispatchRequests =
                toSend
                |> List.map (fun recommendation ->
                    let specId = recommendation.Selection.Feature.Id
                    let priorityLabel = recommendation.Selection.Task.Priority |> Option.defaultValue "P?"
                    let formattedScore = recommendation.Score.ToString("F3")
                    let formattedPriorityWeight = recommendation.PriorityWeight.ToString("F2")
                    let formattedSimilarity = recommendation.SimilarityScore.ToString("F2")
                    let failureSignalValue = recommendation.FailureSignal.ToString()
                    { AuggieDispatchRequest.Title = Some $"Spec {recommendation.Selection.Feature.Id}"
                      Instruction = buildInstruction recommendation
                      Metadata =
                        [ "specId", specId
                          "taskPriority", priorityLabel
                          "score", formattedScore
                          "priorityWeight", formattedPriorityWeight
                          "similarity", formattedSimilarity
                          "failureSignal", failureSignalValue ]
                        |> Map.ofList })

            let! dispatchResults =
                dispatchRequests
                |> List.map (fun request -> AuggieIntegration.dispatchAsync dispatchLogger cliSettings request)
                |> Async.Parallel

            dispatchResults
            |> Array.iter (fun result ->
                let specId = result.Request.Metadata |> Map.tryFind "specId" |> Option.defaultValue "<unknown>"
                if result.Succeeded then
                    dispatchLogger.LogInformation("Auggie CLI completed successfully for {SpecId}.", specId)
                else
                    let errorMessage = result.ErrorMessage |> Option.defaultValue "no details"
                    dispatchLogger.LogWarning("Auggie CLI failed (exit code {ExitCode}) for {SpecId}: {Error}", result.ExitCode, specId, errorMessage))

            return recommendations, enqueued, dispatchResults |> Array.toList
        }

    member this.EnsureRoadmapSpecAsync(loggerFactory: ILoggerFactory, template: RoadmapMaintenance.RoadmapSpecTemplate, ?tasks: string list, ?allowAuggieFallback: bool) =
        async {
            let logger = loggerFactory.CreateLogger("RoadmapMaintenance")
            try
                RoadmapMaintenance.ensureRoadmap logger template tasks |> ignore
                return true
            with ex ->
                logger.LogError(ex, "Failed to ensure roadmap spec {SpecId}.", template.Id)
                let fallbackAllowed =
                    defaultArg allowAuggieFallback (SelfImprovementHelpers.shouldAutoDispatch ())
                if fallbackAllowed then
                    let dispatchLogger = loggerFactory.CreateLogger("AuggieFallback")
                    let cliSettings = SelfImprovementHelpers.resolveAuggieSettings None
                    let storiesSummary =
                        template.Stories
                        |> List.map (fun story -> $"- {story.Title} (Priority {story.Priority})")
                        |> String.concat Environment.NewLine
                    let instruction =
                        [ "## Roadmap Maintenance Assistance"
                          $"Spec Id: {template.Id}"
                          $"Title: {template.Title}"
                          $"Existing Stories:"
                          storiesSummary
                          ""
                          "Please create or repair the roadmap spec within `.specify/specs` using the supplied context."
                          ""
                          "Failure Details:"
                          ex.ToString() ]
                        |> String.concat Environment.NewLine
                    let request =
                        { AuggieDispatchRequest.Title = Some $"Repair roadmap spec {template.Id}"
                          Instruction = instruction
                          Metadata = Map.ofList [ "specId", template.Id; "action", "ensureRoadmap" ] }
                    let! result = AuggieIntegration.dispatchAsync dispatchLogger cliSettings request
                    return result.Succeeded
                else
                    return false
        }

    member this.UpdateRoadmapTaskStatusAsync(loggerFactory: ILoggerFactory, specId: string, taskId: string, completed: bool) =
        async {
            let logger = loggerFactory.CreateLogger("RoadmapMaintenance")
            try
                let success = RoadmapMaintenance.setTaskStatus logger specId taskId completed
                if success then
                    return true
                else if SelfImprovementHelpers.shouldAutoDispatch () then
                    let dispatchLogger = loggerFactory.CreateLogger("AuggieFallback")
                    let cliSettings = SelfImprovementHelpers.resolveAuggieSettings None
                    let statusLabel = if completed then "completed" else "open"
                    let instruction =
                        [ "## Roadmap Task Update Assistance"
                          $"Spec Id: {specId}"
                          $"Task Id: {taskId}"
                          $"Desired Status: {statusLabel}"
                          ""
                          "The automated updater could not locate the task entry. Please adjust the tasks file under `.specify/specs` accordingly."
                          "" ]
                        |> String.concat Environment.NewLine
                    let request =
                        { AuggieDispatchRequest.Title = Some $"Update roadmap task {specId}:{taskId}"
                          Instruction = instruction
                          Metadata = Map.ofList [ "specId", specId; "taskId", taskId; "action", "updateTask"; "completed", completed.ToString() ] }
                    let! result = AuggieIntegration.dispatchAsync dispatchLogger cliSettings request
                    return result.Succeeded
                else
                    return false
            with ex ->
                logger.LogError(ex, "Unexpected failure updating roadmap task {SpecId}:{TaskId}", specId, taskId)
                if SelfImprovementHelpers.shouldAutoDispatch () then
                    let dispatchLogger = loggerFactory.CreateLogger("AuggieFallback")
                    let cliSettings = SelfImprovementHelpers.resolveAuggieSettings None
                    let statusLabel = if completed then "completed" else "open"
                    let instruction =
                        [ "## Roadmap Task Update Assistance"
                          $"Spec Id: {specId}"
                          $"Task Id: {taskId}"
                          $"Desired Status: {statusLabel}"
                          ""
                          "An exception occurred while attempting to update this task automatically. Please correct the tasks file."
                          ""
                          "Error:"
                          ex.ToString() ]
                        |> String.concat Environment.NewLine
                    let request =
                        { AuggieDispatchRequest.Title = Some $"Repair roadmap task {specId}:{taskId}"
                          Instruction = instruction
                          Metadata = Map.ofList [ "specId", specId; "taskId", taskId; "action", "updateTask"; "completed", completed.ToString() ] }
                    let! result = AuggieIntegration.dispatchAsync dispatchLogger cliSettings request
                    return result.Succeeded
                else
                    return false
        }
    
    interface ISelfImprovementService with
        member this.AnalyzeFileAsync(filePath) = this.AnalyzeFileAsync(filePath)
        member this.AnalyzeDirectoryAsync(directoryPath) = this.AnalyzeDirectoryAsync(directoryPath)
        member this.ApplyImprovementsAsync(filePath, improvements) = this.ApplyImprovementsAsync(filePath, improvements)
        member this.GetImprovementHistoryAsync(filePath) = this.GetImprovementHistoryAsync(filePath)
        member this.RunExecutionHarnessAsync(config, ?executor) = this.RunExecutionHarnessAsync(config, ?executor = executor)
        member this.GetAgentFeedbackSummaryAsync(memoryPath, ?limit) = this.GetAgentFeedbackSummaryAsync(memoryPath, ?limit = limit)
        member this.GetRoleDirectivesAsync(memoryPath, ?limit) = this.GetRoleDirectivesAsync(memoryPath, ?limit = limit)
        member this.DiscoverSpecKitFeaturesAsync(baseDirectory) = this.DiscoverSpecKitFeaturesAsync(baseDirectory)
        member this.DecomposeSpecKitFeatureAsync(featureId, ?baseDirectory) =
            this.DecomposeSpecKitFeatureAsync(featureId, ?baseDirectory = baseDirectory)
        member this.GetSpecKitIterationConfigAsync(featureId, ?baseDirectory, ?options) =
            this.GetSpecKitIterationConfigAsync(featureId, ?baseDirectory = baseDirectory, ?options = options)
        member this.RunSpecKitIterationAsync(featureId, loggerFactory, ?baseDirectory, ?options, ?executor) =
            this.RunSpecKitIterationAsync(featureId, loggerFactory, ?baseDirectory = baseDirectory, ?options = options, ?executor = executor)
        member this.RunNextSpecKitIterationAsync(loggerFactory, ?baseDirectory, ?options, ?executor) =
            this.RunNextSpecKitIterationAsync(loggerFactory, ?baseDirectory = baseDirectory, ?options = options, ?executor = executor)
        member this.RunTeamCycleAsync(loggerFactory, ?options) =
            this.RunTeamCycleAsync(loggerFactory, ?options = options)
        member this.PlanNextSpecKitGoalsAsync(loggerFactory, ?topCandidates, ?recentMemory) =
            this.PlanNextSpecKitGoalsAsync(loggerFactory, ?topCandidates = topCandidates, ?recentMemory = recentMemory)
        member this.DispatchPlannerRecommendationsAsync(loggerFactory, ?topCandidates, ?recentMemory, ?dispatchCount, ?settings) =
            this.DispatchPlannerRecommendationsAsync(loggerFactory, ?topCandidates = topCandidates, ?recentMemory = recentMemory, ?dispatchCount = dispatchCount, ?settings = settings)
        member this.EnsureRoadmapSpecAsync(loggerFactory, template, ?tasks, ?allowAuggieFallback) =
            this.EnsureRoadmapSpecAsync(loggerFactory, template, ?tasks = tasks, ?allowAuggieFallback = allowAuggieFallback)
        member this.UpdateRoadmapTaskStatusAsync(loggerFactory, specId, taskId, completed) =
            this.UpdateRoadmapTaskStatusAsync(loggerFactory, specId, taskId, completed)







