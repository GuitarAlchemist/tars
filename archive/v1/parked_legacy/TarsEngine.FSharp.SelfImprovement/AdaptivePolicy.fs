namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Text.Json
open TarsEngine.FSharp.Core.Services.CrossAgentValidation
open TarsEngine.FSharp.Core.Services.ReasoningTrace
open TarsEngine.FSharp.SelfImprovement
open PersistentAdaptiveMemory
open MetaReasoningCritic
open CrossAgentFeedback
open AutonomousSpecHarness

/// Adaptive heuristics for updating autonomous execution settings based on feedback.
module AdaptivePolicy =
    type StoredFeedback = { consensus: string option; critic: string option; timestamp: DateTime }

    type FeedbackRecord =
        { Consensus: ConsensusOutcome option
          CriticVerdict: CriticVerdict option }

    let private consensusToLabel = function
        | None -> None
        | Some (ConsensusPassed _) -> Some "passed"
        | Some (ConsensusNeedsReview _) -> Some "needs_review"
        | Some (ConsensusFailed _) -> Some "failed"

    let private criticToLabel = function
        | None -> None
        | Some CriticVerdict.Accept -> Some "accept"
        | Some (CriticVerdict.NeedsReview _) -> Some "needs_review"
        | Some (CriticVerdict.Reject _) -> Some "reject"

    let private labelToConsensus = function
        | Some "failed" -> Some (ConsensusFailed ([], "persisted"))
        | Some "needs_review" -> Some (ConsensusNeedsReview ([], "persisted"))
        | Some "passed" -> Some (ConsensusPassed [])
        | _ -> None

    let private labelToCritic = function
        | Some "accept" -> Some CriticVerdict.Accept
        | Some "needs_review" -> Some (CriticVerdict.NeedsReview "persisted")
        | Some "reject" -> Some (CriticVerdict.Reject "persisted")
        | _ -> None

    let private loadStored path =
        if String.IsNullOrWhiteSpace(path) || not (File.Exists(path)) then []
        else
            let json = File.ReadAllText(path)
            if String.IsNullOrWhiteSpace(json) then []
            else JsonSerializer.Deserialize<StoredFeedback list>(json)

    let private loadHistory path =
        loadStored path
        |> List.map (fun entry ->
            ({ Consensus = labelToConsensus entry.consensus
               CriticVerdict = labelToCritic entry.critic } : FeedbackRecord))

    let private appendHistory path (record: FeedbackRecord) =
        if String.IsNullOrWhiteSpace(path) then ()
        else
            let storedRecord =
                { consensus = consensusToLabel record.Consensus
                  critic = criticToLabel record.CriticVerdict
                  timestamp = DateTime.UtcNow }

            let updatedStored = loadStored path @ [ storedRecord ]
            let options = JsonSerializerOptions(WriteIndented = true)
            File.WriteAllText(path, JsonSerializer.Serialize(updatedStored, options))

    let private defaultConsensusRule =
        { MinimumPassCount = 2
          RequiredRoles = [ AgentRole.Reasoner; AgentRole.Reviewer ]
          AllowNeedsReview = false
          MinimumConfidence = Some 0.6
          MaxFailureCount = Some 0 }

    let private capturePolicySnapshot (config: SpecDrivenIterationConfig) =
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
          consensusRule = captureConsensusRule config.ConsensusRule }

    let private resolveSpecIdentifier (config: SpecDrivenIterationConfig) (result: AutonomousSpecHarness.SpecDrivenIterationResult) =
        let fromPath =
            if String.IsNullOrWhiteSpace(config.SpecPath) then None
            else
                let candidate = Path.GetFileNameWithoutExtension(config.SpecPath)
                if String.IsNullOrWhiteSpace(candidate) then None else Some candidate

        fromPath |> Option.defaultValue result.SpecSummary.Title

    let private resolveSpecPath (config: SpecDrivenIterationConfig) =
        if String.IsNullOrWhiteSpace(config.SpecPath) then ""
        else
            try
                Path.GetFullPath(config.SpecPath)
            with
            | _ -> config.SpecPath

    let private persistAdaptiveSnapshot
        (config: SpecDrivenIterationConfig)
        (result: AutonomousSpecHarness.SpecDrivenIterationResult)
        (updated: SpecDrivenIterationConfig)
        (fallbackPath: string option) : (string * TarsEngine.FSharp.SelfImprovement.PersistentAdaptiveMemory.MemoryEntry) option =

        let storePath =
            match config.AdaptiveMemoryPath |> Option.filter (String.IsNullOrWhiteSpace >> not) with
            | Some explicit -> explicit
            | None ->
                match fallbackPath |> Option.filter (String.IsNullOrWhiteSpace >> not) with
                | Some fallback -> fallback
                | None -> Path.Combine(Environment.CurrentDirectory, "output", "adaptive_memory_default.jsonl")

        let policyBefore = capturePolicySnapshot config
        let policyAfter = capturePolicySnapshot updated
        let entry: TarsEngine.FSharp.SelfImprovement.PersistentAdaptiveMemory.MemoryEntry =
            { runId = Guid.NewGuid()
              specId = resolveSpecIdentifier config result
              specPath = resolveSpecPath config
              description = config.Description
              timestamp = DateTime.UtcNow
              consensus = captureConsensus result.Consensus
              critic = captureCritic result.CriticVerdict None
              reasoning = result.ReasoningTraces |> Option.defaultValue [] |> captureReasoning
              policyBefore = policyBefore
              policyAfter = policyAfter
              policyChanges = diffPolicies policyBefore policyAfter
              inferenceTelemetry = PersistentAdaptiveMemory.captureTelemetry result.ClosureResult.EvolutionData
              agentFeedback = result.CrossAgentFeedback |> Option.defaultValue [] |> captureAgentFeedback
              validatorFindings = []
              validatorDisagreements = []
              harness = captureHarness result.HarnessReport }

        append storePath entry
        Some (storePath, entry)

    let private trainMetaCritic storeInfo updated =
        match storeInfo with
        | None -> updated
        | Some (path, entry) ->
            let entries =
                let fromDisk = loadRecent path 256
                if fromDisk |> List.isEmpty then [ entry ] else fromDisk

            match MetaReasoningCritic.train entries with
            | None -> updated
            | Some model ->
                let criticFunction = MetaReasoningCritic.buildCritic model

                let configWithCritic =
                    match updated.ReasoningCritic with
                    | Some _ -> updated
                    | None -> { updated with ReasoningCritic = Some criticFunction }

                let aggregateWeight =
                    model.NegativeIndicators |> List.sumBy snd

                let configWithGuard =
                    if aggregateWeight >= 0.8 && not configWithCritic.RequireCriticApproval then
                        { configWithCritic with RequireCriticApproval = true }
                    else
                        configWithCritic

                let promptSibling = Path.ChangeExtension(path, ".prompts.json")
                MetaReasoningCritic.persistPromptAdjustments promptSibling model
                MetaReasoningCritic.persistPromptAdjustments MetaReasoningCritic.defaultPromptPath model

                configWithGuard

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

    let private clamp01 value =
        if value < 0.0 then 0.0
        elif value > 1.0 then 1.0
        else value

    let private configToGenome (config: SpecDrivenIterationConfig) =
        { requireConsensus = config.RequireConsensusForExecution
          requireCritic = config.RequireCriticApproval
          stopOnFailure = config.StopOnFailure
          captureLogs = config.CaptureLogs
          minimumPassCount = config.ConsensusRule |> Option.map (fun rule -> rule.MinimumPassCount)
          allowNeedsReview = config.ConsensusRule |> Option.map (fun rule -> rule.AllowNeedsReview)
          minimumConfidence = config.ConsensusRule |> Option.bind (fun rule -> rule.MinimumConfidence)
          maxFailureCount = config.ConsensusRule |> Option.bind (fun rule -> rule.MaxFailureCount) }

    let private consensusRuleFromGenome (genome: PolicyGenome) =
        let hasExplicitRule =
            Option.isSome genome.minimumPassCount
            || Option.isSome genome.allowNeedsReview
            || Option.isSome genome.minimumConfidence
            || Option.isSome genome.maxFailureCount

        if not genome.requireConsensus && not hasExplicitRule then
            None
        else
            Some
                { MinimumPassCount = genome.minimumPassCount |> Option.defaultValue 2
                  RequiredRoles = [ AgentRole.Reasoner; AgentRole.Reviewer ]
                  AllowNeedsReview = genome.allowNeedsReview |> Option.defaultValue false
                  MinimumConfidence = genome.minimumConfidence |> Option.map clamp01
                  MaxFailureCount = genome.maxFailureCount }

    let private applyGenome (template: SpecDrivenIterationConfig) (genome: PolicyGenome) =
        { template with
            RequireConsensusForExecution = genome.requireConsensus
            RequireCriticApproval = genome.requireCritic
            StopOnFailure = genome.stopOnFailure
            CaptureLogs = genome.captureLogs
            ConsensusRule = consensusRuleFromGenome genome }

    let private distinctGenomes genomes =
        let folder acc genome =
            let key = genomeKey genome
            if List.exists (fun existing -> genomeKey existing = key) acc then acc else genome :: acc
        genomes |> List.fold folder [] |> List.rev

    let private mutateGenome (genome: PolicyGenome) =
        let toggledConsensus =
            if genome.requireConsensus then
                { genome with
                    requireConsensus = false
                    minimumPassCount = None
                    allowNeedsReview = None
                    minimumConfidence = None
                    maxFailureCount = None }
            else
                let minimum =
                    match genome.minimumPassCount with
                    | Some value -> value
                    | None -> 2
                { genome with
                    requireConsensus = true
                    minimumPassCount = Some minimum }

        let tightenedConfidence =
            let value =
                match genome.minimumConfidence with
                | Some v -> clamp01 (v + 0.1)
                | None -> 0.7
            { genome with minimumConfidence = Some value }

        let relaxedConfidence =
            let value =
                match genome.minimumConfidence with
                | Some v -> clamp01 (v - 0.1)
                | None -> 0.5
            { genome with minimumConfidence = Some value }

        let allowNeedsReview =
            { genome with allowNeedsReview = Some true }

        let disallowNeedsReview =
            { genome with allowNeedsReview = Some false }

        let loosenFailures =
            let value =
                match genome.maxFailureCount with
                | Some v -> v + 1
                | None -> 1
            { genome with maxFailureCount = Some value }

        let zeroFailureTolerance =
            { genome with maxFailureCount = Some 0 }

        let flippedCritic =
            { genome with requireCritic = not genome.requireCritic }

        let flippedLogging =
            { genome with captureLogs = not genome.captureLogs }

        let flippedStopOnFailure =
            { genome with stopOnFailure = not genome.stopOnFailure }

        [ toggledConsensus
          tightenedConfidence
          relaxedConfidence
          allowNeedsReview
          disallowNeedsReview
          loosenFailures
          zeroFailureTolerance
          flippedCritic
          flippedLogging
          flippedStopOnFailure ]
        |> List.filter ((<>) genome)
        |> distinctGenomes

    let private scoreGenome (summaryMap: Map<string, PolicyPerformanceSummary>) (genome: PolicyGenome) =
        match summaryMap |> Map.tryFind (genomeKey genome) with
        | None -> 0.5 // optimistic prior
        | Some summary ->
            let trials = float summary.Trials
            let successes = float summary.Successes
            let failures = float summary.Failures
            let successScore = (successes + 1.0) / (trials + 2.0)
            let failurePenalty = (failures + 0.5) / (trials + 2.0) * 0.3
            let criticPenalty = summary.CriticRejectRate * 0.25 + summary.CriticNeedsReviewRate * 0.1
            successScore - failurePenalty - criticPenalty

    let private tuneConfigWithEntries (entries: MemoryEntry list) (baseConfig: SpecDrivenIterationConfig) =
        if entries.IsEmpty then baseConfig
        else
            let summaries = summarizePolicyOutcomes entries
            let summaryMap =
                summaries
                |> List.map (fun summary -> genomeKey summary.Genome, summary)
                |> Map.ofList

            let currentGenome = configToGenome baseConfig
            let historicalTopGenomes =
                summaries
                |> List.sortByDescending (fun summary -> summary.Successes - summary.Failures)
                |> List.truncate 3
                |> List.map (fun summary -> summary.Genome)

            let candidates =
                currentGenome
                :: mutateGenome currentGenome
                |> List.append historicalTopGenomes
                |> distinctGenomes

            let scored =
                candidates
                |> List.map (fun genome -> genome, scoreGenome summaryMap genome)

            let bestGenome, bestScore = scored |> List.maxBy snd
            let currentScore = scoreGenome summaryMap currentGenome

            let finalGenome =
                if genomeKey bestGenome <> genomeKey currentGenome && bestScore - currentScore < 0.05 then
                    currentGenome
                else
                    bestGenome

            applyGenome baseConfig finalGenome

    let private autoTunePolicy (storeInfo: (string * MemoryEntry) option) (latest: AutonomousSpecHarness.SpecDrivenIterationResult) (config: SpecDrivenIterationConfig) =
        let candidatePaths =
            [ storeInfo |> Option.map fst
              config.AdaptiveMemoryPath
              getLastWritePath () ]
            |> List.choose id
            |> List.distinct

        let entries =
            candidatePaths
            |> List.tryPick (fun path ->
                try
                    let items = loadRecent path 256
                    if items.IsEmpty then None else Some items
                with
                | :? IOException -> None
                | :? UnauthorizedAccessException -> None)
            |> Option.defaultValue []

        let tuned =
            if entries.IsEmpty then config
            else tuneConfigWithEntries entries config

        let consensusFailed =
            match latest.Consensus with
            | Some (ConsensusFailed _) -> true
            | _ -> false

        let criticBlocking =
            match latest.CriticVerdict with
            | Some (CriticVerdict.Reject _)
            | Some (CriticVerdict.NeedsReview _) -> true
            | _ -> false

        let withConsensus =
            if consensusFailed then
                let consensusRule =
                    match tuned.ConsensusRule with
                    | Some rule -> Some rule
                    | None -> Some defaultConsensusRule

                { tuned with
                    RequireConsensusForExecution = true
                    ConsensusRule = consensusRule }
            else
                tuned

        let withCritic =
            if criticBlocking then
                { withConsensus with RequireCriticApproval = true }
            else
                withConsensus

        let applyCrossAgentSignals config feedback =
            let reviewerEscalated =
                feedback
                |> List.exists (fun fb ->
                    match fb.Role, fb.Verdict with
                    | AgentRole.Reviewer, (CrossAgentFeedback.FeedbackVerdict.Reject _ | CrossAgentFeedback.FeedbackVerdict.Escalate _)
                    | AgentRole.SpecGuardian, (CrossAgentFeedback.FeedbackVerdict.Reject _ | CrossAgentFeedback.FeedbackVerdict.Escalate _) -> true
                    | _ -> false)

            let implementerNeedsWork =
                feedback
                |> List.exists (fun fb ->
                    match fb.Role, fb.Verdict with
                    | AgentRole.Custom roleName, (CrossAgentFeedback.FeedbackVerdict.NeedsWork _ | CrossAgentFeedback.FeedbackVerdict.Reject _)
                        when roleName.Equals("Implementer", StringComparison.OrdinalIgnoreCase) -> true
                    | _ -> false)

            let escalationPresent =
                feedback
                |> List.exists (fun fb ->
                    match fb.Verdict with
                    | CrossAgentFeedback.FeedbackVerdict.Escalate _ -> true
                    | _ -> false)

            let enforcedConsensus =
                if reviewerEscalated then
                    { config with
                        RequireConsensusForExecution = true
                        RequireCriticApproval = true
                        ConsensusRule = config.ConsensusRule |> Option.orElse (Some defaultConsensusRule) }
                else
                    config

            let implementerAdjusted =
                if implementerNeedsWork then
                    { enforcedConsensus with StopOnFailure = false }
                else
                    enforcedConsensus

            if escalationPresent then
                { implementerAdjusted with CaptureLogs = true }
            else
                implementerAdjusted

        match latest.CrossAgentFeedback with
        | Some feedback when not feedback.IsEmpty -> applyCrossAgentSignals withCritic feedback
        | _ -> withCritic

    module Internal =
        /// Exposed for testing: applies policy tuning using the supplied memory entries.
        let tuneWithEntries entries config = tuneConfigWithEntries entries config

    let update (config: SpecDrivenIterationConfig) (history: FeedbackRecord list) =
        if history.IsEmpty then config
        else
            let consensusFailures =
                history
                |> List.filter (fun record ->
                    match record.Consensus with
                    | Some (ConsensusFailed _) -> true
                    | _ -> false)
                |> List.length

            let criticRejections =
                history
                |> List.filter (fun record ->
                    match record.CriticVerdict with
                    | Some (CriticVerdict.Reject _) -> true
                    | _ -> false)
                |> List.length

            let criticNeedsReview =
                history
                |> List.filter (fun record ->
                    match record.CriticVerdict with
                    | Some (CriticVerdict.NeedsReview _) -> true
                    | _ -> false)
                |> List.length

            let updatedForConsensus =
                if consensusFailures >= 2 then
                    { config with
                        ConsensusRule = Some (config.ConsensusRule |> Option.defaultValue defaultConsensusRule)
                        RequireConsensusForExecution = true }
                elif consensusFailures = 0 then
                    { config with RequireConsensusForExecution = false }
                else
                    config

            let updatedForCritic =
                if criticRejections + criticNeedsReview >= 2 then
                    { updatedForConsensus with RequireCriticApproval = true }
                elif criticRejections = 0 && criticNeedsReview = 0 then
                    { updatedForConsensus with RequireCriticApproval = false }
                else
                    updatedForConsensus

            updatedForCritic

    let private buildFeedbackRecord (result: AutonomousSpecHarness.SpecDrivenIterationResult) =
        { Consensus = result.Consensus
          CriticVerdict = result.CriticVerdict }

    let runAdaptiveIteration loggerFactory history config =
        async {
            let! result = AutonomousSpecHarness.runIteration loggerFactory config
            let feedback = buildFeedbackRecord result
            let updated = update config (feedback :: history)
            let storeInfo = persistAdaptiveSnapshot config result updated None
            let tuned = trainMetaCritic storeInfo updated
            let evolved = autoTunePolicy storeInfo result tuned
            return result, evolved
        }

    let runAdaptiveIterationWithExecutor loggerFactory history config executor =
        async {
            let! result = AutonomousSpecHarness.runIterationWithExecutor loggerFactory config executor
            let feedback = buildFeedbackRecord result
            let updated = update config (feedback :: history)
            let storeInfo = persistAdaptiveSnapshot config result updated None
            let tuned = trainMetaCritic storeInfo updated
            let evolved = autoTunePolicy storeInfo result tuned
            return result, evolved
        }

    let runAdaptiveIterationWithHistory loggerFactory historyPath config =
        async {
            let history = loadHistory historyPath
            let! result = AutonomousSpecHarness.runIteration loggerFactory config
            let feedback = buildFeedbackRecord result
            appendHistory historyPath feedback
            let updated = update config (feedback :: history)
            let fallback = if String.IsNullOrWhiteSpace(historyPath) then None else Some(Path.ChangeExtension(historyPath, ".memory.jsonl"))
            let storeInfo = persistAdaptiveSnapshot config result updated fallback
            let tuned = trainMetaCritic storeInfo updated
            let evolved = autoTunePolicy storeInfo result tuned
            return result, evolved
        }

    let runAdaptiveIterationWithHistoryAndExecutor loggerFactory historyPath config executor =
        async {
            let history = loadHistory historyPath
            let! result = AutonomousSpecHarness.runIterationWithExecutor loggerFactory config executor
            let feedback = buildFeedbackRecord result
            appendHistory historyPath feedback
            let updated = update config (feedback :: history)
            let fallback = if String.IsNullOrWhiteSpace(historyPath) then None else Some(Path.ChangeExtension(historyPath, ".memory.jsonl"))
            let storeInfo = persistAdaptiveSnapshot config result updated fallback
            let tuned = trainMetaCritic storeInfo updated
            let evolved = autoTunePolicy storeInfo result tuned
            return result, evolved
        }
