namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Text.Json
open System.Text.Json.Nodes
open System.Globalization
open Microsoft.Extensions.Logging
open Tars.Engine.VectorStore

/// Planner that analyses Spec Kit tasks, adaptive memory, and roadmap objectives to recommend the next autonomous steps.
module AutonomousNextStepPlanner =

    open SpecKitWorkspace
    open PersistentAdaptiveMemory

    type PlannerConfig =
        { TopCandidates: int
          RecentMemoryLimit: int
          MemoryPath: string option
          VectorStoreConfig: VectorStoreConfig
          SimilarityGoal: string }

    type PlannerRecommendation =
        { Selection: SpecKitSelection
          Score: float
          SimilarityScore: float
          FailureSignal: int
          PriorityWeight: float
          Rationale: string list }

    type private BacklogEntry =
        { id: string
          title: string
          priority: string
          specPath: string
          createdAt: DateTime
          sourceSpec: string
          sourceRunId: string option
          status: string }

    let private serializerOptions =
        let options = JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase)
        options

    let defaultVectorStoreConfig =
        { RawDimension = 512
          EnableFFT = true
          EnableDual = false
          EnableProjective = false
          EnableHyperbolic = false
          EnableWavelet = false
          EnableMinkowski = false
          EnablePauli = false
          SpaceWeights = Map.ofList [ ("raw", 1.0); ("fft", 0.4); ("phase", 0.2) ]
          PersistToDisk = false
          StoragePath = None }

    let defaultConfig =
        { TopCandidates = 3
          RecentMemoryLimit = 64
          MemoryPath = None
          VectorStoreConfig = defaultVectorStoreConfig
          SimilarityGoal =
            "Maximise delivery velocity, minimise defect escape, improve recursive self-improvement stability, honour CUDA performance targets, and close high priority Spec Kit tasks with measurable outcomes." }

    let private backlogPath () =
        Path.Combine(Environment.CurrentDirectory, ".specify", "backlog.json")

    let private loadPendingSpecs () =
        let path = backlogPath ()
        if File.Exists(path) then
            try
                File.ReadAllText(path)
                |> fun json -> JsonSerializer.Deserialize<BacklogEntry list>(json, serializerOptions)
                |> Option.ofObj
                |> Option.defaultValue []
                |> List.filter (fun entry -> String.Equals(entry.status, "pending", StringComparison.OrdinalIgnoreCase))
                |> List.map (fun entry -> entry.sourceSpec)
                |> Set.ofList
            with _ ->
                Set.empty
        else
            Set.empty

    type GovernanceMetrics =
        { CapabilityPassRatio: float option
          CapabilityTrend: float option
          SafetyConsensusConfidence: float option
          SafetyConfidenceTrend: float option
          SafetyCriticStatus: string option
          SafetyCriticRejectRate: float option
          InferenceLatency: float option
          InferenceLatencyTrend: float option
          InferenceTokenCount: float option
          InferenceTokenTrend: float option
          InferenceUsedCuda: bool option }

    type private GovernanceSample =
        { Capability: float option
          ConsensusConfidence: float option
          CriticStatus: string option
          InferenceLatency: float option
          InferenceTokens: float option
          InferenceUsedCuda: bool option }

    let private tryAsValue (node: JsonNode) =
        if isNull node then None else
        let value = node.AsValue()
        if isNull value then None else Some value

    let private parseDouble (node: JsonNode) =
        match tryAsValue node with
        | None -> None
        | Some value ->
            match value.GetValueKind() with
            | JsonValueKind.Number ->
                try
                    Some (value.GetValue<double>())
                with _ -> None
            | JsonValueKind.String ->
                match Double.TryParse(value.GetValue<string>(), NumberStyles.Float, CultureInfo.InvariantCulture) with
                | true, parsed -> Some parsed
                | _ -> None
            | _ -> None

    let private parseString (node: JsonNode) =
        match tryAsValue node with
        | Some value when value.GetValueKind() = JsonValueKind.String -> Some(value.GetValue<string>())
        | _ -> None

    let private parseBool (node: JsonNode) =
        match tryAsValue node with
        | None -> None
        | Some value ->
            match value.GetValueKind() with
            | JsonValueKind.True -> Some true
            | JsonValueKind.False -> Some false
            | JsonValueKind.String ->
                match value.GetValue<string>().Trim().ToLowerInvariant() with
                | "true"
                | "yes"
                | "1"
                | "gpu" -> Some true
                | "false"
                | "no"
                | "0"
                | "cpu" -> Some false
                | _ -> None
            | JsonValueKind.Number ->
                let mutable parsed = 0.0
                if value.TryGetValue<double>(&parsed) then Some (Math.Abs(parsed) > Double.Epsilon) else None
            | _ -> None

    let private iterationsDirectory =
        Path.Combine(Environment.CurrentDirectory, ".specify", "ledger", "iterations")

    let private latestLedgerPath () =
        Path.Combine(iterationsDirectory, "latest.json")

    let private loadLedgerFiles limit =
        if not (Directory.Exists(iterationsDirectory)) then
            []
        else
            Directory.GetFiles(iterationsDirectory, "*.json", SearchOption.TopDirectoryOnly)
            |> Array.filter (fun path -> not (path.EndsWith("latest.json", StringComparison.OrdinalIgnoreCase)))
            |> Array.sortDescending
            |> Array.truncate limit
            |> Array.toList

    let private parseMetricsFromNode (node: JsonNode) =
        if isNull node then None
        else
            let metricsNode = node.["metrics"]
            if isNull metricsNode then None
            else
                let capability = parseDouble metricsNode.["capability.pass_ratio"]
                let consensus = parseDouble metricsNode.["safety.consensus_avg_confidence"]
                let critic = parseString metricsNode.["safety.critic_status"]
                let inferenceLatency = parseDouble metricsNode.["inference.metrics.analysis_elapsed_ms"]
                let inferenceTokens = parseDouble metricsNode.["inference.metrics.token_count"]
                let inferenceCuda = parseBool metricsNode.["inference.metrics.used_cuda"]
                Some
                    { Capability = capability
                      ConsensusConfidence = consensus
                      CriticStatus = critic
                      InferenceLatency = inferenceLatency
                      InferenceTokens = inferenceTokens
                      InferenceUsedCuda = inferenceCuda }

    let private loadGovernanceHistory limit =
        let files = loadLedgerFiles limit
        files
        |> List.choose (fun path ->
            try
                JsonNode.Parse(File.ReadAllText(path)) |> parseMetricsFromNode
            with _ ->
                None)

    let private loadLatestGovernanceMetrics () =
        let path = latestLedgerPath ()
        if not (File.Exists(path)) then
            { CapabilityPassRatio = None
              CapabilityTrend = None
              SafetyConsensusConfidence = None
              SafetyConfidenceTrend = None
              SafetyCriticStatus = None
              SafetyCriticRejectRate = None
              InferenceLatency = None
              InferenceLatencyTrend = None
              InferenceTokenCount = None
              InferenceTokenTrend = None
              InferenceUsedCuda = None }
        else
            try
                let json = JsonNode.Parse(File.ReadAllText(path))
                let metricsNode = if isNull json then null else json.["metrics"]
                let capability = if isNull metricsNode then None else parseDouble metricsNode.["capability.pass_ratio"]
                let consensus = if isNull metricsNode then None else parseDouble metricsNode.["safety.consensus_avg_confidence"]
                let critic = if isNull metricsNode then None else parseString metricsNode.["safety.critic_status"]
                let inferenceLatency = if isNull metricsNode then None else parseDouble metricsNode.["inference.metrics.analysis_elapsed_ms"]
                let inferenceTokens = if isNull metricsNode then None else parseDouble metricsNode.["inference.metrics.token_count"]
                let inferenceCuda = if isNull metricsNode then None else parseBool metricsNode.["inference.metrics.used_cuda"]
                let history = loadGovernanceHistory 10
                let capabilityTrend =
                    history
                    |> List.choose (fun sample -> sample.Capability)
                    |> fun values ->
                        if values.IsEmpty then None else Some (values |> List.average)
                let confidenceTrend =
                    history
                    |> List.choose (fun sample -> sample.ConsensusConfidence)
                    |> fun values ->
                        if values.IsEmpty then None else Some (values |> List.average)
                let criticRejectRate =
                    history
                    |> List.map (fun sample -> sample.CriticStatus)
                    |> fun statuses ->
                        if statuses.IsEmpty then None
                        else
                            let rejectCount =
                                statuses
                                |> List.filter (fun status ->
                                    status
                                    |> Option.exists (fun value -> value.Equals("reject", StringComparison.OrdinalIgnoreCase)))
                                |> List.length
                            Some (float rejectCount / float statuses.Length)
                let inferenceLatencyTrend =
                    history
                    |> List.choose (fun sample -> sample.InferenceLatency)
                    |> fun values -> if values.IsEmpty then None else Some (values |> List.average)
                let inferenceTokenTrend =
                    history
                    |> List.choose (fun sample -> sample.InferenceTokens)
                    |> fun values -> if values.IsEmpty then None else Some (values |> List.average)
                let inferenceCudaRecent =
                    history
                    |> List.choose (fun sample -> sample.InferenceUsedCuda)
                    |> List.tryHead
                { CapabilityPassRatio = capability
                  CapabilityTrend = capabilityTrend
                  SafetyConsensusConfidence = consensus
                  SafetyConfidenceTrend = confidenceTrend
                  SafetyCriticStatus = critic
                  SafetyCriticRejectRate = criticRejectRate
                  InferenceLatency = inferenceLatency
                  InferenceLatencyTrend = inferenceLatencyTrend
                  InferenceTokenCount = inferenceTokens
                  InferenceTokenTrend = inferenceTokenTrend
                  InferenceUsedCuda = inferenceCuda |> Option.orElse inferenceCudaRecent }
            with _ ->
                { CapabilityPassRatio = None
                  CapabilityTrend = None
                  SafetyConsensusConfidence = None
                  SafetyConfidenceTrend = None
                  SafetyCriticStatus = None
                  SafetyCriticRejectRate = None
                  InferenceLatency = None
                  InferenceLatencyTrend = None
                  InferenceTokenCount = None
                  InferenceTokenTrend = None
                  InferenceUsedCuda = None }

    let private priorityRank (priority: string option) =
        match priority |> Option.map (fun value -> value.Trim().ToUpperInvariant()) with
        | Some "P0" -> 0
        | Some "P1" -> 1
        | Some "P2" -> 2
        | Some "P3" -> 3
        | Some value when value.StartsWith("P") ->
            match Int32.TryParse(value.Substring(1)) with
            | true, number when number >= 0 -> number
            | _ -> 99
        | _ -> 100

    let private priorityWeight rank =
        match rank with
        | 0 -> 1.0
        | 1 -> 0.85
        | 2 -> 0.65
        | 3 -> 0.45
        | _ -> 0.3

    let private collectCandidates () =
        SpecKitWorkspace.discoverFeatures None
        |> List.collect (fun feature ->
            feature.Tasks
            |> List.filter (fun task -> not (String.Equals(task.Status, "done", StringComparison.OrdinalIgnoreCase)))
            |> List.map (fun task ->
                { Feature = feature
                  Task = task
                  PriorityRank = priorityRank task.Priority }))

    let private loadMemoryEntries (config: PlannerConfig) =
        let path =
            config.MemoryPath
            |> Option.defaultValue (Path.Combine(Environment.CurrentDirectory, "output", "adaptive_memory_spec-kit.jsonl"))

        if File.Exists(path) then
            PersistentAdaptiveMemory.loadRecent path config.RecentMemoryLimit
        else
            []

    let private computeFailureSignals (entries: PersistentAdaptiveMemory.MemoryEntry list) =
        entries
        |> List.fold (fun state entry ->
            let status =
                entry.harness
                |> Option.map (fun harness -> harness.status)
                |> Option.defaultValue "unknown"

            let failure =
                status.StartsWith("failed", StringComparison.OrdinalIgnoreCase)
                || (entry.consensus |> Option.exists (fun c -> not (String.Equals(c.status, "passed", StringComparison.OrdinalIgnoreCase))))

            if failure then
                let current =
                    state
                    |> Map.tryFind entry.specId
                    |> Option.defaultValue 0
                state |> Map.add entry.specId (current + 1)
            else
                state) Map.empty

    let private summarizeMemoryForSpec (entries: PersistentAdaptiveMemory.MemoryEntry list) =
        entries
        |> List.sortByDescending (fun entry -> entry.timestamp)
        |> List.truncate 3
        |> List.map (fun entry ->
            let harnessStatus =
                entry.harness
                |> Option.map (fun harness -> harness.status)
                |> Option.defaultValue "unknown"

            let harnessDetail =
                entry.harness
                |> Option.bind (fun harness -> harness.failureReason)
                |> Option.defaultValue ""

            let consensusSummary =
                entry.consensus
                |> Option.map (fun consensus ->
                    let agentStatuses =
                        consensus.agents
                        |> List.map (fun agent -> $"{agent.role}:{agent.outcome}[{agent.confidence |> Option.defaultValue 0.0:F2}]")
                        |> String.concat ", "
                    $"status={consensus.status}; agents={agentStatuses}")
                |> Option.defaultValue "no consensus summary"

            let criticSummary =
                entry.critic
                |> Option.map (fun critic ->
                    let criticMessage = critic.message |> Option.defaultValue ""
                    $"{critic.status}:{criticMessage}")
                |> Option.defaultValue "critic=none"

            let feedbackHighlights =
                entry.agentFeedback
                |> List.take (min 3 entry.agentFeedback.Length)
                |> List.map (fun feedback ->
                    let actionsText =
                        match feedback.suggestedActions with
                        | [] -> ""
                        | xs ->
                            let concatenated = String.concat "|" xs
                            $" -> {concatenated}"
                    let confidence = feedback.confidence |> Option.defaultValue 0.0
                    $"{feedback.role}:{feedback.verdict}[{confidence:F2}]{actionsText}")
                |> String.concat "; "

            let timestamp = entry.timestamp.ToString("o")
            $"[{timestamp}] harness={harnessStatus} {harnessDetail} // consensus {consensusSummary} // critic {criticSummary} // feedback {feedbackHighlights}"
        )
        |> String.concat Environment.NewLine

    let private buildSpecContext (entries: PersistentAdaptiveMemory.MemoryEntry list) =
        entries
        |> List.groupBy (fun entry -> entry.specId)
        |> List.map (fun (specId, grouped) -> specId, summarizeMemoryForSpec grouped)
        |> Map.ofList

    let private candidateDocumentText (selection: SpecKitSelection) (context: string option) =
        let feature = selection.Feature
        let summary = feature.Summary
        let stories =
            summary.UserStories
            |> List.map (fun story ->
                let storyPriority = story.Priority |> Option.defaultValue "N/A"
                let acceptance = String.concat "; " story.AcceptanceCriteria
                $"Story: {story.Title} Priority={storyPriority} Acceptance={acceptance}")
            |> String.concat Environment.NewLine

        let edgeCases = summary.EdgeCases |> String.concat "; "

        [
            $"Spec: {summary.Title}"
            $"Task: {selection.Task.Description}"
            $"PriorityRank: {selection.PriorityRank}"
            $"Stories: {stories}"
            $"EdgeCases: {edgeCases}"
            $"SpecPath: {feature.SpecPath}"
            match context with
            | Some ctx when not (String.IsNullOrWhiteSpace ctx) ->
                $"RecentRuns:{Environment.NewLine}{ctx}"
            | _ -> ""
        ]
        |> String.concat Environment.NewLine
        |> fun text -> text.Trim()

    let private createVectorStore (config: PlannerConfig) =
        let store = VectorStoreFactory.createInMemory config.VectorStoreConfig
        let generator = MultiSpaceEmbeddingGenerator(config.VectorStoreConfig) :> IEmbeddingGenerator
        store, generator

    let private indexCandidates
        (store: IVectorStore)
        (generator: IEmbeddingGenerator)
        (specContext: Map<string, string>)
        (candidates: SpecKitSelection list) =
        let addAsync candidate =
            async {
                let context = specContext |> Map.tryFind candidate.Feature.Id
                let text = candidateDocumentText candidate context
                let! embedding = generator.GenerateEmbedding text
                let doc =
                    { VectorDocument.Id = $"{candidate.Feature.Id}:{candidate.Task.LineNumber}"
                      Content = text
                      Embedding = embedding
                      Tags =
                        [ candidate.Feature.Id
                          candidate.Task.Priority |> Option.defaultValue "P?" ]
                      Timestamp = DateTime.UtcNow
                      Source = Some candidate.Feature.SpecPath }
                do! store.AddDocument doc
                return ()
            }

        candidates
        |> List.map addAsync
        |> Async.Parallel
        |> Async.Ignore

    let private scoreCandidates
        (logger: ILogger)
        (config: PlannerConfig)
        (store: IVectorStore)
        (generator: IEmbeddingGenerator)
        (candidates: SpecKitSelection list)
        (failureSignals: Map<string, int>)
        (governance: GovernanceMetrics) =
        async {
            let! queryEmbedding = generator.GenerateEmbedding config.SimilarityGoal
            let query =
                { VectorQuery.Text = config.SimilarityGoal
                  Embedding = queryEmbedding
                  Filters = Map.empty
                  MaxResults = candidates.Length
                  MinScore = 0.0 }

            let! results = store.Search query
            logger.LogInformation("Vector store produced {ResultCount} similarity results for planner query.", results.Length)

            let similarityMap =
                results
                |> List.map (fun result -> result.Document.Id, result.FinalScore)
                |> Map.ofList

            let maxFailure =
                if Map.isEmpty failureSignals then 0 else failureSignals |> Map.toSeq |> Seq.map snd |> Seq.max

            let capabilityWeight =
                governance.CapabilityPassRatio
                |> Option.map (fun ratio -> ratio |> max 0.0 |> min 1.0)
                |> Option.defaultValue 0.5

            let capabilityTrendWeight =
                governance.CapabilityTrend
                |> Option.map (fun ratio -> ratio |> max 0.0 |> min 1.0)
                |> Option.defaultValue capabilityWeight

            let safetyWeight =
                governance.SafetyConsensusConfidence
                |> Option.map (fun value -> value |> max 0.0 |> min 1.0)
                |> Option.defaultValue 0.5

            let safetyTrendWeight =
                governance.SafetyConfidenceTrend
                |> Option.map (fun value -> value |> max 0.0 |> min 1.0)
                |> Option.defaultValue safetyWeight

            let criticModifier =
                governance.SafetyCriticStatus
                |> Option.map (fun status ->
                    if status.Equals("reject", StringComparison.OrdinalIgnoreCase) then 0.5
                    elif status.Equals("needs_review", StringComparison.OrdinalIgnoreCase) then 0.75
                    else 1.0)
                |> Option.defaultValue 1.0

            let inferenceKeywords =
                [ "inference"
                  "ollama"
                  "telemetry"
                  "latency"
                  "model"
                  "prompt"
                  "token"
                  "vector store"
                  "embedding"
                  "gpu" ]

            let cudaKeywords =
                [ "cuda"
                  "gpu"
                  "device"
                  "kernel"
                  "vector store"
                  "accelerat" ]

            let normalizeLatency value =
                value |> fun ms -> ms / 1200.0 |> max 0.0 |> min 1.0

            let normalizeTokens value =
                value |> fun tokens -> tokens / 1024.0 |> max 0.0 |> min 1.0

            let inferenceLatencyPressure =
                [ governance.InferenceLatency; governance.InferenceLatencyTrend ]
                |> List.choose id
                |> List.map normalizeLatency
                |> fun values -> if values.IsEmpty then 0.0 else values |> List.max

            let inferenceTokenPressure =
                [ governance.InferenceTokenCount; governance.InferenceTokenTrend ]
                |> List.choose id
                |> List.map normalizeTokens
                |> fun values -> if values.IsEmpty then 0.0 else values |> List.max

            let inferenceUrgency = max inferenceLatencyPressure inferenceTokenPressure

            let cudaUrgency =
                match governance.InferenceUsedCuda with
                | Some false -> 0.6
                | _ -> 0.0

            let containsAny (text: string) (keywords: string list) =
                keywords |> List.exists (fun keyword -> text.Contains(keyword))

            let recommendations =
                candidates
                |> List.map (fun selection ->
                    let id = $"{selection.Feature.Id}:{selection.Task.LineNumber}"
                    let similarity = similarityMap |> Map.tryFind id |> Option.defaultValue 0.0
                    let failureCount = failureSignals |> Map.tryFind selection.Feature.Id |> Option.defaultValue 0
                    let failureWeight =
                        if maxFailure = 0 then 0.0 else float failureCount / float maxFailure
                    let pWeight = priorityWeight selection.PriorityRank
                    let capabilityBonus = ((capabilityWeight + capabilityTrendWeight) / 2.0) * 0.12
                    let safetyBonus = ((safetyWeight + safetyTrendWeight) / 2.0) * 0.12
                    let criticPenalty =
                        governance.SafetyCriticRejectRate
                        |> Option.map (fun rate -> 1.0 - (rate |> max 0.0 |> min 1.0) * 0.5)
                        |> Option.defaultValue 1.0
                    let adjustedSimilarity = similarity * criticModifier
                    let candidateText =
                        [ selection.Feature.Id
                          selection.Feature.Summary.Title
                          selection.Task.Description ]
                        |> List.choose (fun text -> if String.IsNullOrWhiteSpace(text) then None else Some(text.ToLowerInvariant()))
                        |> String.concat " "
                    let inferenceFocus =
                        not (String.IsNullOrWhiteSpace(candidateText))
                        && containsAny candidateText inferenceKeywords
                    let cudaFocus =
                        not (String.IsNullOrWhiteSpace(candidateText))
                        && containsAny candidateText cudaKeywords
                    let inferenceBonus =
                        if inferenceFocus then
                            (inferenceUrgency * 0.18)
                            + (if cudaFocus then cudaUrgency * 0.12 else 0.0)
                        else 0.0
                    let inferencePenalty =
                        if inferenceFocus then 0.0 else inferenceUrgency * 0.08
                    let cudaPenalty =
                        if cudaFocus || cudaUrgency = 0.0 then 0.0 else cudaUrgency * 0.05
                    let score =
                        (pWeight * 0.38)
                        + (failureWeight * 0.20)
                        + (adjustedSimilarity * 0.22)
                        + capabilityBonus
                        + safetyBonus
                        + inferenceBonus
                        - inferencePenalty
                        - cudaPenalty
                        |> fun total -> total * criticPenalty
                        |> max 0.0

                    let rationale =
                        [ $"Priority weight {pWeight:F2}"
                          if failureCount > 0 then $"Recent failure count: {failureCount}" else "Stable harness history"
                          if similarity > 0.0 then $"Roadmap similarity {similarity:F2}" else "Weak roadmap similarity"
                          $"Capability pass ratio {capabilityWeight:F2} (trend {capabilityTrendWeight:F2})"
                          $"Consensus confidence {safetyWeight:F2} (trend {safetyTrendWeight:F2})"
                          if inferenceUrgency > 0.0 then
                              $"Inference latency pressure {inferenceLatencyPressure:F2}; token pressure {inferenceTokenPressure:F2}"
                          else
                              "Inference telemetry nominal"
                          if cudaUrgency > 0.0 then
                              if cudaFocus then "Candidate tackles CUDA degradation (GPU offline)"
                              else "CUDA degradation detected; favour GPU remediation"
                          else "CUDA acceleration healthy"
                          match governance.SafetyCriticStatus with
                          | Some status -> $"Critic status {status}"
                          | None -> "Critic status unknown"
                          governance.SafetyCriticRejectRate
                          |> Option.map (fun rate -> $"Critic reject trend {rate:F2}")
                          |> Option.defaultValue "Critic reject trend unavailable" ]

                    { Selection = selection
                      Score = score
                      SimilarityScore = similarity
                      FailureSignal = failureCount
                      PriorityWeight = pWeight
                      Rationale = rationale })
                |> List.sortByDescending (fun recommendation -> recommendation.Score)

            return recommendations
        }

    let plan (logger: ILogger) (config: PlannerConfig option) =
        async {
            let config = defaultArg config defaultConfig
            let candidates = collectCandidates ()

            if candidates.IsEmpty then
                logger.LogInformation("Planner found no pending Spec Kit tasks to evaluate.")
                return []
            else
                logger.LogInformation("Planner evaluating {CandidateCount} pending tasks.", candidates.Length)

                let memoryEntries = loadMemoryEntries config
                let failureSignals = computeFailureSignals memoryEntries
                let specContext = buildSpecContext memoryEntries
                let governance = loadLatestGovernanceMetrics ()

                let store, generator = createVectorStore config
                do! indexCandidates store generator specContext candidates

                let! scored = scoreCandidates logger config store generator candidates failureSignals governance
                let top =
                    scored
                    |> List.truncate config.TopCandidates

                top
                |> List.iter (fun recommendation ->
                    logger.LogInformation(
                        "Planner candidate {SpecId} :: {TaskDesc} => Score {Score:F3} (priority={PriorityWeight:F2}, failures={Failure}, similarity={Similarity:F2})",
                        recommendation.Selection.Feature.Id,
                        recommendation.Selection.Task.Description,
                        recommendation.Score,
                        recommendation.PriorityWeight,
                        recommendation.FailureSignal,
                        recommendation.SimilarityScore))

                return top
        }

    let planAndEnqueue (logger: ILogger) (config: PlannerConfig option) =
        async {
            let config = defaultArg config defaultConfig
            let! recommendations = plan logger (Some config)

            let pendingSpecs = loadPendingSpecs ()

            let mutable enqueued = []

            for recommendation in recommendations do
                let specId = recommendation.Selection.Feature.Id
                if pendingSpecs.Contains(specId) then
                    logger.LogInformation("Skipping {SpecId} because backlog already has pending follow-ups.", specId)
                else
                    SpecKitGoalPlanner.recordGoal recommendation.Selection (Some(Guid.NewGuid()))
                    logger.LogInformation("Enqueued follow-up spec for {SpecId} :: {Task}", specId, recommendation.Selection.Task.Description)
                    enqueued <- recommendation :: enqueued

            return recommendations, List.rev enqueued
        }
