namespace Tars.Evolution

open System
open System.IO
open System.Text.Json
open Tars.Llm
open Tars.Core.WorkflowOfThought
open Tars.Cortex
open Tars.Cortex.WoTTypes

/// <summary>
/// Phase 17: Retroaction Loop - the core self-improvement feedback cycle.
/// Connects: WoT execution -> trace analysis -> pattern compilation -> storage -> serving.
/// This is the WIRING that turns individual executions into cumulative learning.
/// </summary>
module RetroactionLoop =

    // =========================================================================
    // Configuration
    // =========================================================================

    type RetroactionConfig =
        { /// Minimum score to compile a trace into a new pattern
          PatternThreshold: float
          /// Minimum score to promote a trace to golden status
          GoldenThreshold: float
          /// Number of variants to generate during amplification
          AmplifyCount: int
          /// Maximum patterns in library before pruning
          MaxLibrarySize: int
          /// Minimum diversity score (0-1) before coherence check intervenes
          MinDiversity: float }

    let defaultConfig =
        { PatternThreshold = 0.6
          GoldenThreshold = 0.85
          AmplifyCount = 3
          MaxLibrarySize = 50
          MinDiversity = 0.3 }

    // =========================================================================
    // Scoring
    // =========================================================================

    /// Score an execution result based on success, validation, and efficiency.
    type ExecutionScore =
        { Success: bool
          ValidationPassed: bool
          Confidence: float
          Efficiency: float
          Overall: float }

    let scoreExecution (result: string) (problem: Problem) : ExecutionScore =
        let validationPassed =
            PatternLibrary.validateResult result problem.ValidationCriteria

        let hasContent = result.Length > 10
        let isReasonable = result.Length < 50000

        let confidence =
            if validationPassed then 0.9
            elif hasContent then 0.4
            else 0.1

        let efficiency =
            if isReasonable && hasContent then 0.8
            elif hasContent then 0.5
            else 0.2

        let overall =
            if validationPassed then
                0.5 * 1.0 + 0.3 * confidence + 0.2 * efficiency
            else
                0.5 * 0.0 + 0.3 * confidence + 0.2 * efficiency

        { Success = validationPassed && hasContent
          ValidationPassed = validationPassed
          Confidence = confidence
          Efficiency = efficiency
          Overall = overall }

    // =========================================================================
    // Pattern Persistence (extends PatternLibrary)
    // =========================================================================

    let private jsonOptions =
        let o = JsonSerializerOptions(JsonSerializerDefaults.General)
        o.Converters.Add(System.Text.Json.Serialization.JsonFSharpConverter())
        o

    let private getPatternsDir () =
        let dir = Path.Combine(Environment.CurrentDirectory, ".tars", "patterns")
        if not (Directory.Exists dir) then
            Directory.CreateDirectory dir |> ignore
        dir

    /// Save a pattern definition to disk.
    let savePattern (pattern: PatternDefinition) : Result<string, string> =
        try
            let dir = getPatternsDir ()
            let safeName = pattern.Name.Replace(" ", "_").Replace("/", "_")
            let path = Path.Combine(dir, sprintf "%s.json" safeName)
            let json = JsonSerializer.Serialize(pattern, jsonOptions)
            File.WriteAllText(path, json)
            Ok path
        with ex ->
            Error (sprintf "Failed to save pattern: %s" ex.Message)

    // =========================================================================
    // Core: executeAndLearn
    // =========================================================================

    /// Execute a problem using the best available pattern and learn from the result.
    let executeAndLearn
        (llm: ILlmService)
        (config: RetroactionConfig)
        (problem: Problem)
        : Async<Result<string * ExecutionScore * PatternDefinition option, string>> =
        async {
            let runId = Guid.NewGuid()
            Console.WriteLine($"[Retroaction] executeAndLearn: {problem.Title} (run={runId})")

            // 1. Load existing patterns and find the best match
            let existingPatterns = PatternLibrary.loadAll ()
            let! bestMatch = PatternLibrary.findMatch llm problem.Description existingPatterns

            // 2. Execute using matched pattern or default prompt
            let! executionResult =
                match bestMatch with
                | Some pattern ->
                    async {
                        Console.WriteLine($"[Retroaction] Using pattern: {pattern.Name}")
                        let context = Map.ofList [ "goal", problem.Description; "input", problem.Description ]
                        let hydrated = PatternLibrary.hydrate pattern context
                        return! PatternLibrary.executePattern llm pattern hydrated problem.Description
                    }
                | None ->
                    async {
                        Console.WriteLine("[Retroaction] No matching pattern, using direct reasoning")
                        let req =
                            { ModelHint = Some "reasoning"
                              Model = None
                              SystemPrompt = Some "You are TARS, an autonomous reasoning agent. Solve the following problem step by step."
                              MaxTokens = Some 2000
                              Temperature = Some 0.7
                              Stop = []
                              Messages = [ { Role = Role.User; Content = problem.Description } ]
                              Tools = []
                              ToolChoice = None
                              ResponseFormat = None
                              Stream = false
                              JsonMode = false
                              Seed = None
                              ContextWindow = None }
                        let! resp = llm.CompleteAsync req |> Async.AwaitTask
                        return Ok resp.Text
                    }

            match executionResult with
            | Error err ->
                return Error (sprintf "Execution failed: %s" err)
            | Ok resultText ->
                // 3. Score the result
                let score = scoreExecution resultText problem
                Console.WriteLine($"[Retroaction] Score: overall={score.Overall:F2}, validation={score.ValidationPassed}, confidence={score.Confidence:F2}")

                // 4. Create a synthetic trace for pattern compilation
                let trace : CanonicalTraceEvent list =
                    [ { StepId = "reason_1"
                        Kind = "reason"
                        ToolName = None
                        ResolvedArgs = Some [ ("goal", problem.Description) ]
                        Outputs = [ resultText |> fun s -> if s.Length > 500 then s.Substring(0, 500) else s ]
                        Status = if score.Success then StepStatus.Ok else StepStatus.Error
                        Error = if score.Success then None else Some "Validation failed"
                        Usage = None
                        Metadata = None } ]

                // 5. If score exceeds threshold, compile into a new pattern
                let mutable newPattern = None

                if score.Overall >= config.PatternThreshold then
                    Console.WriteLine($"[Retroaction] Score {score.Overall:F2} >= threshold {config.PatternThreshold}, compiling pattern...")
                    let! compiled = PatternCompiler.compileFromTrace llm runId trace problem.Description
                    match compiled with
                    | Ok pattern ->
                        let scoredPattern = { pattern with Score = score.Overall }
                        match savePattern scoredPattern with
                        | Ok path ->
                            Console.WriteLine($"[Retroaction] Pattern saved: {path}")
                            newPattern <- Some scoredPattern
                        | Error err ->
                            Console.WriteLine($"[Retroaction] Failed to save pattern: {err}")
                    | Error err ->
                        Console.WriteLine($"[Retroaction] Pattern compilation failed: {err}")

                // 6. If score exceeds golden threshold, save as golden trace
                if score.Overall >= config.GoldenThreshold then
                    Console.WriteLine($"[Retroaction] Score {score.Overall:F2} >= golden threshold {config.GoldenThreshold}, saving golden trace...")
                    let goldenName = sprintf "%s_%s" (problem.Title.Replace(" ", "_")) (runId.ToString().Substring(0, 8))
                    // Build a minimal WoTTrace for golden storage
                    let wotTrace : WoTTrace =
                        { RunId = runId
                          Plan =
                            { Id = runId
                              Nodes = []
                              Edges = []
                              EntryNode = "reason_1"
                              Metadata =
                                { Kind = WorkflowOfThought
                                  SourceGoal = problem.Description
                                  CompiledAt = DateTime.UtcNow
                                  EstimatedTokens = None
                                  EstimatedSteps = Some 1 }
                              Policy = [] }
                          Steps =
                            [ { NodeId = "reason_1"
                                NodeType = "Reason"
                                StartedAt = DateTime.UtcNow
                                Status = NodeStatus.Completed(resultText, 0L)
                                Input = Some problem.Description
                                Output = Some resultText
                                Confidence = Some score.Confidence
                                TokensUsed = None } ]
                          StartedAt = DateTime.UtcNow
                          CompletedAt = Some DateTime.UtcNow
                          FinalStatus = if score.Success then "Success" else "Partial" }

                    match GoldenTraceStore.saveAsBaseline goldenName wotTrace with
                    | Ok path -> Console.WriteLine($"[Retroaction] Golden trace saved: {path}")
                    | Error err -> Console.WriteLine($"[Retroaction] Failed to save golden: {err}")

                return Ok (resultText, score, newPattern)
        }

    // =========================================================================
    // Amplify: Stimulated Emission of Patterns
    // =========================================================================

    /// Create N variants of a successful pattern by mutating instruction templates via LLM.
    /// Keep variants that score higher than the original.
    let amplify
        (llm: ILlmService)
        (config: RetroactionConfig)
        (original: PatternDefinition)
        (problem: Problem)
        (originalScore: float)
        : Async<PatternDefinition list> =
        async {
            Console.WriteLine($"[Retroaction] Amplifying pattern '{original.Name}' (score={originalScore:F2}, variants={config.AmplifyCount})")

            let mutable improvements = []

            for i in 1 .. config.AmplifyCount do
                let mutationPrompt =
                    $"""You are TARS's Pattern Mutator.
Given a reasoning pattern template, create a VARIANT that might perform better.

ORIGINAL PATTERN:
Name: {original.Name}
Description: {original.Description}
Template: {original.Template}

PROBLEM CONTEXT: {problem.Description}

MUTATION #{i}: Apply one of these strategies:
- Reorder reasoning steps for better flow
- Add a verification/sanity-check step
- Make instructions more specific to the problem domain
- Add decomposition before the main reasoning step
- Strengthen the synthesis/conclusion step

Output the COMPLETE modified template as valid JSON (same schema as input template).
Output JSON ONLY, no explanation.
"""

                let req =
                    { ModelHint = None
                      Model = None
                      SystemPrompt = Some "You are a pattern mutation engine. Output valid JSON only."
                      MaxTokens = Some 2000
                      Temperature = Some (0.5 + float i * 0.15) // Increasing temperature for diversity
                      Stop = []
                      Messages = [ { Role = Role.User; Content = mutationPrompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = false
                      Seed = None
                      ContextWindow = None }

                try
                    let! resp = llm.CompleteAsync req |> Async.AwaitTask
                    let mutable text = resp.Text.Trim()
                    let firstBrace = text.IndexOf('{')
                    let lastBrace = text.LastIndexOf('}')

                    if firstBrace >= 0 && lastBrace > firstBrace then
                        text <- text.Substring(firstBrace, lastBrace - firstBrace + 1)

                    // Validate it's parseable JSON
                    use _doc = JsonDocument.Parse(text)

                    let variant =
                        { original with
                            Name = sprintf "%s_v%d" original.Name i
                            Template = text
                            Score = 0.0
                            CreatedFromRunId = Some (Guid.NewGuid()) }

                    // Test the variant
                    let context = Map.ofList [ "goal", problem.Description; "input", problem.Description ]
                    let hydrated = PatternLibrary.hydrate variant context
                    let! variantResult = PatternLibrary.executePattern llm variant hydrated problem.Description

                    match variantResult with
                    | Ok variantText ->
                        let variantScore = scoreExecution variantText problem
                        Console.WriteLine($"[Retroaction] Variant {i} score: {variantScore.Overall:F2} (original: {originalScore:F2})")

                        if variantScore.Overall > originalScore then
                            let scoredVariant = { variant with Score = variantScore.Overall }
                            match savePattern scoredVariant with
                            | Ok _ ->
                                Console.WriteLine($"[Retroaction] Variant {i} is an improvement! Saved.")
                                improvements <- scoredVariant :: improvements
                            | Error err ->
                                Console.WriteLine($"[Retroaction] Failed to save variant: {err}")
                    | Error err ->
                        Console.WriteLine($"[Retroaction] Variant {i} execution failed: {err}")
                with ex ->
                    Console.WriteLine($"[Retroaction] Variant {i} mutation failed: {ex.Message}")

            return improvements |> List.rev
        }

    // =========================================================================
    // Coherence Check
    // =========================================================================

    /// Ensures the pattern library maintains diversity and doesn't converge
    /// to a single strategy. Returns warnings and optionally prunes duplicates.
    let coherenceCheck
        (config: RetroactionConfig)
        : string list =
        let patterns = PatternLibrary.loadAll ()
        let mutable warnings = []

        // 1. Check library size
        if patterns.Length > config.MaxLibrarySize then
            warnings <- sprintf "Library size %d exceeds max %d. Consider pruning low-scoring patterns." patterns.Length config.MaxLibrarySize :: warnings

        if patterns.IsEmpty then
            warnings <- "Pattern library is empty. No learning has occurred yet." :: warnings
        else
            // 2. Check diversity: count distinct goals
            let distinctGoals = patterns |> List.map (fun p -> p.Goal) |> List.distinct |> List.length
            let diversity = float distinctGoals / float (max 1 patterns.Length)

            if diversity < config.MinDiversity then
                warnings <- sprintf "Library diversity %.2f is below minimum %.2f. Patterns are converging." diversity config.MinDiversity :: warnings

            // 3. Check for near-duplicate names (possible redundancy)
            let nameClusters =
                patterns
                |> List.groupBy (fun p ->
                    // Strip version suffixes like _v1, _v2
                    let name = p.Name
                    let underscoreV = name.LastIndexOf("_v")
                    if underscoreV > 0 then name.Substring(0, underscoreV) else name)
                |> List.filter (fun (_, group) -> group.Length > 3)

            for (baseName, group) in nameClusters do
                warnings <- sprintf "Pattern '%s' has %d variants. Consider keeping only the top scorer." baseName group.Length :: warnings

            // 4. Check average fitness
            let avgScore = patterns |> List.averageBy (fun p -> p.Score)
            if avgScore < 0.3 then
                warnings <- sprintf "Average pattern fitness %.2f is low. Learning may not be effective." avgScore :: warnings

            // 5. Prune if over limit: remove lowest-scoring patterns
            if patterns.Length > config.MaxLibrarySize then
                let toRemove =
                    patterns
                    |> List.sortBy (fun p -> p.Score)
                    |> List.take (patterns.Length - config.MaxLibrarySize)

                for p in toRemove do
                    let dir = getPatternsDir ()
                    let safeName = p.Name.Replace(" ", "_").Replace("/", "_")
                    let path = Path.Combine(dir, sprintf "%s.json" safeName)
                    if File.Exists path then
                        File.Delete path
                        warnings <- sprintf "Pruned low-scoring pattern '%s' (score=%.2f)" p.Name p.Score :: warnings

        warnings

    // =========================================================================
    // Promotion Pipeline Integration
    // =========================================================================

    /// Feed a completed execution into the promotion pipeline.
    /// Converts cycle data into TraceArtifacts and runs the 7-step loop.
    let feedPromotionPipeline
        (problem: Problem)
        (score: ExecutionScore)
        (pattern: PatternDefinition)
        : PromotionPipeline.PipelineResult list =
        let artifact : PromotionPipeline.TraceArtifact =
            { TaskId = problem.Title
              PatternName = pattern.Name
              PatternTemplate = pattern.Template
              Context = problem.Description
              Score = score.Overall
              Timestamp = System.DateTime.UtcNow
              RollbackExpansion = pattern.RollbackExpansion }
        PromotionPipeline.run 3 [ artifact ]

    // =========================================================================
    // Cycle Metrics
    // =========================================================================

    type CycleResult =
        { Problem: Problem
          Result: string option
          Score: ExecutionScore option
          NewPattern: PatternDefinition option
          Improvements: PatternDefinition list
          PromotionResults: PromotionPipeline.PipelineResult list
          CoherenceWarnings: string list
          LibrarySize: int
          AvgFitness: float
          CurriculumState: CurriculumState }

    // =========================================================================
    // runCycle: The Main Loop Iteration
    // =========================================================================

    /// Run a single retroaction cycle:
    /// 1. Get next problem from curriculum
    /// 2. executeAndLearn
    /// 3. If successful, amplify the pattern
    /// 4. coherenceCheck the library
    /// 5. Record metrics
    let runCycle
        (llm: ILlmService)
        (config: RetroactionConfig)
        (curriculumState: CurriculumState)
        (allProblems: Problem list)
        : Async<Result<CycleResult, string>> =
        async {
            // 1. Get next problem
            match CurriculumManager.getNextProblem curriculumState allProblems with
            | None ->
                return Error "No more problems available in curriculum"
            | Some problem ->
                Console.WriteLine($"[Retroaction] === Cycle Start: {problem.Title} ===")

                // 2. Execute and learn
                let! learnResult = executeAndLearn llm config problem

                match learnResult with
                | Error err ->
                    // Record failure in curriculum
                    let (ProblemId pid) = problem.Id
                    let newState = CurriculumManager.recordFailure curriculumState problem.Id
                    return Ok
                        { Problem = problem
                          Result = None
                          Score = None
                          NewPattern = None
                          Improvements = []
                          PromotionResults = []
                          CoherenceWarnings = [ sprintf "Execution failed: %s" err ]
                          LibrarySize = (PatternLibrary.loadAll ()).Length
                          AvgFitness = 0.0
                          CurriculumState = newState }

                | Ok (resultText, score, newPattern) ->
                    // 3. Update curriculum
                    let newCurrState =
                        if score.Success then
                            CurriculumManager.recordSuccess curriculumState problem.Id
                        else
                            CurriculumManager.recordFailure curriculumState problem.Id

                    // 4. Amplify if we got a good pattern
                    let! improvements =
                        match newPattern with
                        | Some pattern when score.Overall >= config.PatternThreshold ->
                            amplify llm config pattern problem score.Overall
                        | _ ->
                            async { return [] }

                    // 5. Feed promotion pipeline & persist index
                    let promotionResults =
                        match newPattern with
                        | Some pattern ->
                            Console.WriteLine("[Retroaction] Feeding promotion pipeline...")
                            let results = feedPromotionPipeline problem score pattern
                            for r in results do
                                Console.WriteLine(r.AuditReport)

                            // Close the loop: rebuild & persist promotion index
                            // so pattern selectors pick up newly promoted patterns
                            let index = PromotionIndex.refresh ()
                            Console.WriteLine($"[Retroaction] Promotion index refreshed: {index.PatternCount} patterns")

                            results
                        | None -> []

                    // 6. Coherence check
                    let coherenceWarnings = coherenceCheck config

                    // 7. Compute library metrics
                    let allPatterns = PatternLibrary.loadAll ()
                    let avgFitness =
                        if allPatterns.IsEmpty then 0.0
                        else allPatterns |> List.averageBy (fun p -> p.Score)

                    Console.WriteLine($"[Retroaction] === Cycle End: success={score.Success}, library={allPatterns.Length}, avgFitness={avgFitness:F2} ===")

                    // Emit structured cycle output
                    let cycleJson =
                        StructuredOutput.buildCycleOutput
                            (Guid.NewGuid().ToString("N").[..7])
                            problem.Title
                            score.Success
                            score.Overall
                            score.ValidationPassed
                            (newPattern |> Option.map (fun p -> p.Name))
                            newPattern.IsSome
                            improvements.Length
                            promotionResults
                            allPatterns.Length
                            newCurrState.MasteryScore
                        |> StructuredOutput.cycleToJson
                    Console.Error.WriteLine($"[Retroaction] Structured output: {cycleJson.Length} chars")

                    return Ok
                        { Problem = problem
                          Result = Some resultText
                          Score = Some score
                          NewPattern = newPattern
                          Improvements = improvements
                          PromotionResults = promotionResults
                          CoherenceWarnings = coherenceWarnings
                          LibrarySize = allPatterns.Length
                          AvgFitness = avgFitness
                          CurriculumState = newCurrState }
        }

    // =========================================================================
    // Multi-Cycle Runner
    // =========================================================================

    /// Run multiple retroaction cycles, accumulating learning.
    let runCycles
        (llm: ILlmService)
        (config: RetroactionConfig)
        (initialState: CurriculumState)
        (problems: Problem list)
        (maxCycles: int)
        : Async<CycleResult list> =
        async {
            let mutable results = []
            let mutable state = initialState
            let mutable cycleCount = 0

            while cycleCount < maxCycles do
                let! cycleResult = runCycle llm config state problems
                match cycleResult with
                | Ok result ->
                    results <- result :: results
                    state <- result.CurriculumState
                    cycleCount <- cycleCount + 1
                | Error err ->
                    Console.WriteLine($"[Retroaction] Cycle stopped: {err}")
                    cycleCount <- maxCycles // Break

            return results |> List.rev
        }
