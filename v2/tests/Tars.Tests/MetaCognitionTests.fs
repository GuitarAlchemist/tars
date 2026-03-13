namespace Tars.Tests

open System
open Xunit
open Tars.Core.MetaCognition

/// Tests for the MetaCognition pure-logic layer (no LLM required).
module MetaCognitionTests =

    // =========================================================================
    // Test data helpers
    // =========================================================================

    let makeFailure runId goal pattern error tags score =
        { RunId = runId
          Goal = goal
          PatternUsed = pattern
          ErrorMessage = error
          TraceStepCount = 3
          FailedAtStep = None
          Timestamp = DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc)
          Tags = tags
          Score = score }

    // =========================================================================
    // FailureClustering tests
    // =========================================================================

    [<Fact>]
    let ``Jaccard similarity of identical sets is 1.0`` () =
        let result = FailureClustering.jaccardSimilarity [ "a"; "b"; "c" ] [ "a"; "b"; "c" ]
        Assert.Equal(1.0, result)

    [<Fact>]
    let ``Jaccard similarity of disjoint sets is 0.0`` () =
        let result = FailureClustering.jaccardSimilarity [ "a"; "b" ] [ "c"; "d" ]
        Assert.Equal(0.0, result)

    [<Fact>]
    let ``Jaccard similarity of empty sets is 1.0`` () =
        let result = FailureClustering.jaccardSimilarity [] []
        Assert.Equal(1.0, result)

    [<Fact>]
    let ``Jaccard similarity of overlapping sets`` () =
        let result = FailureClustering.jaccardSimilarity [ "a"; "b"; "c" ] [ "b"; "c"; "d" ]
        // intersection={b,c}=2, union={a,b,c,d}=4, ratio=0.5
        Assert.Equal(0.5, result)

    [<Fact>]
    let ``Error similarity detects similar messages`` () =
        let sim = FailureClustering.errorSimilarity
                    "tool not found: file_reader"
                    "tool not found: code_analyzer"
        Assert.True(sim > 0.5)

    [<Fact>]
    let ``Error similarity of identical messages is 1.0`` () =
        let sim = FailureClustering.errorSimilarity "timeout error" "timeout error"
        Assert.Equal(1.0, sim)

    [<Fact>]
    let ``Failure similarity weights error, goal, and pattern`` () =
        let f1 = makeFailure "1" "analyze code" "react" "tool not found" [] 0.0
        let f2 = makeFailure "2" "analyze code" "react" "tool not found" [] 0.0
        let sim = FailureClustering.failureSimilarity f1 f2
        Assert.Equal(1.0, sim)

    [<Fact>]
    let ``Clustering groups similar failures`` () =
        let failures =
            [ makeFailure "1" "analyze code" "react" "tool not found: analyzer" [] 0.0
              makeFailure "2" "analyze tests" "react" "tool not found: test_runner" [] 0.0
              makeFailure "3" "write poem" "cot" "timeout waiting for response" [] 0.0 ]

        let clusters = FailureClustering.cluster 0.3 failures
        // First two should cluster together (similar error + pattern), third is separate
        Assert.True(clusters.Length >= 2)

    [<Fact>]
    let ``Clustering empty list returns empty`` () =
        let clusters = FailureClustering.cluster 0.5 []
        Assert.Empty(clusters)

    [<Fact>]
    let ``Root cause classification detects missing tool`` () =
        let failures =
            [ makeFailure "1" "analyze" "react" "tool not found: file_reader" [] 0.0 ]
        let cause = FailureClustering.classifyRootCause failures
        match cause with
        | FailureRootCause.MissingTool _ -> ()
        | other -> Assert.Fail(sprintf "Expected MissingTool, got %A" other)

    [<Fact>]
    let ``Root cause classification detects external failure`` () =
        let failures =
            [ makeFailure "1" "search" "react" "connection refused to API" [] 0.0 ]
        let cause = FailureClustering.classifyRootCause failures
        match cause with
        | FailureRootCause.ExternalFailure _ -> ()
        | other -> Assert.Fail(sprintf "Expected ExternalFailure, got %A" other)

    [<Fact>]
    let ``Root cause classification detects wrong pattern`` () =
        let failures =
            [ makeFailure "1" "compare" "cot" "validation failed: no match" [] 0.0 ]
        let cause = FailureClustering.classifyRootCause failures
        match cause with
        | FailureRootCause.WrongPattern _ -> ()
        | other -> Assert.Fail(sprintf "Expected WrongPattern, got %A" other)

    [<Fact>]
    let ``Build clusters assigns IDs and metadata`` () =
        let failures =
            [ makeFailure "1" "analyze" "react" "tool not found" [] 0.2
              makeFailure "2" "analyze" "react" "tool not found" [] 0.3 ]

        let clusters = FailureClustering.buildClusters 0.3 failures
        Assert.True(clusters.Length >= 1)
        let c = clusters |> List.head
        Assert.True(c.ClusterId.StartsWith("cluster-"))
        Assert.True(c.Frequency >= 1)

    // =========================================================================
    // GapDetection tests
    // =========================================================================

    [<Fact>]
    let ``Extract domain tags from goal`` () =
        let tags = GapDetection.extractDomainTags "analyze and refactor the code"
        Assert.Contains("analysis", tags)
        Assert.Contains("refactoring", tags)
        Assert.Contains("code-generation", tags)

    [<Fact>]
    let ``Extract domain tags defaults to general`` () =
        let tags = GapDetection.extractDomainTags "hello world"
        Assert.Equal<string list>([ "general" ], tags)

    [<Fact>]
    let ``Failure rate by domain computes correctly`` () =
        let successes = [ "analyze code", [ "analysis" ]; "write code", [ "code-generation" ] ]
        let failures =
            [ makeFailure "1" "analyze bugs" "react" "failed" [ "analysis" ] 0.0
              makeFailure "2" "analyze perf" "react" "failed" [ "analysis" ] 0.0 ]

        let rates = GapDetection.failureRateByDomain successes failures
        let (rate, failCount, total) = rates.["analysis"]
        Assert.Equal(2, failCount)
        Assert.Equal(3, total)
        Assert.True(rate > 0.6)

    [<Fact>]
    let ``Detect gaps finds high-failure domains`` () =
        let failures =
            [ makeFailure "1" "test code" "react" "failed" [ "testing" ] 0.0
              makeFailure "2" "test more" "react" "failed" [ "testing" ] 0.0
              makeFailure "3" "test again" "cot" "failed" [ "testing" ] 0.0 ]

        let successes = [ "write code", [ "code-generation" ] ]
        let clusters = FailureClustering.buildClusters 0.3 failures

        let gaps = GapDetection.detectGaps 0.5 clusters successes failures
        Assert.True(gaps.Length >= 1)
        let gap = gaps |> List.head
        Assert.Equal("testing", gap.Domain)
        Assert.True(gap.FailureRate >= 0.5)

    [<Fact>]
    let ``Rank gaps sorts by severity`` () =
        let gaps =
            [ { GapId = "gap-1"; Domain = "testing"; Description = ""; FailureRate = 0.8
                SampleSize = 10; RelatedClusters = []; SuggestedRemedy = GapRemedy.LearnPattern "x"
                DetectedAt = DateTime.UtcNow; Confidence = 0.9 }
              { GapId = "gap-2"; Domain = "code"; Description = ""; FailureRate = 0.3
                SampleSize = 5; RelatedClusters = []; SuggestedRemedy = GapRemedy.LearnPattern "y"
                DetectedAt = DateTime.UtcNow; Confidence = 0.5 } ]

        let ranked = GapDetection.rankGaps gaps
        Assert.Equal("gap-1", ranked.[0].GapId)  // Higher severity first

    [<Fact>]
    let ``Suggest remedy for missing tool`` () =
        let remedy = GapDetection.suggestRemedy (FailureRootCause.MissingTool "analyzer")
        match remedy with
        | GapRemedy.AcquireTool(name, _) -> Assert.Equal("analyzer", name)
        | other -> Assert.Fail(sprintf "Expected AcquireTool, got %A" other)

    [<Fact>]
    let ``Suggest remedy for knowledge gap`` () =
        let remedy = GapDetection.suggestRemedy (FailureRootCause.KnowledgeGap "quantum computing")
        match remedy with
        | GapRemedy.IngestKnowledge(domain, _) -> Assert.Equal("quantum computing", domain)
        | other -> Assert.Fail(sprintf "Expected IngestKnowledge, got %A" other)

    // =========================================================================
    // AdaptiveSignals tests
    // =========================================================================

    [<Fact>]
    let ``No signals for successful steps`` () =
        let steps =
            [ { StepId = "s1"; Kind = "Reason"; Succeeded = true; ErrorMessage = None
                DurationMs = 100L; Confidence = Some 0.9 }
              { StepId = "s2"; Kind = "Tool"; Succeeded = true; ErrorMessage = None
                DurationMs = 200L; Confidence = Some 0.85 } ]

        let signals = AdaptiveSignals.evaluateProgress AdaptiveSignals.defaultConfig steps None
        Assert.Empty(signals)

    [<Fact>]
    let ``Confidence dropping signal fires below threshold`` () =
        let steps =
            [ { StepId = "s1"; Kind = "Reason"; Succeeded = true; ErrorMessage = None
                DurationMs = 100L; Confidence = Some 0.2 } ]

        let signals = AdaptiveSignals.evaluateProgress AdaptiveSignals.defaultConfig steps None
        let hasConfidenceSignal =
            signals |> List.exists (fun s ->
                match s with
                | AdaptationSignal.ConfidenceDropping _ -> true
                | _ -> false)
        Assert.True(hasConfidenceSignal)

    [<Fact>]
    let ``Consecutive failures signal fires`` () =
        let steps =
            [ { StepId = "s1"; Kind = "Tool"; Succeeded = false; ErrorMessage = Some "err"
                DurationMs = 100L; Confidence = None }
              { StepId = "s2"; Kind = "Tool"; Succeeded = false; ErrorMessage = Some "err"
                DurationMs = 100L; Confidence = None }
              { StepId = "s3"; Kind = "Tool"; Succeeded = false; ErrorMessage = Some "err"
                DurationMs = 100L; Confidence = None } ]

        let signals = AdaptiveSignals.evaluateProgress AdaptiveSignals.defaultConfig steps None
        let hasConsecutive =
            signals |> List.exists (fun s ->
                match s with
                | AdaptationSignal.ConsecutiveFailures n when n >= 3 -> true
                | _ -> false)
        Assert.True(hasConsecutive)

    [<Fact>]
    let ``Budget exhaustion signal fires`` () =
        let steps =
            [ for i in 1..9 do
                { StepId = sprintf "s%d" i; Kind = "Reason"; Succeeded = true
                  ErrorMessage = None; DurationMs = 100L; Confidence = Some 0.8 } ]

        let signals = AdaptiveSignals.evaluateProgress AdaptiveSignals.defaultConfig steps (Some 10)
        let hasBudget =
            signals |> List.exists (fun s ->
                match s with
                | AdaptationSignal.BudgetExhausting _ -> true
                | _ -> false)
        Assert.True(hasBudget)

    [<Fact>]
    let ``Decide action returns ContinueCurrent for no signals`` () =
        let action = AdaptiveSignals.decideAction [] [ "react"; "got" ]
        Assert.Equal(AdaptationAction.ContinueCurrent, action)

    [<Fact>]
    let ``Decide action switches pattern on consecutive failures`` () =
        let signals = [ AdaptationSignal.ConsecutiveFailures 3 ]
        let action = AdaptiveSignals.decideAction signals [ "got"; "tot" ]
        match action with
        | AdaptationAction.SwitchPattern(pattern, _) -> Assert.Equal("got", pattern)
        | other -> Assert.Fail(sprintf "Expected SwitchPattern, got %A" other)

    [<Fact>]
    let ``Decide action aborts on many consecutive failures`` () =
        let signals = [ AdaptationSignal.ConsecutiveFailures 5 ]
        let action = AdaptiveSignals.decideAction signals [ "got" ]
        match action with
        | AdaptationAction.Abort _ -> ()
        | other -> Assert.Fail(sprintf "Expected Abort, got %A" other)

    [<Fact>]
    let ``Decide action inserts recovery on step failure`` () =
        let signals = [ AdaptationSignal.StepFailing("s1", "timeout") ]
        let action = AdaptiveSignals.decideAction signals []
        match action with
        | AdaptationAction.InsertRecoveryStep _ -> ()
        | other -> Assert.Fail(sprintf "Expected InsertRecoveryStep, got %A" other)

    // =========================================================================
    // ReflectionEngine pure tests
    // =========================================================================

    module private TE =
        open Tars.Core.WorkflowOfThought

        let make stepId kind (status: StepStatus) : TraceEvent =
            { StepId = stepId
              Kind = kind
              StartedAtUtc = DateTime.UtcNow
              EndedAtUtc = DateTime.UtcNow
              DurationMs = 100L
              ToolName = None
              ResolvedArgs = None
              Outputs = [ "output" ]
              Status = status
              Error = None
              Usage = None
              Metadata = Map.empty }

    [<Fact>]
    let ``Compare intent vs outcome with perfect execution`` () =
        let planned = [ "s1"; "s2"; "s3" ]
        let trace =
            [ TE.make "s1" "Reason" Tars.Core.WorkflowOfThought.StepStatus.Ok
              TE.make "s2" "Tool" Tars.Core.WorkflowOfThought.StepStatus.Ok
              TE.make "s3" "Reason" Tars.Core.WorkflowOfThought.StepStatus.Ok ]

        let comparison = Tars.Evolution.ReflectionEngine.compareIntentVsOutcome planned trace "test goal"
        Assert.Equal(3, comparison.PlannedSteps)
        Assert.Equal(3, comparison.ExecutedSteps)
        Assert.Empty(comparison.SkippedSteps)
        Assert.Empty(comparison.FailedSteps)
        Assert.True(comparison.OverallAlignment >= 0.9)

    [<Fact>]
    let ``Compare intent vs outcome with failures`` () =
        let planned = [ "s1"; "s2"; "s3" ]
        let trace =
            [ TE.make "s1" "Reason" Tars.Core.WorkflowOfThought.StepStatus.Ok
              TE.make "s2" "Tool" Tars.Core.WorkflowOfThought.StepStatus.Error ]

        let comparison = Tars.Evolution.ReflectionEngine.compareIntentVsOutcome planned trace "test goal"
        Assert.Equal(1, comparison.FailedSteps.Length)
        Assert.Equal(1, comparison.SkippedSteps.Length)  // s3 was skipped
        Assert.True(comparison.OverallAlignment < 0.9)

    [<Fact>]
    let ``Reflect pure generates meaningful report`` () =
        let comparison =
            { PlannedSteps = 5
              ExecutedSteps = 3
              SkippedSteps = [ "s4"; "s5" ]
              FailedSteps = [ "s3" ]
              UnexpectedSteps = []
              OverallAlignment = 0.4 }

        let report = Tars.Evolution.ReflectionEngine.reflectPure comparison "analyze code quality"
        Assert.Equal("analyze code quality", report.Goal)
        Assert.True(report.Surprises.Length > 0)
        Assert.True(report.LessonsLearned.Length > 0)
        Assert.True(report.SuggestedImprovements.Length > 0)

    [<Fact>]
    let ``Classify outcome detects as-expected`` () =
        let comparison =
            { PlannedSteps = 3; ExecutedSteps = 3; SkippedSteps = []
              FailedSteps = []; UnexpectedSteps = []; OverallAlignment = 0.95 }
        let outcome = Tars.Evolution.ReflectionEngine.classifyOutcome comparison
        Assert.Equal(ReflectionOutcome.AsExpected, outcome)

    [<Fact>]
    let ``Classify outcome detects worse-than-expected`` () =
        let comparison =
            { PlannedSteps = 4; ExecutedSteps = 4; SkippedSteps = []
              FailedSteps = [ "s1"; "s2"; "s3" ]; UnexpectedSteps = []; OverallAlignment = 0.3 }
        let outcome = Tars.Evolution.ReflectionEngine.classifyOutcome comparison
        match outcome with
        | ReflectionOutcome.WorseThanExpected _ -> ()
        | other -> Assert.Fail(sprintf "Expected WorseThanExpected, got %A" other)

    [<Fact>]
    let ``Synthesize lessons deduplicates and counts`` () =
        let reports =
            [ { RunId = "1"; Goal = "a"; IntendedStrategy = ""; ActualBehavior = ""
                Outcome = ReflectionOutcome.AsExpected; Surprises = []
                LessonsLearned = [ "Add validation"; "Check prerequisites" ]
                SuggestedImprovements = []; Timestamp = DateTime.UtcNow }
              { RunId = "2"; Goal = "b"; IntendedStrategy = ""; ActualBehavior = ""
                Outcome = ReflectionOutcome.AsExpected; Surprises = []
                LessonsLearned = [ "Add validation"; "Use better prompts" ]
                SuggestedImprovements = []; Timestamp = DateTime.UtcNow } ]

        let lessons = Tars.Evolution.ReflectionEngine.synthesizeLessons reports
        Assert.True(lessons.Length >= 2)
        // "Add validation" should appear first (mentioned 2x)
        Assert.Contains("observed 2 times", lessons.[0])

    // =========================================================================
    // CurriculumPlanner tests
    // =========================================================================

    [<Fact>]
    let ``Generate tasks from templates`` () =
        let gaps =
            [ { GapId = "gap-testing"; Domain = "testing"; Description = "80% failure rate"
                FailureRate = 0.8; SampleSize = 10; RelatedClusters = []
                SuggestedRemedy = GapRemedy.LearnPattern "test patterns"
                DetectedAt = DateTime.UtcNow; Confidence = 0.9 } ]

        let tasks = Tars.Evolution.CurriculumPlanner.generateTasksFromTemplates gaps 5
        Assert.Equal(1, tasks.Length)
        let task = tasks.[0]
        Assert.Equal("gap-testing", task.GapId)
        Assert.True(task.Description.Contains("testing"))
        Assert.True(task.Priority > 0.0)

    [<Fact>]
    let ``Convert targeted task to Problem`` () =
        let task =
            { TaskId = "t-001"; GapId = "gap-code"; Description = "Practice code generation with new pattern"
              Difficulty = 3; ExpectedOutcome = "success"; ValidationCriteria = Some "no errors"
              Priority = 0.7 }

        let problem = Tars.Evolution.CurriculumPlanner.toProblem task
        Assert.True(problem.Tags |> List.contains "meta-cognitive")
        Assert.True(problem.Tags |> List.contains "gap-code")

    [<Fact>]
    let ``Merge curriculum puts targeted first`` () =
        let existing : Tars.Evolution.Problem list =
            [ { Id = Tars.Evolution.ProblemId "old-1"
                Source = Tars.Evolution.Custom "test"
                Title = "Old task"; Description = "existing"
                Difficulty = Tars.Evolution.Intermediate; Tags = []
                ValidationCriteria = None; ReferenceSolution = None } ]

        let targeted =
            [ { TaskId = "t-001"; GapId = "gap-x"; Description = "targeted task"
                Difficulty = 2; ExpectedOutcome = ""; ValidationCriteria = None; Priority = 0.9 } ]

        let merged = Tars.Evolution.CurriculumPlanner.mergeCurriculum existing targeted
        Assert.Equal(2, merged.Length)
        // Targeted should be first
        Assert.True(merged.[0].Tags |> List.contains "meta-cognitive")
