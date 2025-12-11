/// Trace Analyzer - QA Agent for debugging agent execution
/// Native Polly-equivalent for analyzing traces and refining prompts
module Tars.Cortex.TraceAnalyzer

open System

// ============================================================================
// Types
// ============================================================================

/// Severity level for identified issues
type IssueSeverity =
    | Info
    | Warning
    | Error
    | Critical

/// A single issue found in execution trace
type Issue =
    { StepIndex: int
      Severity: IssueSeverity
      Category: string
      Description: string
      SuggestedFix: string option }

/// Analysis result for a single trace
type TraceAnalysis =
    { TraceId: string
      Summary: string
      Issues: Issue list
      Suggestions: string list
      EfficiencyScore: float // 0.0 - 1.0
      AnalyzedAt: DateTime }

/// Pattern found across multiple traces
type PatternInfo =
    { Name: string
      Frequency: int
      Description: string
      Impact: string }

/// Report of patterns across traces
type PatternReport =
    { TotalTraces: int
      Patterns: PatternInfo list
      CommonIssues: Issue list
      Recommendations: string list }

/// Execution step for analysis
type ExecutionStep =
    { Index: int
      Action: string
      Input: string option
      Output: string option
      Duration: TimeSpan
      TokensUsed: int option }

/// Execution trace to analyze
type ExecutionTrace =
    { Id: string
      StartTime: DateTime
      EndTime: DateTime
      Steps: ExecutionStep list
      FinalResult: string option
      Success: bool }

// ============================================================================
// Analysis Functions (Pure, no LLM dependency)
// ============================================================================

/// Analyze a single trace for issues
let analyzeTrace (trace: ExecutionTrace) : TraceAnalysis =
    let issues = ResizeArray<Issue>()

    // Check for failure
    if not trace.Success then
        issues.Add
            { StepIndex = trace.Steps.Length - 1
              Severity = IssueSeverity.Error
              Category = "Failure"
              Description = "Trace ended in failure"
              SuggestedFix = Some "Review final step for errors" }

    // Check for slow steps (> 5 seconds)
    trace.Steps
    |> List.iteri (fun i step ->
        if step.Duration.TotalSeconds > 5.0 then
            issues.Add
                { StepIndex = i
                  Severity = Warning
                  Category = "Performance"
                  Description = $"Step '{step.Action}' took {step.Duration.TotalSeconds:F1}s"
                  SuggestedFix = Some "Consider optimizing or parallelizing" })

    // Check for high token usage
    trace.Steps
    |> List.iteri (fun i step ->
        match step.TokensUsed with
        | Some tokens when tokens > 4000 ->
            issues.Add
                { StepIndex = i
                  Severity = Warning
                  Category = "Cost"
                  Description = $"Step '{step.Action}' used {tokens} tokens"
                  SuggestedFix = Some "Consider prompt compression" }
        | _ -> ())

    let totalDuration = (trace.EndTime - trace.StartTime).TotalSeconds

    let efficiency =
        if trace.Success && totalDuration < 10.0 then 0.9
        elif trace.Success then 0.7
        else 0.3

    { TraceId = trace.Id
      Summary = $"Trace with {trace.Steps.Length} steps, duration {totalDuration:F1}s"
      Issues = issues |> Seq.toList
      Suggestions =
        if not trace.Success then
            [ "Review error handling"; "Add retry logic" ]
        elif issues.Count > 0 then
            [ "Optimize slow steps"; "Reduce token usage" ]
        else
            []
      EfficiencyScore = efficiency
      AnalyzedAt = DateTime.UtcNow }

/// Find patterns across multiple traces
let findPatterns (traces: ExecutionTrace list) : PatternReport =
    let successCount = traces |> List.filter (fun t -> t.Success) |> List.length
    let successRate = float successCount / float traces.Length
    let pct = sprintf "%.1f" (successRate * 100.0)

    let avgDuration =
        traces
        |> List.map (fun t -> (t.EndTime - t.StartTime).TotalSeconds)
        |> List.average

    let patterns =
        [ { Name = "Success Rate"
            Frequency = traces.Length
            Description = $"{pct}%% of traces succeeded"
            Impact = if successRate > 0.8 then "positive" else "negative" }
          { Name = "Avg Duration"
            Frequency = traces.Length
            Description = sprintf "Average duration: %.1fs" avgDuration
            Impact = if avgDuration < 10.0 then "positive" else "neutral" } ]

    { TotalTraces = traces.Length
      Patterns = patterns
      CommonIssues = []
      Recommendations =
        if successRate < 0.8 then
            [ "Consider adding retry logic"; "Review error handling" ]
        else
            [] }

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a basic execution step
let createStep index action input output duration tokens =
    { Index = index
      Action = action
      Input = input
      Output = output
      Duration = duration
      TokensUsed = tokens }

/// Create a trace from steps
let createTrace id startTime endTime steps success result =
    { Id = id
      StartTime = startTime
      EndTime = endTime
      Steps = steps
      FinalResult = result
      Success = success }

/// Analyze multiple traces and return combined report
let analyzeTraces (traces: ExecutionTrace list) =
    let analyses = traces |> List.map analyzeTrace
    let patterns = findPatterns traces
    (analyses, patterns)
