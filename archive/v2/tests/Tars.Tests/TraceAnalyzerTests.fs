module Tars.Tests.TraceAnalyzerTests

open System
open Xunit
open Tars.Cortex.TraceAnalyzer

// ============================================================================
// ExecutionTrace Tests
// ============================================================================

[<Fact>]
let ``createStep creates step with all fields`` () =
    let step =
        createStep 0 "think" (Some "input") (Some "output") (TimeSpan.FromSeconds(1.0)) (Some 100)

    Assert.Equal(0, step.Index)
    Assert.Equal("think", step.Action)
    Assert.Equal(Some "input", step.Input)
    Assert.Equal(Some "output", step.Output)
    Assert.Equal(100, step.TokensUsed.Value)

[<Fact>]
let ``createTrace creates trace with steps`` () =
    let start = DateTime.UtcNow
    let endTime = start.AddSeconds(5.0)
    let steps = [ createStep 0 "action1" None None (TimeSpan.FromSeconds(1.0)) None ]

    let trace = createTrace "trace-1" start endTime steps true (Some "result")

    Assert.Equal("trace-1", trace.Id)
    Assert.Equal(1, trace.Steps.Length)
    Assert.True(trace.Success)
    Assert.Equal(Some "result", trace.FinalResult)

// ============================================================================
// analyzeTrace Tests
// ============================================================================

[<Fact>]
let ``analyzeTrace identifies failure in failed trace`` () =
    let trace =
        createTrace "failed-trace" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] false None

    let analysis = analyzeTrace trace

    Assert.Contains(analysis.Issues, fun i -> i.Severity = IssueSeverity.Error)
    Assert.True(analysis.EfficiencyScore < 0.5)

[<Fact>]
let ``analyzeTrace gives high efficiency for successful fast trace`` () =
    let trace =
        createTrace "fast-trace" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(2.0)) [] true None

    let analysis = analyzeTrace trace

    Assert.True(analysis.EfficiencyScore >= 0.8)
    Assert.Empty(analysis.Issues)

[<Fact>]
let ``analyzeTrace flags slow steps`` () =
    let slowStep =
        createStep 0 "slow-action" None None (TimeSpan.FromSeconds(10.0)) None

    let trace =
        createTrace "slow-trace" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(10.0)) [ slowStep ] true None

    let analysis = analyzeTrace trace

    Assert.Contains(analysis.Issues, fun i -> i.Category = "Performance")

[<Fact>]
let ``analyzeTrace flags high token usage`` () =
    let expensiveStep =
        createStep 0 "expensive-action" None None (TimeSpan.FromSeconds(1.0)) (Some 5000)

    let trace =
        createTrace "expensive-trace" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [ expensiveStep ] true None

    let analysis = analyzeTrace trace

    Assert.Contains(analysis.Issues, fun i -> i.Category = "Cost")

[<Fact>]
let ``analyzeTrace includes suggestions for failed traces`` () =
    let trace =
        createTrace "failed-trace" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] false None

    let analysis = analyzeTrace trace

    Assert.NotEmpty(analysis.Suggestions)

// ============================================================================
// findPatterns Tests
// ============================================================================

[<Fact>]
let ``findPatterns calculates success rate`` () =
    let traces =
        [ createTrace "t1" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] true None
          createTrace "t2" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] true None
          createTrace "t3" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] false None ]

    let report = findPatterns traces

    Assert.Equal(3, report.TotalTraces)
    Assert.Contains(report.Patterns, fun p -> p.Name = "Success Rate")

[<Fact>]
let ``findPatterns recommends improvements for low success rate`` () =
    let traces =
        [ createTrace "t1" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] false None
          createTrace "t2" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] false None ]

    let report = findPatterns traces

    Assert.NotEmpty(report.Recommendations)

[<Fact>]
let ``findPatterns returns empty recommendations for high success rate`` () =
    let traces =
        [ createTrace "t1" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] true None
          createTrace "t2" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] true None
          createTrace "t3" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] true None ]

    let report = findPatterns traces

    Assert.Empty(report.Recommendations)

// ============================================================================
// analyzeTraces Tests
// ============================================================================

[<Fact>]
let ``analyzeTraces returns both analyses and patterns`` () =
    let traces =
        [ createTrace "t1" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] true None
          createTrace "t2" DateTime.UtcNow (DateTime.UtcNow.AddSeconds(1.0)) [] false None ]

    let (analyses, patterns) = analyzeTraces traces

    Assert.Equal(2, analyses.Length)
    Assert.Equal(2, patterns.TotalTraces)
