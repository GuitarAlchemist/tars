module Tars.Tests.ToleranceEngineeringTests

open System
open Xunit
open Tars.Core.ToleranceEngineering

/// Tests for Tolerance Engineering module
module ToleranceTests =

    [<Fact>]
    let ``getSpec returns correct values for Critical profile`` () =
        let spec = getSpec Critical
        Assert.Equal(0.95, spec.ConfidenceThreshold)
        Assert.Equal(1.5, spec.SafetyFactor)
        Assert.False(spec.AllowDegraded)

    [<Fact>]
    let ``getSpec returns correct values for Exploratory profile`` () =
        let spec = getSpec Exploratory
        Assert.Equal(0.40, spec.ConfidenceThreshold)
        Assert.Equal(1.0, spec.SafetyFactor)
        Assert.True(spec.AllowDegraded)

    [<Fact>]
    let ``checkTolerance returns WithinTolerance for high confidence`` () =
        let result = checkTolerance Standard 0.95 1
        match result with
        | WithinTolerance(conf, _) -> Assert.Equal(0.95, conf)
        | _ -> Assert.Fail("Expected WithinTolerance")

    [<Fact>]
    let ``checkTolerance returns BelowTolerance with Retry for low confidence`` () =
        // Precise: threshold=0.85, humanReview=0.75
        // Use 0.78 which is above human review but below threshold
        let result = checkTolerance Precise 0.78 1
        match result with
        | BelowTolerance(conf, Retry remaining) -> 
            Assert.Equal(0.78, conf)
            Assert.True(remaining > 0)
        | RequiresHumanReview _ -> 
            Assert.Fail("Confidence 0.78 should be above human review threshold of 0.75")
        | _ -> Assert.Fail("Expected BelowTolerance with Retry")

    [<Fact>]
    let ``checkTolerance returns RequiresHumanReview for very low confidence`` () =
        let result = checkTolerance Critical 0.50 1
        match result with
        | RequiresHumanReview _ -> ()
        | _ -> Assert.Fail("Expected RequiresHumanReview")

    [<Fact>]
    let ``checkTolerance returns ApproachingLimit for confidence in safety buffer`` () =
        // For Standard: threshold=0.70, safety=1.1, effective=0.77
        // A confidence of 0.72 should be in the safety buffer
        let result = checkTolerance Standard 0.72 1
        match result with
        | ApproachingLimit(conf, msg) -> 
            Assert.Equal(0.72, conf)
            Assert.Contains("safety margin", msg)
        | _ -> Assert.Fail("Expected ApproachingLimit")

    [<Fact>]
    let ``checkVariance returns None for insufficient data`` () =
        let result = checkVariance Standard ["only one response"]
        Assert.True(result.IsNone)

    [<Fact>]
    let ``checkVariance returns HighVariance for dissimilar responses`` () =
        let responses = [
            "The quick brown fox jumps over the lazy dog"
            "A completely different unrelated sentence about machines"
        ]
        let result = checkVariance Critical responses
        // Critical has max variance of 5%, this should exceed it
        match result with
        | Some (HighVariance(measured, max)) ->
            Assert.True(measured > max)
        | _ -> Assert.Fail("Expected HighVariance")

    [<Fact>]
    let ``checkVariance returns None for similar responses`` () =
        let responses = [
            "The function returns a list of integers"
            "The function returns a sequence of integers"
        ]
        let result = checkVariance Exploratory responses
        // Exploratory has max variance of 50%, similar responses should be ok
        match result with
        | None -> ()
        | Some (HighVariance(v, _)) when v <= 0.50 -> ()
        | _ -> Assert.Fail("Expected None or low variance")

    [<Fact>]
    let ``custom creates correct tolerance spec`` () =
        let profile = custom 0.80 0.15 4 0.60 1.3
        let spec = getSpec profile
        Assert.Equal(0.80, spec.ConfidenceThreshold)
        Assert.Equal(0.15, spec.MaxVariance)
        Assert.Equal(4, spec.MaxRetries)
        Assert.Equal(0.60, spec.HumanReviewThreshold)
        Assert.Equal(1.3, spec.SafetyFactor)

module VarianceTrackerTests =

    [<Fact>]
    let ``VarianceTracker records responses`` () =
        let tracker = VarianceTracker()
        tracker.RecordResponse("op1", "query", "response 1", 0.85, Standard)
        tracker.RecordResponse("op1", "query", "response 2", 0.90, Standard)
        
        let variance = tracker.GetVariance("op1")
        Assert.True(variance.IsSome)

    [<Fact>]
    let ``VarianceTracker returns None for unknown operation`` () =
        let tracker = VarianceTracker()
        let variance = tracker.GetVariance("unknown")
        Assert.True(variance.IsNone)

    [<Fact>]
    let ``GetHighVarianceOperations returns empty for low variance`` () =
        let tracker = VarianceTracker()
        tracker.RecordResponse("op1", "query", "response", 0.85, Standard)
        tracker.RecordResponse("op1", "query", "response", 0.90, Standard)  // Same response
        
        let highVar = tracker.GetHighVarianceOperations(0.9)
        Assert.Empty(highVar)

module MetricsAggregatorTests =

    [<Fact>]
    let ``MetricsAggregator tracks operations`` () =
        let metrics = MetricsAggregator()
        metrics.Record(WithinTolerance(0.95, 0.1))
        metrics.Record(WithinTolerance(0.85, 0.05))
        metrics.Record(ApproachingLimit(0.75, "warning"))
        
        let m = metrics.GetMetrics()
        Assert.Equal(3, m.TotalOperations)
        Assert.Equal(2, m.WithinTolerance)
        Assert.Equal(1, m.ApproachingLimit)

    [<Fact>]
    let ``MetricsAggregator calculates averages correctly`` () =
        let metrics = MetricsAggregator()
        metrics.Record(WithinTolerance(0.90, 0.1))
        metrics.Record(WithinTolerance(0.80, 0.05))
        
        let m = metrics.GetMetrics()
        Assert.Equal(0.85, m.AverageConfidence, 2)

    [<Fact>]
    let ``GenerateReport produces formatted output`` () =
        let metrics = MetricsAggregator()
        metrics.Record(WithinTolerance(0.95, 0.1))
        
        let report = metrics.GenerateReport()
        Assert.Contains("NONDETERMINISM TOLERANCE REPORT", report)
        Assert.Contains("Total Operations", report)
        Assert.Contains("HEALTHY", report)

    [<Fact>]
    let ``GenerateReport shows WARNING for low success rate`` () =
        let metrics = MetricsAggregator()
        // Add mostly failures
        for _ in 1..8 do
            metrics.Record(BelowTolerance(0.50, Abort "low"))
        for _ in 1..2 do
            metrics.Record(WithinTolerance(0.90, 0.1))
        
        let report = metrics.GenerateReport()
        Assert.Contains("WARNING", report)
