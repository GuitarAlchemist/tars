/// <summary>
/// Tolerance Engineering - Managing Nondeterminism with Engineering Rigor
/// =======================================================================
/// 
/// Inspired by Martin Fowler's insight that LLMs require "tolerances" 
/// similar to structural engineering - knowing how much variance is 
/// acceptable before failure.
/// 
/// "What are the tolerances of nondeterminism that we have to deal with?"
/// - Martin Fowler, December 2024
/// 
/// Just as a structural engineer calculates safety factors for bridges,
/// TARS calculates safety factors for LLM operations to ensure reliability.
/// </summary>
namespace Tars.Core

open System
open System.Collections.Concurrent

/// Tolerance Engineering - bringing structural engineering rigor to AI
module ToleranceEngineering =

    // =========================================================================
    // Core Types
    // =========================================================================

    /// Tolerance specification - like a structural engineer's safety margins
    type ToleranceSpec = {
        /// Minimum confidence threshold for this operation (0.0-1.0)
        ConfidenceThreshold: float
        
        /// Maximum acceptable variance between runs (0.0-1.0)
        MaxVariance: float
        
        /// Number of retries before escalation/failure
        MaxRetries: int
        
        /// Confidence below which human review is required
        HumanReviewThreshold: float
        
        /// Safety factor (like structural engineering's 1.5-2.0x margins)
        /// Applied: effectiveThreshold = ConfidenceThreshold * SafetyFactor
        SafetyFactor: float
        
        /// Whether to allow degraded operation below threshold
        AllowDegraded: bool
    }

    /// Tolerance profiles for different operation types
    type ToleranceProfile =
        | Critical      // Safety-critical: code execution, system changes
        | Precise       // High precision needed: code generation, refactoring
        | Standard      // Normal operations: analysis, explanation
        | Exploratory   // Fuzzy is fine: brainstorming, ideation
        | Custom of ToleranceSpec

    /// Track variance across LLM calls
    type VarianceRecord = {
        OperationId: string
        Query: string
        Responses: (string * float * DateTime) list  // response, confidence, timestamp
        MeasuredVariance: float option
        Profile: ToleranceProfile
    }

    /// Result of a tolerance check
    type ToleranceResult =
        | WithinTolerance of actualConfidence: float * margin: float
        | ApproachingLimit of actualConfidence: float * warningMessage: string
        | BelowTolerance of actualConfidence: float * requiredAction: ToleranceAction
        | RequiresHumanReview of reason: string
        | HighVariance of measured: float * max: float

    and ToleranceAction =
        | Retry of attemptsRemaining: int
        | Escalate of reason: string
        | Degrade of fallbackStrategy: string
        | Abort of reason: string

    /// Nondeterminism metrics for a session
    type NondeterminismMetrics = {
        TotalOperations: int
        WithinTolerance: int
        ApproachingLimit: int
        BelowTolerance: int
        HumanReviewsRequested: int
        AverageConfidence: float
        AverageVariance: float
        SafetyMarginUsed: float  // How much of the safety buffer we're using
    }

    // =========================================================================
    // Tolerance Profile Definitions
    // =========================================================================

    /// Get ToleranceSpec for a profile
    let getSpec (profile: ToleranceProfile) : ToleranceSpec =
        match profile with
        | Critical ->
            { ConfidenceThreshold = 0.95
              MaxVariance = 0.05
              MaxRetries = 5
              HumanReviewThreshold = 0.90
              SafetyFactor = 1.5  // 50% safety margin
              AllowDegraded = false }
        
        | Precise ->
            { ConfidenceThreshold = 0.85
              MaxVariance = 0.10
              MaxRetries = 3
              HumanReviewThreshold = 0.75
              SafetyFactor = 1.2  // 20% safety margin
              AllowDegraded = false }
        
        | Standard ->
            { ConfidenceThreshold = 0.70
              MaxVariance = 0.20
              MaxRetries = 2
              HumanReviewThreshold = 0.50
              SafetyFactor = 1.1
              AllowDegraded = true }
        
        | Exploratory ->
            { ConfidenceThreshold = 0.40
              MaxVariance = 0.50
              MaxRetries = 1
              HumanReviewThreshold = 0.20
              SafetyFactor = 1.0  // No safety margin needed
              AllowDegraded = true }
        
        | Custom spec -> spec

    /// Create a custom tolerance spec
    let custom threshold variance retries humanThreshold safety =
        Custom {
            ConfidenceThreshold = threshold
            MaxVariance = variance
            MaxRetries = retries
            HumanReviewThreshold = humanThreshold
            SafetyFactor = safety
            AllowDegraded = threshold < 0.70
        }

    // =========================================================================
    // Tolerance Checking
    // =========================================================================

    /// Check if a confidence value is within tolerance
    let checkTolerance 
        (profile: ToleranceProfile) 
        (actualConfidence: float) 
        (currentRetries: int) 
        : ToleranceResult =
        
        let spec = getSpec profile
        let effectiveThreshold = spec.ConfidenceThreshold * spec.SafetyFactor
        
        // Calculate margin from threshold
        let margin = actualConfidence - effectiveThreshold
        
        if actualConfidence < spec.HumanReviewThreshold then
            RequiresHumanReview (
                sprintf "Confidence %.1f%% is below human review threshold %.1f%%" 
                    (actualConfidence * 100.0) 
                    (spec.HumanReviewThreshold * 100.0))
        
        elif actualConfidence < spec.ConfidenceThreshold then
            // Below base threshold
            if currentRetries < spec.MaxRetries then
                BelowTolerance(actualConfidence, Retry(spec.MaxRetries - currentRetries))
            elif spec.AllowDegraded then
                BelowTolerance(actualConfidence, Degrade "Operating in degraded mode due to low confidence")
            else
                BelowTolerance(actualConfidence, Abort "Confidence too low and degraded operation not allowed")
        
        elif actualConfidence < effectiveThreshold then
            // Below effective threshold but above base - within safety buffer
            ApproachingLimit(
                actualConfidence,
                sprintf "Using %.0f%% of safety margin (conf: %.1f%%, threshold: %.1f%%)" 
                    ((1.0 - margin / (effectiveThreshold - spec.ConfidenceThreshold)) * 100.0)
                    (actualConfidence * 100.0)
                    (effectiveThreshold * 100.0))
        
        else
            // Fully within tolerance
            WithinTolerance(actualConfidence, margin)

    /// Check variance across multiple runs
    let checkVariance 
        (profile: ToleranceProfile) 
        (responses: string list) 
        : ToleranceResult option =
        
        if responses.Length < 2 then
            None
        else
            let spec = getSpec profile
            
            // Calculate semantic variance (simplified - in practice use embeddings)
            // Here we approximate with string similarity
            let calculateSimilarity (s1: string) (s2: string) =
                let words1 = s1.ToLowerInvariant().Split([|' '; '.'; ','; '\n'|], StringSplitOptions.RemoveEmptyEntries) |> Set.ofArray
                let words2 = s2.ToLowerInvariant().Split([|' '; '.'; ','; '\n'|], StringSplitOptions.RemoveEmptyEntries) |> Set.ofArray
                if Set.isEmpty words1 || Set.isEmpty words2 then 0.0
                else
                    let intersection = Set.intersect words1 words2 |> Set.count |> float
                    let union = Set.union words1 words2 |> Set.count |> float
                    intersection / union  // Jaccard similarity
            
            // Calculate pairwise variance
            let pairs = 
                [for i in 0..responses.Length-2 do
                    for j in i+1..responses.Length-1 do
                        yield calculateSimilarity responses.[i] responses.[j]]
            
            let avgSimilarity = if pairs.IsEmpty then 1.0 else List.average pairs
            let measuredVariance = 1.0 - avgSimilarity
            
            if measuredVariance > spec.MaxVariance then
                Some (HighVariance(measuredVariance, spec.MaxVariance))
            else
                None

    // =========================================================================
    // Variance Tracker
    // =========================================================================

    /// Tracks variance across operations
    type VarianceTracker() =
        let records = ConcurrentDictionary<string, VarianceRecord>()
        
        /// Record a response for variance tracking
        member _.RecordResponse(operationId: string, query: string, response: string, confidence: float, profile: ToleranceProfile) =
            let entry = (response, confidence, DateTime.UtcNow)
            records.AddOrUpdate(
                operationId,
                (fun _ -> 
                    { OperationId = operationId
                      Query = query
                      Responses = [entry]
                      MeasuredVariance = None
                      Profile = profile }),
                (fun _ existing -> 
                    let newResponses = existing.Responses @ [entry] |> List.truncate 10  // Keep last 10
                    let responses = newResponses |> List.map (fun (r, _, _) -> r)
                    let variance = 
                        if responses.Length >= 2 then
                            match checkVariance profile responses with
                            | Some (HighVariance(v, _)) -> Some v
                            | _ -> 
                                // Calculate actual variance even if within tolerance
                                let similarities = 
                                    [for i in 0..responses.Length-2 do
                                        let words1 = responses.[i].Split(' ') |> Set.ofArray
                                        let words2 = responses.[i+1].Split(' ') |> Set.ofArray
                                        if Set.isEmpty words1 || Set.isEmpty words2 then 0.0
                                        else float (Set.intersect words1 words2 |> Set.count) / float (Set.union words1 words2 |> Set.count)]
                                if similarities.IsEmpty then None
                                else Some (1.0 - List.average similarities)
                        else None
                    { existing with 
                        Responses = newResponses
                        MeasuredVariance = variance })
            ) |> ignore
        
        /// Get variance for an operation
        member _.GetVariance(operationId: string) : float option =
            match records.TryGetValue(operationId) with
            | true, record -> record.MeasuredVariance
            | false, _ -> None
        
        /// Get all high-variance operations
        member _.GetHighVarianceOperations(threshold: float) : VarianceRecord list =
            records.Values
            |> Seq.filter (fun r -> 
                match r.MeasuredVariance with
                | Some v -> v > threshold
                | None -> false)
            |> Seq.toList
        
        /// Clear old records
        member _.Cleanup(maxAge: TimeSpan) =
            let cutoff = DateTime.UtcNow - maxAge
            for kvp in records do
                let record = kvp.Value
                match record.Responses with
                | [] -> records.TryRemove(kvp.Key) |> ignore
                | responses ->
                    let latest = responses |> List.map (fun (_, _, t) -> t) |> List.max
                    if latest < cutoff then
                        records.TryRemove(kvp.Key) |> ignore

    // =========================================================================
    // Metrics Aggregation
    // =========================================================================

    /// Session-level metrics tracker
    type MetricsAggregator() =
        let mutable totalOps = 0
        let mutable withinTolerance = 0
        let mutable approachingLimit = 0
        let mutable belowTolerance = 0
        let mutable humanReviews = 0
        let mutable confidenceSum = 0.0
        let mutable varianceSum = 0.0
        let mutable varianceCount = 0
        let mutable safetyMarginUsed = 0.0
        let lockObj = obj()
        
        /// Record a tolerance result
        member _.Record(result: ToleranceResult, ?variance: float) =
            lock lockObj (fun () ->
                totalOps <- totalOps + 1
                match result with
                | WithinTolerance(conf, margin) ->
                    withinTolerance <- withinTolerance + 1
                    confidenceSum <- confidenceSum + conf
                | ApproachingLimit(conf, _) ->
                    approachingLimit <- approachingLimit + 1
                    confidenceSum <- confidenceSum + conf
                    safetyMarginUsed <- safetyMarginUsed + 0.5  // Approximate
                | BelowTolerance(conf, _) ->
                    belowTolerance <- belowTolerance + 1
                    confidenceSum <- confidenceSum + conf
                    safetyMarginUsed <- safetyMarginUsed + 1.0
                | RequiresHumanReview _ ->
                    humanReviews <- humanReviews + 1
                | HighVariance(v, _) ->
                    varianceSum <- varianceSum + v
                    varianceCount <- varianceCount + 1
                
                match variance with
                | Some v ->
                    varianceSum <- varianceSum + v
                    varianceCount <- varianceCount + 1
                | None -> ()
            )
        
        /// Get current metrics
        member _.GetMetrics() : NondeterminismMetrics =
            lock lockObj (fun () ->
                { TotalOperations = totalOps
                  WithinTolerance = withinTolerance
                  ApproachingLimit = approachingLimit
                  BelowTolerance = belowTolerance
                  HumanReviewsRequested = humanReviews
                  AverageConfidence = if totalOps > 0 then confidenceSum / float totalOps else 0.0
                  AverageVariance = if varianceCount > 0 then varianceSum / float varianceCount else 0.0
                  SafetyMarginUsed = if totalOps > 0 then safetyMarginUsed / float totalOps else 0.0 }
            )
        
        /// Generate a tolerance report
        member this.GenerateReport() : string =
            let m = this.GetMetrics()
            let successRate = if m.TotalOperations > 0 then float m.WithinTolerance / float m.TotalOperations * 100.0 else 0.0
            
            sprintf """
╔══════════════════════════════════════════════════════════════════════╗
║               NONDETERMINISM TOLERANCE REPORT                        ║
╠══════════════════════════════════════════════════════════════════════╣
║  Total Operations:        %5d                                       ║
║  Within Tolerance:        %5d (%.1f%%)                              ║
║  Approaching Limit:       %5d                                       ║
║  Below Tolerance:         %5d                                       ║
║  Human Reviews Requested: %5d                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  Average Confidence:      %.1f%%                                     ║
║  Average Variance:        %.1f%%                                     ║
║  Safety Margin Usage:     %.1f%%                                     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Status: %s                                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""
                m.TotalOperations
                m.WithinTolerance successRate
                m.ApproachingLimit
                m.BelowTolerance
                m.HumanReviewsRequested
                (m.AverageConfidence * 100.0)
                (m.AverageVariance * 100.0)
                (m.SafetyMarginUsed * 100.0)
                (if successRate >= 90.0 then "✅ HEALTHY - Operating within tolerances"
                 elif successRate >= 70.0 then "⚠️ CAUTION - Approaching tolerance limits"
                 else "🔴 WARNING - Significant tolerance violations")

    // =========================================================================
    // Tolerance-Aware Execution
    // =========================================================================

    /// Execute an operation with tolerance checking
    let executeWithTolerance<'T>
        (profile: ToleranceProfile)
        (getConfidence: 'T -> float)
        (operation: unit -> Async<'T>)
        (metrics: MetricsAggregator)
        : Async<Result<'T, string>> =
        async {
            let spec = getSpec profile
            let mutable attempts = 0
            let mutable lastResult: 'T option = None
            let mutable lastError = ""
            
            while attempts < spec.MaxRetries && Option.isNone lastResult do
                attempts <- attempts + 1
                try
                    let! result = operation()
                    let confidence = getConfidence result
                    
                    match checkTolerance profile confidence attempts with
                    | WithinTolerance _ ->
                        metrics.Record(WithinTolerance(confidence, 0.0))
                        lastResult <- Some result
                    | ApproachingLimit(_, msg) ->
                        metrics.Record(ApproachingLimit(confidence, msg))
                        lastResult <- Some result  // Accept but log warning
                        printfn $"⚠️ Tolerance warning: %s{msg}"
                    | BelowTolerance(_, Retry remaining) ->
                        printfn "🔄 Retrying (attempt %d/%d) - confidence %.1f%% below threshold" attempts spec.MaxRetries (confidence * 100.0)
                        lastError <- sprintf "Confidence too low: %.1f%%" (confidence * 100.0)
                    | BelowTolerance(_, Degrade msg) ->
                        metrics.Record(BelowTolerance(confidence, Degrade msg))
                        printfn $"⚠️ Degraded operation: %s{msg}"
                        lastResult <- Some result  // Accept in degraded mode
                    | BelowTolerance(_, Abort reason) ->
                        metrics.Record(BelowTolerance(confidence, Abort reason))
                        lastError <- reason
                        attempts <- spec.MaxRetries  // Stop retrying
                    | BelowTolerance(_, Escalate reason) ->
                        metrics.Record(BelowTolerance(confidence, Escalate reason))
                        lastError <- reason
                        attempts <- spec.MaxRetries
                    | RequiresHumanReview reason ->
                        metrics.Record(RequiresHumanReview reason)
                        lastError <- $"Human review required: %s{reason}"
                        attempts <- spec.MaxRetries
                    | HighVariance(v, max) ->
                        metrics.Record(HighVariance(v, max))
                        lastError <- sprintf "High variance detected: %.1f%% (max %.1f%%)" (v * 100.0) (max * 100.0)
                        
                with ex ->
                    lastError <- ex.Message
                    printfn $"❌ Operation failed: %s{ex.Message}"

            match lastResult with
            | Some r -> return Result<'T, string>.Ok r
            | None -> return Result<'T, string>.Error lastError
        }
