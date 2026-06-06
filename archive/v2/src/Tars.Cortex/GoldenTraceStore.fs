namespace Tars.Cortex

open System
open System.IO
open System.Text.Json
open Tars.Cortex.WoTTypes

/// <summary>
/// Golden Trace storage and comparison module.
/// Persists canonical traces for regression testing and comparison.
/// </summary>
module GoldenTraceStore =

    // =========================================================================
    // Canonical Trace Types
    // =========================================================================

    /// Canonical representation of a WoT trace step for diffing.
    type CanonicalTraceStep =
        { NodeId: string
          NodeKind: string
          Status: string
          OutputPrefix: string option
          DurationBucket: string }

    /// Canonical representation of a complete WoT trace.
    type CanonicalGolden =
        { TraceId: Guid
          PatternKind: string
          Goal: string
          TotalSteps: int
          SuccessfulSteps: int
          FailedSteps: int
          Steps: CanonicalTraceStep list
          FinalStatus: string
          CreatedAt: DateTime
          Version: string }

    /// Difference between two golden traces.
    type TraceDiff =
        { Field: string
          Expected: string
          Actual: string }

    /// Result of comparing two traces.
    type ComparisonResult =
        { IsMatch: bool
          Diffs: TraceDiff list
          ExpectedPath: string option
          ActualPath: string option }

    // =========================================================================
    // Conversion Functions
    // =========================================================================

    let private durationBucket (ms: int64) =
        if ms < 100L then "fast"
        elif ms < 1000L then "medium"
        else "slow"

    let private statusToString status =
        match status with
        | Completed _ -> "Completed"
        | Failed(err, _) -> sprintf "Failed: %s" (if err.Length > 50 then err.Substring(0, 50) else err)
        | Skipped r -> sprintf "Skipped: %s" r
        | Pending -> "Pending"
        | Running -> "Running"

    let private durationFromStatus status =
        match status with
        | Completed(_, d) -> d
        | Failed(_, d) -> d
        | _ -> 0L

    /// Convert a WoTTrace to canonical form for storage/comparison.
    let toCanonical (trace: WoTTrace) : CanonicalGolden =
        let canonicalSteps =
            trace.Steps 
            |> List.map (fun step ->
                let durationMs = durationFromStatus step.Status
                let outputPrefix = 
                    step.Output 
                    |> Option.map (fun o -> if o.Length > 100 then o.Substring(0, 100) else o)
                
                { NodeId = step.NodeId
                  NodeKind = step.NodeType
                  Status = statusToString step.Status
                  OutputPrefix = outputPrefix
                  DurationBucket = durationBucket durationMs })
        
        let successCount = 
            trace.Steps 
            |> List.filter (fun s -> match s.Status with Completed _ -> true | _ -> false) 
            |> List.length
        
        let failedCount = 
            trace.Steps 
            |> List.filter (fun s -> match s.Status with Failed _ -> true | _ -> false) 
            |> List.length
        
        { TraceId = trace.RunId
          PatternKind = trace.Plan.Metadata.Kind.ToString()
          Goal = trace.Plan.Metadata.SourceGoal
          TotalSteps = trace.Steps.Length
          SuccessfulSteps = successCount
          FailedSteps = failedCount
          Steps = canonicalSteps
          FinalStatus = trace.FinalStatus
          CreatedAt = DateTime.UtcNow
          Version = "1.0" }

    // =========================================================================
    // Serialization
    // =========================================================================

    let private jsonOptions = 
        JsonSerializerOptions(
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase)

    /// Serialize a canonical golden to JSON.
    let serialize (golden: CanonicalGolden) : string =
        JsonSerializer.Serialize(golden, jsonOptions)

    /// Deserialize JSON to a canonical golden.
    let deserialize (json: string) : Result<CanonicalGolden, string> =
        try
            Ok (JsonSerializer.Deserialize<CanonicalGolden>(json, jsonOptions))
        with ex ->
            Error (sprintf "Failed to deserialize golden: %s" ex.Message)

    // =========================================================================
    // Storage Operations
    // =========================================================================

    let private goldenDir = ".wot/goldens"

    let private ensureDir () =
        if not (Directory.Exists goldenDir) then
            Directory.CreateDirectory goldenDir |> ignore

    let private goldenPath (name: string) =
        Path.Combine(goldenDir, sprintf "%s.golden.json" name)

    /// Save a golden trace to disk.
    let save (name: string) (golden: CanonicalGolden) : Result<string, string> =
        try
            ensureDir ()
            let path = goldenPath name
            let json = serialize golden
            File.WriteAllText(path, json)
            Ok path
        with ex ->
            Error (sprintf "Failed to save golden: %s" ex.Message)

    /// Load a golden trace from disk.
    let load (name: string) : Result<CanonicalGolden, string> =
        let path = goldenPath name
        if File.Exists path then
            let json = File.ReadAllText path
            deserialize json
        else
            Error (sprintf "Golden '%s' not found at %s" name path)

    /// List all available golden traces.
    let listAll () : string list =
        ensureDir ()
        Directory.GetFiles(goldenDir, "*.golden.json")
        |> Array.map (fun p -> 
            Path.GetFileName(p).Replace(".golden.json", ""))
        |> Array.toList

    // =========================================================================
    // Comparison
    // =========================================================================

    /// Compare two canonical goldens, ignoring timing and IDs.
    let diff (expected: CanonicalGolden) (actual: CanonicalGolden) : TraceDiff list =
        let mutable diffs = []
        
        if expected.PatternKind <> actual.PatternKind then
            diffs <- { Field = "PatternKind"; Expected = expected.PatternKind; Actual = actual.PatternKind } :: diffs
        
        if expected.TotalSteps <> actual.TotalSteps then
            diffs <- { Field = "TotalSteps"; Expected = string expected.TotalSteps; Actual = string actual.TotalSteps } :: diffs
        
        if expected.SuccessfulSteps <> actual.SuccessfulSteps then
            diffs <- { Field = "SuccessfulSteps"; Expected = string expected.SuccessfulSteps; Actual = string actual.SuccessfulSteps } :: diffs
        
        if expected.FailedSteps <> actual.FailedSteps then
            diffs <- { Field = "FailedSteps"; Expected = string expected.FailedSteps; Actual = string actual.FailedSteps } :: diffs
        
        if expected.FinalStatus <> actual.FinalStatus then
            diffs <- { Field = "FinalStatus"; Expected = expected.FinalStatus; Actual = actual.FinalStatus } :: diffs
        
        // Compare step kinds (order matters)
        let expectedKinds = expected.Steps |> List.map (fun s -> s.NodeKind)
        let actualKinds = actual.Steps |> List.map (fun s -> s.NodeKind)
        
        if expectedKinds <> actualKinds then
            diffs <- { Field = "StepKinds"; Expected = sprintf "%A" expectedKinds; Actual = sprintf "%A" actualKinds } :: diffs
        
        // Compare step statuses
        let expectedStatuses = expected.Steps |> List.map (fun s -> s.Status)
        let actualStatuses = actual.Steps |> List.map (fun s -> s.Status)
        
        if expectedStatuses <> actualStatuses then
            diffs <- { Field = "StepStatuses"; Expected = sprintf "%A" expectedStatuses; Actual = sprintf "%A" actualStatuses } :: diffs
        
        diffs |> List.rev

    /// Compare an actual trace against a stored golden.
    let compareAgainstGolden (goldenName: string) (actual: CanonicalGolden) : ComparisonResult =
        match load goldenName with
        | Ok expected ->
            let diffs = diff expected actual
            { IsMatch = diffs.IsEmpty
              Diffs = diffs
              ExpectedPath = Some (goldenPath goldenName)
              ActualPath = None }
        | Error msg ->
            { IsMatch = false
              Diffs = [{ Field = "Error"; Expected = "Golden exists"; Actual = msg }]
              ExpectedPath = None
              ActualPath = None }

    /// Save a trace as a new golden (baseline for future comparisons).
    let saveAsBaseline (name: string) (trace: WoTTrace) : Result<string, string> =
        let canonical = toCanonical trace
        save name canonical

    /// Compare an actual trace against a stored golden and return a formatted report.
    let compareAndReport (goldenName: string) (actual: WoTTrace) : string =
        let canonicalActual = toCanonical actual
        let result = compareAgainstGolden goldenName canonicalActual
        
        if result.IsMatch then
            sprintf "✅ Golden '%s' matches (TotalSteps: %d, Success: %d)" 
                goldenName canonicalActual.TotalSteps canonicalActual.SuccessfulSteps
        else
            let diffLines = 
                result.Diffs 
                |> List.map (fun d -> sprintf "  - %s: expected '%s', got '%s'" d.Field d.Expected d.Actual)
                |> String.concat "\n"
            sprintf "❌ Golden '%s' differs:\n%s" goldenName diffLines
