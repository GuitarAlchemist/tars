namespace Tars.Evolution

open System

/// Category of benchmark problem
type ProblemCategory =
    | StringManipulation
    | Algorithms
    | DataStructures
    | ErrorHandling
    | AsyncPatterns
    | TypeDesign
    | PatternMatching
    /// Music-theory reasoning grounded in the sibling Guitar Alchemist domain
    /// (pitch-class arithmetic, scales, triad quality, Tn set-class equivalence).
    | MusicTheory

/// A curated benchmark problem with deterministic validation.
/// The solution is compiled and validated via dotnet fsi — no LLM-as-judge.
type BenchmarkProblem =
    { Id: string
      Title: string
      Description: string
      Difficulty: ProblemDifficulty
      Category: ProblemCategory
      /// The expected function signature, e.g. "let reverse (s: string) : string = ..."
      ExpectedSignature: string
      /// F# code appended after the solution. Must print "PASS" or "FAIL: reason".
      ValidationCode: string
      /// Optional hints for the LLM prompt
      Hints: string list
      /// Time limit for LLM generation in seconds
      TimeLimitSeconds: int
      /// Optional F# performance harness, appended after a *validated* solution.
      /// It must run the solution over a representative workload (with warmup) and
      /// print one line `ELAPSED_NS: <median-nanoseconds>`. When present, the loop
      /// gains a continuous speed reward on top of the binary PASS/FAIL signal.
      PerfHarness: string option
      /// Optional FsCheck property body (no `#r`/`open` — the runner supplies the
      /// FsCheck header + the solution). It must print `PROP PASS` or `PROP FAIL:`.
      /// Adversarial generated inputs catch solutions that pass the fixed example
      /// cases but violate an invariant (i.e. overfit the test set).
      Properties: string option }

/// Result of attempting a single benchmark problem
type BenchmarkAttempt =
    { ProblemId: string
      Difficulty: ProblemDifficulty
      Category: ProblemCategory
      GeneratedCode: string
      Compiled: bool
      Validated: bool
      CompileErrors: string list
      ValidationOutput: string
      GenerationTimeMs: int64
      ValidationTimeMs: int64
      /// Median execution time of the validated solution over a warmed workload,
      /// in nanoseconds. None unless the problem defines a PerfHarness and the
      /// solution validated. This is the continuous speed-reward signal.
      ExecutionNs: int64 option
      /// Result of FsCheck property testing: Some true (held), Some false (an
      /// invariant was falsified — overfit/buggy despite passing examples), or
      /// None (no Properties defined, or not run).
      PropertiesValidated: bool option
      Timestamp: DateTime }

/// Aggregate results for a benchmark run
type BenchmarkRunSummary =
    { RunId: Guid
      Timestamp: DateTime
      ModelUsed: string
      TotalProblems: int
      Compiled: int
      Validated: int
      CompileRate: float
      PassRate: float
      Attempts: BenchmarkAttempt list
      TotalDurationMs: int64 }
