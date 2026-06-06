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
      TimeLimitSeconds: int }

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
