module Tars.Tests.EvalSplitTests

open System
open Xunit
open Tars.Evolution

// Issue #294: self-train exported SFT data from the same problems its A/B benchmark
// scored, so a post-fine-tune gain could be recall. Held-out problems are never exported.

let private bank = ProblemBank.all () @ GaProblemBank.all ()
let private byId = bank |> List.map (fun p -> p.Id, p) |> Map.ofList

[<Fact>]
let ``the held-out split is pinned`` () =
    // Changing the salt or these ids silently invalidates earlier held-out comparisons.
    Assert.Equal<string>(
        [ "basic-reverse-string"
          "inter-safe-divide"
          "adv-matrix-multiply"
          "exp-result-ce"
          "ga-pc-interval" ],
        EvalSplit.heldOut bank |> List.map (fun p -> p.Id)
    )

[<Fact>]
let ``every difficulty and the GA bank keep problems on both sides of the split`` () =
    for difficulty in [ Beginner; Intermediate; Advanced; Expert ] do
        let tier = ProblemBank.byDifficulty difficulty
        Assert.NotEmpty(EvalSplit.heldOut tier)
        Assert.NotEmpty(EvalSplit.training tier)

    Assert.NotEmpty(EvalSplit.heldOut (GaProblemBank.all ()))
    Assert.NotEmpty(EvalSplit.training (GaProblemBank.all ()))

[<Fact>]
let ``timed problems stay in training so fastest-variant data survives`` () =
    let timed = bank |> List.filter (fun p -> p.PerfHarness.IsSome)
    Assert.NotEmpty timed
    Assert.All(timed, fun p -> Assert.False(EvalSplit.isHeldOut p, p.Id))

[<Fact>]
let ``self-train never selects a held-out problem for export`` () =
    let attempt (p: BenchmarkProblem) : BenchmarkAttempt =
        { ProblemId = p.Id
          Difficulty = p.Difficulty
          Category = p.Category
          GeneratedCode = $"let solution () = \"{p.Id}\""
          Compiled = true
          Validated = true
          CompileErrors = []
          ValidationOutput = "PASS"
          GenerationTimeMs = 1L
          ValidationTimeMs = 1L
          ExecutionNs = None
          PropertiesValidated = None
          Timestamp = DateTime.UtcNow }

    let summary: BenchmarkRunSummary =
        { RunId = Guid.NewGuid()
          Timestamp = DateTime.UtcNow
          ModelUsed = "stub"
          TotalProblems = bank.Length
          Compiled = bank.Length
          Validated = bank.Length
          CompileRate = 1.0
          PassRate = 1.0
          Attempts = bank |> List.map attempt
          TotalDurationMs = 0L }

    let chosen, skipped = SelfTrain.selectTrainingAttempts byId [ summary ] None
    let chosenIds = chosen |> List.map (fun (a, _) -> a.ProblemId) |> Set.ofList
    let heldOutIds = EvalSplit.heldOut bank |> List.map (fun p -> p.Id) |> Set.ofList

    Assert.Empty(Set.intersect chosenIds heldOutIds)
    Assert.Equal(heldOutIds.Count, skipped)
    Assert.Equal<Set<string>>(EvalSplit.training bank |> List.map (fun p -> p.Id) |> Set.ofList, chosenIds)
