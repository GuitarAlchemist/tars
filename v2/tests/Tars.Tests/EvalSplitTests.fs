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

let private attempt (p: BenchmarkProblem) : BenchmarkAttempt =
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

/// A saved run in which every one of `problems` was solved.
let private solvedRun (problems: BenchmarkProblem list) : BenchmarkRunSummary =
    { RunId = Guid.NewGuid()
      Timestamp = DateTime.UtcNow
      ModelUsed = "stub"
      CycleId = None
      TotalProblems = problems.Length
      Compiled = problems.Length
      Validated = problems.Length
      CompileRate = 1.0
      PassRate = 1.0
      Attempts = problems |> List.map attempt
      TotalDurationMs = 0L }

[<Fact>]
let ``self-train never selects a held-out problem for export`` () =
    let chosen, skipped = SelfTrain.selectTrainingAttempts byId [ solvedRun bank ] None
    let chosenIds = chosen |> List.map (fun (a, _) -> a.ProblemId) |> Set.ofList
    let heldOutIds = EvalSplit.heldOut bank |> List.map (fun p -> p.Id) |> Set.ofList

    Assert.Empty(Set.intersect chosenIds heldOutIds)
    Assert.Equal(heldOutIds.Count, skipped)
    Assert.Equal<Set<string>>(EvalSplit.training bank |> List.map (fun p -> p.Id) |> Set.ofList, chosenIds)

// ADR 0005, backlog item 3: a second, harder bank, with its own held-out split.

let private hard = ProblemBank.hard ()

[<Fact>]
let ``the hard bank shares no id with the other banks`` () =
    let ids = hard |> List.map (fun p -> p.Id)
    Assert.Equal<string>(List.distinct ids, ids)
    Assert.All(ids, fun id -> Assert.False(byId.ContainsKey id, id))

[<Fact>]
let ``the hard bank's held-out split is pinned, and leaves problems to train on`` () =
    Assert.Equal<string>(
        [ "hard-wrap-text"
          "hard-throttled-map"
          "hard-min-window"
          "hard-huffman"
          "hard-life-step" ],
        EvalSplit.heldOut hard |> List.map (fun p -> p.Id)
    )

    Assert.Equal(11, (EvalSplit.training hard).Length)

[<Fact>]
let ``self-train exports the hard bank's training problems and none of its held-out ones`` () =
    // With the map self-train itself uses: a problem missing from it is not held out for it,
    // and exportDataset drops a chosen attempt whose problem it does not know.
    let chosen, skipped =
        SelfTrain.selectTrainingAttempts SelfTrain.problemsById [ solvedRun hard ] None

    let chosenIds = chosen |> List.map (fun (a, _) -> a.ProblemId) |> Set.ofList

    Assert.Equal<Set<string>>(EvalSplit.training hard |> List.map (fun p -> p.Id) |> Set.ofList, chosenIds)
    Assert.Equal((EvalSplit.heldOut hard).Length, skipped)
    Assert.All(chosenIds, fun id -> Assert.True(SelfTrain.problemsById.ContainsKey id, id))
