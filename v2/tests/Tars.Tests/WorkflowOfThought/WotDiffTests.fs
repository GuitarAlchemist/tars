namespace Tars.Tests.WorkflowOfThought

open Xunit
open Tars.Core.WorkflowOfThought

module WotDiffTests =

    let mkGolden toolCalls mode steps =
        { SchemaVersion = "wot.golden.v1"
          Summary =
            { ToolCalls = toolCalls
              VerifyPassed = Some true
              FirstError = None
              OutputKeys = []
              Mode = mode
              PassRate = Some 1.0
              EstimatedCost = 0.0m
              DiffCount = 0
              TotalTokens = 0 }
          Steps = steps }

    let mkStep id kind status =
        { StepId = id
          Kind = kind
          ToolName = None
          ResolvedArgs = None
          Outputs = []
          Status = status
          Error = None
          Usage = None
          Metadata = None }

    [<Fact>]
    let ``GoldenDiff - Identical goldens have no changes`` () =
        let s1 = mkStep "s1" "reason" StepStatus.Ok
        let g1 = mkGolden 1 "Stub" [ s1 ]
        let g2 = mkGolden 1 "Stub" [ s1 ]

        let diff = GoldenDiff.compute g1 g2
        Assert.False(diff.HasChanges)

    [<Fact>]
    let ``GoldenDiff - Detects summary changes`` () =
        let s1 = mkStep "s1" "reason" StepStatus.Ok
        let g1 = mkGolden 1 "Stub" [ s1 ]
        let g2 = mkGolden 2 "Llm" [ s1 ] // Changed calls and mode

        let diff = GoldenDiff.compute g1 g2
        Assert.True(diff.HasChanges)
        Assert.True(diff.SummaryChanges.ContainsKey "ToolCalls")
        Assert.True(diff.SummaryChanges.ContainsKey "Mode")
        Assert.Equal("Stub", fst diff.SummaryChanges.["Mode"])
        Assert.Equal("Llm", snd diff.SummaryChanges.["Mode"])

    [<Fact>]
    let ``GoldenDiff - Detects step differences`` () =
        let s1 = mkStep "s1" "reason" StepStatus.Ok

        let s1Changed =
            { s1 with
                Status = StepStatus.Error
                Error = Some "Fail" }

        let g1 = mkGolden 1 "Stub" [ s1 ]
        let g2 = mkGolden 1 "Stub" [ s1Changed ]

        let diff = GoldenDiff.compute g1 g2
        Assert.True(diff.HasChanges)
        Assert.True(diff.StepChanges.ContainsKey "s1")

        match diff.StepChanges.["s1"] with
        | Changed changes ->
            Assert.True(changes.ContainsKey "Status")
            Assert.True(changes.ContainsKey "Error")
        | _ -> failwith "Expected Changed"

    [<Fact>]
    let ``GoldenDiff - Detects missing and extra steps`` () =
        let s1 = mkStep "s1" "reason" StepStatus.Ok
        let s2 = mkStep "s2" "work" StepStatus.Ok

        let g1 = mkGolden 1 "Stub" [ s1 ] // Has s1
        let g2 = mkGolden 1 "Stub" [ s2 ] // Has s2 (so s1 missing, s2 extra)

        let diff = GoldenDiff.compute g1 g2
        Assert.True(diff.HasChanges)

        Assert.Equal(MissingInNew, diff.StepChanges.["s1"])
        Assert.Equal(ExtraInNew, diff.StepChanges.["s2"])
