module Tars.Tests.PromotionIndexTests

open Xunit
open Xunit.Abstractions
open Tars.Evolution

type PromotionIndexTests(output: ITestOutputHelper) =

    [<Fact>]
    let ``PromotionIndex builds and persists from live stores`` () =
        let index = PromotionIndex.refresh ()
        output.WriteLine($"Index has {index.PatternCount} patterns")

        if index.PatternCount = 0 then
            output.WriteLine("SKIP: No promotion data on disk (CI environment)")
        else

        for entry in index.Entries do
            output.WriteLine($"  {entry.PatternName}: level={PromotionLevel.label entry.Level} (rank={entry.LevelRank}), score={entry.Score:F3}, weight={entry.Weight:F3}")

        let gaEntries = index.Entries |> List.filter (fun e -> e.PatternName.StartsWith("ga."))
        Assert.True(gaEntries.Length >= 5, $"Expected at least 5 GA patterns, got {gaEntries.Length}")

        let ranks = index.Entries |> List.map (fun e -> e.LevelRank)
        let isSorted = ranks |> List.pairwise |> List.forall (fun (a, b) -> a >= b)
        Assert.True(isSorted, "Index entries should be sorted by level rank descending")

    [<Fact>]
    let ``PromotionIndex can be loaded from disk`` () =
        let original = PromotionIndex.refresh ()

        if original.PatternCount = 0 then
            output.WriteLine("SKIP: No promotion data on disk (CI environment)")
        else

        match PromotionIndex.load () with
        | Some loaded ->
            Assert.Equal(original.PatternCount, loaded.PatternCount)
            output.WriteLine($"Loaded {loaded.PatternCount} patterns from disk")
        | None ->
            Assert.Fail("Expected to load index from disk after refresh")

    [<Fact>]
    let ``PromotionIndex findForGoal scores GA patterns`` () =
        let index = PromotionIndex.refresh ()

        if index.PatternCount = 0 then
            output.WriteLine("SKIP: No promotion data on disk (CI environment)")
        else

        let routingMatch = PromotionIndex.findForGoal "route this query to the best agent" index
        let routingName = routingMatch |> Option.map (fun e -> e.PatternName) |> Option.defaultValue "none"
        output.WriteLine($"Routing goal match: {routingName}")

        let skillMatch = PromotionIndex.findForGoal "compute the scale notes for C major" index
        let skillName = skillMatch |> Option.map (fun e -> e.PatternName) |> Option.defaultValue "none"
        output.WriteLine($"Skill goal match: {skillName}")

        Assert.True(routingMatch.IsSome, "Expected a match for routing goal")
        Assert.True(skillMatch.IsSome, "Expected a match for skill goal")
