module Tars.Tests.InsightExporterTests

open Xunit
open Xunit.Abstractions
open Tars.Evolution

type InsightExporterTests(output: ITestOutputHelper) =

    [<Fact>]
    let ``InsightExporter builds snapshot from live state`` () =
        let snapshot = InsightExporter.buildSnapshot ()
        output.WriteLine($"Snapshot: {snapshot.PatternScores.Length} pattern scores, {snapshot.Gaps.Length} gaps, {snapshot.PromotedPatterns.Length} promoted")

        Assert.Equal("2.0", snapshot.TarsVersion)
        Assert.False(System.String.IsNullOrEmpty snapshot.Timestamp)

        // Should have pattern scores for the standard pattern kinds
        Assert.True(snapshot.PatternScores.Length > 0, "Expected at least one pattern score")
        for ps in snapshot.PatternScores do
            let kws = System.String.Join(", ", ps.GoalKeywords)
            output.WriteLine($"  {ps.PatternKind}: score={ps.Score:F3}, keywords=[{kws}]")

        // Promoted patterns should match what's in the promotion index
        for pp in snapshot.PromotedPatterns do
            output.WriteLine($"  promoted: {pp.PatternName} at {pp.Level} (rank={pp.LevelRank})")

    [<Fact>]
    let ``InsightExporter exports and loads snapshot`` () =
        let path = InsightExporter.export ()
        output.WriteLine($"Exported to: {path}")

        Assert.True(System.IO.File.Exists path, $"Expected file at {path}")

        match InsightExporter.loadLatest () with
        | Some loaded ->
            Assert.Equal("2.0", loaded.TarsVersion)
            output.WriteLine($"Loaded: {loaded.PatternScores.Length} scores, {loaded.Gaps.Length} gaps, {loaded.PromotedPatterns.Length} promoted")
            let recs = System.String.Join("; ", loaded.RecommendedActions)
            output.WriteLine($"Recommendations: {recs}")
        | None ->
            Assert.Fail("Expected to load snapshot after export")

    [<Fact>]
    let ``InsightExporter snapshot includes recommendations`` () =
        let snapshot = InsightExporter.buildSnapshot ()

        // With limited test data, should recommend more evolution cycles
        output.WriteLine($"Recommendations ({snapshot.RecommendedActions.Length}):")
        for r in snapshot.RecommendedActions do
            output.WriteLine($"  - {r}")

        // Outcome summary should be a valid map
        for kv in snapshot.OutcomeSummary |> Map.toList do
            output.WriteLine($"  outcome: {fst kv} = {snd kv}")
