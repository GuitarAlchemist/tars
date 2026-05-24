module Tars.Tests.GaPatternSeederTests

open Xunit
open Xunit.Abstractions
open Tars.Evolution

/// Integration test that runs GA pattern seeding through the real promotion pipeline.
/// This test modifies ~/.tars/promotion/ state — it's the actual cross-repo discovery.
type GaPatternSeederTests(output: ITestOutputHelper) =

    [<Fact>]
    let ``GA patterns seed into promotion pipeline`` () =
        let artifacts = GaPatternSeeder.gaTraceArtifacts ()
        output.WriteLine($"GA trace artifacts: {artifacts.Length}")

        // Verify we have the 5 expected pattern families
        let patternNames = artifacts |> List.map (fun a -> a.PatternName) |> List.distinct
        output.WriteLine($"Distinct patterns: {patternNames.Length}")
        for name in patternNames do
            let count = artifacts |> List.filter (fun a -> a.PatternName = name) |> List.length
            output.WriteLine($"  {name}: {count} occurrences")

        Assert.Equal(5, patternNames.Length)
        Assert.Contains("ga.confidence_evidence_response", patternNames)
        Assert.Contains("ga.domain_skill_fastpath", patternNames)
        Assert.Contains("ga.routing_fallback_cascade", patternNames)
        Assert.Contains("ga.hook_lifecycle_fsm", patternNames)
        Assert.Contains("ga.orchestrator_pipeline", patternNames)

        // Run the pipeline with minOccurrences=3
        let results = GaPatternSeeder.seed 3
        output.WriteLine($"\nPipeline results: {results.Length}")

        for r in results do
            let decision = match r.Decision with
                           | GovernanceDecision.Approve reason -> $"APPROVED: {reason}"
                           | GovernanceDecision.Reject reason -> $"REJECTED: {reason}"
                           | GovernanceDecision.Defer reason -> $"DEFERRED: {reason}"
            let level = PromotionLevel.label r.Candidate.ProposedLevel
            let criteria = PromotionCriteria.score r.Candidate.Criteria
            output.WriteLine($"  {r.Candidate.Record.PatternName}")
            output.WriteLine($"    Level: {PromotionLevel.label r.Candidate.Record.CurrentLevel} → {level}")
            output.WriteLine($"    Criteria: {criteria}/8")
            output.WriteLine($"    Decision: {decision}")
            match r.RoundtripValidation with
            | Some rt -> output.WriteLine($"    Roundtrip: passed={rt.Passed}, semantic={rt.SemanticMatch:F2}")
            | None -> ()
            output.WriteLine("")

        // Pipeline may return 0 results if patterns already reached max level
        // from a previous run — that's valid. The recurrence store check below
        // verifies the artifacts were processed regardless.
        output.WriteLine($"(Pipeline returning 0 results means patterns already at max level)")

        // Verify recurrence records were created
        let records = PromotionPipeline.getRecurrenceRecords ()
        let gaRecords = records |> List.filter (fun r -> r.PatternName.StartsWith("ga."))
        output.WriteLine($"GA recurrence records in store: {gaRecords.Length}")
        for r in gaRecords do
            output.WriteLine($"  {r.PatternName}: occurrences={r.OccurrenceCount}, level={PromotionLevel.label r.CurrentLevel}, score={r.AverageScore:F3}")

        Assert.True(gaRecords.Length >= 3, $"Expected at least 3 GA patterns in recurrence store, got {gaRecords.Length}")

    [<Fact>]
    let ``GA patterns have valid rollback expansions`` () =
        let artifacts = GaPatternSeeder.gaTraceArtifacts ()
        let withRollback = artifacts |> List.filter (fun a -> a.RollbackExpansion.IsSome)

        // At least one artifact per pattern family should have a rollback expansion
        let patternsWithRollback =
            withRollback
            |> List.map (fun a -> a.PatternName)
            |> List.distinct

        Assert.Equal(5, patternsWithRollback.Length) // All 5 pattern families have rollback

        // Each rollback should contain step descriptions
        for a in withRollback do
            let rb = a.RollbackExpansion.Value
            Assert.Contains("step:", rb)
            Assert.Contains("goal:", rb)
            Assert.Contains("output:", rb)
