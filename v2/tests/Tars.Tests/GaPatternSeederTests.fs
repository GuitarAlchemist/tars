module Tars.Tests.GaPatternSeederTests

open Xunit
open Xunit.Abstractions
open Tars.Evolution

/// Runs GA pattern seeding through the real promotion pipeline, against an in-memory
/// store: it must never write to the user's ~/.tars/promotion store (#247).
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

        // Run the pipeline with minOccurrences=3 against a fresh, isolated store
        let store = InMemoryPromotionStore() :> IPromotionStore
        let results = GaPatternSeeder.seed store 3
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

        // The store starts empty, so every GA pattern family must land in it
        let gaRecords =
            store.GetRecurrence () |> List.filter (fun r -> r.PatternName.StartsWith("ga."))
        output.WriteLine($"GA recurrence records in store: {gaRecords.Length}")
        for r in gaRecords do
            output.WriteLine($"  {r.PatternName}: occurrences={r.OccurrenceCount}, level={PromotionLevel.label r.CurrentLevel}, score={r.AverageScore:F3}")

        Assert.Equal(5, gaRecords.Length)
        Assert.NotEmpty(results)

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
