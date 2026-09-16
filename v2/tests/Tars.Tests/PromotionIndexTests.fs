module Tars.Tests.PromotionIndexTests

open System
open System.IO
open Xunit
open Tars.Evolution

// These tests exercise PromotionIndex against an ISOLATED, seeded promotion
// store (a unique temp directory), not the process-global ~/.tars store.
// The store-from-live variants are shared, mutable, and lazily cached, so
// reading them under xUnit's parallel runner is order-dependent and flaky:
// a partial ambient state (PatternCount > 0 but < 5 ga. patterns) slips past
// a naive "skip if empty" guard. Seeding deterministic data in a private
// directory makes the assertions test PromotionIndex.build's *logic* on known
// input — hermetic, parallel-safe, and meaningful (no green-but-dead skip).

let private mkRecord name level score context : RecurrenceRecord =
    { PatternId = name
      PatternName = name
      FirstSeen = DateTime(2026, 1, 1)
      LastSeen = DateTime(2026, 6, 1)
      OccurrenceCount = 5
      TaskIds = [ name + "-t1"; name + "-t2"; name + "-t3" ]
      Contexts = [ context ]
      CurrentLevel = level
      PromotionHistory = [ (level, DateTime(2026, 3, 1)) ]
      AverageScore = score }

/// Deterministic seed: 5 distinct ga.* patterns spanning every promotion
/// level (so the rank-descending sort is actually exercised) plus one
/// non-ga pattern (so the ga. filter is meaningful).
let private seededRecords () : RecurrenceRecord list =
    [ mkRecord "ga.confidence_evidence_response" Helper 0.92
          "TheoryAgent returns structured JSON with confidence score and evidence list"
      mkRecord "ga.domain_compute" Builder 0.88
          "compute the scale notes for C major key from the deterministic domain model"
      mkRecord "ga.embedding_router" DslClause 0.90
          "route this query to the best agent via embedding similarity scoring"
      mkRecord "ga.lifecycle_hooks" Implementation 0.75
          "per-request state lifecycle hooks keyed by correlation id"
      mkRecord "ga.orchestrator_pipeline" GrammarRule 0.85
          "orchestrator pipeline pre hooks skill scan agent dispatch post hooks"
      mkRecord "tars.meta_reflection" Helper 0.70
          "tars self-reflection on completed task outcomes" ]

/// Run a test body against a private, seeded promotion directory; always
/// cleaned up afterwards.
let private withSeededStore (f: string -> unit) =
    let dir =
        Path.Combine(Path.GetTempPath(), "tars-promo-test-" + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory dir |> ignore
    try
        PromotionPipeline.saveStoresTo dir (seededRecords ()) []
        f dir
    finally
        try Directory.Delete(dir, true) with _ -> ()

[<Fact>]
let ``PromotionIndex builds and persists from an isolated store`` () =
    withSeededStore (fun dir ->
        let index = PromotionIndex.refreshIn dir

        Assert.Equal(6, index.PatternCount)

        let gaEntries =
            index.Entries |> List.filter (fun e -> e.PatternName.StartsWith("ga."))
        Assert.True(gaEntries.Length >= 5, $"Expected at least 5 GA patterns, got {gaEntries.Length}")

        let ranks = index.Entries |> List.map (fun e -> e.LevelRank)
        let isSorted = ranks |> List.pairwise |> List.forall (fun (a, b) -> a >= b)
        Assert.True(isSorted, "Index entries should be sorted by level rank descending")

        Assert.True(
            File.Exists(Path.Combine(dir, "index.json")),
            "refreshIn should persist index.json to the store directory"))

[<Fact>]
let ``PromotionIndex can be loaded from disk`` () =
    withSeededStore (fun dir ->
        let original = PromotionIndex.refreshIn dir

        match PromotionIndex.loadFrom dir with
        | Some loaded -> Assert.Equal(original.PatternCount, loaded.PatternCount)
        | None -> Assert.Fail("Expected to load index from disk after refresh"))

[<Fact>]
let ``PromotionIndex findForGoal scores GA patterns`` () =
    withSeededStore (fun dir ->
        let index = PromotionIndex.buildFromDir dir

        let routingMatch = PromotionIndex.findForGoal "route this query to the best agent" index
        let skillMatch = PromotionIndex.findForGoal "compute the scale notes for C major" index

        Assert.True(routingMatch.IsSome, "Expected a match for routing goal")
        Assert.True(skillMatch.IsSome, "Expected a match for skill goal"))

// Issue #265: Tars.Cortex cannot reference Tars.Evolution, so PatternSelector re-reads the
// index JSON by field name, with silent defaults. Write through PromotionIndex, read through
// PatternSelector, using non-default values so a renamed field cannot pass as a default.
[<Fact>]
let ``PatternSelector reads back every field PromotionIndex writes`` () =
    let dir = Path.Combine(Path.GetTempPath(), $"tars-promotion-roundtrip-{Guid.NewGuid():N}")

    let entry: PromotionIndex.IndexEntry =
        { PatternId = "ga.roundtrip"
          PatternName = "ga.roundtrip"
          Level = DslClause
          LevelRank = PromotionLevel.rank DslClause
          Score = 0.87
          OccurrenceCount = 7
          Contexts = [ "route a query"; "score a voicing" ]
          RollbackExpansion = None
          Weight = 0.31
          LastPromoted = DateTime(2026, 9, 1) }

    try
        PromotionIndex.saveTo
            dir
            { Entries = [ entry ]
              GeneratedAt = DateTime(2026, 9, 2)
              PatternCount = 1 }

        let json = File.ReadAllText(Path.Combine(dir, "index.json"))
        let read = Tars.Cortex.PatternSelector.parsePromotedEntries json

        let expected: Tars.Cortex.PatternSelector.PromotedEntry list =
            [ { PatternName = entry.PatternName
                LevelRank = entry.LevelRank
                Score = entry.Score
                Weight = entry.Weight
                Contexts = entry.Contexts } ]

        Assert.NotEqual(0, entry.LevelRank)
        Assert.Equal<Tars.Cortex.PatternSelector.PromotedEntry list>(expected, read)
    finally
        if Directory.Exists dir then
            Directory.Delete(dir, true)
