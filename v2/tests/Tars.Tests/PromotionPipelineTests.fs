module Tars.Tests.PromotionPipelineTests

open Xunit
open System
open Tars.Evolution

// =============================================================================
// Promotion Types Tests
// =============================================================================

[<Fact>]
let ``PromotionLevel.next returns correct progression`` () =
    Assert.Equal(Some Helper, PromotionLevel.next Implementation)
    Assert.Equal(Some Builder, PromotionLevel.next Helper)
    Assert.Equal(Some DslClause, PromotionLevel.next Builder)
    Assert.Equal(Some GrammarRule, PromotionLevel.next DslClause)
    Assert.Equal(None, PromotionLevel.next GrammarRule)

[<Fact>]
let ``PromotionLevel.rank is monotonically increasing`` () =
    let levels = [ Implementation; Helper; Builder; DslClause; GrammarRule ]
    let ranks = levels |> List.map PromotionLevel.rank
    Assert.Equal<int list>([0; 1; 2; 3; 4], ranks)

[<Fact>]
let ``PromotionCriteria.score counts true criteria`` () =
    let allTrue = {
        MinOccurrences = true; RemovesComplexity = true
        MoreReadable = true; StableSemantics = true
        AutoValidatable = true; NoOverlap = true
        ComposesCleanly = true; ImprovesPlanning = true
    }
    Assert.Equal(8, PromotionCriteria.score allTrue)

    let allFalse = PromotionCriteria.empty
    Assert.Equal(0, PromotionCriteria.score allFalse)

    let partial = { PromotionCriteria.empty with MinOccurrences = true; RemovesComplexity = true; MoreReadable = true }
    Assert.Equal(3, PromotionCriteria.score partial)

// =============================================================================
// Grammar Governor Tests
// =============================================================================

let private makeRecord name count level score =
    { PatternId = Guid.NewGuid().ToString("N").[..7]
      PatternName = name
      FirstSeen = DateTime.UtcNow.AddDays(-7.0)
      LastSeen = DateTime.UtcNow
      OccurrenceCount = count
      TaskIds = [ for i in 1..count -> $"task_{i}" ]
      Contexts = [ "context_a"; "context_b" ]
      CurrentLevel = level
      PromotionHistory = [ (Implementation, DateTime.UtcNow.AddDays(-7.0)) ]
      AverageScore = score }

let private makeCandidate record proposed criteria template rollback =
    { Record = record
      ProposedLevel = proposed
      Criteria = criteria
      Evidence = [ "evidence_1"; "evidence_2" ]
      PatternTemplate = template
      RollbackExpansion = rollback }

[<Fact>]
let ``Governor rejects pattern with insufficient occurrences`` () =
    let record = makeRecord "rare_pattern" 1 Implementation 0.8
    let criteria = { PromotionCriteria.empty with
                       MinOccurrences = false; RemovesComplexity = true
                       MoreReadable = true; StableSemantics = true
                       AutoValidatable = true; NoOverlap = true
                       ComposesCleanly = true; ImprovesPlanning = true }
    let candidate = makeCandidate record Helper criteria "template" None
    let decision = GrammarGovernor.evaluate [] candidate
    match decision with
    | Reject reason -> Assert.Contains("occurrences", reason)
    | _ -> Assert.Fail("Expected rejection for low occurrences")

[<Fact>]
let ``Governor approves pattern meeting 6+ criteria`` () =
    let record = makeRecord "good_pattern" 5 Implementation 0.85
    let criteria = { MinOccurrences = true; RemovesComplexity = true
                     MoreReadable = true; StableSemantics = true
                     AutoValidatable = true; NoOverlap = true
                     ComposesCleanly = false; ImprovesPlanning = false }
    let candidate = makeCandidate record Helper criteria "template" (Some "expanded code")
    let decision = GrammarGovernor.evaluate [] candidate
    match decision with
    | Approve reason -> Assert.Contains("6/8", reason)
    | _ -> Assert.Fail("Expected approval for 6/8 criteria")

[<Fact>]
let ``Governor defers pattern meeting 4-5 criteria`` () =
    let record = makeRecord "maybe_pattern" 4 Implementation 0.7
    let criteria = { MinOccurrences = true; RemovesComplexity = true
                     MoreReadable = true; StableSemantics = true
                     AutoValidatable = true; NoOverlap = false
                     ComposesCleanly = false; ImprovesPlanning = false }
    let candidate = makeCandidate record Helper criteria "template" (Some "expanded")
    let decision = GrammarGovernor.evaluate [] candidate
    match decision with
    | Defer reason -> Assert.Contains("5/8", reason)
    | _ -> Assert.Fail("Expected deferral for 5/8 criteria")

[<Fact>]
let ``Governor rejects pattern with fewer than 4 criteria`` () =
    let record = makeRecord "weak_pattern" 3 Implementation 0.5
    let criteria = { MinOccurrences = true; RemovesComplexity = true
                     MoreReadable = false; StableSemantics = false
                     AutoValidatable = false; NoOverlap = true
                     ComposesCleanly = false; ImprovesPlanning = false }
    let candidate = makeCandidate record Helper criteria "template" None
    let decision = GrammarGovernor.evaluate [] candidate
    match decision with
    | Reject reason -> Assert.Contains("3/8", reason)
    | _ -> Assert.Fail("Expected rejection for 3/8 criteria")

[<Fact>]
let ``Governor rejects DslClause promotion without rollback`` () =
    let record = makeRecord "no_rollback" 10 Builder 0.9
    let criteria = { MinOccurrences = true; RemovesComplexity = true
                     MoreReadable = true; StableSemantics = true
                     AutoValidatable = true; NoOverlap = true
                     ComposesCleanly = true; ImprovesPlanning = true }
    let candidate = makeCandidate record DslClause criteria "template" None
    let decision = GrammarGovernor.evaluate [] candidate
    match decision with
    | Reject reason -> Assert.Contains("rollback", reason)
    | _ -> Assert.Fail("Expected rejection for missing rollback at DslClause level")

[<Fact>]
let ``Audit report contains all sections`` () =
    let record = makeRecord "audit_test" 5 Implementation 0.8
    let criteria = { MinOccurrences = true; RemovesComplexity = true
                     MoreReadable = true; StableSemantics = true
                     AutoValidatable = true; NoOverlap = true
                     ComposesCleanly = false; ImprovesPlanning = false }
    let candidate = makeCandidate record Helper criteria "template" (Some "rollback code")
    let decision = Approve "test"
    let report = GrammarGovernor.auditReport candidate decision
    Assert.Contains("GRAMMAR GOVERNOR", report)
    Assert.Contains("audit_test", report)
    Assert.Contains("CRITERIA", report)
    Assert.Contains("DECISION", report)

// =============================================================================
// Promotion Pipeline Tests
// =============================================================================

[<Fact>]
let ``Pipeline extracts and tracks recurrence`` () =
    let artifacts : PromotionPipeline.TraceArtifact list = [
        { TaskId = "t1"
          PatternName = "extract_test"
          PatternTemplate = "template"
          Context = "math"
          Score = 0.8
          Timestamp = DateTime.UtcNow
          RollbackExpansion = None }
        { TaskId = "t2"; PatternName = "extract_test"
          PatternTemplate = "template"; Context = "math"
          Score = 0.9; Timestamp = DateTime.UtcNow
          RollbackExpansion = None }
    ]
    let records = PromotionPipeline.extract (PromotionPipeline.inspect artifacts)
    Assert.True(records.Length >= 1)
    let record = records |> List.find (fun r -> r.PatternName = "extract_test")
    Assert.True(record.OccurrenceCount >= 2)

[<Fact>]
let ``Pipeline classify returns None for low occurrence`` () =
    let record = makeRecord "low_occ" 1 Implementation 0.5
    let result = PromotionPipeline.classify 3 record
    Assert.True(result.IsNone)

[<Fact>]
let ``Pipeline classify returns candidate for sufficient occurrence`` () =
    let record = makeRecord "high_occ" 5 Implementation 0.8
    let result = PromotionPipeline.classify 3 record
    Assert.True(result.IsSome)
    Assert.Equal(Helper, result.Value.ProposedLevel)

[<Fact>]
let ``Pipeline full run handles empty input`` () =
    let results = PromotionPipeline.run (InMemoryPromotionStore()) 3 []
    Assert.Empty(results)

[<Fact>]
let ``EvolutionMetadata.empty has correct defaults`` () =
    let meta = EvolutionMetadata.empty
    Assert.True(meta.PromotionLevel.IsNone)
    Assert.Equal(0, meta.OccurrenceCount)
    Assert.Equal(0.0, meta.Confidence)
    Assert.Empty(meta.MutationHistory)
    Assert.Empty(meta.Effects)

// =============================================================================
// Pattern identity stability
// =============================================================================
// PatternId used to be `Guid.NewGuid().ToString("N").[..7]`. Weights are keyed
// by PatternId while the store keys recurrence by PatternName, so a random id
// made the two correlatable only through the store instance that minted it.

[<Fact>]
let ``patternIdOf is deterministic for the same name`` () =
    let a = PromotionPipeline.patternIdOf "extract_method"
    let b = PromotionPipeline.patternIdOf "extract_method"
    Assert.Equal(a, b)
    Assert.Equal(8, a.Length)
    Assert.NotEqual<string>(a, PromotionPipeline.patternIdOf "extract_method ")
    Assert.NotEqual<string>(a, PromotionPipeline.patternIdOf "inline_method")

[<Fact>]
let ``two independent stores agree on a pattern's identity`` () =
    // The property that matters: an id minted in one store is the id another
    // store mints for the same name. Under the old random scheme these differed,
    // so weights could never be carried between stores — or survive a reset.
    let artifacts name : PromotionPipeline.TraceArtifact list =
        [ for i in 1..3 ->
            { TaskId = $"t{i}"
              PatternName = name
              PatternTemplate = "tmpl"
              Context = "ctx"
              Score = 0.8
              Timestamp = DateTime.UtcNow
              RollbackExpansion = None } ]

    let idFrom () =
        let store = InMemoryPromotionStore() :> IPromotionStore
        PromotionPipeline.extractInto store (artifacts "shared_pattern")
        |> List.head
        |> fun r -> r.PatternId

    Assert.Equal(idFrom (), idFrom ())
    Assert.Equal(PromotionPipeline.patternIdOf "shared_pattern", idFrom ())

[<Fact>]
let ``weights stay matched to records after recurrence state is rebuilt`` () =
    // The failure this prevents: recurrence.json is lost or rebuilt while
    // weights.json survives. Under random ids every lookup in classifyWeighted
    // missed, every rule fell back to 0.0, ranking silently degraded to input
    // order, and the old weights were orphaned forever — SaveWeights appends
    // and never prunes.
    let artifacts : PromotionPipeline.TraceArtifact list =
        [ for i in 1..3 ->
            { TaskId = $"t{i}"
              PatternName = "durable_pattern"
              PatternTemplate = "tmpl"
              Context = "ctx"
              Score = 0.8
              Timestamp = DateTime.UtcNow
              RollbackExpansion = None } ]

    let firstStore = InMemoryPromotionStore() :> IPromotionStore
    PromotionPipeline.run firstStore 3 artifacts |> ignore
    let carriedWeights = firstStore.LoadWeights()
    Assert.NotEmpty(carriedWeights)

    // Fresh store: weights carried over, recurrence starts empty.
    let rebuilt = InMemoryPromotionStore() :> IPromotionStore
    rebuilt.SaveWeights carriedWeights
    let records = PromotionPipeline.extractInto rebuilt artifacts

    let record = records |> List.find (fun r -> r.PatternName = "durable_pattern")

    Assert.True(
        carriedWeights |> List.exists (fun w -> w.PatternId = record.PatternId),
        "weights orphaned after a recurrence rebuild — pattern identity is not stable"
    )
