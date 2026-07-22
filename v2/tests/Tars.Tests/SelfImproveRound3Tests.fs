module Tars.Tests.SelfImproveRound3Tests

open System
open System.Reflection
open Xunit
open Tars.Evolution
open Tars.Evolution.WeightedGrammar
open Tars.Evolution.ReplicatorDynamics

// Round-3 self-improve backlog seeds (ADR 0002 D5), sourced from the round-1
// fundamental research program (docs/research/): each test pins the corrected
// contract for a defect that was adversarially double-verified against the live
// code and ~/.tars state. Tagged `Category=SelfImproveBacklog` so CI excludes
// in-flight seeds; the trait is removed once the loop closes each.

// ─────────────────────────────────────────────────────────────────────────────
// Seed 1 — outcome-store codec (CLOSED by human implementation, wave-1 item 1,
// docs/plans/2026-07-21-research-agenda-deepened.md §3). The store previously
// serialized PatternKind with `sprintf "%A"` and parsed it with a lowercasing
// substring matcher whose Custom fallthrough re-wrapped the whole string, adding
// one quoting layer per load/append cycle (live records nested ~89 layers deep).
// The fix: `kindToString` is a total, injective print and `parseKind` its exact
// left inverse — parseKind (kindToString k) = k for every case, incl. adversarial
// Custom payloads. Reflection binds both private functions by name.
// ─────────────────────────────────────────────────────────────────────────────

[<Theory>]
[<InlineData("chain-of-thought")>]          // fieldless cases exercised via the loop below;
[<InlineData("Custom:benchmark:easy")>]     // these InlineData drive the adversarial Custom payloads
[<InlineData("Custom \"quoted\"")>]          // a literal that looks like the old %A form
[<InlineData("ChainOfThought")>]             // payload equal to a case name
[<InlineData("with\nnewline")>]
[<InlineData("")>]
[<InlineData("supply-chain-audit")>]         // would misclassify as ChainOfThought under the old matcher
let ``PatternKind round-trips through the outcome store codec without corruption`` (customPayload: string) =
    let storeModule = typeof<Tars.Cortex.PatternOutcomeStore.PatternOutcome>.DeclaringType
    let bind name =
        let mi = storeModule.GetMethod(name, BindingFlags.Static ||| BindingFlags.NonPublic ||| BindingFlags.Public)
        Assert.True(not (isNull mi), $"PatternOutcomeStore.{name} not found — keep the function name when editing the codec")
        mi
    let kindToString = bind "kindToString"
    let parseKind = bind "parseKind"
    let print (k: Tars.Cortex.WoTTypes.PatternKind) = kindToString.Invoke(null, [| box k |]) :?> string
    let parse (s: string) = parseKind.Invoke(null, [| box s |]) :?> Tars.Cortex.WoTTypes.PatternKind
    // Exact-inverse over every fieldless case plus the adversarial Custom payload.
    let cases: Tars.Cortex.WoTTypes.PatternKind list =
        [ Tars.Cortex.WoTTypes.PatternKind.ChainOfThought
          Tars.Cortex.WoTTypes.PatternKind.ReAct
          Tars.Cortex.WoTTypes.PatternKind.PlanAndExecute
          Tars.Cortex.WoTTypes.PatternKind.GraphOfThoughts
          Tars.Cortex.WoTTypes.PatternKind.TreeOfThoughts
          Tars.Cortex.WoTTypes.PatternKind.WorkflowOfThought
          Tars.Cortex.WoTTypes.PatternKind.Custom customPayload ]
    for kind in cases do
        Assert.Equal(kind, parse (print kind))
        // Idempotent: parsing an already-canonical string is a fixed point.
        Assert.Equal(parse (print kind), parse (print (parse (print kind))))

// ─────────────────────────────────────────────────────────────────────────────
// Seed 2 — degenerate Beta prior (docs/research/theory-bayesian-grammar-induction.md).
// bayesianUpdate with priorCount = 0 (every rule from fromRecurrenceRecords:
// SelectionCount = 0) collapses the posterior to exactly 1.0 or 0.0 on the first
// observation. A Beta(1,1)-style pseudo-count prior must keep one observation
// from being treated as certainty.
// ─────────────────────────────────────────────────────────────────────────────

[<Fact>]
[<Trait("Category", "SelfImproveBacklog")>]
let ``bayesianUpdate keeps a fresh rule's posterior non-degenerate`` () =
    let afterSuccess, _ = bayesianUpdate 0.5 0 true 0.95
    Assert.True(afterSuccess > 0.5 && afterSuccess < 1.0,
        sprintf "one success from a fresh rule must not yield certainty: got %.4f" afterSuccess)
    let afterFailure, _ = bayesianUpdate 0.5 0 false 0.95
    Assert.True(afterFailure > 0.0 && afterFailure < 0.5,
        sprintf "one failure from a fresh rule must not yield certainty: got %.4f" afterFailure)

// ─────────────────────────────────────────────────────────────────────────────
// Seed 3 — posterior never reaches the ranking (theory-bayesian-grammar-induction.md,
// empirical-promotion-dynamics.md: live r(SuccessRate, Weight) = −0.85).
// updateWeight advances SuccessRate/Confidence but leaves Weight — the field
// classifyWeighted ranks by — untouched, so execution outcomes can never change
// promotion order. After repeated failures the ranking Weight must fall.
// ─────────────────────────────────────────────────────────────────────────────

[<Fact>]
[<Trait("Category", "SelfImproveBacklog")>]
let ``updateWeight folds the Bayesian posterior into the ranking Weight`` () =
    let rule : WeightedRule =
        { PatternId = "seed3"
          PatternName = "posterior_to_weight"
          Level = Helper
          RawScore = 6
          Weight = 0.5
          Confidence = 0.75
          SuccessRate = 0.5
          SelectionCount = 10
          Source = Tars
          LastUpdated = DateTime.UtcNow }
    let afterFailures =
        (rule, [ 1..10 ]) ||> List.fold (fun r _ -> updateWeight WeightedGrammar.defaultConfig r false)
    Assert.True(afterFailures.Weight < rule.Weight,
        sprintf "10 consecutive failures must lower the ranking Weight: %.4f -> %.4f"
            rule.Weight afterFailures.Weight)

// ─────────────────────────────────────────────────────────────────────────────
// Seed 4 — smoothing floor broken (docs/research/theory-replicator-dynamics.md).
// step clamps to the floor BEFORE renormalization, so a floored species is
// rescaled below the floor and the documented 1% diversity guarantee does not
// hold. The floor must be a post-normalization invariant.
// ─────────────────────────────────────────────────────────────────────────────

[<Fact>]
let ``replicator step preserves the smoothing floor after renormalization`` () =
    let mkSpecies id proportion fitness : GrammarSpecies =
        { PatternId = id
          PatternName = id
          Level = Helper
          Proportion = proportion
          Fitness = fitness
          IsStable = false }
    let floor = 0.01
    let next = step 0.1 floor [ mkSpecies "fit" 0.99 1.0; mkSpecies "rare" 0.01 0.0 ]
    for s in next do
        Assert.True(s.Proportion >= floor,
            sprintf "species %s at %.6f is below the smoothing floor %.2f" s.PatternName s.Proportion floor)

// ─────────────────────────────────────────────────────────────────────────────
// Seed 5 — a name is not a template (docs/research/frontier-program-synthesis.md,
// empirical-promotion-dynamics.md). The live pipeline proposes candidates with
// PatternTemplate = the pattern's *name* (PromotionPipeline.run: `propose
// candidate.Record.PatternName`), and RemovesComplexity is a bare non-emptiness
// check — so every named pattern "removes complexity" by construction, one of
// the criteria inflations behind the governor's 95.7% approval rate. A candidate
// whose template is just its own name must not pass RemovesComplexity.
// ─────────────────────────────────────────────────────────────────────────────

[<Fact>]
let ``a pattern name alone does not satisfy RemovesComplexity`` () =
    let record =
        { PatternId = "seed5"
          PatternName = "extract_test"
          FirstSeen = DateTime.UtcNow.AddDays(-7.0)
          LastSeen = DateTime.UtcNow
          OccurrenceCount = 6
          TaskIds = [ for i in 1..6 -> $"task_{i}" ]
          Contexts = [ "context_a"; "context_b" ]
          CurrentLevel = Implementation
          PromotionHistory = []
          AverageScore = 0.9 }
    let candidate =
        { Record = record
          ProposedLevel = Helper
          Criteria = PromotionCriteria.empty
          Evidence = []
          PatternTemplate = record.PatternName
          RollbackExpansion = None }
    let criteria = PromotionPipeline.validateDeterministic [] candidate
    Assert.False(criteria.RemovesComplexity,
        "a candidate whose template is literally its own name encodes no abstraction and must not count as removing complexity")
