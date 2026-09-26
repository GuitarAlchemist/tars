namespace Tars.Tests

open Xunit
open Tars.Cortex
open Tars.Cortex.ReasoningPattern
open Tars.Cortex.PatternSelector

type PatternSelectorTests() =

    let selector = PatternLibraryService()

    [<Fact>]
    member _.``Selects linear CoT by default``() =
        let pattern = selector.Select("What is the capital of France?")
        Assert.Equal("Linear Chain of Thought", pattern.Name)
        Assert.Equal(ReasoningPattern.Linear, pattern.Kind)

    [<Fact>]
    member _.``Selects Parallel Brainstorming for idea generation``() =
        let pattern = selector.Select("Brainstorm ideas for a marketing campaign")
        Assert.Equal("Parallel Brainstorming", pattern.Name)
        Assert.Equal(ReasoningPattern.Graph, pattern.Kind)

    [<Fact>]
    member _.``Selects Parallel Brainstorming for generation keywords``() =
        let pattern = selector.Select("Generate 5 variants of a slogan")
        Assert.Equal("Parallel Brainstorming", pattern.Name)

    [<Fact>]
    member _.``Selects Critic Refinement for verification tasks``() =
        let pattern = selector.Select("Double check this code for bugs")
        Assert.Equal("Critic Refinement", pattern.Name)
        Assert.Equal(ReasoningPattern.Loop, pattern.Kind)

    [<Fact>]
    member _.``Selects Critic Refinement for critique tasks``() =
        let pattern = selector.Select("Critique this essay")
        Assert.Equal("Critic Refinement", pattern.Name)

    [<Fact>]
    member _.``Can register and retrieve custom patterns``() =
        let custom =
            { ReasoningPattern.empty with
                Name = "Custom Strategy"
                Kind = ReasoningPattern.TreeSearch }

        selector.Register(custom)

        match selector.GetByName("Custom Strategy") with
        | Some p -> Assert.Equal(ReasoningPattern.TreeSearch, p.Kind)
        | None -> Assert.Fail("Custom pattern not found")

// ─────────────────────────────────────────────────────────────────────
// HistoryAwareSelector tests
// ─────────────────────────────────────────────────────────────────────

open Tars.Cortex.WoTTypes

type HistoryAwareSelectorTests() =

    let selector = PatternSelector.HistoryAwareSelector() :> IPatternSelector

    let defaultState : WoTCognitiveState =
        { Mode = Exploratory
          Eigenvalue = 0.5
          Entropy = 0.5
          BranchingFactor = 1.0
          ActivePattern = None
          WoTRunId = None
          StepCount = 0
          TokenBudget = None
          LastTransition = System.DateTime.UtcNow
          ConstraintScore = None
          SuccessRate = 0.5 }

    [<Fact>]
    member _.``Recommend returns valid PatternKind for explain goal``() =
        let result = selector.Recommend("explain how sorting works", defaultState)
        // Should select ChainOfThought for explain-type goals
        Assert.Equal(ChainOfThought, result)

    [<Fact>]
    member _.``Recommend returns valid PatternKind for search goal``() =
        let result = selector.Recommend("search for relevant papers on ML", defaultState)
        Assert.Equal(ReAct, result)

    [<Fact>]
    member _.``Recommend returns valid PatternKind for comparison goal``() =
        let result = selector.Recommend("compare alternative approaches to caching", defaultState)
        Assert.Equal(GraphOfThoughts, result)

    [<Fact>]
    member _.``Recommend returns valid PatternKind for exploration goal``() =
        let result = selector.Recommend("explore different brainstorm strategies", defaultState)
        Assert.Equal(TreeOfThoughts, result)

    [<Fact>]
    member _.``Recommend returns valid PatternKind for pipeline goal``() =
        let result = selector.Recommend("build a workflow pipeline for data processing", defaultState)
        Assert.Equal(WorkflowOfThought, result)

    [<Fact>]
    member _.``Recommend can return PlanAndExecute``() =
        // `Patterns.fs` implements PlanAndExecute in full, but it had no entry in the
        // selector's heuristic map, and `combineScores` maps over that map — so the
        // kind was unreachable no matter what the goal said or what the promotion
        // index had learned.
        let result = selector.Recommend("plan and implement the migration", defaultState)
        Assert.Equal(PlanAndExecute, result)

    [<Fact>]
    member _.``A goal about planets is not a goal about plans``() =
        // "planet", "plant" and "airplane" all contain "plan". Without a whole-word
        // match, "list known planets" scores PlanAndExecute at 0.8 against nothing
        // else above 0.4 and is routed through a five-call planning workflow.
        for goal in [ "list known planets"; "identify this plant"; "how does an airplane fly" ] do
            Assert.NotEqual(PlanAndExecute, selector.Recommend(goal, defaultState))

        for goal in [ "plan the release"; "implement the parser"; "migration of the store"; "deploy to staging" ] do
            Assert.Equal(PlanAndExecute, selector.Recommend(goal, defaultState))

    [<Fact>]
    member _.``A custom pattern whose name contains plan is not PlanAndExecute``() =
        // "explanation" contains "plan". Kind ids were matched by substring, and
        // `GoldenTraceStore` persists `PatternKind.ToString()`, so an unrelated custom
        // pattern could collect PlanAndExecute's history and bandit weight.
        Assert.Equal(Some PlanAndExecute, PatternSelector.keyToKind "PlanAndExecute")
        Assert.Equal(Some PlanAndExecute, PatternSelector.keyToKind "planandexecute")
        Assert.Equal(None, PatternSelector.keyToKind "explanation")
        Assert.Equal(None, PatternSelector.keyToKind "Custom:explanation")
        Assert.Equal(None, PatternSelector.keyToKind "my-planner")
        Assert.Equal(None, PatternSelector.keyToKind "")

    [<Fact>]
    member _.``Score returns scores for all pattern kinds``() =
        let scores = selector.Score("explain something step by step")
        // Should have entries for all 6 standard pattern kinds
        Assert.True(scores.ContainsKey(ChainOfThought), "Should have ChainOfThought score")
        Assert.True(scores.ContainsKey(ReAct), "Should have ReAct score")
        Assert.True(scores.ContainsKey(GraphOfThoughts), "Should have GraphOfThoughts score")
        Assert.True(scores.ContainsKey(TreeOfThoughts), "Should have TreeOfThoughts score")
        Assert.True(scores.ContainsKey(WorkflowOfThought), "Should have WorkflowOfThought score")
        Assert.True(scores.ContainsKey(PlanAndExecute), "Should have PlanAndExecute score")
        // All scores should be positive
        for KeyValue(_, score) in scores do
            Assert.True(score > 0.0, $"All scores should be positive, got {score}")

    [<Fact>]
    member _.``Score gives highest value to matching heuristic``() =
        let scores = selector.Score("search for files and find bugs")
        let reactScore = scores.[ReAct]
        // ReAct should score highest for search/find goals
        let otherMax =
            scores
            |> Map.filter (fun k _ -> k <> ReAct)
            |> Map.values
            |> Seq.max
        Assert.True(reactScore >= otherMax,
            $"ReAct ({reactScore}) should score >= other max ({otherMax}) for search goals")
