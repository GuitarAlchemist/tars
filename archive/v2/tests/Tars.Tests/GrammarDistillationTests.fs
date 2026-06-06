namespace Tars.Tests

open Xunit
open Tars.Core.WorkflowOfThought
open Tars.Evolution

module GrammarDistillationTests =

    // ── Test helpers ─────────────────────────────────────────────────────────

    let makeEvent stepId kind toolName outputs status =
        { StepId = stepId
          Kind = kind
          ToolName = toolName
          ResolvedArgs = None
          Outputs = outputs
          Status = status
          Error = None
          Usage = None
          Metadata = None }

    let simpleTrace = [
        makeEvent "analyse" "reason" None ["analysis of the problem"] StepStatus.Ok
        makeEvent "research" "reason" None ["supporting evidence"] StepStatus.Ok
        makeEvent "validate" "work" (Some "validate_puzzle_answer") ["PASS"] StepStatus.Ok
    ]

    let complexTrace = [
        makeEvent "plan" "reason" None ["step-by-step plan"] StepStatus.Ok
        makeEvent "search" "work" (Some "search_web") ["search results"] StepStatus.Ok
        makeEvent "analyse" "reason" None ["analysis"] StepStatus.Ok
        makeEvent "verify" "work" (Some "fsharp_compile") ["compiled ok"] StepStatus.Ok
        makeEvent "summarise" "reason" None ["final summary"] StepStatus.Ok
    ]

    // ── Structural distillation ──────────────────────────────────────────────

    [<Fact>]
    let ``structural distillation extracts DAG shape`` () =
        let facet = GrammarDistillation.extractStructural simpleTrace

        match facet with
        | Structural (slots, edges) ->
            Assert.Equal(3, slots.Length)
            Assert.Equal(2, edges.Length)
            Assert.Equal("analyse", slots.[0].Name)
            Assert.Equal("reason", slots.[0].Kind)
            Assert.Equal("validate", slots.[2].Name)
            Assert.Equal("work", slots.[2].Kind)
        | _ -> failwith "Expected Structural facet"

    [<Fact>]
    let ``structural distillation infers sequential edges`` () =
        let facet = GrammarDistillation.extractStructural simpleTrace

        match facet with
        | Structural (_, edges) ->
            Assert.Equal("analyse", edges.[0].From)
            Assert.Equal("research", edges.[0].To)
            Assert.Equal("research", edges.[1].From)
            Assert.Equal("validate", edges.[1].To)
        | _ -> failwith "Expected Structural facet"

    [<Fact>]
    let ``structural distillation patches input types from predecessors`` () =
        let facet = GrammarDistillation.extractStructural simpleTrace

        match facet with
        | Structural (slots, _) ->
            Assert.Equal("Context", slots.[0].InputType)
            Assert.Equal(slots.[0].OutputType, slots.[1].InputType)
            Assert.Equal(slots.[1].OutputType, slots.[2].InputType)
        | _ -> failwith "Expected Structural facet"

    // ── Typed distillation ───────────────────────────────────────────────────

    [<Fact>]
    let ``typed distillation checks composability`` () =
        let facet = GrammarDistillation.extractTyped simpleTrace

        match facet with
        | Typed (slots, composable) ->
            Assert.Equal(3, slots.Length)
            Assert.True(composable)
        | _ -> failwith "Expected Typed facet"

    [<Fact>]
    let ``typed distillation includes tool type for work nodes`` () =
        let facet = GrammarDistillation.extractTyped simpleTrace

        match facet with
        | Typed (slots, _) ->
            let workSlot = slots |> List.find (fun s -> s.Kind = "work")
            Assert.Contains("ToolResult", workSlot.OutputType)
            Assert.Contains("validate_puzzle_answer", workSlot.OutputType)
        | _ -> failwith "Expected Typed facet"

    // ── Behavioral distillation ──────────────────────────────────────────────

    [<Fact>]
    let ``behavioral distillation extracts tools and conditions`` () =
        let facet = GrammarDistillation.extractBehavioral simpleTrace

        match facet with
        | Behavioral (conditions, tools) ->
            Assert.Equal(1, tools.Length)
            Assert.Contains("validate_puzzle_answer", tools)
            Assert.True(conditions.Length >= 3) // at least one per completed step
        | _ -> failwith "Expected Behavioral facet"

    [<Fact>]
    let ``behavioral distillation captures tool requirements`` () =
        let facet = GrammarDistillation.extractBehavioral complexTrace

        match facet with
        | Behavioral (_, tools) ->
            Assert.Equal(2, tools.Length)
            Assert.Contains("search_web", tools)
            Assert.Contains("fsharp_compile", tools)
        | _ -> failwith "Expected Behavioral facet"

    // ── Full distillation ────────────────────────────────────────────────────

    [<Fact>]
    let ``distillTrace produces all three facets`` () =
        let production = GrammarDistillation.distillTrace simpleTrace "solve puzzle"

        Assert.True(production.IsSome)
        let p = production.Value
        Assert.Equal(3, p.Facets.Length)
        Assert.Equal(1, p.TraceCount)
        Assert.Equal("solve puzzle", p.Name)
        Assert.True(p.SuccessRate > 0.9)

    [<Fact>]
    let ``distillTrace returns None for empty trace`` () =
        let result = GrammarDistillation.distillTrace [] "nothing"
        Assert.True(result.IsNone)

    [<Fact>]
    let ``fingerprint is stable for same structure`` () =
        let a = GrammarDistillation.distillTrace simpleTrace "goal A"
        let b = GrammarDistillation.distillTrace simpleTrace "goal B"

        Assert.True(a.IsSome && b.IsSome)
        Assert.Equal(a.Value.Id, b.Value.Id)

    [<Fact>]
    let ``fingerprint differs for different structures`` () =
        let a = GrammarDistillation.distillTrace simpleTrace "goal"
        let b = GrammarDistillation.distillTrace complexTrace "goal"

        Assert.True(a.IsSome && b.IsSome)
        Assert.NotEqual<string>(a.Value.Id, b.Value.Id)

    // ── Merge ────────────────────────────────────────────────────────────────

    [<Fact>]
    let ``merge combines trace counts and averages success rates`` () =
        let a = (GrammarDistillation.distillTrace simpleTrace "goal").Value
        let b = { a with TraceCount = 2; SuccessRate = 0.8 }

        let merged = GrammarDistillation.merge a b

        Assert.Equal(3, merged.TraceCount)
        let expectedRate = (1.0 * 1.0 + 0.8 * 2.0) / 3.0
        Assert.InRange(merged.SuccessRate, expectedRate - 0.01, expectedRate + 0.01)

    [<Fact>]
    let ``merge promotes level based on trace count and success`` () =
        let a = (GrammarDistillation.distillTrace simpleTrace "goal").Value
        let many = { a with TraceCount = 4; SuccessRate = 0.85 }

        let merged = GrammarDistillation.merge a many

        Assert.True(PromotionLevel.rank merged.SuggestedLevel >= PromotionLevel.rank Helper)

    // ── Batch distillation ───────────────────────────────────────────────────

    [<Fact>]
    let ``distillAll merges traces with same structure`` () =
        let traces = [
            (simpleTrace, "goal 1")
            (simpleTrace, "goal 2")
            (complexTrace, "different goal")
        ]

        let result = GrammarDistillation.distillAll traces

        Assert.Equal(3, result.TracesProcessed)
        Assert.Equal(2, result.Productions.Length) // 2 distinct structures
        let merged = result.Productions |> List.find (fun p -> p.TraceCount > 1)
        Assert.Equal(2, merged.TraceCount)

    // ── WeightedRule bridge ──────────────────────────────────────────────────

    [<Fact>]
    let ``toWeightedRule converts production to Bayesian weight`` () =
        let production = (GrammarDistillation.distillTrace simpleTrace "goal").Value
        let rule = GrammarDistillation.toWeightedRule production

        Assert.Equal(production.Id, rule.PatternId)
        Assert.Equal(production.SuccessRate, rule.SuccessRate)
        Assert.Equal(WeightedGrammar.Evolved, rule.Source)
        Assert.True(rule.Weight > 0.0)
        Assert.True(rule.RawScore >= 3)

    // ── Render ───────────────────────────────────────────────────────────────

    [<Fact>]
    let ``render produces readable grammar rule`` () =
        let production = (GrammarDistillation.distillTrace complexTrace "solve complex problem").Value
        let text = GrammarDistillation.render production

        Assert.Contains("solve complex problem", text)
        Assert.Contains("Structural", text)
        Assert.Contains("Typed", text)
        Assert.Contains("Behavioral", text)
        Assert.Contains("search_web", text)
        Assert.Contains("fsharp_compile", text)
        Assert.Contains("->", text) // type arrows
