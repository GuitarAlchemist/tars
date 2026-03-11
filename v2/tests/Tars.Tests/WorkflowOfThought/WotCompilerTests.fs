namespace Tars.Tests.WorkflowOfThought

open Xunit
open Tars.DSL.Wot
open Tars.DSL.Wot.WotCompiler
open Tars.Core.WorkflowOfThought

module WotCompilerTests =

    let private mkNode id kind outputs =
        { DslConvert.defaultNode id kind with
            Outputs = outputs |> List.map SimpleOutput }

    [<Fact>]
    let ``compileWorkflowToSteps - golden path compiles Verify + ToolCall`` () =
        // Arrange
        let wf =
            { Name = "test_wf"
              Description = Some "Test workflow"
              Domain = None
              Difficulty = None
              Version = "1.0.0"
              Risk = "low"
              Inputs = Map.ofList [ "target_file", "src/Foo.fs" ]
              Policy =
                { AllowedTools = Set.empty
                  MaxToolCalls = 0
                  MaxTokens = 0
                  MaxTimeMs = 0 }
              Nodes =
                [ mkNode "plan" NodeKind.Reason [ "draft_plan" ]
                  mkNode "critique" NodeKind.Reason [ "reviewed_plan" ]
                  { mkNode "analyze" NodeKind.Work [ "analysis_md" ] with
                      Tool = Some "analyze_file_complexity"
                      Args = Some(Map.ofList [ "path", box "src/Foo.fs" ]) }
                  { mkNode "verify" NodeKind.Work [ "verified" ] with
                      Checks =
                          [ WotCheck.NonEmpty "analysis_md"
                            WotCheck.Contains("analysis_md", "| Cyclomatic Complexity |") ] }
                  mkNode "distill" NodeKind.Reason [ "final_idea" ] ]
              Edges =
                [ "plan", "critique"
                  "critique", "analyze"
                  "analyze", "verify"
                  "verify", "distill" ]
              ParallelGroups = [] }

        // Act
        let result = compileWorkflowToPlanParsed wf

        // Assert
        match result with
        | Error errs ->
            let msg = errs |> List.map string |> String.concat "\n"
            failwith $"Expected Ok, got errors:\n{msg}"
        | Ok plan ->
            Assert.Equal(5, plan.Steps.Length)

            // Find analyze step -> ToolCall
            let hasAnalyzeTool =
                plan.Steps
                |> List.exists (fun s ->
                    match s.Action with
                    | StepAction.Work(WorkOperation.ToolCall(tool, args)) ->
                        tool = "analyze_file_complexity" && args.ContainsKey("path")
                    | _ -> false)

            Assert.True(hasAnalyzeTool, "Expected a ToolCall step for analyze_file_complexity.")

            // Find verify step -> Verify [NonEmpty; Contains]
            let hasVerifyChecks =
                plan.Steps
                |> List.exists (fun s ->
                    match s.Action with
                    | StepAction.Work(WorkOperation.Verify checks) ->
                        checks |> List.contains (WotCheck.NonEmpty "analysis_md")
                        && checks
                           |> List.contains (WotCheck.Contains("analysis_md", "| Cyclomatic Complexity |"))
                    | _ -> false)

            Assert.True(hasVerifyChecks, "Expected a Verify step with NonEmpty + Contains checks.")

    [<Fact>]
    let ``compileWorkflowToSteps - rejects non linear graphs`` () =
        let wf =
            { Name = "test_wf_bad"
              Description = Some "Test workflow bad"
              Domain = None
              Difficulty = None
              Version = "1.0.0"
              Risk = "low"
              Inputs = Map.empty
              Policy =
                { AllowedTools = Set.empty
                  MaxToolCalls = 0
                  MaxTokens = 0
                  MaxTimeMs = 0 }
              Nodes =
                [ mkNode "A" NodeKind.Reason []
                  mkNode "B" NodeKind.Reason []
                  mkNode "C" NodeKind.Reason [] ]
              Edges = [ "A", "B"; "B", "C" ]
              ParallelGroups = [] }

        // Correct baseline
        match compileWorkflowToPlanParsed wf with
        | Ok _ -> ()
        | Error e -> failwith $"Baseline should valid, got {e}"

        // Introduce extra edge (A -> C) -> Not a chain
        let wfBad =
            { wf with
                Edges = ("A", "C") :: wf.Edges }

        match compileWorkflowToPlanParsed wfBad with
        | Ok _ -> failwith "Expected compile error for non-linear graph."
        | Error errs ->
            Assert.True(errs.Length > 0)

            Assert.Contains(
                errs,
                (fun e ->
                    match e with
                    | InvalidGraph _ -> true
                    | _ -> false)
            )

    // ─────────────────────────────────────────────────────────────────────
    // Condition metadata tests
    // ─────────────────────────────────────────────────────────────────────

    [<Fact>]
    let ``condition metadata is stored when node has a Condition`` () =
        let wf =
            { Name = "cond_wf"
              Description = Some "Condition test"
              Domain = None
              Difficulty = None
              Version = "1.0.0"
              Risk = "low"
              Inputs = Map.empty
              Policy =
                { AllowedTools = Set.empty
                  MaxToolCalls = 0
                  MaxTokens = 0
                  MaxTimeMs = 0 }
              Nodes =
                [ mkNode "A" NodeKind.Reason [ "a_out" ]
                  { mkNode "B" NodeKind.Reason [ "b_out" ] with
                      Condition = Some "${confidence} > 0.7" } ]
              Edges = [ "A", "B" ]
              ParallelGroups = [] }

        match compileWorkflowToPlanParsed wf with
        | Error errs -> failwith $"Expected Ok, got errors: {errs}"
        | Ok plan ->
            let stepB = plan.Steps |> List.find (fun s -> s.Id = "B")
            Assert.True(stepB.Metadata.ContainsKey("condition"), "Step B should have 'condition' metadata key.")
            match stepB.Metadata.["condition"] with
            | MStr c -> Assert.Equal("${confidence} > 0.7", c)
            | other -> failwith $"Expected MStr, got {other}"

    [<Fact>]
    let ``no condition metadata when node has no Condition`` () =
        let wf =
            { Name = "no_cond_wf"
              Description = None
              Domain = None
              Difficulty = None
              Version = "1.0.0"
              Risk = "low"
              Inputs = Map.empty
              Policy =
                { AllowedTools = Set.empty
                  MaxToolCalls = 0
                  MaxTokens = 0
                  MaxTimeMs = 0 }
              Nodes =
                [ mkNode "X" NodeKind.Reason [ "x_out" ]
                  mkNode "Y" NodeKind.Reason [ "y_out" ] ]
              Edges = [ "X", "Y" ]
              ParallelGroups = [] }

        match compileWorkflowToPlanParsed wf with
        | Error errs -> failwith $"Expected Ok, got errors: {errs}"
        | Ok plan ->
            for step in plan.Steps do
                Assert.False(step.Metadata.ContainsKey("condition"),
                    $"Step {step.Id} should not have 'condition' metadata.")

    // ─────────────────────────────────────────────────────────────────────
    // Parallel group fan-out / fan-in edge tests
    // ─────────────────────────────────────────────────────────────────────

    [<Fact>]
    let ``parallel groups generate correct fan-out and fan-in edges`` () =
        // Workflow: A -> [B, C] (parallel) -> D
        let wf =
            { Name = "parallel_wf"
              Description = Some "Parallel group test"
              Domain = None
              Difficulty = None
              Version = "1.0.0"
              Risk = "low"
              Inputs = Map.empty
              Policy =
                { AllowedTools = Set.empty
                  MaxToolCalls = 0
                  MaxTokens = 0
                  MaxTimeMs = 0 }
              Nodes =
                [ mkNode "A" NodeKind.Reason [ "a_out" ]
                  mkNode "B" NodeKind.Reason [ "b_out" ]
                  mkNode "C" NodeKind.Reason [ "c_out" ]
                  mkNode "D" NodeKind.Reason [ "d_out" ] ]
              Edges = [ "A", "B"; "B", "C"; "C", "D" ]
              ParallelGroups =
                [ { GroupId = "parallel_1"; NodeIds = [ "B"; "C" ] } ] }

        match compileWorkflowToPlanParsed wf with
        | Error errs -> failwith $"Expected Ok, got errors: {errs}"
        | Ok plan ->
            // Should have 4 steps (A, B, C, D)
            Assert.Equal(4, plan.Steps.Length)

            // Steps B and C should have parallel_group metadata
            let stepB = plan.Steps |> List.find (fun s -> s.Id = "B")
            let stepC = plan.Steps |> List.find (fun s -> s.Id = "C")
            Assert.True(stepB.Metadata.ContainsKey("parallel_group"),
                "Step B should have 'parallel_group' metadata.")
            Assert.True(stepC.Metadata.ContainsKey("parallel_group"),
                "Step C should have 'parallel_group' metadata.")
            match stepB.Metadata.["parallel_group"] with
            | MStr g -> Assert.Equal("parallel_1", g)
            | other -> failwith $"Expected MStr, got {other}"

    [<Fact>]
    let ``expandParallelEdges creates fan-out from predecessor and fan-in to successor`` () =
        let nodes =
            [ mkNode "start" NodeKind.Reason []
              mkNode "p1" NodeKind.Reason []
              mkNode "p2" NodeKind.Reason []
              mkNode "end_node" NodeKind.Reason [] ]
        let edges = [ ("start", "p1"); ("p1", "p2"); ("p2", "end_node") ]
        let groups = [ { GroupId = "g1"; NodeIds = [ "p1"; "p2" ] } ]

        let expanded = expandParallelEdges nodes edges groups

        // Should have fan-out: start -> p1 and start -> p2
        Assert.Contains(("start", "p1"), expanded)
        Assert.Contains(("start", "p2"), expanded)
        // Should have fan-in: p1 -> end_node and p2 -> end_node
        Assert.Contains(("p1", "end_node"), expanded)
        Assert.Contains(("p2", "end_node"), expanded)

    // ─────────────────────────────────────────────────────────────────────
    // Variable interpolation tests
    // ─────────────────────────────────────────────────────────────────────

    [<Fact>]
    let ``variable interpolation resolves inputs in node goals`` () =
        let wf =
            { Name = "var_wf"
              Description = Some "Variable interpolation test"
              Domain = None
              Difficulty = None
              Version = "1.0.0"
              Risk = "low"
              Inputs = Map.ofList [ "target_file", "src/Main.fs"; "language", "F#" ]
              Policy =
                { AllowedTools = Set.empty
                  MaxToolCalls = 0
                  MaxTokens = 0
                  MaxTimeMs = 0 }
              Nodes =
                [ { mkNode "analyze" NodeKind.Reason [ "analysis" ] with
                      Goal = Some "Analyze ${target_file} written in ${language}" }
                  mkNode "report" NodeKind.Reason [ "report_out" ] ]
              Edges = [ "analyze", "report" ]
              ParallelGroups = [] }

        match compileWorkflowToPlanParsed wf with
        | Error errs -> failwith $"Expected Ok, got errors: {errs}"
        | Ok plan ->
            let analyzeStep = plan.Steps |> List.find (fun s -> s.Id = "analyze")
            match analyzeStep.Action with
            | StepAction.Reason(ReasonOperation.Plan goal) ->
                Assert.Contains("src/Main.fs", goal)
                Assert.Contains("F#", goal)
                Assert.DoesNotContain("${target_file}", goal)
                Assert.DoesNotContain("${language}", goal)
            | other -> failwith $"Expected Reason(Plan ...), got {other}"

    [<Fact>]
    let ``variable interpolation leaves unknown variables for runtime`` () =
        let wf =
            { Name = "unresolved_wf"
              Description = None
              Domain = None
              Difficulty = None
              Version = "1.0.0"
              Risk = "low"
              Inputs = Map.ofList [ "known", "value1" ]
              Policy =
                { AllowedTools = Set.empty
                  MaxToolCalls = 0
                  MaxTokens = 0
                  MaxTimeMs = 0 }
              Nodes =
                [ { mkNode "step1" NodeKind.Reason [ "out1" ] with
                      Goal = Some "Process ${known} and ${unknown_var}" }
                  mkNode "step2" NodeKind.Reason [ "out2" ] ]
              Edges = [ "step1", "step2" ]
              ParallelGroups = [] }

        match compileWorkflowToPlanParsed wf with
        | Error errs -> failwith $"Expected Ok, got errors: {errs}"
        | Ok plan ->
            let step1 = plan.Steps |> List.find (fun s -> s.Id = "step1")
            match step1.Action with
            | StepAction.Reason(ReasonOperation.Plan goal) ->
                // Known variable should be resolved
                Assert.Contains("value1", goal)
                // Unknown variable should remain as-is for runtime resolution
                Assert.Contains("${unknown_var}", goal)
            | other -> failwith $"Expected Reason(Plan ...), got {other}"
