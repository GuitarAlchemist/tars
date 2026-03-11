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
                  "verify", "distill" ] }

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
              Edges = [ "A", "B"; "B", "C" ] }

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
