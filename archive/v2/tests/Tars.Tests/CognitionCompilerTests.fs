namespace Tars.Tests

open System
open System.IO
open Xunit
open Tars.Core.HybridBrain
open Tars.Metascript

type CognitionCompilerTests(output: Xunit.Abstractions.ITestOutputHelper) =

    let trsxContent =
        """
GOAL "Refactor Domain.fs"
NODE "Plan" REASON Plan "Refactor Domain.fs to improve maintainability"
NODE "Critique" REASON Critique "Plan"
NODE "Build" WORK ToolCall "dotnet_build"
NODE "Distill" REASON Synthesize ["Plan", "Build"]

EDGE "Plan" -> "Critique"
EDGE "Critique" -> "Build"
EDGE "Build" -> "Distill"
"""

    [<Fact>]
    member this.``End-to-End: Trsx -> Graph -> TypedIR -> Executable``() =
        // 1. Parse Surface DSL (.trsx) into WorkflowGraph (Knowledge Graph)
        let graph = TrsxParser.parse trsxContent
        output.WriteLine($"Parsed Graph: {graph.Name} with {graph.Nodes.Count} nodes")
        Assert.Equal(4, graph.Nodes.Count)

        // 2. Compile Graph into Typed IR Draft Plan
        let draftPlan = IrCompiler.compileFromGraph graph
        output.WriteLine($"Compiled Draft Plan: {draftPlan.Description}")
        Assert.Equal(graph.Nodes.Count, draftPlan.Steps.Length)

        // 3. Verify Steps are correct
        let planStep = draftPlan.Steps |> List.find (fun s -> s.Name = "Plan")

        match planStep.Action with
        | UseTool(LlmCall(prompt, _)) ->
            output.WriteLine($"Plan Step Action: {prompt}")
            Assert.Contains("Plan:", prompt)
        | _ -> Assert.Fail("Plan step should be LlmCall")

        // 4. Validate and Transform to Executable (Cognition Compiler Pipeline)
        let pipelineResult = StateTransitions.fullPipeline ValidationContext.Empty draftPlan

        match pipelineResult with
        | Ok executablePlan ->
            output.WriteLine("Pipeline Success: Plan is Executable")
            Assert.IsAssignableFrom<Plan<Executable>>(executablePlan) |> ignore
            // Verify content persists
            Assert.Equal(draftPlan.Steps.Length, executablePlan.Steps.Length)
        | Error critique ->
            output.WriteLine("Pipeline Failed with Critique:")
            let text = CritiqueFormatter.formatForLlm critique
            output.WriteLine(text)
            Assert.Fail("Pipeline validation failed")
