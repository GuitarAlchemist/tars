namespace Tars.Tests

open Xunit
open Tars.Cortex
open Tars.Cortex.ReasoningPattern
open Tars.Cortex.PatternCompiler
open Tars.Cortex.WoTTypes

type ReasoningPatternTests() =

    let compiler = DefaultPatternCompiler() :> IPatternCompiler

    [<Fact>]
    member _.``Can compile Linear CoT pattern``() =
        let pattern = Library.linearCoT
        let goal = "Solve world hunger"

        let plan = compiler.CompilePattern(pattern, goal)

        Assert.Equal(4, plan.Nodes.Length)
        Assert.Equal(3, plan.Edges.Length)

        // decompose -> reason -> synthesize -> verify
        let decompose = plan.Nodes |> List.find (fun n -> nodeId n = "decompose")
        let reason = plan.Nodes |> List.find (fun n -> nodeId n = "reason")

        Assert.Contains(decompose.Id, plan.EntryNode)

        match decompose.Payload with
        | :? ReasonPayload as rp ->
            Assert.Contains("Decompose the problem into key components: Solve world hunger", rp.Prompt)
        | _ -> Assert.Fail("Wrong payload type")

    [<Fact>]
    member _.``Can compile Critic Refinement loop pattern``() =
        let pattern = Library.criticRefinement
        let goal = "Write a poem"

        let plan = compiler.CompilePattern(pattern, goal)

        Assert.Equal(4, plan.Nodes.Length)
        // draft -> critique -> refine -> final_check
        Assert.Equal(3, plan.Edges.Length)

        let draft = plan.Nodes |> List.find (fun n -> nodeId n = "draft")
        let critique = plan.Nodes |> List.find (fun n -> nodeId n = "critique")

        // Connects draft -> critique
        let edge =
            plan.Edges |> List.find (fun e -> e.From = draft.Id && e.To = critique.Id)

        Assert.NotNull(edge)

    [<Fact>]
    member _.``Can compile Parallel Brainstorming graph pattern``() =
        let pattern = Library.parallelBrainstorming
        let goal = "Design a logo"

        let plan = compiler.CompilePattern(pattern, goal)

        // 5 steps defined in pattern but compilation creates a synthetic start node if multiple roots?
        // Let's check logic:
        // decompose has no deps.
        // idea_1 deps decompose
        // idea_2 deps decompose
        // idea_3 deps decompose
        // synthesize deps ideas

        // Roots: decompose (no incoming edges within graph)
        // So just 1 root. No synthetic start needed.
        // Total nodes: 5.

        Assert.Equal(5, plan.Nodes.Length)

        // idea_1, 2, 3 should all point to synthesize
        let synthesize = plan.Nodes |> List.find (fun n -> nodeId n = "synthesize")

        let incomingToSynthesize = plan.Edges |> List.filter (fun e -> e.To = synthesize.Id)
        Assert.Equal(3, incomingToSynthesize.Length)

    [<Fact>]
    member _.``Compilation replaces goal placeholder``() =
        let step =
            { Id = "test"
              Role = "Reason"
              InstructionTemplate = Some "Solve: {goal}"
              Parameters = Map.empty
              Dependencies = [] }

        let pattern =
            { ReasoningPattern.empty with
                Steps = [ step ] }

        let goal = "XYZ"

        let plan = compiler.CompilePattern(pattern, goal)
        let node = plan.Nodes.Head

        match node.Payload with
        | :? ReasonPayload as rp -> Assert.Equal("Solve: XYZ", rp.Prompt)
        | _ -> Assert.Fail("Wrong payload")
