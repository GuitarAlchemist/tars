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
