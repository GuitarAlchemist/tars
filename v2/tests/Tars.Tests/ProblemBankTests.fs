module Tars.Tests.ProblemBankTests

open Xunit
open Xunit.Abstractions
open Tars.Evolution

type ProblemBankTests(output: ITestOutputHelper) =

    [<Fact>]
    let ``all returns non-empty list`` () =
        let problems = ProblemBank.all ()
        output.WriteLine($"ProblemBank.all() returned {problems.Length} problems")
        Assert.NotEmpty(problems)

    [<Fact>]
    let ``all problems have unique IDs`` () =
        let problems = ProblemBank.all ()
        let ids = problems |> List.map (fun p -> p.Id)
        let uniqueIds = ids |> Set.ofList
        Assert.Equal(ids.Length, uniqueIds.Count)

    [<Fact>]
    let ``all problems have non-empty Title`` () =
        for p in ProblemBank.all () do
            Assert.False(System.String.IsNullOrWhiteSpace(p.Title), $"Problem '{p.Id}' has empty Title")

    [<Fact>]
    let ``all problems have non-empty Description`` () =
        for p in ProblemBank.all () do
            Assert.False(System.String.IsNullOrWhiteSpace(p.Description), $"Problem '{p.Id}' has empty Description")

    [<Fact>]
    let ``all problems have non-empty ExpectedSignature`` () =
        for p in ProblemBank.all () do
            Assert.False(System.String.IsNullOrWhiteSpace(p.ExpectedSignature), $"Problem '{p.Id}' has empty ExpectedSignature")

    [<Fact>]
    let ``all problems have non-empty ValidationCode`` () =
        for p in ProblemBank.all () do
            Assert.False(System.String.IsNullOrWhiteSpace(p.ValidationCode), $"Problem '{p.Id}' has empty ValidationCode")

    [<Fact>]
    let ``summary total matches all length`` () =
        let problems = ProblemBank.all ()
        let summary = ProblemBank.summary ()
        output.WriteLine($"Summary: Total={summary.Total}, Basic={summary.Basic}, Intermediate={summary.Intermediate}, Advanced={summary.Advanced}, Expert={summary.Expert}")
        Assert.Equal(problems.Length, summary.Total)

    [<Fact>]
    let ``summary difficulty counts sum to total`` () =
        let summary = ProblemBank.summary ()
        let summed = summary.Basic + summary.Intermediate + summary.Advanced + summary.Expert
        Assert.Equal(summary.Total, summed)

    [<Fact>]
    let ``each difficulty level has at least one problem`` () =
        let summary = ProblemBank.summary ()
        Assert.True(summary.Basic > 0, "Expected at least one Beginner problem")
        Assert.True(summary.Intermediate > 0, "Expected at least one Intermediate problem")
        Assert.True(summary.Advanced > 0, "Expected at least one Advanced problem")
        Assert.True(summary.Expert > 0, "Expected at least one Expert problem")

    [<Fact>]
    let ``byDifficulty returns correct subsets`` () =
        for diff in [ Beginner; Intermediate; Advanced; Expert ] do
            let filtered = ProblemBank.byDifficulty diff
            Assert.NotEmpty(filtered)
            for p in filtered do
                Assert.Equal(diff, p.Difficulty)

    [<Fact>]
    let ``each used category is represented`` () =
        let problems = ProblemBank.all ()
        let categories = problems |> List.map (fun p -> p.Category) |> Set.ofList
        output.WriteLine($"Categories found: {categories}")
        // These categories are all used in the current bank
        for cat in [ StringManipulation; Algorithms; DataStructures; ErrorHandling; PatternMatching; TypeDesign ] do
            Assert.True(categories.Contains(cat), $"Expected category {cat} to be represented")

    [<Fact>]
    let ``byCategory returns correct subsets`` () =
        let problems = ProblemBank.all ()
        let categories = problems |> List.map (fun p -> p.Category) |> Set.ofList
        for cat in categories do
            let filtered = ProblemBank.byCategory cat
            Assert.NotEmpty(filtered)
            for p in filtered do
                Assert.Equal(cat, p.Category)

    [<Fact>]
    let ``tryFind returns Some for existing ID`` () =
        let first = (ProblemBank.all ()).Head
        let found = ProblemBank.tryFind first.Id
        Assert.True(found.IsSome, $"Expected to find problem '{first.Id}'")
        Assert.Equal(first.Id, found.Value.Id)

    [<Fact>]
    let ``tryFind returns None for unknown ID`` () =
        let found = ProblemBank.tryFind "nonexistent-problem-id"
        Assert.True(found.IsNone, "Expected None for unknown ID")

    [<Fact>]
    let ``problem IDs follow prefix naming convention`` () =
        let knownPrefixes = [ "basic-"; "inter-"; "adv-"; "exp-" ]
        for p in ProblemBank.all () do
            let hasKnownPrefix = knownPrefixes |> List.exists (fun prefix -> p.Id.StartsWith(prefix))
            Assert.True(hasKnownPrefix, $"Problem ID '{p.Id}' does not start with a known prefix ({knownPrefixes})")

    [<Fact>]
    let ``all problems have positive time limits`` () =
        for p in ProblemBank.all () do
            Assert.True(p.TimeLimitSeconds > 0, $"Problem '{p.Id}' has non-positive TimeLimitSeconds={p.TimeLimitSeconds}")
