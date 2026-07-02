module Tars.Tests.GaProblemBankTests

open Xunit
open Tars.Evolution

type GaProblemBankTests() =

    [<Fact>]
    let ``all returns non-empty GA-domain problems`` () =
        Assert.NotEmpty(GaProblemBank.all ())

    [<Fact>]
    let ``all problems are MusicTheory category`` () =
        for p in GaProblemBank.all () do
            Assert.Equal(MusicTheory, p.Category)

    [<Fact>]
    let ``all problems have unique IDs`` () =
        let ids = GaProblemBank.all () |> List.map (fun p -> p.Id)
        Assert.Equal(ids.Length, (ids |> Set.ofList).Count)

    [<Fact>]
    let ``all problems carry a PASS-emitting validation and a signature`` () =
        for p in GaProblemBank.all () do
            Assert.False(System.String.IsNullOrWhiteSpace(p.ExpectedSignature), $"'{p.Id}' has empty signature")
            Assert.Contains("PASS", p.ValidationCode)

    [<Fact>]
    let ``GA problems are distinct from the generic ProblemBank`` () =
        let gaIds = GaProblemBank.all () |> List.map (fun p -> p.Id) |> Set.ofList
        let codeIds = ProblemBank.all () |> List.map (fun p -> p.Id) |> Set.ofList
        Assert.Empty(Set.intersect gaIds codeIds)
