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
    let ``all problems carry examples in place of a validation script, and a signature`` () =
        for p in GaProblemBank.all () do
            Assert.False(System.String.IsNullOrWhiteSpace(p.ExpectedSignature), $"'{p.Id}' has empty signature")
            Assert.False(p.Examples.IsEmpty, $"'{p.Id}' has no examples")
            Assert.Equal("", p.ValidationCode)

    [<Fact>]
    let ``GA problems are distinct from the generic ProblemBank`` () =
        let gaIds = GaProblemBank.all () |> List.map (fun p -> p.Id) |> Set.ofList
        let codeIds = ProblemBank.all () |> List.map (fun p -> p.Id) |> Set.ofList
        Assert.Empty(Set.intersect gaIds codeIds)

// #394: the parent process compares the solution's values with the examples' expected values.
module GaExamples =

    let private references: Map<string, string> =
        Map.ofList
            [ "ga-pc-interval", "let pcInterval (a: int) (b: int) : int = ((b - a) % 12 + 12) % 12"
              "ga-transpose", "let transpose (pcs: int list) (n: int) : int list = pcs |> List.map (fun pc -> ((pc + n) % 12 + 12) % 12)"
              "ga-major-scale", "let majorScale (root: int) : int list = [ 0; 2; 4; 5; 7; 9; 11 ] |> List.map (fun o -> (root + o) % 12)"
              "ga-triad-quality", "let triadQuality (root: int) (third: int) (fifth: int) : string =\n    match ((third - root) % 12 + 12) % 12, ((fifth - root) % 12 + 12) % 12 with\n    | 4, 7 -> \"major\"\n    | 3, 7 -> \"minor\"\n    | 3, 6 -> \"diminished\"\n    | 4, 8 -> \"augmented\"\n    | _ -> \"unknown\""
              "ga-is-transposition", "let isTransposition (a: int list) (b: int list) : bool =\n    let norm xs = xs |> List.map (fun x -> (x % 12 + 12) % 12) |> Set.ofList\n    let sa, sb = norm a, norm b\n    sa.Count = sb.Count && [ 0 .. 11 ] |> List.exists (fun n -> Set.map (fun x -> (x + n) % 12) sa = sb)" ]

    // Each is well typed and wrong on at least one case, so the examples compile against it and
    // reject it by comparing values, not by failing to compile.
    let private wrongSolutions: Map<string, string> =
        Map.ofList
            [ "ga-pc-interval", "let pcInterval (a: int) (b: int) : int = b - a"
              "ga-transpose", "let transpose (pcs: int list) (n: int) : int list = pcs |> List.map (fun pc -> pc + n)"
              "ga-major-scale", "let majorScale (root: int) : int list = [ 0; 2; 4; 5; 7; 9; 11 ] |> List.map (fun o -> root + o)"
              "ga-triad-quality", "let triadQuality (root: int) (third: int) (fifth: int) : string = \"major\""
              "ga-is-transposition", "let isTransposition (a: int list) (b: int list) : bool = a = b" ]

    let private problem (id: string) =
        GaProblemBank.all () |> List.find (fun p -> p.Id = id)

    // ga-transpose's FsCheck properties are left out: they run apart from the examples, and only
    // after the examples have validated the solution.
    [<Theory>]
    [<InlineData("ga-pc-interval")>]
    [<InlineData("ga-transpose")>]
    [<InlineData("ga-major-scale")>]
    [<InlineData("ga-triad-quality")>]
    [<InlineData("ga-is-transposition")>]
    let ``a GA problem's examples accept its reference solution`` (id: string) =
        let llm = EvolutionBenchmarkTests.BenchmarkLlm references.[id]
        let attempt = (BenchmarkRunner.runProblem llm { problem id with Properties = None } false ignore).Result
        Assert.True(attempt.Validated, $"{id}: {attempt.ValidationOutput}")

    [<Theory>]
    [<InlineData("ga-pc-interval")>]
    [<InlineData("ga-transpose")>]
    [<InlineData("ga-major-scale")>]
    [<InlineData("ga-triad-quality")>]
    [<InlineData("ga-is-transposition")>]
    let ``a GA problem's examples reject a wrong solution by its values`` (id: string) =
        let llm = EvolutionBenchmarkTests.BenchmarkLlm wrongSolutions.[id]
        let attempt = (BenchmarkRunner.runProblem llm (problem id) false ignore).Result
        Assert.False(attempt.Validated, $"{id}: {attempt.ValidationOutput}")
        Assert.StartsWith("Example failed:", attempt.ValidationOutput)
