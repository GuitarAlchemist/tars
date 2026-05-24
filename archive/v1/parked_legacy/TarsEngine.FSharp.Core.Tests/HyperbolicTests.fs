
module TarsEngine.FSharp.Core.Tests.HyperbolicTests

open Xunit
open TarsEngine.FSharp.Core.VectorStore
open TarsEngine.FSharp.Core.VectorStore.EmbeddingTransforms
open TarsEngine.FSharp.Core.VectorStore.SimilarityComputer

[<TestFixture>]
type HyperbolicTests() =

    let (|Tolerance|_|) (tolerance: float) (x: float, y: float) =
        if abs (x - y) < tolerance then Some () else None

    [<Fact>]
    member _.``computeHyperbolic should scale vector norm to be less than 1``() =
        let vector = [| 1.0; 2.0; 3.0 |]
        let hyperbolicVector = computeHyperbolic vector
        let norm = sqrt (Array.sumBy (fun x -> x * x) hyperbolicVector)
        Assert.True(norm < 1.0)

    [<Fact>]
    member _.``computeHyperbolic should handle zero vector``() =
        let vector = [| 0.0; 0.0; 0.0 |]
        let hyperbolicVector = computeHyperbolic vector
        Assert.Equal(vector, hyperbolicVector)

    [<Fact>]
    member _.``computeHyperbolic should slightly scale vector already in disk``() =
        let vector = [| 0.5; 0.5 |]
        let hyperbolicVector = computeHyperbolic vector
        let norm = sqrt (Array.sumBy (fun x -> x * x) hyperbolicVector)
        Assert.True(norm < 1.0)
        Assert.True(norm > 0.0)

    [<Fact>]
    member _.``hyperbolicSimilarity should be 1 for identical vectors``() =
        let vector = [| 0.1; 0.2; 0.3 |]
        let similarity = hyperbolicSimilarity vector vector
        match similarity, 1.0 with
        | Tolerance 1e-9 -> Assert.True(true)
        | _ -> Assert.True(false, $"Expected similarity of 1.0, but got {similarity}")

    [<Fact>]
    member _.``hyperbolicSimilarity should be less than 1 for different vectors``() =
        let vector1 = [| 0.1; 0.2; 0.3 |]
        let vector2 = [| 0.4; 0.5; 0.6 |]
        let similarity = hyperbolicSimilarity vector1 vector2
        Assert.True(similarity < 1.0)

    [<Fact>]
    member _.``hyperbolicSimilarity should decrease as vectors move apart``() =
        let vector1 = [| 0.1; 0.1 |]
        let vector2 = [| 0.3; 0.3 |]
        let vector3 = [| 0.8; 0.8 |]
        let similarity12 = hyperbolicSimilarity vector1 vector2
        let similarity13 = hyperbolicSimilarity vector1 vector3
        Assert.True(similarity12 > similarity13)
