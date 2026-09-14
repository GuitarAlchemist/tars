module Tars.Tests.MetricSpaceTests

open Xunit
open Tars.Core

[<Fact>]
let ``cosineSimilarity returns 1.0 for identical vectors`` () =
    let v = [| 1.0f; 2.0f; 3.0f |]
    let similarity = MetricSpace.cosineSimilarity v v
    Assert.Equal(1.0f, similarity, 4)

[<Fact>]
let ``cosineSimilarity returns 0.0 for orthogonal vectors`` () =
    let v1 = [| 1.0f; 0.0f |]
    let v2 = [| 0.0f; 1.0f |]
    let similarity = MetricSpace.cosineSimilarity v1 v2
    Assert.Equal(0.0f, similarity, 4)

[<Fact>]
let ``cosineSimilarity returns -1.0 for opposite vectors`` () =
    let v1 = [| 1.0f; 2.0f |]
    let v2 = [| -1.0f; -2.0f |]
    let similarity = MetricSpace.cosineSimilarity v1 v2
    Assert.Equal(-1.0f, similarity, 4)

[<Fact>]
let ``cosineSimilarity matches a scalar reference across SIMD and remainder lanes`` () =
    let reference (a: float32[]) (b: float32[]) =
        let dot = Array.fold2 (fun acc x y -> acc + float x * float y) 0.0 a b
        let norm (v: float32[]) = sqrt (v |> Array.sumBy (fun x -> float x * float x))
        dot / (norm a * norm b)

    let rng = System.Random(42)

    for dims in [ 3; 8; 37; 384; 1024 ] do
        let a = Array.init dims (fun _ -> float32 (rng.NextDouble() - 0.5))
        let b = Array.init dims (fun _ -> float32 (rng.NextDouble() - 0.5))
        Assert.Equal(reference a b, float (MetricSpace.cosineSimilarity a b), 5)

[<Fact>]
let ``euclideanDistance calculates correctly`` () =
    let v1 = [| 1.0f; 5.0f |]
    let v2 = [| 4.0f; 1.0f |]
    // Distance = sqrt((1-4)^2 + (5-1)^2) = sqrt(9 + 16) = 5
    let dist = MetricSpace.euclideanDistance v1 v2
    Assert.Equal(5.0f, dist, 4)

[<Fact>]
let ``centroid calculates average vector`` () =
    let v1 = [| 2.0f; 4.0f |]
    let v2 = [| 4.0f; 8.0f |]
    let center = MetricSpace.centroid [ v1; v2 ]
    Assert.Equal<float32>([| 3.0f; 6.0f |], center)
