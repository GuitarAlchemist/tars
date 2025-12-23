namespace Tars.Core

open System

/// Operations in high-dimensional semantic vector space
module MetricSpace =

    type Vector = float32[]

    /// Calculates the Euclidean distance between two vectors
    let euclideanDistance (v1: Vector) (v2: Vector) =
        if v1.Length <> v2.Length then
            invalidArg "v2" "Vectors must have the same length"

        let mutable sum = 0.0f

        for i in 0 .. v1.Length - 1 do
            let diff = v1.[i] - v2.[i]
            sum <- sum + (diff * diff)

        sqrt sum

    /// Calculates the Cosine Similarity between two vectors
    /// Returns 1.0 for identical vectors, 0.0 for orthogonal, -1.0 for opposite
    let cosineSimilarity (v1: Vector) (v2: Vector) =
        if v1.Length <> v2.Length then
            invalidArg "v2" "Vectors must have the same length"

        let mutable dotProduct = 0.0f
        let mutable mag1 = 0.0f
        let mutable mag2 = 0.0f

        for i in 0 .. v1.Length - 1 do
            dotProduct <- dotProduct + (v1.[i] * v2.[i])
            mag1 <- mag1 + (v1.[i] * v1.[i])
            mag2 <- mag2 + (v2.[i] * v2.[i])

        if mag1 = 0.0f || mag2 = 0.0f then
            0.0f
        else
            dotProduct / (sqrt (mag1) * sqrt (mag2))

    /// Calculates the centroid (average) of a set of vectors
    let centroid (vectors: Vector list) : Vector =
        if vectors.IsEmpty then
            Array.empty
        else
            let dim = vectors.Head.Length
            let count = float32 vectors.Length
            let result = Array.zeroCreate dim

            for v in vectors do
                if v.Length <> dim then
                    invalidArg "vectors" "All vectors must have the same dimension"

                for i in 0 .. dim - 1 do
                    result.[i] <- result.[i] + v.[i]

            for i in 0 .. dim - 1 do
                result.[i] <- result.[i] / count

            result

    /// Normalizes a vector to unit length
    let normalize (v: Vector) : Vector =
        let mag = v |> Array.sumBy (fun x -> x * x) |> sqrt
        if mag = 0.0f then v else v |> Array.map (fun x -> x / mag)
