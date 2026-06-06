namespace Tars.Cortex

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Tars.Core

/// Simple approximate nearest neighbor vector store using random hyperplane LSH.
/// Suitable for demo-sized workloads; falls back to full scan when buckets are empty.
type AnnVectorStore(hashBits: int) =
    do if hashBits <= 0 || hashBits > 30 then invalidArg "hashBits" "hashBits must be between 1 and 30"

    let buckets = ConcurrentDictionary<int, ConcurrentBag<string * float32[] * Map<string, string>>>()
    let vectors = ConcurrentDictionary<string, float32[] * Map<string, string>>()
    let mutable projections : float32[][] option = None
    let rng = Random()

    let ensureProjections (dim: int) =
        match projections with
        | Some p when p.Length = hashBits && p[0].Length = dim -> p
        | _ ->
            let proj =
                Array.init hashBits (fun _ ->
                    let v = Array.init dim (fun _ -> float32 (rng.NextDouble() * 2.0 - 1.0))
                    // normalize
                    let norm = Math.Sqrt(float (Array.sumBy (fun x -> float (x * x)) v))
                    if norm > 0.0 then v |> Array.map (fun x -> x / float32 norm) else v)
            projections <- Some proj
            proj

    let hashVector (vector: float32[]) =
        let proj = ensureProjections vector.Length
        let mutable h = 0
        for i = 0 to proj.Length - 1 do
            let dot =
                Array.fold2 (fun acc a b -> acc + a * b) 0.0f proj[i] vector
            if dot >= 0.0f then
                h <- h ||| (1 <<< i)
        h

    let cosineSimilarity (a: float32[]) (b: float32[]) =
        if a.Length <> b.Length || a.Length = 0 then 0.0f else
        let mutable dot = 0.0f
        let mutable magA = 0.0f
        let mutable magB = 0.0f
        for i = 0 to a.Length - 1 do
            dot <- dot + a[i] * b[i]
            magA <- magA + a[i] * a[i]
            magB <- magB + b[i] * b[i]
        let denom = Math.Sqrt(float magA) * Math.Sqrt(float magB)
        if denom = 0.0 then 0.0f else dot / float32 denom

    interface IVectorStore with
        member _.SaveAsync(collection: string, id: string, vector: float32[], payload: Map<string, string>) =
            task {
                // collection ignored for now; future: bucket per collection
                vectors[id] <- (vector, payload)
                let bucketId = hashVector vector
                let bag = buckets.GetOrAdd(bucketId, fun _ -> ConcurrentBag())
                bag.Add((id, vector, payload))
            }
            :> Task

        member _.SearchAsync(collection: string, queryVector: float32[], limit: int) =
            task {
                let bucketId = hashVector queryVector
                let candidates : seq<string * float32[] * Map<string, string>> =
                    match buckets.TryGetValue(bucketId) with
                    | true, bag when bag.Count > 0 -> bag :> seq<_>
                    | _ ->
                        // fallback: scan all stored vectors if bucket empty
                        vectors
                        |> Seq.map (fun kvp -> kvp.Key, fst kvp.Value, snd kvp.Value)

                let scored =
                    candidates
                    |> Seq.map (fun (id, vec, payload) ->
                        let sim = cosineSimilarity queryVector vec
                        let dist = 1.0f - sim
                        let cleanId =
                            if String.IsNullOrEmpty id then Guid.NewGuid().ToString("N") else id
                        (cleanId, dist, payload))
                    |> Seq.sortBy (fun (_, d, _) -> d)
                    |> Seq.truncate (max 1 limit)
                    |> Seq.toList

                return scored
            }

    /// Expose bucket count for testing/introspection
    member _.BucketCount = buckets.Count
    member _.VectorCount = vectors.Count
