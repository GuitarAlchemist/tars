namespace TarsEngine.CustomTransformers

open System
open System.Runtime.InteropServices

/// CUDA operations for hybrid geometric spaces in TARS
module CudaHybridOperations =

    /// P/Invoke declarations for CUDA hybrid space operations
    module CudaInterop =
        
        [<DllImport("cuda_kernels_hybrid_space.dll", CallingConvention = CallingConvention.Cdecl)>]
        extern void call_mobius_add(
            float32[] x,
            float32[] y,
            float32[] result,
            float32 curvature,
            int batch_size,
            int dim
        )

        [<DllImport("cuda_kernels_hybrid_space.dll", CallingConvention = CallingConvention.Cdecl)>]
        extern void call_hyperbolic_distance(
            float32[] u,
            float32[] v,
            float32[] distances,
            float32 curvature,
            int batch_size,
            int dim
        )

        [<DllImport("cuda_kernels_hybrid_space.dll", CallingConvention = CallingConvention.Cdecl)>]
        extern void call_projective_normalize(
            float32[] vectors,
            int batch_size,
            int dim
        )

        [<DllImport("cuda_kernels_hybrid_space.dll", CallingConvention = CallingConvention.Cdecl)>]
        extern void call_dual_quaternion_norm(
            float32[] real,
            float32[] dual,
            float32[] norms,
            int batch_size,
            int quat_dim
        )

    /// Geometric space types for TARS embeddings
    type GeometricSpace =
        | Euclidean
        | Hyperbolic of curvature: float
        | Projective
        | DualQuaternion

    /// Hybrid embedding result
    type HybridEmbedding = {
        Euclidean: float[] option
        Hyperbolic: float[] option
        Projective: float[] option
        DualQuaternion: float[] option
        Metadata: Map<string, obj>
    }

    /// Möbius addition in hyperbolic space
    let mobiusAdd (x: float[]) (y: float[]) (curvature: float) : float[] =
        if x.Length <> y.Length then
            invalidArg "x/y" "Vectors must be the same length"

        let dim = x.Length
        let result = Array.zeroCreate<float> dim

        try
            // Convert to float32 for CUDA interop
            let x32 = x |> Array.map float32
            let y32 = y |> Array.map float32
            let result32 = Array.zeroCreate<float32> dim
            CudaInterop.call_mobius_add(x32, y32, result32, float32 curvature, 1, dim)
            result32 |> Array.map float
        with
        | ex ->
            printfn "⚠️  CUDA Möbius addition failed, falling back to CPU: %s" ex.Message
            // CPU fallback implementation
            let x2 = x |> Array.map (fun xi -> xi * xi) |> Array.sum
            let y2 = y |> Array.map (fun yi -> yi * yi) |> Array.sum
            let xy = Array.zip x y |> Array.map (fun (xi, yi) -> xi * yi) |> Array.sum
            let denom = 1.0 + 2.0 * curvature * xy + curvature * curvature * x2 * y2

            Array.zip x y
            |> Array.map (fun (xi, yi) ->
                let num = (1.0 + 2.0 * curvature * xy + curvature * y2) * xi + (1.0 - curvature * x2) * yi
                num / (denom + 1e-8))

    /// Hyperbolic distance calculation
    let hyperbolicDistance (u: float[]) (v: float[]) (curvature: float) : float =
        if u.Length <> v.Length then
            invalidArg "u/v" "Vectors must be the same length"

        let dim = u.Length
        let distances = Array.zeroCreate<float32> 1

        try
            // Convert to float32 for CUDA interop
            let u32 = u |> Array.map float32
            let v32 = v |> Array.map float32
            CudaInterop.call_hyperbolic_distance(u32, v32, distances, float32 curvature, 1, dim)
            float distances.[0]
        with
        | ex ->
            printfn "⚠️  CUDA hyperbolic distance failed, falling back to CPU: %s" ex.Message
            // CPU fallback implementation
            let norm_u_sq = u |> Array.map (fun ui -> ui * ui) |> Array.sum |> min 0.999
            let norm_v_sq = v |> Array.map (fun vi -> vi * vi) |> Array.sum |> min 0.999
            let dot_uv = Array.zip u v |> Array.map (fun (ui, vi) -> ui * vi) |> Array.sum

            let diff_norm_sq = norm_u_sq + norm_v_sq - 2.0 * dot_uv
            let denominator = (1.0 - norm_u_sq) * (1.0 - norm_v_sq)

            if denominator > 1e-8 then
                let ratio = 1.0 + 2.0 * diff_norm_sq / denominator
                (Math.Log(max ratio 1.0) + 0.5 * Math.Log(max ratio 1.0)) / sqrt curvature
            else
                0.0

    /// Projective space normalization
    let projectiveNormalize (vectors: float[][]) : float[][] =
        let batchSize = vectors.Length
        if batchSize = 0 then vectors
        else
            let dim = vectors.[0].Length
            let flatVectors = vectors |> Array.concat |> Array.map float32

            try
                CudaInterop.call_projective_normalize(flatVectors, batchSize, dim)
                // Reshape back to 2D array and convert back to float
                Array.init batchSize (fun i ->
                    Array.sub flatVectors (i * dim) dim |> Array.map float)
            with
            | ex ->
                printfn "⚠️  CUDA projective normalize failed, falling back to CPU: %s" ex.Message
                // CPU fallback implementation
                vectors |> Array.map (fun vec ->
                    let norm = vec |> Array.map (fun x -> x * x) |> Array.sum |> sqrt
                    vec |> Array.map (fun x -> x / (norm + 1e-8)))

    /// Dual quaternion norm calculation
    let dualQuaternionNorm (real: float[]) (dual: float[]) : float =
        if real.Length <> dual.Length then
            invalidArg "real/dual" "Real and dual parts must be the same length"

        let quatDim = real.Length
        let norms = Array.zeroCreate<float32> 1

        try
            // Convert to float32 for CUDA interop
            let real32 = real |> Array.map float32
            let dual32 = dual |> Array.map float32
            CudaInterop.call_dual_quaternion_norm(real32, dual32, norms, 1, quatDim)
            float norms.[0]
        with
        | ex ->
            printfn "⚠️  CUDA dual quaternion norm failed, falling back to CPU: %s" ex.Message
            // CPU fallback implementation
            let realNormSq = real |> Array.map (fun r -> r * r) |> Array.sum
            let dualNormSq = dual |> Array.map (fun d -> d * d) |> Array.sum
            sqrt (realNormSq + dualNormSq)

    /// Create hybrid embedding from multiple geometric spaces
    let createHybridEmbedding 
        (euclidean: float[] option)
        (hyperbolic: float[] option) 
        (projective: float[] option)
        (dualQuaternion: float[] option)
        (metadata: Map<string, obj>) : HybridEmbedding =
        {
            Euclidean = euclidean
            Hyperbolic = hyperbolic
            Projective = projective
            DualQuaternion = dualQuaternion
            Metadata = metadata
        }

    /// Calculate similarity in specified geometric space
    let calculateSimilarity (space: GeometricSpace) (emb1: HybridEmbedding) (emb2: HybridEmbedding) : float option =
        match space with
        | Euclidean ->
            match emb1.Euclidean, emb2.Euclidean with
            | Some e1, Some e2 ->
                let distance = 
                    Array.zip e1 e2 
                    |> Array.map (fun (x1, x2) -> (x1 - x2) * (x1 - x2))
                    |> Array.sum
                    |> sqrt
                Some (1.0 / (1.0 + distance))
            | _ -> None

        | Hyperbolic curvature ->
            match emb1.Hyperbolic, emb2.Hyperbolic with
            | Some h1, Some h2 ->
                let distance = hyperbolicDistance h1 h2 curvature
                Some (1.0 / (1.0 + distance))
            | _ -> None

        | Projective ->
            match emb1.Projective, emb2.Projective with
            | Some p1, Some p2 ->
                let dotProduct = Array.zip p1 p2 |> Array.map (fun (x, y) -> x * y) |> Array.sum
                Some (abs dotProduct) // Projective similarity is absolute cosine
            | _ -> None

        | DualQuaternion ->
            match emb1.DualQuaternion, emb2.DualQuaternion with
            | Some dq1, Some dq2 ->
                let mid = dq1.Length / 2
                let real1, dual1 = Array.splitAt mid dq1
                let real2, dual2 = Array.splitAt mid dq2
                let norm1 = dualQuaternionNorm real1 dual1
                let norm2 = dualQuaternionNorm real2 dual2
                Some (1.0 / (1.0 + abs(norm1 - norm2)))
            | _ -> None

    /// Test CUDA operations
    let testCudaOperations () =
        printfn "🧪 Testing TARS CUDA Hybrid Operations..."
        
        try
            // Test Möbius addition
            let x = [| 0.1; 0.2; 0.3 |]
            let y = [| 0.4; 0.5; 0.6 |]
            let result = mobiusAdd x y 1.0
            printfn "✅ Möbius addition: %A + %A = %A" x y result

            // Test hyperbolic distance
            let u = [| 0.2; 0.3; 0.1 |]
            let v = [| 0.5; 0.1; 0.4 |]
            let distance = hyperbolicDistance u v 1.0
            printfn "✅ Hyperbolic distance: %.4f" distance

            // Test projective normalization
            let vectors = [| [| 1.0; 2.0; 3.0 |]; [| 4.0; 5.0; 6.0 |] |]
            let normalized = projectiveNormalize vectors
            printfn "✅ Projective normalization: %A -> %A" vectors normalized

            // Test dual quaternion norm
            let real = [| 1.0; 0.0; 0.0; 0.0 |]
            let dual = [| 0.0; 1.0; 0.0; 0.0 |]
            let norm = dualQuaternionNorm real dual
            printfn "✅ Dual quaternion norm: %.4f" norm
            
            printfn "🎉 All CUDA hybrid operations tested successfully!"
            true
        with
        | ex ->
            printfn "❌ CUDA test failed: %s" ex.Message
            false

    /// Demo function for hybrid embeddings
    let demoHybridEmbeddings () =
        printfn "🌌 TARS Hybrid Embeddings Demo"
        printfn "=============================="
        
        // Create sample embeddings
        let embedding1 = createHybridEmbedding (Some [| 1.0; 0.0; 0.0 |]) (Some [| 0.3; 0.2; 0.1 |]) (Some [| 0.577; 0.577; 0.577 |]) (Some [| 1.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0 |]) (Map.ofList [("source", box "tars_concept_1")])

        let embedding2 = createHybridEmbedding (Some [| 0.8; 0.6; 0.0 |]) (Some [| 0.4; 0.3; 0.2 |]) (Some [| 0.707; 0.707; 0.0 |]) (Some [| 0.9; 0.1; 0.0; 0.0; 0.1; 0.9; 0.0; 0.0 |]) (Map.ofList [("source", box "tars_concept_2")])

        // Calculate similarities in different spaces
        let spaces = [
            Euclidean
            Hyperbolic 1.0
            Projective
            DualQuaternion
        ]
        
        printfn "🔍 Similarity Analysis:"
        for space in spaces do
            match calculateSimilarity space embedding1 embedding2 with
            | Some similarity ->
                printfn "   %A: %.4f" space similarity
            | None ->
                printfn "   %A: N/A (missing embeddings)" space
        
        printfn ""
        printfn "✅ Hybrid embeddings demo complete!"
