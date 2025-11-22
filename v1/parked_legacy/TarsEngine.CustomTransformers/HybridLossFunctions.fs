namespace TarsEngine.CustomTransformers

open System
open System.Collections.Generic
open CudaHybridOperations

/// Advanced loss functions for TARS hybrid transformer training
module HybridLossFunctions =

    /// Loss function types for different geometric spaces
    type LossType =
        | MeanSquaredError
        | ContrastiveLoss of margin: float
        | TripletLoss of margin: float
        | BeliefAlignmentLoss
        | EntropyRegularization
        | HyperbolicContrastive of curvature: float * margin: float

    /// Training sample with multiple space embeddings
    type TrainingSample = {
        Text: string
        EmbeddingTargets: Map<GeometricSpace, float[]>
        BeliefState: float option  // For belief alignment
        ContradictionFlag: bool
        Metadata: Map<string, obj>
    }

    /// Loss computation result
    type LossResult = {
        TotalLoss: float
        ComponentLosses: Map<string, float>
        Gradients: Map<GeometricSpace, float[]> option
        Metrics: Map<string, float>
    }

    /// Euclidean mean squared error loss
    let meanSquaredError (predicted: float[]) (target: float[]) : float =
        if predicted.Length <> target.Length then
            invalidArg "predicted/target" "Arrays must have same length"
        
        Array.zip predicted target
        |> Array.map (fun (p, t) -> (p - t) ** 2.0)
        |> Array.average

    /// Contrastive loss for similar/dissimilar pairs
    let contrastiveLoss (anchor: float[]) (other: float[]) (isSimilar: bool) (margin: float) : float =
        let distance = 
            Array.zip anchor other
            |> Array.map (fun (a, o) -> (a - o) ** 2.0)
            |> Array.sum
            |> sqrt
        
        if isSimilar then
            distance ** 2.0  // Minimize distance for similar pairs
        else
            max 0.0 (margin - distance) ** 2.0  // Maximize distance for dissimilar pairs

    /// Triplet loss for anchor-positive-negative triplets
    let tripletLoss (anchor: float[]) (positive: float[]) (negative: float[]) (margin: float) : float =
        let distancePos = 
            Array.zip anchor positive
            |> Array.map (fun (a, p) -> (a - p) ** 2.0)
            |> Array.sum
            |> sqrt
        
        let distanceNeg = 
            Array.zip anchor negative
            |> Array.map (fun (a, n) -> (a - n) ** 2.0)
            |> Array.sum
            |> sqrt
        
        max 0.0 (distancePos - distanceNeg + margin)

    /// Hyperbolic contrastive loss using Poincaré distance
    let hyperbolicContrastiveLoss (anchor: float[]) (other: float[]) (isSimilar: bool) (curvature: float) (margin: float) : float =
        let distance = hyperbolicDistance anchor other curvature
        
        if isSimilar then
            distance ** 2.0
        else
            max 0.0 (margin - distance) ** 2.0

    /// Belief alignment loss for contradiction detection
    let beliefAlignmentLoss (embedding1: float[]) (embedding2: float[]) (beliefAlignment: float) : float =
        let similarity = 
            Array.zip embedding1 embedding2
            |> Array.map (fun (e1, e2) -> e1 * e2)
            |> Array.sum
        
        let normalizedSimilarity = similarity / (sqrt (Array.sumBy (fun x -> x ** 2.0) embedding1) * sqrt (Array.sumBy (fun x -> x ** 2.0) embedding2))
        
        // Loss increases when similarity doesn't match belief alignment
        (normalizedSimilarity - beliefAlignment) ** 2.0

    /// Entropy regularization to encourage diverse embeddings
    let entropyRegularization (embeddings: float[][]) : float =
        let numEmbeddings = embeddings.Length
        if numEmbeddings <= 1 then 0.0
        else
            let avgEmbedding = 
                Array.init embeddings.[0].Length (fun i ->
                    embeddings |> Array.averageBy (fun emb -> emb.[i]))
            
            let variance = 
                embeddings
                |> Array.map (fun emb ->
                    Array.zip emb avgEmbedding
                    |> Array.map (fun (e, avg) -> (e - avg) ** 2.0)
                    |> Array.sum)
                |> Array.average
            
            // Encourage higher variance (lower entropy loss)
            1.0 / (1.0 + variance)

    /// Weighted hybrid loss combiner
    type HybridLossWeights = {
        Euclidean: float
        Hyperbolic: float
        Projective: float
        DualQuaternion: float
        BeliefAlignment: float
        Entropy: float
        Contrastive: float
    }

    let defaultWeights = {
        Euclidean = 1.0
        Hyperbolic = 1.0
        Projective = 0.5
        DualQuaternion = 0.3
        BeliefAlignment = 2.0
        Entropy = 0.1
        Contrastive = 1.5
    }

    /// Compute hybrid loss across multiple geometric spaces
    let computeHybridLoss 
        (predicted: HybridEmbedding) 
        (target: HybridEmbedding) 
        (sample: TrainingSample)
        (weights: HybridLossWeights) : LossResult =
        
        let mutable totalLoss = 0.0
        let componentLosses = Dictionary<string, float>()
        let metrics = Dictionary<string, float>()
        
        // Euclidean loss
        match predicted.Euclidean, target.Euclidean with
        | Some predEuc, Some targEuc ->
            let eucLoss = meanSquaredError predEuc targEuc
            componentLosses.["euclidean"] <- eucLoss
            totalLoss <- totalLoss + weights.Euclidean * eucLoss
        | _ -> ()
        
        // Hyperbolic loss
        match predicted.Hyperbolic, target.Hyperbolic with
        | Some predHyp, Some targHyp ->
            let hypLoss = hyperbolicContrastiveLoss predHyp targHyp true 1.0 0.5
            componentLosses.["hyperbolic"] <- hypLoss
            totalLoss <- totalLoss + weights.Hyperbolic * hypLoss
        | _ -> ()
        
        // Projective loss
        match predicted.Projective, target.Projective with
        | Some predProj, Some targProj ->
            let projLoss = meanSquaredError predProj targProj
            componentLosses.["projective"] <- projLoss
            totalLoss <- totalLoss + weights.Projective * projLoss
        | _ -> ()
        
        // Dual quaternion loss
        match predicted.DualQuaternion, target.DualQuaternion with
        | Some predDQ, Some targDQ ->
            let dqLoss = meanSquaredError predDQ targDQ
            componentLosses.["dual_quaternion"] <- dqLoss
            totalLoss <- totalLoss + weights.DualQuaternion * dqLoss
        | _ -> ()
        
        // Belief alignment loss
        match predicted.Euclidean, target.Euclidean, sample.BeliefState with
        | Some predEuc, Some targEuc, Some beliefState ->
            let beliefLoss = beliefAlignmentLoss predEuc targEuc beliefState
            componentLosses.["belief_alignment"] <- beliefLoss
            totalLoss <- totalLoss + weights.BeliefAlignment * beliefLoss
        | _ -> ()
        
        // Contradiction penalty
        if sample.ContradictionFlag then
            let contradictionPenalty = 1.0
            componentLosses.["contradiction_penalty"] <- contradictionPenalty
            totalLoss <- totalLoss + contradictionPenalty
        
        // Compute metrics
        metrics.["total_loss"] <- totalLoss
        metrics.["num_components"] <- float componentLosses.Count
        
        {
            TotalLoss = totalLoss
            ComponentLosses = componentLosses |> Seq.map (|KeyValue|) |> Map.ofSeq
            Gradients = None  // Would be computed by automatic differentiation
            Metrics = metrics |> Seq.map (|KeyValue|) |> Map.ofSeq
        }

    /// Batch loss computation for multiple samples
    let computeBatchLoss 
        (predictedBatch: HybridEmbedding[])
        (targetBatch: HybridEmbedding[])
        (sampleBatch: TrainingSample[])
        (weights: HybridLossWeights) : LossResult =
        
        if predictedBatch.Length <> targetBatch.Length || targetBatch.Length <> sampleBatch.Length then
            invalidArg "batch" "All batches must have the same length"
        
        let batchSize = predictedBatch.Length
        let batchResults = 
            Array.zip3 predictedBatch targetBatch sampleBatch
            |> Array.map (fun (pred, targ, sample) -> computeHybridLoss pred targ sample weights)
        
        let totalLoss = batchResults |> Array.averageBy (fun r -> r.TotalLoss)
        
        let componentLosses = 
            batchResults
            |> Array.collect (fun r -> r.ComponentLosses |> Map.toArray)
            |> Array.groupBy fst
            |> Array.map (fun (key, values) -> key, values |> Array.averageBy snd)
            |> Map.ofArray
        
        let metrics = 
            Map.ofList [
                ("batch_size", float batchSize)
                ("avg_total_loss", totalLoss)
                ("loss_std", 
                    let losses = batchResults |> Array.map (fun r -> r.TotalLoss)
                    let mean = Array.average losses
                    sqrt (losses |> Array.averageBy (fun l -> (l - mean) ** 2.0)))
            ]
        
        {
            TotalLoss = totalLoss
            ComponentLosses = componentLosses
            Gradients = None
            Metrics = metrics
        }

    /// Adaptive loss weight adjustment based on performance
    let adaptLossWeights (currentWeights: HybridLossWeights) (lossHistory: LossResult[]) : HybridLossWeights =
        if lossHistory.Length < 2 then currentWeights
        else
            let recentLoss = lossHistory |> Array.last
            let previousLoss = lossHistory.[lossHistory.Length - 2]
            
            // Increase weights for components that are improving slowly
            let adaptWeight currentWeight componentName =
                match recentLoss.ComponentLosses.TryFind componentName, previousLoss.ComponentLosses.TryFind componentName with
                | Some recent, Some previous ->
                    let improvement = (previous - recent) / previous
                    if improvement < 0.01 then  // Less than 1% improvement
                        min 2.0 (currentWeight * 1.1)  // Increase weight by 10%
                    else
                        max 0.1 (currentWeight * 0.95)  // Decrease weight by 5%
                | _ -> currentWeight
            
            {
                Euclidean = adaptWeight currentWeights.Euclidean "euclidean"
                Hyperbolic = adaptWeight currentWeights.Hyperbolic "hyperbolic"
                Projective = adaptWeight currentWeights.Projective "projective"
                DualQuaternion = adaptWeight currentWeights.DualQuaternion "dual_quaternion"
                BeliefAlignment = adaptWeight currentWeights.BeliefAlignment "belief_alignment"
                Entropy = currentWeights.Entropy  // Keep entropy weight stable
                Contrastive = adaptWeight currentWeights.Contrastive "contrastive"
            }

    /// Demo function for loss computation
    let demoLossFunctions () =
        printfn "🧮 TARS Hybrid Loss Functions Demo"
        printfn "=================================="
        
        // Create sample embeddings
        let predicted = createHybridEmbedding (Some [| 0.8f; 0.6f; 0.2f |]) (Some [| 0.3f; 0.4f; 0.1f |]) (Some [| 0.707f; 0.707f; 0.0f |]) (Some [| 0.9f; 0.1f; 0.0f; 0.0f; 0.1f; 0.9f; 0.0f; 0.0f |]) (Map.empty)

        let target = createHybridEmbedding (Some [| 1.0f; 0.5f; 0.3f |]) (Some [| 0.2f; 0.5f; 0.2f |]) (Some [| 0.577f; 0.577f; 0.577f |]) (Some [| 1.0f; 0.0f; 0.0f; 0.0f; 0.0f; 1.0f; 0.0f; 0.0f |]) (Map.empty)
        
        let sample = {
            Text = "belief_graph { contradiction_rate: 0.05 }"
            EmbeddingTargets = Map.empty
            BeliefState = Some 0.8f
            ContradictionFlag = false
            Metadata = Map.empty
        }
        
        // Compute hybrid loss
        let lossResult = computeHybridLoss predicted target sample defaultWeights
        
        printfn "📊 Loss Computation Results:"
        printfn "   Total Loss: %.4f" lossResult.TotalLoss
        printfn "   Component Losses:"
        for kvp in lossResult.ComponentLosses do
            printfn "      %s: %.4f" kvp.Key kvp.Value
        
        printfn "   Metrics:"
        for kvp in lossResult.Metrics do
            printfn "      %s: %.4f" kvp.Key kvp.Value
        
        printfn ""
        printfn "✅ Hybrid loss functions demo complete!"
