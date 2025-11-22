namespace Tars.Engine.VectorStore

open System
open System.Numerics
open MathNet.Numerics
open MathNet.Numerics.IntegralTransforms

/// Similarity computation implementations for different mathematical spaces
module SimilarityComputer =
    
    /// Compute cosine similarity between two vectors
    let cosineSimilarity (v1: FloatVector) (v2: FloatVector) : float =
        if v1.Length <> v2.Length then 0.0
        else
            let dot = Array.map2 (*) v1 v2 |> Array.sum
            let mag1 = sqrt (Array.sumBy (fun x -> x * x) v1)
            let mag2 = sqrt (Array.sumBy (fun x -> x * x) v2)
            if mag1 > 1e-10 && mag2 > 1e-10 then
                dot / (mag1 * mag2)
            else
                0.0
    
    /// Compute similarity in frequency domain (FFT space)
    let fftSimilarity (v1: ComplexVector) (v2: ComplexVector) : float =
        if v1.Length <> v2.Length then 0.0
        else
            let dot = Array.map2 (fun (a: Complex) (b: Complex) -> a.Real * b.Real + a.Imaginary * b.Imaginary) v1 v2 |> Array.sum
            let mag1 = sqrt (Array.sumBy (fun (x: Complex) -> x.Magnitude * x.Magnitude) v1)
            let mag2 = sqrt (Array.sumBy (fun (x: Complex) -> x.Magnitude * x.Magnitude) v2)
            if mag1 > 1e-10 && mag2 > 1e-10 then
                dot / (mag1 * mag2)
            else
                0.0
    
    /// Compute phase correlation for pattern detection
    let phaseCorrelation (v1: ComplexVector) (v2: ComplexVector) : float =
        if v1.Length <> v2.Length then 0.0
        else
            let phases1 = v1 |> Array.map (fun (c: Complex) -> c.Phase)
            let phases2 = v2 |> Array.map (fun (c: Complex) -> c.Phase)
            let phaseDiffs = Array.map2 (fun p1 p2 -> abs(p1 - p2)) phases1 phases2
            let avgPhaseDiff = Array.average phaseDiffs
            1.0 / (1.0 + avgPhaseDiff)
    
    /// Compute dual space similarity (functional alignment)
    let dualSimilarity (v1: FloatVector) (v2: FloatVector) : float =
        // In dual space, we measure how well one vector "probes" the other
        cosineSimilarity v1 v2

    /// Compute projective similarity (direction-based, scale-invariant)
    let projectiveSimilarity (v1: FloatVector) (v2: FloatVector) : float =
        if v1.Length <> v2.Length then 0.0
        else
            // Normalize to unit vectors for scale invariance
            let norm1 = sqrt (Array.sumBy (fun x -> x * x) v1)
            let norm2 = sqrt (Array.sumBy (fun x -> x * x) v2)
            if norm1 > 1e-10 && norm2 > 1e-10 then
                let unit1 = v1 |> Array.map (fun x -> x / norm1)
                let unit2 = v2 |> Array.map (fun x -> x / norm2)
                abs (cosineSimilarity unit1 unit2)
            else
                0.0
    
    /// Compute hyperbolic similarity (for hierarchical data)
    let hyperbolicSimilarity (v1: FloatVector) (v2: FloatVector) : float =
        if v1.Length <> v2.Length then 0.0
        else
            // Poincaré disk model distance
            let diff = Array.map2 (fun a b -> a - b) v1 v2
            let norm = sqrt (Array.sumBy (fun x -> x * x) diff)
            let norm1 = sqrt (Array.sumBy (fun x -> x * x) v1)
            let norm2 = sqrt (Array.sumBy (fun x -> x * x) v2)
            
            // Ensure we're in the unit disk
            let clamp x = max -0.99 (min 0.99 x)
            let r1 = clamp norm1
            let r2 = clamp norm2
            
            // Hyperbolic distance in Poincaré disk
            let numerator = 2.0 * norm * norm
            let denominator = (1.0 - r1 * r1) * (1.0 - r2 * r2)
            if denominator > 1e-10 then
                let distance = log (1.0 + numerator / denominator + sqrt ((1.0 + numerator / denominator) ** 2.0 - 1.0))
                1.0 / (1.0 + distance)
            else
                0.0
    
    /// Compute wavelet similarity (multi-resolution)
    let waveletSimilarity (v1: FloatVector) (v2: FloatVector) : float =
        // Simple wavelet approximation using windowed averages
        if v1.Length <> v2.Length then 0.0
        else
            let windowSize = max 3 (v1.Length / 8)
            let windows1 = Array.windowed windowSize v1 |> Array.map Array.average
            let windows2 = Array.windowed windowSize v2 |> Array.map Array.average
            cosineSimilarity windows1 windows2
    
    /// Compute Minkowski spacetime similarity
    let minkowskiSimilarity (v1: FloatVector) (v2: FloatVector) : float =
        if v1.Length <> v2.Length || v1.Length = 0 then 0.0
        else
            // Separate spatial and temporal components
            let spatial1 = v1.[0..v1.Length-2]
            let spatial2 = v2.[0..v2.Length-2]
            let time1 = v1.[v1.Length-1]
            let time2 = v2.[v2.Length-1]
            
            // Minkowski metric: ds² = dx² + dy² + dz² - c²dt²
            let spatialDist = Array.map2 (fun a b -> (a - b) ** 2.0) spatial1 spatial2 |> Array.sum
            let timeDist = (time1 - time2) ** 2.0
            let minkowskiDist = sqrt (abs (spatialDist - timeDist))
            
            1.0 / (1.0 + minkowskiDist)
    
    /// Compute Pauli matrix similarity
    let pauliSimilarity ((a1,b1,c1,d1): Matrix2x2) ((a2,b2,c2,d2): Matrix2x2) : float =
        let dotProduct (c1: Complex) (c2: Complex) = c1.Real * c2.Real + c1.Imaginary * c2.Imaginary
        let sum = dotProduct a1 a2 + dotProduct b1 b2 + dotProduct c1 c2 + dotProduct d1 d2
        let norm1 = sqrt (a1.Magnitude**2.0 + b1.Magnitude**2.0 + c1.Magnitude**2.0 + d1.Magnitude**2.0)
        let norm2 = sqrt (a2.Magnitude**2.0 + b2.Magnitude**2.0 + c2.Magnitude**2.0 + d2.Magnitude**2.0)
        if norm1 > 1e-10 && norm2 > 1e-10 then
            sum / (norm1 * norm2)
        else
            0.0
    
    /// Adjust score based on belief state
    let adjustForBelief (score: float) (belief: TruthValue) : float =
        match belief with
        | True -> score
        | False -> 0.0
        | Both -> score * 0.7  // Partial confidence
        | Neither -> score * 0.3  // Low confidence

/// Multi-space similarity computer implementation
type MultiSpaceSimilarityComputer(config: VectorStoreConfig) =
    
    interface ISimilarityComputer with
        
        member _.ComputeSimilarity (emb1: MultiSpaceEmbedding) (emb2: MultiSpaceEmbedding) : LabeledScore list =
            let scores = ResizeArray<LabeledScore>()
            
            // Raw cosine similarity
            let score = SimilarityComputer.cosineSimilarity emb1.Raw emb2.Raw
            let adjustedScore = SimilarityComputer.adjustForBelief score emb2.Belief
            scores.Add({
                Space = "raw"
                Score = adjustedScore
                Reason = "Direct cosine similarity in embedding space"
                Confidence = 0.9
            })

            // FFT similarity
            if config.EnableFFT then
                let score = SimilarityComputer.fftSimilarity emb1.FFT emb2.FFT
                let adjustedScore = SimilarityComputer.adjustForBelief score emb2.Belief
                scores.Add({
                    Space = "fft"
                    Score = adjustedScore
                    Reason = "Frequency domain cosine similarity"
                    Confidence = 0.8
                })
                
                // Phase correlation for pattern detection
                let phaseScore = SimilarityComputer.phaseCorrelation emb1.FFT emb2.FFT
                let adjustedPhaseScore = SimilarityComputer.adjustForBelief phaseScore emb2.Belief
                scores.Add({
                    Space = "phase"
                    Score = adjustedPhaseScore
                    Reason = "Phase correlation for pattern detection"
                    Confidence = 0.7
                })
            
            // Dual space similarity
            if config.EnableDual then
                let score = SimilarityComputer.dualSimilarity emb1.Dual emb2.Dual
                let adjustedScore = SimilarityComputer.adjustForBelief score emb2.Belief
                scores.Add({
                    Space = "dual"
                    Score = adjustedScore
                    Reason = "Functional probing in dual space"
                    Confidence = 0.8
                })
            
            // Projective similarity
            if config.EnableProjective then
                let score = SimilarityComputer.projectiveSimilarity emb1.Projective emb2.Projective
                let adjustedScore = SimilarityComputer.adjustForBelief score emb2.Belief
                scores.Add({
                    Space = "projective"
                    Score = adjustedScore
                    Reason = "Scale-invariant directional alignment"
                    Confidence = 0.8
                })
            
            // Hyperbolic similarity
            if config.EnableHyperbolic then
                let score = SimilarityComputer.hyperbolicSimilarity emb1.Hyperbolic emb2.Hyperbolic
                let adjustedScore = SimilarityComputer.adjustForBelief score emb2.Belief
                scores.Add({
                    Space = "hyperbolic"
                    Score = adjustedScore
                    Reason = "Hierarchical distance in hyperbolic space"
                    Confidence = 0.7
                })
            
            // Wavelet similarity
            if config.EnableWavelet then
                let score = SimilarityComputer.waveletSimilarity emb1.Wavelet emb2.Wavelet
                let adjustedScore = SimilarityComputer.adjustForBelief score emb2.Belief
                scores.Add({
                    Space = "wavelet"
                    Score = adjustedScore
                    Reason = "Multi-resolution pattern matching"
                    Confidence = 0.7
                })
            
            // Minkowski similarity
            if config.EnableMinkowski then
                let score = SimilarityComputer.minkowskiSimilarity emb1.Minkowski emb2.Minkowski
                let adjustedScore = SimilarityComputer.adjustForBelief score emb2.Belief
                scores.Add({
                    Space = "minkowski"
                    Score = adjustedScore
                    Reason = "Spacetime interval similarity"
                    Confidence = 0.6
                })
            
            // Pauli matrix similarity
            if config.EnablePauli then
                let score = SimilarityComputer.pauliSimilarity emb1.Pauli emb2.Pauli
                let adjustedScore = SimilarityComputer.adjustForBelief score emb2.Belief
                scores.Add({
                    Space = "pauli"
                    Score = adjustedScore
                    Reason = "Quantum-like transformation similarity"
                    Confidence = 0.5
                })
            
            scores |> Seq.toList
        
        member _.AggregateSimilarity (scores: LabeledScore list) : float =
            if List.isEmpty scores then 0.0
            else
                // Weighted average based on confidence and configured weights
                let weightedSum = 
                    scores 
                    |> List.sumBy (fun s -> 
                        let weight = config.SpaceWeights.TryFind(s.Space) |> Option.defaultValue 1.0
                        s.Score * s.Confidence * weight)
                
                let totalWeight = 
                    scores 
                    |> List.sumBy (fun s -> 
                        let weight = config.SpaceWeights.TryFind(s.Space) |> Option.defaultValue 1.0
                        s.Confidence * weight)
                
                if totalWeight > 1e-10 then
                    weightedSum / totalWeight
                else
                    0.0
