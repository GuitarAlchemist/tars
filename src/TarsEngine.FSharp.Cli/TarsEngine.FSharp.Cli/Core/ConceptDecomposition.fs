// ============================================================================
// TARS Sparse Concept Decomposition Module
// Implements interpretable vector decomposition for reasoning transparency
// ============================================================================

namespace TarsEngine.FSharp.Cli.Core

open System
open System.Collections.Generic
open System.Runtime.InteropServices
open System.Threading.Tasks

// ============================================================================
// CORE TYPES AND STRUCTURES
// ============================================================================

/// Represents a concept in the basis dictionary
type ConceptBasis = {
    Id: string
    Name: string
    Description: string
    Vector: float[]
    Category: string
    Weight: float
    CreatedAt: DateTime
}

/// Sparse representation of a vector in terms of concept basis
type SparseConceptRepresentation = {
    OriginalVector: float[]
    ConceptWeights: Map<string, float>
    ReconstructionError: float
    Sparsity: float
    InterpretationText: string
    Timestamp: DateTime
}

/// Configuration for sparse coding algorithms
type SparseCodingConfig = {
    Lambda: float              // L1 regularization strength
    MaxIterations: int         // Maximum iterations for optimization
    Tolerance: float           // Convergence tolerance
    UseCuda: bool             // Whether to use CUDA acceleration
    SparsityTarget: float     // Target sparsity level (0.0 to 1.0)
}

/// Result of concept decomposition analysis
type ConceptDecompositionResult = {
    SparseRepresentation: SparseConceptRepresentation
    DominantConcepts: (string * float) list
    SemanticSummary: string
    QualityMetrics: Map<string, float>
    Success: bool
    ErrorMessage: string option
}

// ============================================================================
// CUDA INTEROP (OPTIONAL ACCELERATION)
// ============================================================================

module CudaInterop =
    [<DllImport("libminimal_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int sparse_lasso_cuda(
        double[] dictionary,
        int dictRows,
        int dictCols,
        double[] target,
        int targetLength,
        double lambda,
        double[] output,
        double[] residual)

    [<DllImport("libminimal_cuda", CallingConvention = CallingConvention.Cdecl)>]
    extern int vector_similarity_batch_cuda(
        double[] vectors,
        int numVectors,
        int vectorDim,
        double[] query,
        double[] similarities)

// ============================================================================
// SPARSE CODING ALGORITHMS
// ============================================================================

module SparseCoding =
    
    /// CPU-based LASSO implementation using coordinate descent
    let lassoCoordinateDescent (dictionary: float[][]) (target: float[]) (lambda: float) (maxIter: int) (tolerance: float) : float[] =
        let numConcepts = dictionary.Length
        let vectorDim = target.Length
        let weights = Array.zeroCreate numConcepts
        let residual = Array.copy target
        
        let mutable converged = false
        let mutable iteration = 0
        
        while not converged && iteration < maxIter do
            let mutable maxChange = 0.0
            
            for j in 0 .. numConcepts - 1 do
                let oldWeight = weights.[j]
                
                // Compute correlation with residual
                let correlation = 
                    Array.zip residual dictionary.[j]
                    |> Array.sumBy (fun (r, d) -> r * d)
                
                // Soft thresholding
                let newWeight = 
                    if correlation > lambda then correlation - lambda
                    elif correlation < -lambda then correlation + lambda
                    else 0.0
                
                weights.[j] <- newWeight
                let change = newWeight - oldWeight
                maxChange <- max maxChange (abs change)
                
                // Update residual
                if abs change > 1e-10 then
                    for i in 0 .. vectorDim - 1 do
                        residual.[i] <- residual.[i] - change * dictionary.[j].[i]
            
            converged <- maxChange < tolerance
            iteration <- iteration + 1
        
        weights

    /// CUDA-accelerated LASSO (if available)
    let lassoCuda (dictionary: float[][]) (target: float[]) (lambda: float) : float[] option =
        try
            let dictFlat = dictionary |> Array.collect id
            let dictRows = dictionary.Length
            let dictCols = if dictRows > 0 then dictionary.[0].Length else 0
            let output = Array.zeroCreate dictRows
            let residual = Array.zeroCreate dictCols
            
            let result = CudaInterop.sparse_lasso_cuda(dictFlat, dictRows, dictCols, target, target.Length, lambda, output, residual)
            
            if result = 0 then Some output else None
        with
        | _ -> None

    /// Main sparse coding function with fallback
    let sparseCoding (config: SparseCodingConfig) (dictionary: float[][]) (target: float[]) : float[] =
        if config.UseCuda then
            match lassoCuda dictionary target config.Lambda with
            | Some result -> result
            | None -> 
                // Fallback to CPU implementation
                lassoCoordinateDescent dictionary target config.Lambda config.MaxIterations config.Tolerance
        else
            lassoCoordinateDescent dictionary target config.Lambda config.MaxIterations config.Tolerance

// ============================================================================
// CONCEPT BASIS MANAGEMENT
// ============================================================================

module ConceptBasisManager =
    
    /// Default concept basis for common semantic dimensions
    let createDefaultConceptBasis () : ConceptBasis[] = [|
        { Id = "positive_sentiment"; Name = "Positive Sentiment"; Description = "Positive emotional tone"; 
          Vector = [| 0.8; 0.6; 0.2; -0.1; 0.7; 0.5; 0.3; 0.9 |]; Category = "sentiment"; Weight = 1.0; CreatedAt = DateTime.UtcNow }
        { Id = "negative_sentiment"; Name = "Negative Sentiment"; Description = "Negative emotional tone"; 
          Vector = [| -0.7; -0.5; -0.8; 0.9; -0.6; -0.4; -0.2; -0.8 |]; Category = "sentiment"; Weight = 1.0; CreatedAt = DateTime.UtcNow }
        { Id = "technical_domain"; Name = "Technical Domain"; Description = "Technical or scientific content"; 
          Vector = [| 0.2; 0.9; 0.7; 0.1; 0.8; 0.6; 0.4; 0.3 |]; Category = "domain"; Weight = 1.0; CreatedAt = DateTime.UtcNow }
        { Id = "creative_domain"; Name = "Creative Domain"; Description = "Creative or artistic content"; 
          Vector = [| 0.9; 0.3; 0.8; 0.7; 0.2; 0.6; 0.9; 0.5 |]; Category = "domain"; Weight = 1.0; CreatedAt = DateTime.UtcNow }
        { Id = "logical_reasoning"; Name = "Logical Reasoning"; Description = "Logical or analytical thinking"; 
          Vector = [| 0.1; 0.8; 0.9; 0.2; 0.7; 0.8; 0.1; 0.6 |]; Category = "reasoning"; Weight = 1.0; CreatedAt = DateTime.UtcNow }
        { Id = "intuitive_reasoning"; Name = "Intuitive Reasoning"; Description = "Intuitive or creative thinking"; 
          Vector = [| 0.8; 0.2; 0.3; 0.9; 0.4; 0.1; 0.8; 0.7 |]; Category = "reasoning"; Weight = 1.0; CreatedAt = DateTime.UtcNow }
        { Id = "uncertainty"; Name = "Uncertainty"; Description = "Uncertain or speculative content"; 
          Vector = [| 0.3; 0.5; 0.1; 0.8; 0.2; 0.9; 0.4; 0.6 |]; Category = "confidence"; Weight = 1.0; CreatedAt = DateTime.UtcNow }
        { Id = "confidence"; Name = "Confidence"; Description = "Confident or certain content"; 
          Vector = [| 0.9; 0.8; 0.7; 0.1; 0.8; 0.2; 0.9; 0.3 |]; Category = "confidence"; Weight = 1.0; CreatedAt = DateTime.UtcNow }
    |]

    /// Normalize concept basis vectors
    let normalizeBasis (basis: ConceptBasis[]) : ConceptBasis[] =
        basis |> Array.map (fun concept ->
            let norm = concept.Vector |> Array.sumBy (fun x -> x * x) |> sqrt
            let normalizedVector = concept.Vector |> Array.map (fun x -> x / norm)
            { concept with Vector = normalizedVector }
        )

    /// Add custom concept to basis
    let addCustomConcept (basis: ConceptBasis[]) (newConcept: ConceptBasis) : ConceptBasis[] =
        Array.append basis [| newConcept |]

    /// Get concept basis as dictionary matrix
    let getBasisMatrix (basis: ConceptBasis[]) : float[][] =
        basis |> Array.map (fun concept -> concept.Vector)

// ============================================================================
// CONCEPT DECOMPOSITION ENGINE
// ============================================================================

module ConceptDecompositionEngine =
    
    /// Default configuration for sparse coding
    let defaultConfig = {
        Lambda = 0.1
        MaxIterations = 1000
        Tolerance = 1e-6
        UseCuda = true
        SparsityTarget = 0.3
    }

    /// Calculate sparsity of weight vector
    let calculateSparsity (weights: float[]) : float =
        let nonZeroCount = weights |> Array.filter (fun w -> abs w > 1e-6) |> Array.length
        1.0 - (float nonZeroCount / float weights.Length)

    /// Calculate reconstruction error
    let calculateReconstructionError (original: float[]) (reconstructed: float[]) : float =
        Array.zip original reconstructed
        |> Array.sumBy (fun (o, r) -> (o - r) * (o - r))
        |> sqrt

    /// Reconstruct vector from sparse representation
    let reconstructVector (basis: ConceptBasis[]) (weights: float[]) : float[] =
        let vectorDim = basis.[0].Vector.Length
        let reconstructed = Array.zeroCreate vectorDim
        
        for i in 0 .. basis.Length - 1 do
            let weight = weights.[i]
            if abs weight > 1e-6 then
                for j in 0 .. vectorDim - 1 do
                    reconstructed.[j] <- reconstructed.[j] + weight * basis.[i].Vector.[j]
        
        reconstructed

    /// Generate interpretation text from sparse weights
    let generateInterpretation (basis: ConceptBasis[]) (weights: float[]) : string =
        let significantConcepts = 
            Array.zip basis weights
            |> Array.filter (fun (_, weight) -> abs weight > 0.1)
            |> Array.sortByDescending (fun (_, weight) -> abs weight)
            |> Array.take (min 5 (Array.length basis))
        
        if significantConcepts.Length = 0 then
            "No significant concepts detected"
        else
            let descriptions = 
                significantConcepts
                |> Array.map (fun (concept, weight) ->
                    let strength = if abs weight > 0.5 then "strongly" elif abs weight > 0.3 then "moderately" else "weakly"
                    let direction = if weight > 0.0 then "positive" else "negative"
                    $"{strength} {direction} {concept.Name.ToLower()}")
            
            "This vector represents: " + String.Join(", ", descriptions)

    /// Main decomposition function
    let decomposeVector (config: SparseCodingConfig) (basis: ConceptBasis[]) (targetVector: float[]) : ConceptDecompositionResult =
        try
            // Normalize basis if needed
            let normalizedBasis = ConceptBasisManager.normalizeBasis basis
            let basisMatrix = ConceptBasisManager.getBasisMatrix normalizedBasis
            
            // Perform sparse coding
            let weights = SparseCoding.sparseCoding config basisMatrix targetVector
            
            // Calculate metrics
            let sparsity = calculateSparsity weights
            let reconstructed = reconstructVector normalizedBasis weights
            let reconstructionError = calculateReconstructionError targetVector reconstructed
            
            // Create concept weights map
            let conceptWeights = 
                Array.zip normalizedBasis weights
                |> Array.filter (fun (_, weight) -> abs weight > 1e-6)
                |> Array.map (fun (concept, weight) -> (concept.Id, weight))
                |> Map.ofArray
            
            // Generate interpretation
            let interpretation = generateInterpretation normalizedBasis weights
            
            // Get dominant concepts
            let dominantConcepts = 
                Array.zip normalizedBasis weights
                |> Array.filter (fun (_, weight) -> abs weight > 0.05)
                |> Array.sortByDescending (fun (_, weight) -> abs weight)
                |> Array.map (fun (concept, weight) -> (concept.Name, weight))
                |> Array.toList
            
            // Create sparse representation
            let sparseRep = {
                OriginalVector = targetVector
                ConceptWeights = conceptWeights
                ReconstructionError = reconstructionError
                Sparsity = sparsity
                InterpretationText = interpretation
                Timestamp = DateTime.UtcNow
            }
            
            // Quality metrics
            let qualityMetrics = Map.ofList [
                ("reconstruction_error", reconstructionError)
                ("sparsity", sparsity)
                ("num_active_concepts", float conceptWeights.Count)
                ("max_weight", weights |> Array.map abs |> Array.max)
            ]
            
            {
                SparseRepresentation = sparseRep
                DominantConcepts = dominantConcepts
                SemanticSummary = interpretation
                QualityMetrics = qualityMetrics
                Success = true
                ErrorMessage = None
            }
            
        with
        | ex ->
            {
                SparseRepresentation = {
                    OriginalVector = targetVector
                    ConceptWeights = Map.empty
                    ReconstructionError = Double.MaxValue
                    Sparsity = 0.0
                    InterpretationText = "Decomposition failed"
                    Timestamp = DateTime.UtcNow
                }
                DominantConcepts = []
                SemanticSummary = $"Error: {ex.Message}"
                QualityMetrics = Map.empty
                Success = false
                ErrorMessage = Some ex.Message
            }

// ============================================================================
// TARS INTEGRATION UTILITIES
// ============================================================================

module DemoScenarios =
    // ============================================================================
    // DEMO SCENARIOS FOR CONCEPT ANALYSIS
    // ============================================================================

    /// Get predefined demo scenarios for concept analysis
    let getDemoScenarios () : (string * float[]) list = [
        ("Positive Technical Content", [| 0.8; 0.3; 0.9; 0.1; 0.7; 0.6; 0.4; 0.8 |])
        ("Negative Emotional Response", [| -0.6; -0.4; 0.2; 0.8; -0.5; -0.3; 0.1; -0.7 |])
        ("Creative Problem Solving", [| 0.4; 0.9; 0.6; 0.3; 0.8; 0.7; 0.9; 0.5 |])
        ("Analytical Reasoning", [| 0.7; 0.2; 0.8; 0.4; 0.9; 0.3; 0.6; 0.8 |])
        ("Social Communication", [| 0.3; 0.8; 0.4; 0.6; 0.5; 0.9; 0.7; 0.4 |])
        ("Abstract Mathematical Thinking", [| 0.9; 0.1; 0.7; 0.8; 0.6; 0.2; 0.8; 0.9 |])
        ("Practical Implementation", [| 0.6; 0.7; 0.5; 0.9; 0.4; 0.8; 0.3; 0.6 |])
        ("Mixed Sentiment Complex Problem", [| 0.2; -0.3; 0.8; 0.5; -0.1; 0.7; 0.4; -0.2 |])
    ]

module TarsIntegration =

    /// Convert TARS reasoning trace to vector (placeholder - would integrate with actual TARS vector system)
    let reasoningTraceToVector (reasoningText: string) : float[] =
        // This is a simplified example - in practice would use actual TARS embedding system
        let hash = reasoningText.GetHashCode()
        let random = Random(hash)
        Array.init 8 (fun _ -> random.NextDouble() * 2.0 - 1.0)
    
    /// Create concept decomposition for TARS reasoning trace
    let analyzeReasoningTrace (reasoningText: string) : ConceptDecompositionResult =
        let basis = ConceptBasisManager.createDefaultConceptBasis()
        let vector = reasoningTraceToVector reasoningText
        ConceptDecompositionEngine.decomposeVector ConceptDecompositionEngine.defaultConfig basis vector
    
    /// Format decomposition result for TARS output
    let formatForTarsOutput (result: ConceptDecompositionResult) : string =
        if result.Success then
            let conceptList = 
                result.DominantConcepts
                |> List.take (min 3 result.DominantConcepts.Length)
                |> List.map (fun (name, weight) -> $"{name}: {weight:F2}")
                |> String.concat ", "
            
            let sparsityValue = result.QualityMetrics.["sparsity"]
            let errorValue = result.QualityMetrics.["reconstruction_error"]
            $"🧠 Concept Analysis: {result.SemanticSummary}\n" +
            $"📊 Dominant Concepts: {conceptList}\n" +
            $"🎯 Sparsity: {sparsityValue:F2} | Error: {errorValue:F3}"
        else
            let errorMsg = result.ErrorMessage |> Option.defaultValue "Unknown error"
            $"❌ Concept decomposition failed: {errorMsg}"
