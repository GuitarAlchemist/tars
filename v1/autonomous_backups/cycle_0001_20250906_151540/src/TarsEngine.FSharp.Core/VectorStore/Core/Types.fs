namespace Tars.Engine.VectorStore

open System
open System.Numerics

/// Truth value for tetravalent logic
type TruthValue =
    | True
    | False
    | Both
    | Neither

/// Labeled score with reasoning
type LabeledScore = {
    Space: string
    Score: float
    Reason: string
    Confidence: float
}

/// 2x2 complex matrix for Pauli matrices
type Matrix2x2 = Complex * Complex * Complex * Complex

/// Multi-dimensional vector representation
type FloatVector = float[]

/// Complex vector for frequency domain
type ComplexVector = Complex[]

/// Multi-space embedding combining different mathematical spaces
type MultiSpaceEmbedding = {
    /// Standard high-dimensional embedding (e.g., BERT, OpenAI)
    Raw: FloatVector
    /// Frequency domain representation via FFT
    FFT: ComplexVector
    /// Dual space functional representation
    Dual: FloatVector
    /// Projective/homogeneous coordinates
    Projective: FloatVector
    /// Hyperbolic space embedding for hierarchical data
    Hyperbolic: FloatVector
    /// Wavelet transform for multi-resolution analysis
    Wavelet: FloatVector
    /// Minkowski spacetime representation
    Minkowski: FloatVector
    /// Pauli matrix representation for quantum-like operations
    Pauli: Matrix2x2
    /// Belief state using tetravalent logic
    Belief: TruthValue
    /// Metadata for tracking and debugging
    Metadata: Map<string, string>
}

/// Document with multi-space embedding
type VectorDocument = {
    Id: string
    Content: string
    Embedding: MultiSpaceEmbedding
    Tags: string list
    Timestamp: DateTime
    Source: string option
}

/// Query with multi-space representation
type VectorQuery = {
    Text: string
    Embedding: MultiSpaceEmbedding
    Filters: Map<string, string>
    MaxResults: int
    MinScore: float
}

/// Search result with detailed scoring
type SearchResult = {
    Document: VectorDocument
    Scores: LabeledScore list
    FinalScore: float
    Rank: int
}

/// Vector store configuration
type VectorStoreConfig = {
    /// Dimension of raw embeddings
    RawDimension: int
    /// Enable FFT processing
    EnableFFT: bool
    /// Enable dual space processing
    EnableDual: bool
    /// Enable projective geometry
    EnableProjective: bool
    /// Enable hyperbolic embeddings
    EnableHyperbolic: bool
    /// Enable wavelet transforms
    EnableWavelet: bool
    /// Enable Minkowski spacetime
    EnableMinkowski: bool
    /// Enable Pauli matrices
    EnablePauli: bool
    /// Default aggregation weights for each space
    SpaceWeights: Map<string, float>
    /// Persistence settings
    PersistToDisk: bool
    StoragePath: string option
}

/// Similarity computation interface
type ISimilarityComputer =
    abstract member ComputeSimilarity: MultiSpaceEmbedding -> MultiSpaceEmbedding -> LabeledScore list
    abstract member AggregateSimilarity: LabeledScore list -> float

/// Vector store interface
type IVectorStore =
    abstract member AddDocument: VectorDocument -> Async<unit>
    abstract member AddDocuments: VectorDocument list -> Async<unit>
    abstract member Search: VectorQuery -> Async<SearchResult list>
    abstract member GetDocument: string -> Async<VectorDocument option>
    abstract member UpdateDocument: VectorDocument -> Async<unit>
    abstract member DeleteDocument: string -> Async<unit>
    abstract member GetDocumentCount: unit -> Async<int>
    abstract member Clear: unit -> Async<unit>

/// Embedding generator interface
type IEmbeddingGenerator =
    abstract member GenerateEmbedding: string -> Async<MultiSpaceEmbedding>
    abstract member GenerateEmbeddings: string list -> Async<MultiSpaceEmbedding list>

/// Inference engine interface
type IInferenceEngine =
    abstract member Infer: string -> Map<string, obj> -> Async<obj>
    abstract member InferWithContext: string -> VectorDocument list -> Map<string, obj> -> Async<obj>
    abstract member GetSimilarDocuments: string -> int -> Async<VectorDocument list>

/// Vector store statistics
type VectorStoreStats = {
    DocumentCount: int
    AverageEmbeddingSize: float
    SpaceUsageStats: Map<string, int>
    LastUpdated: DateTime
    IndexSize: int64
}

/// Transform utilities for different mathematical spaces
module TransformUtils =
    
    /// Pauli matrices
    let PauliI = (Complex.One, Complex.Zero, Complex.Zero, Complex.One)
    let PauliX = (Complex.Zero, Complex.One, Complex.One, Complex.Zero)
    let PauliY = (Complex.Zero, Complex(-1.0, 0.0), Complex(1.0, 0.0), Complex.Zero)
    let PauliZ = (Complex.One, Complex.Zero, Complex.Zero, Complex(-1.0, 0.0))
    
    /// Create homogeneous coordinates by appending w=1
    let toHomogeneous (v: FloatVector) : FloatVector =
        Array.append v [|1.0|]

    /// Convert from homogeneous coordinates
    let fromHomogeneous (v: FloatVector) : FloatVector =
        if v.Length > 0 then
            let w = v.[v.Length - 1]
            if abs w > 1e-10 then
                v.[0..v.Length-2] |> Array.map (fun x -> x / w)
            else
                v.[0..v.Length-2]
        else
            v

    /// Create Minkowski spacetime vector (space + time)
    let toMinkowski (spatial: FloatVector) (time: float) : FloatVector =
        Array.append spatial [|time|]

    /// Extract spatial and temporal components
    let fromMinkowski (v: FloatVector) : FloatVector * float =
        if v.Length > 0 then
            let spatial = v.[0..v.Length-2]
            let time = v.[v.Length-1]
            (spatial, time)
        else
            ([||], 0.0)
