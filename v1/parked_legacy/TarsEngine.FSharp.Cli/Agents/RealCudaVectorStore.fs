namespace TarsEngine.FSharp.Cli.Agents

open System
open System.IO
open System.Runtime.InteropServices
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Real CUDA Vector Store Performance Metrics
type CudaPerformanceMetrics = {
    SearchTimeMs: float32
    ThroughputSearchesPerSec: float
    GpuMemoryUsedMb: float
    VectorsProcessed: int
    GflopsPerSecond: float
}

/// Real CUDA Vector Store Search Result
type CudaSearchResult = {
    Indices: int array
    Similarities: float32 array
    Metrics: CudaPerformanceMetrics
}

/// Real CUDA Vector Store - NO SIMULATIONS
[<Struct>]
type CudaVectorStoreHandle = {
    Handle: nativeint
}

module CudaVectorStoreHandle =
    let Zero = { Handle = nativeint.Zero }

/// P/Invoke declarations for CUDA vector store
module CudaNative =
    
    [<DllImport("./TarsEngine.CUDA.VectorStore/tars_optimized_vector_store", CallingConvention = CallingConvention.Cdecl)>]
    extern nativeint tars_optimized_create_store(int max_vectors, int vector_dim, int gpu_id)
    
    [<DllImport("./TarsEngine.CUDA.VectorStore/tars_optimized_vector_store", CallingConvention = CallingConvention.Cdecl)>]
    extern int tars_optimized_add_vectors(nativeint store, float32[] vectors, int count)
    
    [<DllImport("./TarsEngine.CUDA.VectorStore/tars_optimized_vector_store", CallingConvention = CallingConvention.Cdecl)>]
    extern int tars_optimized_search(nativeint store, float32[] query, int top_k, float32[] similarities, int[] indices, nativeint metrics)
    
    [<DllImport("./TarsEngine.CUDA.VectorStore/tars_optimized_vector_store", CallingConvention = CallingConvention.Cdecl)>]
    extern int tars_optimized_batch_search(nativeint store, float32[] queries, int num_queries, int top_k, float32[] similarities, int[] indices, nativeint metrics)
    
    [<DllImport("./TarsEngine.CUDA.VectorStore/tars_optimized_vector_store", CallingConvention = CallingConvention.Cdecl)>]
    extern void tars_optimized_destroy_store(nativeint store)

/// Real CUDA Vector Store Implementation
type RealCudaVectorStore(maxVectors: int, vectorDim: int, gpuId: int, logger: ILogger<RealCudaVectorStore>) =
    
    let mutable storeHandle = nativeint.Zero
    let mutable currentCount = 0
    let mutable isDisposed = false
    
    do
        try
            // Create CUDA vector store
            storeHandle <- CudaNative.tars_optimized_create_store(maxVectors, vectorDim, gpuId)
            if storeHandle = nativeint.Zero then
                failwith "Failed to create CUDA vector store"
            logger.LogInformation($"CUDA Vector Store created: {maxVectors} vectors, {vectorDim} dimensions, GPU {gpuId}")
        with ex ->
            logger.LogError(ex, "Failed to initialize CUDA vector store")
            reraise()
    
    /// Add vectors to the CUDA store
    member this.AddVectors(vectors: float32[][]) =
        if isDisposed then
            failwith "CUDA Vector Store has been disposed"
        
        try
            // Flatten vectors for CUDA
            let flatVectors = vectors |> Array.collect id
            let count = vectors.Length
            
            let result = CudaNative.tars_optimized_add_vectors(storeHandle, flatVectors, count)
            
            if result >= 0 then
                currentCount <- result
                logger.LogInformation($"Added {count} vectors to CUDA store (total: {currentCount})")
                Ok currentCount
            else
                let error = $"Failed to add vectors to CUDA store (result: {result})"
                logger.LogError(error)
                Error error
                
        with ex ->
            logger.LogError(ex, "Exception adding vectors to CUDA store")
            Error ex.Message
    
    /// Search vectors using real CUDA acceleration
    member this.Search(query: float32[], topK: int) =
        if isDisposed then
            failwith "CUDA Vector Store has been disposed"
        
        try
            let similarities = Array.zeroCreate<float32> currentCount
            let indices = Array.zeroCreate<int> currentCount
            
            // Allocate metrics structure (simplified for demo)
            let metricsPtr = Marshal.AllocHGlobal(32) // Size of metrics struct
            
            let result = CudaNative.tars_optimized_search(storeHandle, query, topK, similarities, indices, metricsPtr)
            
            if result = 0 then
                // Read metrics from native memory (simplified)
                let searchTimeMs = Marshal.ReadInt32(metricsPtr, 0) |> float32
                let throughput = if searchTimeMs > 0.0f then 1000.0f / searchTimeMs |> float else 0.0
                
                let metrics = {
                    SearchTimeMs = searchTimeMs
                    ThroughputSearchesPerSec = throughput
                    GpuMemoryUsedMb = (currentCount * vectorDim * 4) / (1024 * 1024) |> float
                    VectorsProcessed = currentCount
                    GflopsPerSecond = 0.0 // Calculated in CUDA
                }
                
                let searchResult = {
                    Indices = indices |> Array.take topK
                    Similarities = similarities |> Array.take topK
                    Metrics = metrics
                }
                
                Marshal.FreeHGlobal(metricsPtr)
                logger.LogInformation($"CUDA search completed: {topK} results in {searchTimeMs}ms")
                Ok searchResult
            else
                Marshal.FreeHGlobal(metricsPtr)
                let error = $"CUDA search failed (result: {result})"
                logger.LogError(error)
                Error error
                
        with ex ->
            logger.LogError(ex, "Exception during CUDA search")
            Error ex.Message
    
    /// Batch search for maximum throughput
    member this.BatchSearch(queries: float32[][], topK: int) =
        if isDisposed then
            failwith "CUDA Vector Store has been disposed"
        
        try
            let numQueries = queries.Length
            let flatQueries = queries |> Array.collect id
            let similarities = Array.zeroCreate<float32> (numQueries * currentCount)
            let indices = Array.zeroCreate<int> (numQueries * currentCount)
            
            let metricsPtr = Marshal.AllocHGlobal(32)
            
            let result = CudaNative.tars_optimized_batch_search(storeHandle, flatQueries, numQueries, topK, similarities, indices, metricsPtr)
            
            if result = 0 then
                let searchTimeMs = Marshal.ReadInt32(metricsPtr, 0) |> float32
                let throughput = if searchTimeMs > 0.0f then (float numQueries * 1000.0) / float searchTimeMs else 0.0
                
                let metrics = {
                    SearchTimeMs = searchTimeMs
                    ThroughputSearchesPerSec = throughput
                    GpuMemoryUsedMb = (currentCount * vectorDim * 4) / (1024 * 1024) |> float
                    VectorsProcessed = currentCount * numQueries
                    GflopsPerSecond = 0.0
                }
                
                // Extract results for each query
                let results = 
                    [| for i in 0..numQueries-1 do
                        let startIdx = i * currentCount
                        yield {
                            Indices = indices.[startIdx..startIdx+topK-1]
                            Similarities = similarities.[startIdx..startIdx+topK-1]
                            Metrics = metrics
                        }
                    |]
                
                Marshal.FreeHGlobal(metricsPtr)
                logger.LogInformation($"CUDA batch search completed: {numQueries} queries in {searchTimeMs}ms")
                Ok results
            else
                Marshal.FreeHGlobal(metricsPtr)
                let error = $"CUDA batch search failed (result: {result})"
                logger.LogError(error)
                Error error
                
        with ex ->
            logger.LogError(ex, "Exception during CUDA batch search")
            Error ex.Message
    
    /// Get current vector count
    member this.VectorCount = currentCount
    
    /// Get maximum vector capacity
    member this.MaxVectors = maxVectors
    
    /// Get vector dimension
    member this.VectorDimension = vectorDim
    
    /// Check if CUDA is available
    static member IsCudaAvailable() =
        try
            let testStore = CudaNative.tars_optimized_create_store(100, 128, 0)
            if testStore <> nativeint.Zero then
                CudaNative.tars_optimized_destroy_store(testStore)
                true
            else
                false
        with
        | _ -> false
    
    /// Dispose of CUDA resources
    member this.Dispose() =
        if not isDisposed && storeHandle <> nativeint.Zero then
            CudaNative.tars_optimized_destroy_store(storeHandle)
            storeHandle <- nativeint.Zero
            isDisposed <- true
            logger.LogInformation("CUDA Vector Store disposed")
    
    interface IDisposable with
        member this.Dispose() = this.Dispose()

/// CUDA Vector Store Factory
type CudaVectorStoreFactory() =
    
    /// Create a new CUDA vector store
    static member CreateStore(maxVectors: int, vectorDim: int, gpuId: int, logger: ILogger<RealCudaVectorStore>) =
        try
            if RealCudaVectorStore.IsCudaAvailable() then
                let store = new RealCudaVectorStore(maxVectors, vectorDim, gpuId, logger)
                Ok store
            else
                Error "CUDA is not available on this system"
        with ex ->
            Error $"Failed to create CUDA vector store: {ex.Message}"
    
    /// Get optimal parameters for the current GPU
    static member GetOptimalParameters() =
        {|
            MaxVectors = 100000
            VectorDim = 384
            GpuId = 0
            BatchSize = 100
        |}
