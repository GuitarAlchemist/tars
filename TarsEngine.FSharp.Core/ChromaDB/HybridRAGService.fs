namespace TarsEngine.FSharp.Core.ChromaDB

open System
open System.Collections.Concurrent
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Hybrid RAG service combining in-memory and ChromaDB storage
type HybridRAGService(chromaClient: IChromaDBClient, logger: ILogger<HybridRAGService>) =
    let inMemoryCache = ConcurrentDictionary<string, ChromaDocument>()
    let collectionName = "tars_knowledge"
    
    // Initialize collection on startup
    let mutable collectionInitialized = false
    
    let ensureCollectionAsync() =
        task {
            if not collectionInitialized then
                logger.LogInformation("Initializing ChromaDB collection: {CollectionName}", collectionName)
                let! collection = chromaClient.CreateCollectionAsync(collectionName)
                collectionInitialized <- true
                logger.LogInformation("Collection initialized: {CollectionName}", collectionName)
        }
    
    interface IHybridRAGService with
        member _.StoreKnowledgeAsync(content: string, metadata: Map<string, obj>) =
            task {
                try
                    do! ensureCollectionAsync()
                    
                    let docId = Guid.NewGuid().ToString()
                    let document = {
                        Id = docId
                        Content = content
                        Metadata = metadata |> Map.add "stored_at" (DateTime.UtcNow :> obj)
                        Embedding = None // TODO: Generate embeddings
                    }
                    
                    // Store in memory cache for fast access
                    inMemoryCache.[docId] <- document
                    logger.LogInformation("Stored document in memory cache: {DocumentId}", docId)
                    
                    // Store in ChromaDB for persistence
                    do! chromaClient.AddDocumentsAsync(collectionName, [document])
                    logger.LogInformation("Stored document in ChromaDB: {DocumentId}", docId)
                    
                    return docId
                with
                | ex ->
                    logger.LogError(ex, "Failed to store knowledge")
                    reraise()
            }
        
        member _.RetrieveKnowledgeAsync(query: string, limit: int) =
            task {
                try
                    do! ensureCollectionAsync()
                    
                    logger.LogInformation("Retrieving knowledge for query: {Query}", query)
                    
                    // First, check in-memory cache for exact matches
                    let memoryResults = 
                        inMemoryCache.Values
                        |> Seq.filter (fun doc -> doc.Content.Contains(query, StringComparison.OrdinalIgnoreCase))
                        |> Seq.take (min limit 5) // Limit memory results
                        |> Seq.toList
                    
                    logger.LogInformation("Found {MemoryResults} results in memory cache", memoryResults.Length)
                    
                    // Then, query ChromaDB for semantic search
                    let! chromaResults = chromaClient.QueryAsync(collectionName, query, limit - memoryResults.Length)
                    
                    // Combine results (memory first for speed)
                    let combinedResults = memoryResults @ chromaResults.Documents
                    
                    logger.LogInformation("Retrieved {TotalResults} total results ({MemoryResults} from cache, {ChromaResults} from ChromaDB)", 
                                        combinedResults.Length, memoryResults.Length, chromaResults.Documents.Length)
                    
                    return combinedResults |> List.take (min limit combinedResults.Length)
                with
                | ex ->
                    logger.LogError(ex, "Failed to retrieve knowledge for query: {Query}", query)
                    return []
            }
        
        member _.SearchSimilarAsync(content: string, limit: int) =
            task {
                try
                    do! ensureCollectionAsync()
                    
                    logger.LogInformation("Searching for similar content (length: {ContentLength})", content.Length)
                    
                    // Use ChromaDB for semantic similarity search
                    let! results = chromaClient.QueryAsync(collectionName, content, limit)
                    
                    logger.LogInformation("Found {SimilarResults} similar documents", results.Documents.Length)
                    return results.Documents
                with
                | ex ->
                    logger.LogError(ex, "Failed to search similar content")
                    return []
            }
        
        member _.GetMemoryStatsAsync() =
            task {
                try
                    let stats = Map.ofList [
                        ("in_memory_documents", inMemoryCache.Count :> obj)
                        ("collection_name", collectionName :> obj)
                        ("collection_initialized", collectionInitialized :> obj)
                        ("last_updated", DateTime.UtcNow :> obj)
                    ]
                    
                    logger.LogInformation("Memory stats: {InMemoryDocs} documents in cache", inMemoryCache.Count)
                    return stats
                with
                | ex ->
                    logger.LogError(ex, "Failed to get memory stats")
                    return Map.empty
            }

