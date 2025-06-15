namespace Tars.Engine.VectorStore

open System
open System.IO
open System.Text.Json
open System.Collections.Concurrent

/// In-memory vector store implementation with optional persistence
type InMemoryVectorStore(config: VectorStoreConfig, similarityComputer: ISimilarityComputer) =
    
    let documents = ConcurrentDictionary<string, VectorDocument>()
    let indexPath = config.StoragePath |> Option.defaultValue ".tars/vector_store"
    
    /// Ensure storage directory exists
    let ensureStorageDirectory () =
        if config.PersistToDisk then
            match config.StoragePath with
            | Some path -> 
                if not (Directory.Exists(path)) then
                    Directory.CreateDirectory(path) |> ignore
            | None -> ()
    
    /// Serialize document to JSON
    let serializeDocument (doc: VectorDocument) : string =
        JsonSerializer.Serialize(doc, JsonSerializerOptions(WriteIndented = true))
    
    /// Deserialize document from JSON
    let deserializeDocument (json: string) : VectorDocument option =
        try
            Some (JsonSerializer.Deserialize<VectorDocument>(json))
        with
        | _ -> None
    
    /// Save document to disk
    let saveDocumentToDisk (doc: VectorDocument) =
        if config.PersistToDisk then
            ensureStorageDirectory()
            let filePath = Path.Combine(indexPath, sprintf "%s.json" doc.Id)
            let json = serializeDocument doc
            File.WriteAllText(filePath, json)
    
    /// Load document from disk
    let loadDocumentFromDisk (id: string) : VectorDocument option =
        if config.PersistToDisk then
            let filePath = Path.Combine(indexPath, sprintf "%s.json" id)
            if File.Exists(filePath) then
                let json = File.ReadAllText(filePath)
                deserializeDocument json
            else
                None
        else
            None
    
    /// Load all documents from disk
    let loadAllDocumentsFromDisk () =
        if config.PersistToDisk && Directory.Exists(indexPath) then
            Directory.GetFiles(indexPath, "*.json")
            |> Array.choose (fun filePath ->
                try
                    let json = File.ReadAllText(filePath)
                    deserializeDocument json
                with
                | _ -> None)
            |> Array.iter (fun doc -> documents.TryAdd(doc.Id, doc) |> ignore)
    
    /// Delete document from disk
    let deleteDocumentFromDisk (id: string) =
        if config.PersistToDisk then
            let filePath = Path.Combine(indexPath, sprintf "%s.json" id)
            if File.Exists(filePath) then
                File.Delete(filePath)
    
    /// Initialize store by loading existing documents
    do loadAllDocumentsFromDisk()
    
    interface IVectorStore with
        
        member _.AddDocument (doc: VectorDocument) : Async<unit> =
            async {
                documents.AddOrUpdate(doc.Id, doc, fun _ _ -> doc) |> ignore
                saveDocumentToDisk doc
            }
        
        member _.AddDocuments (docs: VectorDocument list) : Async<unit> =
            async {
                for doc in docs do
                    documents.AddOrUpdate(doc.Id, doc, fun _ _ -> doc) |> ignore
                    saveDocumentToDisk doc
            }
        
        member _.Search (query: VectorQuery) : Async<SearchResult list> =
            async {
                let results = ResizeArray<SearchResult>()
                
                for kvp in documents do
                    let doc = kvp.Value
                    
                    // Apply filters
                    let passesFilters =
                        query.Filters
                        |> Map.forall (fun key value ->
                            doc.Embedding.Metadata.TryFind(key)
                            |> Option.map (fun v -> v.Contains(value, StringComparison.OrdinalIgnoreCase))
                            |> Option.defaultValue false
                            ||
                            doc.Tags |> List.exists (fun tag -> tag.Contains(value, StringComparison.OrdinalIgnoreCase)))
                    
                    if passesFilters then
                        let scores = similarityComputer.ComputeSimilarity query.Embedding doc.Embedding
                        let finalScore = similarityComputer.AggregateSimilarity scores
                        
                        if finalScore >= query.MinScore then
                            results.Add({
                                Document = doc
                                Scores = scores
                                FinalScore = finalScore
                                Rank = 0  // Will be set after sorting
                            })
                
                // Sort by score and assign ranks
                let sortedResults = 
                    results 
                    |> Seq.sortByDescending (fun r -> r.FinalScore)
                    |> Seq.take (min query.MaxResults results.Count)
                    |> Seq.mapi (fun i r -> { r with Rank = i + 1 })
                    |> Seq.toList
                
                return sortedResults
            }
        
        member _.GetDocument (id: string) : Async<VectorDocument option> =
            async {
                match documents.TryGetValue(id) with
                | true, doc -> return Some doc
                | false, _ -> 
                    // Try loading from disk
                    match loadDocumentFromDisk id with
                    | Some doc -> 
                        documents.TryAdd(id, doc) |> ignore
                        return Some doc
                    | None -> return None
            }
        
        member _.UpdateDocument (doc: VectorDocument) : Async<unit> =
            async {
                documents.AddOrUpdate(doc.Id, doc, fun _ _ -> doc) |> ignore
                saveDocumentToDisk doc
            }
        
        member _.DeleteDocument (id: string) : Async<unit> =
            async {
                documents.TryRemove(id) |> ignore
                deleteDocumentFromDisk id
            }
        
        member _.GetDocumentCount () : Async<int> =
            async {
                return documents.Count
            }
        
        member _.Clear () : Async<unit> =
            async {
                documents.Clear()
                if config.PersistToDisk && Directory.Exists(indexPath) then
                    Directory.GetFiles(indexPath, "*.json")
                    |> Array.iter File.Delete
            }

/// Vector store factory
module VectorStoreFactory =
    
    /// Create an in-memory vector store
    let createInMemory (config: VectorStoreConfig) : IVectorStore =
        let similarityComputer = MultiSpaceSimilarityComputer(config) :> ISimilarityComputer
        InMemoryVectorStore(config, similarityComputer) :> IVectorStore
    
    /// Create vector store with custom similarity computer
    let createWithSimilarityComputer (config: VectorStoreConfig) (similarityComputer: ISimilarityComputer) : IVectorStore =
        InMemoryVectorStore(config, similarityComputer) :> IVectorStore

/// Vector store utilities
module VectorStoreUtils =
    
    /// Create a document from text content
    let createDocument (id: string) (content: string) (embedding: MultiSpaceEmbedding) (tags: string list) (source: string option) : VectorDocument =
        {
            Id = id
            Content = content
            Embedding = embedding
            Tags = tags
            Timestamp = DateTime.Now
            Source = source
        }
    
    /// Create a query from text
    let createQuery (text: string) (embedding: MultiSpaceEmbedding) (maxResults: int) (minScore: float) (filters: Map<string, string>) : VectorQuery =
        {
            Text = text
            Embedding = embedding
            Filters = filters
            MaxResults = maxResults
            MinScore = minScore
        }
    
    /// Get statistics for a vector store
    let getStatistics (store: IVectorStore) : Async<VectorStoreStats> =
        async {
            let! count = store.GetDocumentCount()
            return {
                DocumentCount = count
                AverageEmbeddingSize = 768.0  // Placeholder
                SpaceUsageStats = Map.ofList [
                    ("raw", count)
                    ("fft", count)
                    ("dual", count)
                ]
                LastUpdated = DateTime.Now
                IndexSize = int64 count * 1024L  // Rough estimate
            }
        }
    
    /// Batch add documents with progress reporting
    let batchAddDocuments (store: IVectorStore) (documents: VectorDocument list) (progressCallback: int -> int -> unit) : Async<unit> =
        async {
            let total = documents.Length
            let batchSize = 100
            
            for i in 0..batchSize..total-1 do
                let batch = documents |> List.skip i |> List.take (min batchSize (total - i))
                do! store.AddDocuments batch
                progressCallback (i + batch.Length) total
        }
    
    /// Search with automatic query expansion
    let searchWithExpansion (store: IVectorStore) (query: VectorQuery) (expansionTerms: string list) : Async<SearchResult list> =
        async {
            // First, search with original query
            let! originalResults = store.Search query
            
            // If we don't have enough results, try expanded queries
            if originalResults.Length < query.MaxResults && not (List.isEmpty expansionTerms) then
                let expandedQueries = 
                    expansionTerms 
                    |> List.map (fun term -> 
                        { query with 
                            Text = sprintf "%s %s" query.Text term
                            MaxResults = query.MaxResults - originalResults.Length })
                
                let! expandedResults = 
                    expandedQueries 
                    |> List.map store.Search
                    |> Async.Parallel
                
                let allResults = 
                    originalResults @ (expandedResults |> Array.toList |> List.concat)
                    |> List.distinctBy (fun r -> r.Document.Id)
                    |> List.sortByDescending (fun r -> r.FinalScore)
                    |> List.take query.MaxResults
                    |> List.mapi (fun i r -> { r with Rank = i + 1 })
                
                return allResults
            else
                return originalResults
        }
