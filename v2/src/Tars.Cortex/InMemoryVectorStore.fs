namespace Tars.Cortex

open System
open System.Collections.Concurrent
open System.IO
open System.Numerics
open System.Security.Cryptography
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Tars.Core

/// Represents a stored vector entry with metadata
type VectorEntry = {
    Id: string
    Vector: float32[]
    Payload: Map<string, string>
    Checksum: string
    Version: int
    CreatedAt: DateTime
}

/// Similarity computation utilities with SIMD optimization
module Similarity =
    /// SIMD vector width for float32 operations
    let private simdWidth = Vector<float32>.Count
    
    /// Compute cosine similarity between two vectors using SIMD acceleration
    let cosineSimilarity (a: float32[]) (b: float32[]) =
        if a.Length <> b.Length then 0.0f
        elif a.Length = 0 then 0.0f
        else
            let mutable dotProduct = 0.0f
            let mutable magnitudeA = 0.0f
            let mutable magnitudeB = 0.0f
            
            let length = a.Length
            let simdLength = length - (length % simdWidth)
            
            // SIMD-accelerated loop for aligned portion
            if simdLength > 0 then
                let mutable vDot = Vector<float32>.Zero
                let mutable vMagA = Vector<float32>.Zero
                let mutable vMagB = Vector<float32>.Zero
                
                let mutable i = 0
                while i < simdLength do
                    let va = Vector<float32>(a, i)
                    let vb = Vector<float32>(b, i)
                    vDot <- vDot + va * vb
                    vMagA <- vMagA + va * va
                    vMagB <- vMagB + vb * vb
                    i <- i + simdWidth
                
                // Sum up SIMD vector lanes
                for j in 0 .. simdWidth - 1 do
                    dotProduct <- dotProduct + vDot[j]
                    magnitudeA <- magnitudeA + vMagA[j]
                    magnitudeB <- magnitudeB + vMagB[j]
            
            // Handle remaining elements (scalar fallback)
            for i in simdLength .. length - 1 do
                dotProduct <- dotProduct + a[i] * b[i]
                magnitudeA <- magnitudeA + a[i] * a[i]
                magnitudeB <- magnitudeB + b[i] * b[i]
            
            let magA = sqrt magnitudeA
            let magB = sqrt magnitudeB
            
            if magA = 0.0f || magB = 0.0f then 0.0f
            else dotProduct / (magA * magB)
    
    /// Convert cosine similarity to distance (for consistent API with ChromaDB)
    let similarityToDistance (similarity: float32) =
        1.0f - similarity

/// Thread-safe in-memory vector store implementation
/// Ported and simplified from TARS v1 with improvements
type InMemoryVectorStore() =
    let collections = ConcurrentDictionary<string, ConcurrentDictionary<string, VectorEntry>>()
    
    /// Get or create a collection
    let getOrCreateCollection (name: string) =
        collections.GetOrAdd(name, fun _ -> ConcurrentDictionary<string, VectorEntry>())
    
    /// Compute checksum for a vector entry
    let computeChecksum (vector: float32[]) (payload: Map<string, string>) =
        use sha256 = SHA256.Create()
        let vectorBytes = vector |> Array.collect BitConverter.GetBytes
        let payloadJson = JsonSerializer.Serialize(payload)
        let payloadBytes = Encoding.UTF8.GetBytes(payloadJson)
        let combined = Array.append vectorBytes payloadBytes
        let hash = sha256.ComputeHash(combined)
        Convert.ToHexString(hash).ToLowerInvariant()
    
    interface IVectorStore with
        member this.SaveAsync(collectionName: string, id: string, vector: float32[], payload: Map<string, string>) =
            task {
                let collection = getOrCreateCollection collectionName
                let checksum = computeChecksum vector payload
                
                let entry = {
                    Id = id
                    Vector = vector
                    Payload = payload
                    Checksum = checksum
                    Version = 1
                    CreatedAt = DateTime.UtcNow
                }
                
                // Update version if entry already exists
                let entry =
                    match collection.TryGetValue(id) with
                    | true, existing -> { entry with Version = existing.Version + 1 }
                    | false, _ -> entry
                
                collection[id] <- entry
            } :> Task
        
        member this.SearchAsync(collectionName: string, queryVector: float32[], limit: int) =
            task {
                let collection = getOrCreateCollection collectionName
                
                if collection.IsEmpty then
                    return []
                else
                    let results =
                        collection.Values
                        |> Seq.map (fun entry ->
                            let similarity = Similarity.cosineSimilarity queryVector entry.Vector
                            let distance = Similarity.similarityToDistance similarity
                            (entry.Id, distance, entry.Payload))
                        |> Seq.sortBy (fun (_, distance, _) -> distance)  // Lower distance = more similar
                        |> Seq.truncate limit
                        |> Seq.toList
                    
                    return results
            }
    
    // Additional utility methods beyond IVectorStore interface
    
    /// Get the count of entries in a collection
    member this.GetCountAsync(collectionName: string) =
        task {
            let collection = getOrCreateCollection collectionName
            return collection.Count
        }
    
    /// Delete an entry from a collection
    member this.DeleteAsync(collectionName: string, id: string) =
        task {
            let collection = getOrCreateCollection collectionName
            let mutable removed = Unchecked.defaultof<VectorEntry>
            return collection.TryRemove(id, &removed)
        }
    
    /// Delete an entire collection
    member this.DeleteCollectionAsync(collectionName: string) =
        task {
            let mutable removed = Unchecked.defaultof<ConcurrentDictionary<string, VectorEntry>>
            return collections.TryRemove(collectionName, &removed)
        }
    
    /// Get all collection names
    member this.GetCollectionsAsync() =
        task {
            return collections.Keys |> Seq.toList
        }
    
    /// Get an entry by ID
    member this.GetByIdAsync(collectionName: string, id: string) =
        task {
            let collection = getOrCreateCollection collectionName
            match collection.TryGetValue(id) with
            | true, entry -> return Some entry
            | false, _ -> return None
        }
    
    /// Validate checksum of an entry
    member this.ValidateChecksumAsync(collectionName: string, id: string) =
        task {
            let collection = getOrCreateCollection collectionName
            match collection.TryGetValue(id) with
            | true, entry ->
                let expectedChecksum = computeChecksum entry.Vector entry.Payload
                return entry.Checksum = expectedChecksum
            | false, _ -> return false
        }
    
    /// Clear all data from the store
    member this.ClearAsync() =
        task {
            collections.Clear()
        }
    
    // File persistence methods
    
    /// Persist all collections to a JSON file
    member this.PersistToFileAsync(path: string) =
        task {
            let data =
                collections
                |> Seq.map (fun kvp ->
                    kvp.Key, kvp.Value.Values |> Seq.toArray)
                |> Map.ofSeq
            
            let options = JsonSerializerOptions(WriteIndented = true)
            let json = JsonSerializer.Serialize(data, options)
            do! File.WriteAllTextAsync(path, json)
        }
    
    /// Load collections from a JSON file
    member this.LoadFromFileAsync(path: string) =
        task {
            if File.Exists(path) then
                let! json = File.ReadAllTextAsync(path)
                let data = JsonSerializer.Deserialize<Map<string, VectorEntry[]>>(json)
                
                collections.Clear()
                
                for kvp in data do
                    let collection = ConcurrentDictionary<string, VectorEntry>()
                    for entry in kvp.Value do
                        collection[entry.Id] <- entry
                    collections[kvp.Key] <- collection
                
                return true
            else
                return false
        }
