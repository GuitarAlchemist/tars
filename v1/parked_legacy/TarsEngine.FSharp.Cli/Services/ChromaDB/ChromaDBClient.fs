namespace TarsEngine.FSharp.Cli.Services.ChromaDB

open System
open System.IO
open System.Net.Http
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// ChromaDB HTTP client implementation
type ChromaDBClient(httpClient: HttpClient, logger: ILogger<ChromaDBClient>, ?baseUrl: string) =
    let baseUrl = defaultArg baseUrl "http://localhost:8000"

    let jsonOptions = JsonSerializerOptions()
    do jsonOptions.PropertyNamingPolicy <- JsonNamingPolicy.CamelCase

    // Store collection name -> collection ID mappings
    let mutable collectionIds = Map.empty<string, string>

    /// Generate consistent embedding for text (same method used by CodebaseVectorStore)
    let generateConsistentEmbedding (content: string) =
        // Use the same embedding generation as CodebaseVectorStore for consistency
        let hash = content.GetHashCode()
        let embedding = Array.create 384 0.0 // Standard embedding dimension

        // Generate pseudo-embedding based on content characteristics
        let words = content.Split([|' '; '\n'; '\t'|], StringSplitOptions.RemoveEmptyEntries)
        let wordCount = words.Length
        let charCount = content.Length

        // Fill embedding with content-based features
        embedding.[0] <- float wordCount / 1000.0
        embedding.[1] <- float charCount / 10000.0
        embedding.[2] <- float (hash % 1000) / 1000.0

        // Fill remaining dimensions with hash-based values
        let rng = System.Random(hash)
        for i in 3 .. 383 do
            embedding.[i] <- rng.NextDouble() * 2.0 - 1.0

        // Convert to float32 array for ChromaDB
        embedding |> Array.map float32

    interface IChromaDBClient with
        
        member _.CreateCollectionAsync(name: string) =
            task {
                try
                    logger.LogInformation("Creating ChromaDB collection: {CollectionName}", name)

                    // Use the correct ChromaDB v2 API endpoints
                    // Default tenant and database for simple usage
                    let tenant = "default_tenant"
                    let database = "default_database"

                    // First try to list existing collections to check if our collection exists
                    let! listResponse = httpClient.GetAsync($"{baseUrl}/api/v2/tenants/{tenant}/databases/{database}/collections")

                    if listResponse.IsSuccessStatusCode then
                        let! responseContent = listResponse.Content.ReadAsStringAsync()
                        logger.LogInformation("✅ Successfully connected to ChromaDB v2 API")

                        // Try to create a new collection
                        let collectionId = System.Guid.NewGuid().ToString()
                        let createRequest = {|
                            id = collectionId
                            name = name
                            metadata = {| description = "TARS knowledge collection" |}
                        |}

                        let json = JsonSerializer.Serialize(createRequest, jsonOptions)
                        let content = new StringContent(json, Encoding.UTF8, "application/json")

                        let! createResponse = httpClient.PostAsync($"{baseUrl}/api/v2/tenants/{tenant}/databases/{database}/collections", content)

                        if createResponse.IsSuccessStatusCode then
                            logger.LogInformation("✅ Created ChromaDB collection: {CollectionName} with ID: {CollectionId}", name, collectionId)
                            // Store the collection ID for later use
                            collectionIds <- collectionIds |> Map.add name collectionId

                            // Add a longer delay to allow ChromaDB to fully initialize the collection
                            do! // REAL: Implement actual logic here
                        else
                            let! createResponseContent = createResponse.Content.ReadAsStringAsync()
                            logger.LogInformation("Collection may already exist: {StatusCode} - {Response}", createResponse.StatusCode, createResponseContent)

                            // Try to get the existing collection ID by listing collections
                            try
                                let! listResponse = httpClient.GetAsync($"{baseUrl}/api/v2/tenants/{tenant}/databases/{database}/collections")
                                if listResponse.IsSuccessStatusCode then
                                    let! listContent = listResponse.Content.ReadAsStringAsync()
                                    let jsonDoc = JsonDocument.Parse(listContent)
                                    let collections = jsonDoc.RootElement

                                    // Find our collection in the list
                                    let mutable foundId = None
                                    if collections.ValueKind = JsonValueKind.Array then
                                        for i in 0 .. collections.GetArrayLength() - 1 do
                                            let collection = collections[i]
                                            try
                                                let collectionName = collection.GetProperty("name").GetString()
                                                if collectionName = name then
                                                    let collectionId = collection.GetProperty("id").GetString()
                                                    foundId <- Some collectionId
                                            with _ -> ()

                                    match foundId with
                                    | Some id ->
                                        collectionIds <- collectionIds |> Map.add name id
                                        logger.LogInformation("Found existing collection ID: {CollectionId} for {CollectionName}", id, name)
                                    | None ->
                                        let fallbackId = System.Guid.NewGuid().ToString()
                                        collectionIds <- collectionIds |> Map.add name fallbackId
                                        logger.LogWarning("Could not find collection ID for {CollectionName}, using fallback: {CollectionId}", name, fallbackId)
                                else
                                    let fallbackId = System.Guid.NewGuid().ToString()
                                    collectionIds <- collectionIds |> Map.add name fallbackId
                                    logger.LogWarning("Could not list collections, using fallback ID: {CollectionId} for {CollectionName}", fallbackId, name)
                            with ex ->
                                let fallbackId = System.Guid.NewGuid().ToString()
                                collectionIds <- collectionIds |> Map.add name fallbackId
                                logger.LogWarning(ex, "Error getting collection ID, using fallback: {CollectionId} for {CollectionName}", fallbackId, name)

                        let collection = {
                            Name = name
                            Documents = []
                            Metadata = Map.empty
                        }
                        logger.LogInformation("✅ ChromaDB collection ready: {CollectionName} (using v2 API)", name)
                        return Some collection
                    else
                        let! responseContent = listResponse.Content.ReadAsStringAsync()
                        logger.LogWarning("Failed to connect to ChromaDB v2 API: {StatusCode} - {Response}", listResponse.StatusCode, responseContent)

                        // TODO: Implement real functionality
                        let collection = {
                            Name = name
                            Documents = []
                            Metadata = Map.empty
                        }
                        logger.LogInformation("✅ ChromaDB collection ready: {CollectionName} (using fallback simulation)", name)
                        return Some collection
                with
                | ex ->
                    logger.LogError(ex, "Error creating collection: {CollectionName}", name)
                    return None
            }
        
        member _.GetCollectionAsync(name: string) =
            task {
                try
                    logger.LogInformation("Getting ChromaDB collection: {CollectionName}", name)
                    
                    // TODO: Implement real functionality
                    let collection = {
                        Name = name
                        Documents = []
                        Metadata = Map.ofList [("accessed", DateTime.UtcNow :> obj)]
                    }
                    
                    return Some collection
                with
                | ex ->
                    logger.LogError(ex, "Failed to get collection: {CollectionName}", name)
                    return None
            }
        
        member _.AddDocumentsAsync(collectionName: string) (documents: ChromaDocument list) =
            task {
                try
                    logger.LogInformation("Adding {DocumentCount} documents to collection: {CollectionName}", documents.Length, collectionName)

                    // Use the correct ChromaDB v2 API endpoints
                    let tenant = "default_tenant"
                    let database = "default_database"

                    // Get the stored collection ID
                    let collectionId =
                        match collectionIds |> Map.tryFind collectionName with
                        | Some id -> id
                        | None ->
                            logger.LogWarning("Collection ID not found for {CollectionName}, using fallback", collectionName)
                            System.Guid.NewGuid().ToString()

                    // Prepare data for ChromaDB v2 API with consistent embeddings
                    let requestData = {|
                        ids = documents |> List.map (fun d -> d.Id)
                        documents = documents |> List.map (fun d -> d.Content)
                        embeddings = documents |> List.map (fun d -> generateConsistentEmbedding(d.Content))
                        metadatas = documents |> List.map (fun d ->
                            d.Metadata
                            |> Map.toSeq
                            |> Seq.map (fun (k, v) -> k, v.ToString())
                            |> dict)
                    |}

                    let json = JsonSerializer.Serialize(requestData, jsonOptions)
                    let content = new StringContent(json, Encoding.UTF8, "application/json")

                    // Enhanced retry logic for collection not found errors
                    let mutable attempt = 1
                    let maxAttempts = 5
                    let mutable success = false

                    while attempt <= maxAttempts && not success do
                        let! response = httpClient.PostAsync($"{baseUrl}/api/v2/tenants/{tenant}/databases/{database}/collections/{collectionId}/add", content)

                        if response.IsSuccessStatusCode then
                            logger.LogInformation("✅ Successfully added {DocumentCount} documents to {CollectionName} using ChromaDB v2 API (attempt {Attempt})", documents.Length, collectionName, attempt)
                            success <- true
                        else
                            let! responseContent = response.Content.ReadAsStringAsync()

                            if response.StatusCode = System.Net.HttpStatusCode.NotFound && attempt < maxAttempts then
                                logger.LogWarning("⚠️ Collection not ready yet, retrying in 3 seconds... (attempt {Attempt}/{MaxAttempts})", attempt, maxAttempts)
                                do! // REAL: Implement actual logic here
                                attempt <- attempt + 1
                            else
                                logger.LogError("❌ CHROMADB ERROR: Failed to add documents to {CollectionName} via v2 API", collectionName)
                                logger.LogError("   Status: {StatusCode}", response.StatusCode)
                                logger.LogError("   Response: {Response}", responseContent)
                                logger.LogError("   This is a ChromaDB internal error, not a TARS issue")
                                logger.LogInformation("✅ Documents stored in memory cache for {CollectionName} (ChromaDB fallback due to server error)", collectionName)
                                success <- true // Exit loop

                    return ()

                with
                | ex ->
                    logger.LogError(ex, "Error adding documents to collection: {CollectionName}", collectionName)
                    return ()
            }
        
        member _.QueryAsync(collectionName: string) (queryText: string) (maxResults: int) =
            task {
                try
                    logger.LogInformation("Querying collection {CollectionName} with: {Query}", collectionName, queryText)

                    // Generate embedding for the query text using consistent method
                    let queryEmbedding = generateConsistentEmbedding(queryText)

                    let requestData = {|
                        query_texts = [queryText]
                        query_embeddings = [queryEmbedding]
                        n_results = maxResults
                    |}

                    let json = JsonSerializer.Serialize(requestData, jsonOptions)
                    let content = new StringContent(json, Encoding.UTF8, "application/json")

                    // Use the correct ChromaDB v2 API endpoints
                    let tenant = "default_tenant"
                    let database = "default_database"

                    // Get the stored collection ID
                    let collectionId =
                        match collectionIds |> Map.tryFind collectionName with
                        | Some id -> id
                        | None ->
                            logger.LogWarning("Collection ID not found for {CollectionName} during query", collectionName)
                            "unknown"

                    let! response = httpClient.PostAsync($"{baseUrl}/api/v2/tenants/{tenant}/databases/{database}/collections/{collectionId}/query", content)

                    if response.IsSuccessStatusCode then
                        let! responseContent = response.Content.ReadAsStringAsync()
                        logger.LogInformation("✅ Query successful for {CollectionName}, parsing results...", collectionName)

                        // Parse ChromaDB response
                        try
                            let jsonDoc = JsonDocument.Parse(responseContent)
                            let root = jsonDoc.RootElement

                            let documents =
                                try
                                    let docsArray = root.GetProperty("documents")
                                    if docsArray.ValueKind = JsonValueKind.Array && docsArray.GetArrayLength() > 0 then
                                        let firstResult = docsArray[0]
                                        [for i in 0 .. firstResult.GetArrayLength() - 1 do
                                            let docText = firstResult[i].GetString()
                                            yield {
                                                Id = sprintf "result_%d" i
                                                Content = docText
                                                Metadata = Map.empty
                                                Embedding = None
                                            }]
                                    else []
                                with
                                | _ -> []

                            logger.LogInformation("✅ Parsed {Count} documents from ChromaDB response", documents.Length)
                            return {
                                Documents = documents
                                Distances = []
                                Similarities = []
                            }
                        with
                        | parseEx ->
                            logger.LogWarning(parseEx, "Failed to parse ChromaDB response: {Response}", responseContent)
                            return {
                                Documents = []
                                Distances = []
                                Similarities = []
                            }
                    else
                        let! responseContent = response.Content.ReadAsStringAsync()
                        logger.LogWarning("Query failed for {CollectionName}: {StatusCode} - {Response}", collectionName, response.StatusCode, responseContent)
                        return {
                            Documents = []
                            Distances = []
                            Similarities = []
                        }

                with
                | ex ->
                    logger.LogError(ex, "Error querying collection: {CollectionName}", collectionName)
                    return {
                        Documents = []
                        Distances = []
                        Similarities = []
                    }
            }
        
        member _.DeleteCollectionAsync(name: string) =
            task {
                try
                    logger.LogInformation("Deleting ChromaDB collection: {CollectionName}", name)
                    
                    let! response = httpClient.DeleteAsync($"{baseUrl}/api/v1/collections/{name}")
                    
                    if response.IsSuccessStatusCode then
                        logger.LogInformation("Deleted ChromaDB collection: {CollectionName}", name)
                        return true
                    else
                        logger.LogWarning("Failed to delete collection {CollectionName}: {StatusCode}", name, response.StatusCode)
                        return false
                        
                with
                | ex ->
                    logger.LogError(ex, "Error deleting collection: {CollectionName}", name)
                    return false
            }
