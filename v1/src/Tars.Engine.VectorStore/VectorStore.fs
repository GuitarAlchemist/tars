namespace Tars.Engine.VectorStore

open System
open System.IO
open System.Text.Json
open System.Text.Json.Serialization
open System.Collections.Concurrent

type TruthValueConverter() =
    inherit JsonConverter<TruthValue>()

    override _.Read(reader: byref<Utf8JsonReader>, _typeToConvert, _options) =
        if reader.TokenType <> JsonTokenType.String then
            raise (JsonException("TruthValue must be encoded as a string."))

        match reader.GetString() with
        | null -> raise (JsonException("TruthValue string cannot be null."))
        | value when value.Equals("True", StringComparison.OrdinalIgnoreCase) -> TruthValue.True
        | value when value.Equals("False", StringComparison.OrdinalIgnoreCase) -> TruthValue.False
        | value when value.Equals("Both", StringComparison.OrdinalIgnoreCase) -> TruthValue.Both
        | value when value.Equals("Neither", StringComparison.OrdinalIgnoreCase) -> TruthValue.Neither
        | value -> raise (JsonException($"Unsupported TruthValue '{value}'."))

    override _.Write(writer: Utf8JsonWriter, value: TruthValue, _options) =
        let text =
            match value with
            | TruthValue.True -> "True"
            | TruthValue.False -> "False"
            | TruthValue.Both -> "Both"
            | TruthValue.Neither -> "Neither"

        writer.WriteStringValue(text)

/// In-memory vector store implementation with optional persistence
type InMemoryVectorStore(config: VectorStoreConfig, similarityComputer: ISimilarityComputer) =

    let documents = ConcurrentDictionary<string, VectorDocument>()
    let indexPath = config.StoragePath |> Option.defaultValue ".tars/vector_store"

    let ensureStorageDirectory () =
        if config.PersistToDisk then
            match config.StoragePath with
            | Some path when not (Directory.Exists(path)) ->
                Directory.CreateDirectory(path) |> ignore
            | _ -> ()

    let jsonOptions =
        let options = JsonSerializerOptions(WriteIndented = true)
        options.Converters.Add(TruthValueConverter())
        options

    let serializeDocument (doc: VectorDocument) : string =
        JsonSerializer.Serialize(doc, jsonOptions)

    let deserializeDocument (json: string) : VectorDocument option =
        try
            JsonSerializer.Deserialize<VectorDocument>(json, jsonOptions) |> Option.ofObj
        with
        | _ -> None

    let saveDocumentToDisk (doc: VectorDocument) =
        if config.PersistToDisk then
            ensureStorageDirectory()
            let filePath = Path.Combine(indexPath, $"{doc.Id}.json")
            File.WriteAllText(filePath, serializeDocument doc)

    let loadDocumentFromDisk (id: string) : VectorDocument option =
        if config.PersistToDisk then
            let filePath = Path.Combine(indexPath, $"{id}.json")
            if File.Exists(filePath) then
                File.ReadAllText(filePath) |> deserializeDocument
            else
                None
        else
            None

    let loadAllDocumentsFromDisk () =
        if config.PersistToDisk && Directory.Exists(indexPath) then
            Directory.GetFiles(indexPath, "*.json")
            |> Array.iter (fun filePath ->
                try
                    let json = File.ReadAllText(filePath)
                    match deserializeDocument json with
                    | Some doc -> documents.TryAdd(doc.Id, doc) |> ignore
                    | None -> ()
                with
                | _ -> ())

    let deleteDocumentFromDisk (id: string) =
        if config.PersistToDisk then
            let filePath = Path.Combine(indexPath, $"{id}.json")
            if File.Exists(filePath) then
                File.Delete(filePath)

    do loadAllDocumentsFromDisk()

    interface IVectorStore with

        member _.AddDocument doc =
            async {
                documents.AddOrUpdate(doc.Id, doc, fun _ _ -> doc) |> ignore
                saveDocumentToDisk doc
            }

        member this.AddDocuments docs =
            async {
                for doc in docs do
                    documents.AddOrUpdate(doc.Id, doc, fun _ _ -> doc) |> ignore
                    saveDocumentToDisk doc
            }

        member _.Search query =
            async {
                let results = ResizeArray<SearchResult>()

                for KeyValue(_, doc) in documents do
                    let passesFilters =
                        query.Filters
                        |> Map.forall (fun key value ->
                            let metadataMatches =
                                doc.Embedding.Metadata.TryFind(key)
                                |> Option.exists (fun v -> v.IndexOf(value, StringComparison.OrdinalIgnoreCase) >= 0)

                            let tagMatches =
                                doc.Tags
                                |> List.exists (fun tag -> tag.IndexOf(value, StringComparison.OrdinalIgnoreCase) >= 0)

                            metadataMatches || tagMatches)

                    if passesFilters then
                        let scores = similarityComputer.ComputeSimilarity query.Embedding doc.Embedding
                        let finalScore = similarityComputer.AggregateSimilarity scores

                        if finalScore >= query.MinScore then
                            results.Add({
                                Document = doc
                                Scores = scores
                                FinalScore = finalScore
                                Rank = 0
                            })

                let sorted =
                    results
                    |> Seq.sortByDescending (fun r -> r.FinalScore)
                    |> Seq.truncate query.MaxResults
                    |> Seq.mapi (fun idx r -> { r with Rank = idx + 1 })
                    |> Seq.toList

                return sorted
            }

        member _.GetDocument id =
            async {
                match documents.TryGetValue(id) with
                | true, doc -> return Some doc
                | false, _ ->
                    match loadDocumentFromDisk id with
                    | Some doc ->
                        documents.TryAdd(doc.Id, doc) |> ignore
                        return Some doc
                    | None -> return None
            }

        member _.GetAllDocuments () =
            async {
                return documents.Values |> Seq.toList
            }

        member _.UpdateDocument doc =
            async {
                documents.AddOrUpdate(doc.Id, doc, fun _ _ -> doc) |> ignore
                saveDocumentToDisk doc
            }

        member _.DeleteDocument id =
            async {
                documents.TryRemove(id) |> ignore
                deleteDocumentFromDisk id
            }

        member _.GetDocumentCount () =
            async { return documents.Count }

        member _.Clear () =
            async {
                documents.Clear()
                if config.PersistToDisk && Directory.Exists(indexPath) then
                    Directory.GetFiles(indexPath, "*.json")
                    |> Array.iter File.Delete
            }

module VectorStoreFactory =

    let createInMemory (config: VectorStoreConfig) : IVectorStore =
        let similarityComputer = MultiSpaceSimilarityComputer(config) :> ISimilarityComputer
        InMemoryVectorStore(config, similarityComputer) :> IVectorStore

    let createWithSimilarityComputer (config: VectorStoreConfig) (similarityComputer: ISimilarityComputer) : IVectorStore =
        InMemoryVectorStore(config, similarityComputer) :> IVectorStore

module VectorStoreUtils =

    let createDocument (id: string) (content: string) (embedding: MultiSpaceEmbedding) (tags: string list) (source: string option) : VectorDocument =
        {
            Id = id
            Content = content
            Embedding = embedding
            Tags = tags
            Timestamp = DateTime.UtcNow
            Source = source
        }

    let createQuery (text: string) (embedding: MultiSpaceEmbedding) (maxResults: int) (minScore: float) (filters: Map<string, string>) : VectorQuery =
        {
            Text = text
            Embedding = embedding
            Filters = filters
            MaxResults = maxResults
            MinScore = minScore
        }

    let private averageOrZero values =
        if List.isEmpty values then 0.0 else values |> List.averageBy float

    let private sumLengths (getLength: MultiSpaceEmbedding -> int) (documents: VectorDocument list) =
        documents |> List.sumBy (fun doc -> getLength doc.Embedding)

    let private estimateEmbeddingBytes (embedding: MultiSpaceEmbedding) =
        let floatBytes = 8L
        let complexBytes = 16L

        let floatSpaces =
            [ embedding.Raw.Length
              embedding.Dual.Length
              embedding.Projective.Length
              embedding.Hyperbolic.Length
              embedding.Wavelet.Length
              embedding.Minkowski.Length ]

        let floatsTotalBytes =
            floatSpaces |> List.sum |> int64 |> fun total -> total * floatBytes

        let fftBytes = int64 embedding.FFT.Length * complexBytes
        let pauliBytes = 4L * complexBytes

        floatsTotalBytes + fftBytes + pauliBytes

    let getStatistics (store: IVectorStore) : Async<VectorStoreStats> =
        async {
            let! documents = store.GetAllDocuments()
            let documentCount = documents.Length

            let averageEmbeddingSize =
                documents
                |> List.map (fun doc -> doc.Embedding.Raw.Length)
                |> averageOrZero

            let spaceUsage =
                [
                    "raw", sumLengths (fun e -> e.Raw.Length) documents
                    "fft", documents |> List.sumBy (fun doc -> doc.Embedding.FFT.Length)
                    "dual", sumLengths (fun e -> e.Dual.Length) documents
                    "projective", sumLengths (fun e -> e.Projective.Length) documents
                    "hyperbolic", sumLengths (fun e -> e.Hyperbolic.Length) documents
                    "wavelet", sumLengths (fun e -> e.Wavelet.Length) documents
                    "minkowski", sumLengths (fun e -> e.Minkowski.Length) documents
                    "pauli", documents |> List.length |> fun count -> count * 4
                ]
                |> Map.ofList

            let indexSize =
                documents
                |> List.sumBy (fun doc -> estimateEmbeddingBytes doc.Embedding)

            return {
                DocumentCount = documentCount
                AverageEmbeddingSize = averageEmbeddingSize
                SpaceUsageStats = spaceUsage
                LastUpdated = DateTime.UtcNow
                IndexSize = indexSize
            }
        }

    let batchAddDocuments (store: IVectorStore) (documents: VectorDocument list) (progressCallback: int -> int -> unit) : Async<unit> =
        async {
            let total = documents.Length
            let batchSize = 100

            let rec loop offset =
                async {
                    if offset < total then
                        let batchLength = min batchSize (total - offset)
                        let batch = documents |> List.skip offset |> List.take batchLength
                        do! store.AddDocuments batch
                        progressCallback (offset + batchLength) total
                        return! loop (offset + batchLength)
                }

            do! loop 0
        }

    let searchWithExpansion (store: IVectorStore) (query: VectorQuery) (expansionTerms: string list) : Async<SearchResult list> =
        async {
            let! originalResults = store.Search query

            if originalResults.Length < query.MaxResults && not (List.isEmpty expansionTerms) then
                let remaining = query.MaxResults - originalResults.Length

                let! expandedResults =
                    expansionTerms
                    |> List.map (fun term ->
                        store.Search {
                            query with
                                Text = $"{query.Text} {term}"
                                MaxResults = remaining
                        })
                    |> Async.Parallel

                let combined =
                    originalResults
                    @ (expandedResults |> Array.toList |> List.collect id)
                    |> List.distinctBy (fun r -> r.Document.Id)
                    |> List.sortByDescending (fun r -> r.FinalScore)
                    |> List.truncate query.MaxResults
                    |> List.mapi (fun idx r -> { r with Rank = idx + 1 })

                return combined
            else
                return originalResults
        }
