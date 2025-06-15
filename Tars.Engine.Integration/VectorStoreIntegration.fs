namespace Tars.Engine.Integration

open System
open System.IO
open Tars.Engine.VectorStore

/// Integration module for connecting vector store with existing TARS infrastructure
module VectorStoreIntegration =
    
    /// TARS-specific document types
    type TARSDocumentType =
        | MetascriptFile
        | GrammarFile
        | TraceFile
        | ConfigFile
        | CodeFile
        | DocumentationFile
    
    /// Enhanced document with TARS-specific metadata
    type TARSVectorDocument = {
        BaseDocument: VectorDocument
        DocumentType: TARSDocumentType
        ProjectPath: string option
        AgentName: string option
        ExecutionId: string option
        Dependencies: string list
    }
    
    /// TARS vector store configuration
    let createTARSConfig () : VectorStoreConfig =
        {
            RawDimension = 1024  // Larger for TARS complexity
            EnableFFT = true
            EnableDual = true
            EnableProjective = true
            EnableHyperbolic = true
            EnableWavelet = true
            EnableMinkowski = true
            EnablePauli = true
            SpaceWeights = Map.ofList [
                ("raw", 1.0)
                ("fft", 0.9)      // High weight for pattern detection
                ("dual", 0.95)    // High weight for functional analysis
                ("projective", 0.8)
                ("hyperbolic", 0.85)  // Good for hierarchical agent structures
                ("wavelet", 0.8)
                ("minkowski", 0.7)
                ("pauli", 0.6)
            ]
            PersistToDisk = true
            StoragePath = Some ".tars/vector_store"
        }
    
    /// Create TARS vector store instance
    let createTARSVectorStore () =
        let config = createTARSConfig()
        VectorStoreFactory.createInMemory config
    
    /// Create TARS embedding generator
    let createTARSEmbeddingGenerator () =
        let config = createTARSConfig()
        EmbeddingGeneratorFactory.createEnhanced config None
    
    /// Create TARS inference engine
    let createTARSInferenceEngine () =
        let config = createTARSConfig()
        let vectorStore = createTARSVectorStore()
        let embeddingGenerator = createTARSEmbeddingGenerator()
        InferenceEngineFactory.createAdvanced vectorStore embeddingGenerator config
    
    /// Convert file path to document type
    let inferDocumentType (filePath: string) : TARSDocumentType =
        let extension = Path.GetExtension(filePath).ToLowerInvariant()
        let fileName = Path.GetFileName(filePath).ToLowerInvariant()
        
        match extension, fileName with
        | ".tars", _ -> MetascriptFile
        | ".trsx", _ -> MetascriptFile
        | ".ebnf", _ | ".bnf", _ -> GrammarFile
        | ".json", name when name.Contains("trace") -> TraceFile
        | ".yaml", name when name.Contains("trace") -> TraceFile
        | ".json", name when name.Contains("config") -> ConfigFile
        | ".yaml", name when name.Contains("config") -> ConfigFile
        | ".yml", name when name.Contains("config") -> ConfigFile
        | ".fs", _ | ".fsx", _ | ".fsi", _ -> CodeFile
        | ".cs", _ | ".csx", _ -> CodeFile
        | ".py", _ | ".pyx", _ -> CodeFile
        | ".js", _ | ".ts", _ -> CodeFile
        | ".md", _ | ".txt", _ | ".rst", _ -> DocumentationFile
        | _ -> DocumentationFile
    
    /// Extract TARS-specific metadata from file content
    let extractTARSMetadata (content: string) (filePath: string) : Map<string, string> * string list =
        let metadata = System.Collections.Generic.Dictionary<string, string>()
        let tags = ResizeArray<string>()
        
        // Extract agent name from metascript
        if content.Contains("agent_module") then
            let lines = content.Split('\n')
            for line in lines do
                if line.Trim().StartsWith("name:") then
                    let name = line.Substring(line.IndexOf(':') + 1).Trim().Trim('"')
                    metadata.["agent_name"] <- name
                    tags.Add("agent")
                    tags.Add(name)
                elif line.Trim().StartsWith("goal:") then
                    let goal = line.Substring(line.IndexOf(':') + 1).Trim().Trim('"')
                    metadata.["agent_goal"] <- goal
        
        // Extract grammar information
        if content.Contains("grammar {") || content.Contains("LANG(") then
            tags.Add("grammar")
            if content.Contains("EBNF") then tags.Add("ebnf")
            if content.Contains("BNF") then tags.Add("bnf")
        
        // Extract language blocks
        let languagePattern = @"LANG\(""([^""]+)""\)"
        let matches = System.Text.RegularExpressions.Regex.Matches(content, languagePattern)
        for m in matches do
            let lang = m.Groups.[1].Value.ToLowerInvariant()
            tags.Add(sprintf "lang_%s" lang)
            metadata.[sprintf "uses_%s" lang] <- "true"
        
        // Extract RFC references
        if content.Contains("rfc ") then
            tags.Add("rfc")
            let rfcPattern = @"rfc ""([^""]+)"""
            let rfcMatches = System.Text.RegularExpressions.Regex.Matches(content, rfcPattern)
            for m in rfcMatches do
                let rfcId = m.Groups.[1].Value
                tags.Add(sprintf "rfc_%s" rfcId)
                metadata.[sprintf "references_%s" rfcId] <- "true"
        
        // Add file-based metadata
        metadata.["file_path"] <- filePath
        metadata.["file_extension"] <- Path.GetExtension(filePath)
        metadata.["file_size"] <- content.Length.ToString()
        
        (metadata |> Seq.map (fun kvp -> kvp.Key, kvp.Value) |> Map.ofSeq, tags |> Seq.toList)
    
    /// Create TARS vector document from file
    let createTARSDocumentFromFile (filePath: string) (embeddingGenerator: IEmbeddingGenerator) : Async<TARSVectorDocument> =
        async {
            let content = File.ReadAllText(filePath)
            let! embedding = embeddingGenerator.GenerateEmbedding content
            
            let documentType = inferDocumentType filePath
            let (metadata, tags) = extractTARSMetadata content filePath
            
            // Enhance embedding metadata with TARS-specific info
            let enhancedMetadata = 
                Map.fold (fun acc key value -> Map.add key value acc) embedding.Metadata metadata
            
            let enhancedEmbedding = { embedding with Metadata = enhancedMetadata }
            
            let baseDocument = {
                Id = Path.GetFileNameWithoutExtension(filePath)
                Content = content
                Embedding = enhancedEmbedding
                Tags = tags
                Timestamp = DateTime.Now
                Source = Some filePath
            }
            
            return {
                BaseDocument = baseDocument
                DocumentType = documentType
                ProjectPath = Some (Path.GetDirectoryName(filePath))
                AgentName = metadata.TryFind("agent_name")
                ExecutionId = None
                Dependencies = []
            }
        }
    
    /// Index entire TARS project
    let indexTARSProject (projectPath: string) (vectorStore: IVectorStore) (embeddingGenerator: IEmbeddingGenerator) : Async<int> =
        async {
            let files = ResizeArray<string>()
            
            // Find all relevant files
            let extensions = [".tars"; ".trsx"; ".fs"; ".fsx"; ".cs"; ".md"; ".json"; ".yaml"; ".yml"]
            for ext in extensions do
                let pattern = sprintf "*%s" ext
                let foundFiles = Directory.GetFiles(projectPath, pattern, SearchOption.AllDirectories)
                files.AddRange(foundFiles)
            
            printfn "📁 Found %d files to index in %s" files.Count projectPath
            
            let mutable indexed = 0
            for filePath in files do
                try
                    let! tarsDoc = createTARSDocumentFromFile filePath embeddingGenerator
                    do! vectorStore.AddDocument tarsDoc.BaseDocument
                    indexed <- indexed + 1
                    if indexed % 10 = 0 then
                        printfn "📄 Indexed %d/%d files..." indexed files.Count
                with
                | ex ->
                    printfn "⚠️ Failed to index %s: %s" filePath ex.Message
            
            printfn "✅ Successfully indexed %d files" indexed
            return indexed
        }
    
    /// Search TARS project with context
    let searchTARSProject (query: string) (vectorStore: IVectorStore) (embeddingGenerator: IEmbeddingGenerator) (maxResults: int) : Async<SearchResult list> =
        async {
            let! queryEmbedding = embeddingGenerator.GenerateEmbedding query
            
            let vectorQuery = {
                Text = query
                Embedding = queryEmbedding
                Filters = Map.empty
                MaxResults = maxResults
                MinScore = 0.1
            }
            
            let! results = vectorStore.Search vectorQuery
            return results
        }
    
    /// Get related documents for a TARS file
    let getRelatedTARSDocuments (filePath: string) (vectorStore: IVectorStore) (embeddingGenerator: IEmbeddingGenerator) (maxResults: int) : Async<VectorDocument list> =
        async {
            if File.Exists(filePath) then
                let content = File.ReadAllText(filePath)
                let! embedding = embeddingGenerator.GenerateEmbedding content
                
                let query = {
                    Text = content
                    Embedding = embedding
                    Filters = Map.empty
                    MaxResults = maxResults + 1  // +1 because we'll filter out the original
                    MinScore = 0.2
                }
                
                let! results = vectorStore.Search query
                let fileName = Path.GetFileNameWithoutExtension(filePath)
                
                return 
                    results 
                    |> List.filter (fun r -> r.Document.Id <> fileName)
                    |> List.map (fun r -> r.Document)
                    |> List.take maxResults
            else
                return []
        }
    
    /// Generate TARS project insights
    let generateTARSProjectInsights (vectorStore: IVectorStore) : Async<Map<string, obj>> =
        async {
            let! documentCount = vectorStore.GetDocumentCount()
            
            // This would be expanded with actual analysis
            let insights = Map.ofList [
                ("total_documents", box documentCount)
                ("analysis_timestamp", box (DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")))
                ("vector_spaces_enabled", box ["raw"; "fft"; "dual"; "projective"; "hyperbolic"; "wavelet"; "minkowski"; "pauli"])
            ]
            
            return insights
        }

/// TARS-specific vector store service
type TARSVectorStoreService() =
    let config = VectorStoreIntegration.createTARSConfig()
    let vectorStore = VectorStoreIntegration.createTARSVectorStore()
    let embeddingGenerator = VectorStoreIntegration.createTARSEmbeddingGenerator()
    let inferenceEngine = VectorStoreIntegration.createTARSInferenceEngine()
    
    member _.VectorStore = vectorStore
    member _.EmbeddingGenerator = embeddingGenerator
    member _.InferenceEngine = inferenceEngine
    member _.Config = config
    
    /// Index the current TARS project
    member this.IndexProject(projectPath: string) =
        VectorStoreIntegration.indexTARSProject projectPath this.VectorStore this.EmbeddingGenerator
    
    /// Search the project
    member this.Search(query: string, maxResults: int) =
        VectorStoreIntegration.searchTARSProject query this.VectorStore this.EmbeddingGenerator maxResults
    
    /// Get related documents
    member this.GetRelated(filePath: string, maxResults: int) =
        VectorStoreIntegration.getRelatedTARSDocuments filePath this.VectorStore this.EmbeddingGenerator maxResults
    
    /// Generate insights
    member this.GenerateInsights() =
        VectorStoreIntegration.generateTARSProjectInsights this.VectorStore
    
    /// Perform inference
    member this.Infer(query: string, parameters: Map<string, obj>) =
        this.InferenceEngine.Infer query parameters
