namespace TarsEngine.FSharp.Cli.Services

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Services.ChromaDB
open TarsEngine.FSharp.Cli.Services.CodebaseVectorStore

/// Hybrid RAG (Retrieval-Augmented Generation) service for TARS
module HybridRAGService =
    
    type RAGContext = {
        Query: string
        RetrievedDocuments: SearchResult list
        CodeContext: CodeChunk list
        Metadata: Map<string, obj>
    }
    
    type RAGResponse = {
        Answer: string
        Sources: string list
        Confidence: float
        RetrievalTime: TimeSpan
        GenerationTime: TimeSpan
    }
    
    type RAGConfig = {
        MaxDocuments: int
        MinConfidence: float
        UseCodeContext: bool
        UseSemanticSearch: bool
    }
    
    let defaultConfig = {
        MaxDocuments = 5
        MinConfidence = 0.7
        UseCodeContext = true
        UseSemanticSearch = true
    }
    
    /// Retrieve relevant documents for a query
    let retrieveDocumentsAsync (config: RAGConfig) (query: string) =
        task {
            let startTime = DateTime.UtcNow
            
            // Search ChromaDB for relevant documents
            let! searchResult = ChromaDB.searchSimilarAsync ChromaDB.defaultConfig query config.MaxDocuments
            
            let documents = 
                match searchResult with
                | Ok docs -> docs |> List.filter (fun d -> d.Score >= config.MinConfidence)
                | Error _ -> []
            
            // Search codebase if enabled
            let! codeResults = 
                if config.UseCodeContext then
                    CodebaseVectorStore.searchCodeAsync query config.MaxDocuments
                else
                    Task.FromResult(Ok [])
            
            let codeDocuments = 
                match codeResults with
                | Ok docs -> docs
                | Error _ -> []
            
            let retrievalTime = DateTime.UtcNow - startTime
            
            return {
                Query = query
                RetrievedDocuments = documents @ codeDocuments
                CodeContext = []
                Metadata = Map.ofList [
                    ("retrieval_time", retrievalTime :> obj)
                    ("document_count", documents.Length :> obj)
                    ("code_document_count", codeDocuments.Length :> obj)
                ]
            }
        }
    
    /// Generate response using retrieved context
    let generateResponseAsync (context: RAGContext) =
        task {
            let startTime = DateTime.UtcNow
            
            // Simulate LLM generation
            do! Task.Delay(500)
            
            let sources = context.RetrievedDocuments |> List.map (fun d -> d.Document.Id)
            
            let answer = 
                if context.RetrievedDocuments.IsEmpty then
                    "I don't have enough information to answer that question accurately."
                else
                    let relevantContent = 
                        context.RetrievedDocuments 
                        |> List.take (min 3 context.RetrievedDocuments.Length)
                        |> List.map (fun d -> d.Document.Content)
                        |> String.concat "\n\n"
                    
                    $"Based on the available information:\n\n{relevantContent}\n\nThis provides context for your query: {context.Query}"
            
            let confidence = 
                if context.RetrievedDocuments.IsEmpty then 0.0
                else context.RetrievedDocuments |> List.map (fun d -> d.Score) |> List.average
            
            let generationTime = DateTime.UtcNow - startTime
            
            return {
                Answer = answer
                Sources = sources
                Confidence = confidence
                RetrievalTime = context.Metadata.["retrieval_time"] :?> TimeSpan
                GenerationTime = generationTime
            }
        }
    
    /// Complete RAG pipeline
    let queryAsync (config: RAGConfig) (query: string) =
        task {
            printfn "🔍 TARS Hybrid RAG Processing"
            printfn "Query: %s" query
            printfn ""
            
            // Retrieve relevant documents
            printfn "📚 Retrieving relevant documents..."
            let! context = retrieveDocumentsAsync config query
            
            printfn "Found %d relevant documents" context.RetrievedDocuments.Length
            
            // Generate response
            printfn "🤖 Generating response..."
            let! response = generateResponseAsync context
            
            printfn "✅ RAG processing complete"
            printfn "Confidence: %.2f" response.Confidence
            printfn "Sources: %d" response.Sources.Length
            printfn ""
            
            return response
        }
    
    /// Search similar content
    let searchSimilarAsync (query: string) (limit: int) =
        task {
            let! result = ChromaDB.searchSimilarAsync ChromaDB.defaultConfig query limit
            return result
        }
    
    /// Add knowledge to the RAG system
    let addKnowledgeAsync (content: string) (metadata: Map<string, obj>) =
        task {
            let document = {
                Id = Guid.NewGuid().ToString()
                Content = content
                Metadata = metadata
                Embedding = None
            }
            
            let! result = ChromaDB.addDocumentAsync ChromaDB.defaultConfig document
            return result
        }
    
    /// Get RAG system statistics
    let getStatsAsync () =
        task {
            let! chromaStats = ChromaDB.getStatsAsync ChromaDB.defaultConfig
            let! codebaseStats = CodebaseVectorStore.getStatsAsync ()
            
            return {|
                ChromaDB = chromaStats
                Codebase = codebaseStats
                Status = "Active"
                LastUpdated = DateTime.UtcNow
            |}
        }
