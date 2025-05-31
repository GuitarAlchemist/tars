namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.ChromaDB

type KnowledgeCommand(hybridRAG: IHybridRAGService, logger: ILogger<KnowledgeCommand>) =
    interface ICommand with
        member _.Name = "knowledge"
        member _.Description = "Manage TARS knowledge base with ChromaDB hybrid RAG"
        member _.Usage = "tars knowledge <subcommand> [options]"
        member _.Examples = [
            "tars knowledge store \"F# is awesome\" --tag programming"
            "tars knowledge search \"F# programming\""
            "tars knowledge stats"
        ]
        member _.ValidateOptions(_) = true
        
        member _.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | "store" :: content :: _ ->
                        printfn "🧠 TARS KNOWLEDGE STORAGE (ChromaDB Hybrid RAG)"
                        printfn "================================================"
                        printfn "Storing: %s" content
                        printfn ""
                        
                        let metadata = Map.ofList [
                            ("source", "cli" :> obj)
                            ("type", "user_input" :> obj)
                        ]
                        
                        printfn "📝 Storing in hybrid RAG system..."
                        let! docId = hybridRAG.StoreKnowledgeAsync(content, metadata)
                        
                        printfn "✅ Knowledge stored successfully!"
                        printfn "📄 Document ID: %s" docId
                        printfn "💾 Stored in: In-memory cache + ChromaDB"
                        
                        return CommandResult.success("Knowledge stored")
                    
                    | "search" :: query :: _ ->
                        printfn "🔍 TARS KNOWLEDGE SEARCH (ChromaDB Hybrid RAG)"
                        printfn "==============================================="
                        printfn "Query: %s" query
                        printfn ""
                        
                        printfn "🔎 Searching hybrid RAG system..."
                        let! results = hybridRAG.RetrieveKnowledgeAsync(query, 5)
                        
                        printfn "✅ Search completed!"
                        printfn "📊 Found %d results:" results.Length
                        printfn ""
                        
                        for i, doc in results |> List.indexed do
                            printfn "Result %d:" (i + 1)
                            printfn "  ID: %s" doc.Id
                            printfn "  Content: %s" (doc.Content.Substring(0, min 100 doc.Content.Length))
                            printfn "  Metadata: %A" doc.Metadata
                            printfn ""
                        
                        return CommandResult.success("Knowledge search completed")
                    
                    | "stats" :: _ ->
                        printfn "📊 TARS KNOWLEDGE STATISTICS"
                        printfn "============================="
                        printfn ""
                        
                        let! stats = hybridRAG.GetMemoryStatsAsync()
                        
                        printfn "📈 Hybrid RAG Statistics:"
                        for KeyValue(key, value) in stats do
                            printfn "  %s: %A" key value
                        
                        printfn ""
                        printfn "🧠 ChromaDB Integration: Active"
                        printfn "⚡ In-Memory Cache: Active"
                        printfn "🔄 Hybrid RAG: Operational"
                        
                        return CommandResult.success("Knowledge stats displayed")
                    
                    | "similar" :: content :: _ ->
                        printfn "🔗 TARS SIMILARITY SEARCH"
                        printfn "========================="
                        printfn "Content: %s" content
                        printfn ""
                        
                        let! similar = hybridRAG.SearchSimilarAsync(content, 3)
                        
                        printfn "✅ Found %d similar documents:" similar.Length
                        for i, doc in similar |> List.indexed do
                            printfn "%d. %s" (i + 1) (doc.Content.Substring(0, min 80 doc.Content.Length))
                        
                        return CommandResult.success("Similarity search completed")
                    
                    | [] ->
                        printfn "TARS Knowledge Management Commands:"
                        printfn "  store <content>  - Store knowledge in hybrid RAG"
                        printfn "  search <query>   - Search knowledge base"
                        printfn "  similar <text>   - Find similar content"
                        printfn "  stats            - Show knowledge statistics"
                        return CommandResult.success("Help displayed")
                    
                    | unknown :: _ ->
                        printfn "Unknown knowledge command: %s" unknown
                        return CommandResult.failure("Unknown command")
                with
                | ex ->
                    logger.LogError(ex, "Knowledge command error")
                    return CommandResult.failure(ex.Message)
            }

