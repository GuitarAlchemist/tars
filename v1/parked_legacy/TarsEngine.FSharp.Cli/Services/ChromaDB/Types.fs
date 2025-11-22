namespace TarsEngine.FSharp.Cli.Services.ChromaDB

open System
open System.Collections.Generic
open System.Threading.Tasks

/// ChromaDB document representation
type ChromaDocument = {
    Id: string
    Content: string
    Metadata: Map<string, obj>
    Embedding: float[] option
}

/// ChromaDB collection representation  
type ChromaCollection = {
    Name: string
    Documents: ChromaDocument list
    Metadata: Map<string, obj>
}

/// ChromaDB query result
type ChromaQueryResult = {
    Documents: ChromaDocument list
    Distances: float list
    Similarities: float list
}

/// ChromaDB client interface
type IChromaDBClient =
    abstract member CreateCollectionAsync: string -> Task<ChromaCollection option>
    abstract member GetCollectionAsync: string -> Task<ChromaCollection option>
    abstract member AddDocumentsAsync: string -> ChromaDocument list -> Task<unit>
    abstract member QueryAsync: string -> string -> int -> Task<ChromaQueryResult>
    abstract member DeleteCollectionAsync: string -> Task<bool>

/// Hybrid RAG service interface
type IHybridRAGService =
    abstract member StoreKnowledgeAsync: string -> Map<string, obj> -> Task<string>
    abstract member SearchKnowledgeAsync: string -> int -> Task<ChromaDocument list>
    abstract member GetKnowledgeStatsAsync: unit -> Task<{| InMemoryCount: int; ChromaDBCount: int; TotalSize: int64 |}>
