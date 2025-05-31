namespace TarsEngine.FSharp.Core.ChromaDB

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
    abstract member CreateCollectionAsync: name: string -> Task<ChromaCollection>
    abstract member GetCollectionAsync: name: string -> Task<ChromaCollection option>
    abstract member AddDocumentsAsync: collectionName: string * documents: ChromaDocument list -> Task<unit>
    abstract member QueryAsync: collectionName: string * query: string * limit: int -> Task<ChromaQueryResult>
    abstract member DeleteCollectionAsync: name: string -> Task<unit>

/// Hybrid RAG service interface
type IHybridRAGService =
    abstract member StoreKnowledgeAsync: content: string * metadata: Map<string, obj> -> Task<string>
    abstract member RetrieveKnowledgeAsync: query: string * limit: int -> Task<ChromaDocument list>
    abstract member SearchSimilarAsync: content: string * limit: int -> Task<ChromaDocument list>
    abstract member GetMemoryStatsAsync: unit -> Task<Map<string, obj>>

