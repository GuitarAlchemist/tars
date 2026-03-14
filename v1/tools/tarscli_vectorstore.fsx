#!/usr/bin/env dotnet fsi

// TARS Vector Store CLI Tool
// Usage: dotnet fsi tarscli_vectorstore.fsx <command> [args]

#r "nuget: System.Text.Json"
#r "nuget: MathNet.Numerics"

open System
open System.IO
open System.Text.Json
open System.Numerics
open System.Collections.Generic

// Simplified types for CLI usage (JSON-serializable)
type SimpleDocument = {
    Id: string
    Content: string
    RawEmbedding: float[]
    Tags: string list
    Timestamp: DateTime
    Source: string option
    BeliefState: string
    Metadata: Dictionary<string, string>
}

let vectorStoreDirectory = ".tars/vector_store"

let ensureVectorStoreDirectory () =
    if not (Directory.Exists(vectorStoreDirectory)) then
        Directory.CreateDirectory(vectorStoreDirectory) |> ignore

let generateSimpleEmbedding (text: string) (dimension: int) : float[] =
    let hash = text.GetHashCode()
    let rng = Random(hash)
    Array.init dimension (fun _ -> rng.NextDouble() * 2.0 - 1.0)

let createDocument (id: string) (content: string) (tags: string list) (source: string option) : SimpleDocument =
    let raw = generateSimpleEmbedding content 768
    let metadata = Dictionary<string, string>()
    metadata.["generated_at"] <- DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss")
    metadata.["text_length"] <- content.Length.ToString()
    metadata.["text_hash"] <- content.GetHashCode().ToString()

    {
        Id = id
        Content = content
        RawEmbedding = raw
        Tags = tags
        Timestamp = DateTime.Now
        Source = source
        BeliefState = if Array.average raw > 0.0 then "True" else "False"
        Metadata = metadata
    }

let saveDocument (doc: SimpleDocument) =
    try
        ensureVectorStoreDirectory()
        let filePath = Path.Combine(vectorStoreDirectory, sprintf "%s.json" doc.Id)
        let json = JsonSerializer.Serialize(doc, JsonSerializerOptions(WriteIndented = true))
        File.WriteAllText(filePath, json)
        printfn "✅ Document '%s' saved to %s" doc.Id filePath
        true
    with
    | ex ->
        printfn "❌ Failed to save document '%s': %s" doc.Id ex.Message
        false

let loadDocument (id: string) : SimpleDocument option =
    try
        let filePath = Path.Combine(vectorStoreDirectory, sprintf "%s.json" id)
        if File.Exists(filePath) then
            let json = File.ReadAllText(filePath)
            Some (JsonSerializer.Deserialize<SimpleDocument>(json))
        else
            None
    with
    | ex ->
        printfn "❌ Failed to load document '%s': %s" id ex.Message
        None

let listDocuments () =
    ensureVectorStoreDirectory()
    let files = Directory.GetFiles(vectorStoreDirectory, "*.json")
    
    printfn "📚 Vector Store Documents:"
    printfn "=========================="
    
    if files.Length = 0 then
        printfn "No documents found. Add some with 'add-document'."
    else
        for file in files do
            let id = Path.GetFileNameWithoutExtension(file)
            match loadDocument id with
            | Some doc ->
                let fileInfo = FileInfo(file)
                printfn "📄 %s" doc.Id
                printfn "   Content: %s" (if doc.Content.Length > 50 then doc.Content.Substring(0, 50) + "..." else doc.Content)
                printfn "   Tags: %s" (String.Join(", ", doc.Tags))
                printfn "   Source: %s" (doc.Source |> Option.defaultValue "unknown")
                printfn "   Size: %d bytes" fileInfo.Length
                printfn "   Modified: %s" (fileInfo.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"))
                printfn "   Belief: %s" doc.BeliefState
                printfn ""
            | None ->
                printfn "❌ Failed to load: %s" id

let addDocument (id: string) (content: string) (tags: string) (source: string option) =
    let tagList = 
        if String.IsNullOrWhiteSpace(tags) then []
        else tags.Split([|','; ';'|], StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun s -> s.Trim()) |> Array.toList
    
    let doc = createDocument id content tagList source
    if saveDocument doc then
        printfn "✅ Document added successfully"
        printfn "   ID: %s" doc.Id
        printfn "   Content length: %d characters" doc.Content.Length
        printfn "   Tags: %s" (String.Join(", ", doc.Tags))
        printfn "   Embedding dimensions: %d" doc.RawEmbedding.Length
    else
        printfn "❌ Failed to add document"

let deleteDocument (id: string) =
    try
        let filePath = Path.Combine(vectorStoreDirectory, sprintf "%s.json" id)
        if File.Exists(filePath) then
            File.Delete(filePath)
            printfn "✅ Document '%s' deleted" id
        else
            printfn "❌ Document '%s' not found" id
    with
    | ex ->
        printfn "❌ Failed to delete document '%s': %s" id ex.Message

let cosineSimilarity (v1: float[]) (v2: float[]) : float =
    if v1.Length <> v2.Length then 0.0
    else
        let dot = Array.map2 (*) v1 v2 |> Array.sum
        let mag1 = sqrt (Array.sumBy (fun x -> x * x) v1)
        let mag2 = sqrt (Array.sumBy (fun x -> x * x) v2)
        if mag1 > 1e-10 && mag2 > 1e-10 then
            dot / (mag1 * mag2)
        else
            0.0

let searchDocuments (query: string) (maxResults: int) =
    ensureVectorStoreDirectory()
    let files = Directory.GetFiles(vectorStoreDirectory, "*.json")

    if files.Length = 0 then
        printfn "No documents in vector store to search."
    else
        printfn "🔍 Searching for: '%s'" query
        printfn "=================="

        let queryEmbedding = generateSimpleEmbedding query 768
        let results = ResizeArray<string * SimpleDocument * float>()

        for file in files do
            let id = Path.GetFileNameWithoutExtension(file)
            match loadDocument id with
            | Some doc ->
                let similarity = cosineSimilarity queryEmbedding doc.RawEmbedding
                results.Add((id, doc, similarity))
            | None -> ()

        let sortedResults =
            results
            |> Seq.sortByDescending (fun (_, _, score) -> score)
            |> Seq.take (min maxResults results.Count)
            |> Seq.toList

        if sortedResults.Length = 0 then
            printfn "No results found."
        else
            for i, (id, doc, score) in List.indexed sortedResults do
                printfn "%d. %s (Score: %.3f)" (i + 1) doc.Id score
                printfn "   Content: %s" (if doc.Content.Length > 100 then doc.Content.Substring(0, 100) + "..." else doc.Content)
                printfn "   Tags: %s" (String.Join(", ", doc.Tags))
                printfn "   Belief: %s" doc.BeliefState
                printfn ""

let showDocument (id: string) =
    match loadDocument id with
    | Some doc ->
        printfn "📄 Document Details: %s" doc.Id
        printfn "===================="
        printfn "Content:"
        printfn "%s" doc.Content
        printfn ""
        printfn "Tags: %s" (String.Join(", ", doc.Tags))
        printfn "Source: %s" (doc.Source |> Option.defaultValue "unknown")
        printfn "Timestamp: %s" (doc.Timestamp.ToString("yyyy-MM-dd HH:mm:ss"))
        printfn "Belief State: %s" doc.BeliefState
        printfn "Raw Embedding Dimension: %d" doc.RawEmbedding.Length
        printfn ""
        printfn "Metadata:"
        for kvp in doc.Metadata do
            printfn "  %s: %s" kvp.Key kvp.Value
    | None ->
        printfn "❌ Document '%s' not found" id

let getStats () =
    ensureVectorStoreDirectory()
    let files = Directory.GetFiles(vectorStoreDirectory, "*.json")
    
    printfn "📊 Vector Store Statistics"
    printfn "=========================="
    printfn "Total Documents: %d" files.Length
    
    if files.Length > 0 then
        let totalSize = files |> Array.sumBy (fun f -> (FileInfo(f)).Length)
        let avgSize = totalSize / int64 files.Length
        
        printfn "Total Size: %d bytes" totalSize
        printfn "Average Size: %d bytes" avgSize
        
        let beliefCounts = Dictionary<string, int>()
        let tagCounts = Dictionary<string, int>()

        for file in files do
            let id = Path.GetFileNameWithoutExtension(file)
            match loadDocument id with
            | Some doc ->
                let currentCount = beliefCounts.GetValueOrDefault(doc.BeliefState, 0)
                beliefCounts.[doc.BeliefState] <- currentCount + 1

                for tag in doc.Tags do
                    let currentTagCount = tagCounts.GetValueOrDefault(tag, 0)
                    tagCounts.[tag] <- currentTagCount + 1
            | None -> ()
        
        printfn ""
        printfn "Belief Distribution:"
        for kvp in beliefCounts do
            printfn "  %s: %d" kvp.Key kvp.Value
        
        printfn ""
        printfn "Top Tags:"
        tagCounts
        |> Seq.sortByDescending (fun kvp -> kvp.Value)
        |> Seq.truncate 10
        |> Seq.iter (fun kvp -> printfn "  %s: %d" kvp.Key kvp.Value)

let showHelp () =
    printfn "🚀 TARS Vector Store CLI Tool"
    printfn "============================="
    printfn ""
    printfn "Commands:"
    printfn "  list                                    - List all documents"
    printfn "  add <id> <content> [tags] [source]     - Add a new document"
    printfn "  show <id>                               - Show document details"
    printfn "  search <query> [max_results]           - Search documents"
    printfn "  delete <id>                             - Delete a document"
    printfn "  stats                                   - Show vector store statistics"
    printfn "  help                                    - Show this help"
    printfn ""
    printfn "Examples:"
    printfn "  dotnet fsi tarscli_vectorstore.fsx list"
    printfn "  dotnet fsi tarscli_vectorstore.fsx add doc1 \"Hello world\" \"greeting,test\""
    printfn "  dotnet fsi tarscli_vectorstore.fsx search \"hello\" 5"
    printfn "  dotnet fsi tarscli_vectorstore.fsx show doc1"
    printfn "  dotnet fsi tarscli_vectorstore.fsx delete doc1"

// Main command processing
let args = fsi.CommandLineArgs |> Array.skip 1

match args with
| [||] | [|"help"|] -> showHelp()
| [|"list"|] -> listDocuments()
| [|"add"; id; content|] -> addDocument id content "" None
| [|"add"; id; content; tags|] -> addDocument id content tags None
| [|"add"; id; content; tags; source|] -> addDocument id content tags (Some source)
| [|"show"; id|] -> showDocument id
| [|"search"; query|] -> searchDocuments query 10
| [|"search"; query; maxStr|] -> 
    match Int32.TryParse(maxStr) with
    | true, max -> searchDocuments query max
    | false, _ -> printfn "❌ Invalid max_results: %s" maxStr
| [|"delete"; id|] -> deleteDocument id
| [|"stats"|] -> getStats()
| _ -> 
    printfn "❌ Unknown command or invalid arguments"
    printfn "Use 'help' to see available commands"
    exit 1
