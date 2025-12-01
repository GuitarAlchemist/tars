namespace Tars.Kernel

open System
open System.IO
open System.Text.Json
open Tars.Core

module SemanticMemory =

    type MemoryIndexEntry = {
        Id         : MemorySchemaId
        Embedding  : float32 array
        TaskKind   : string
        Tags       : string list
    }

    type SemanticMemoryConfig = {
        StorageRoot : string
        TopK        : int
    }

    type SemanticMemoryService(config: SemanticMemoryConfig, embedder: Embedder) =
        
        let indexFile = Path.Combine(config.StorageRoot, "index.json")
        let recordsDir = Path.Combine(config.StorageRoot, "records")

        let mutable index : MemoryIndexEntry list = []

        let loadIndex () =
            if File.Exists(indexFile) then
                try
                    let json = File.ReadAllText(indexFile)
                    JsonSerializer.Deserialize<MemoryIndexEntry list>(json)
                with _ -> []
            else []

        let saveIndex () =
            let options = JsonSerializerOptions(WriteIndented = true)
            let json = JsonSerializer.Serialize(index, options)
            File.WriteAllText(indexFile, json)

        let saveSchema (schema: MemorySchema) =
            let path = Path.Combine(recordsDir, schema.Id + ".json")
            let options = JsonSerializerOptions(WriteIndented = true)
            let json = JsonSerializer.Serialize(schema, options)
            File.WriteAllText(path, json)

        let loadSchema (id: string) =
            let path = Path.Combine(recordsDir, id + ".json")
            if File.Exists(path) then
                let json = File.ReadAllText(path)
                try
                    JsonSerializer.Deserialize<MemorySchema>(json) |> Some
                with _ -> None
            else None

        // Cosine similarity
        let similarity (v1: float32 array) (v2: float32 array) =
            let mutable dot = 0.0f
            let mutable n1 = 0.0f
            let mutable n2 = 0.0f
            if v1.Length <> v2.Length then 0.0f
            else
                for i in 0 .. v1.Length - 1 do
                    let a = v1.[i]
                    let b = v2.[i]
                    dot <- dot + a * b
                    n1 <- n1 + a * a
                    n2 <- n2 + b * b
                if n1 = 0.0f || n2 = 0.0f then 0.0f
                else dot / (sqrt n1 * sqrt n2)

        do
            if not (Directory.Exists(config.StorageRoot)) then Directory.CreateDirectory(config.StorageRoot) |> ignore
            if not (Directory.Exists(recordsDir)) then Directory.CreateDirectory(recordsDir) |> ignore
            index <- loadIndex()

        interface ISemanticMemory with
            member _.Retrieve (query: MemoryQuery) = async {
                let textToEmbed = 
                    if String.IsNullOrWhiteSpace query.TextContext then
                        String.Join(" ", query.Tags)
                    else
                        query.TextContext + " " + String.Join(" ", query.Tags)
                
                let! queryEmbedding = embedder textToEmbed
                
                let results =
                    index
                    |> List.map (fun entry -> entry, similarity queryEmbedding entry.Embedding)
                    |> List.sortByDescending snd
                    |> List.truncate config.TopK
                    |> List.choose (fun (entry, score) -> 
                        // Optional: filter by score threshold
                        if score > 0.0f then loadSchema entry.Id else None
                    )
                
                return results
            }

            member _.Grow (traceObj: obj, verifObj: obj) = async {
                let schemaId = Guid.NewGuid().ToString("N")
                
                let taskId, codeStructure = 
                    match traceObj with
                    | :? MemoryTrace as t -> 
                        let cs = 
                            t.Variables 
                            |> Map.tryFind "code_structure" 
                            |> Option.bind (fun o -> match o with | :? CodeStructure as c -> Some c | _ -> None)
                        (t.TaskId, cs)
                    | _ -> ("Unknown Task", None)

                let summary = $"Memory for task: {taskId}"
                let! embedding = embedder summary
                
                let logical = {
                    ProblemSummary = summary
                    StrategySummary = "Executed agent loop"
                    ErrorKinds = []
                    ErrorTags = []
                    OutcomeLabel = "unknown"
                    Score = None
                    CostTokens = None
                    Embedding = embedding
                    Tags = ["auto-generated"; taskId]
                }
                
                let perceptual = 
                    match codeStructure with
                    | Some cs ->
                        Some {
                            TouchedResources = []
                            EnvFingerprint = "todo"
                            ToolsUsed = []
                            CodeStructure = Some cs
                            Embedding = Array.empty // TODO: Embed code structure
                        }
                    | None -> None

                let schema = {
                    Id = schemaId
                    Logical = Some logical
                    Perceptual = perceptual
                    CreatedAt = DateTime.UtcNow
                    LastUsedAt = None
                    UsageCount = 0
                }
                
                saveSchema schema
                
                let entry = {
                    Id = schemaId
                    Embedding = embedding
                    TaskKind = "metascript"
                    Tags = logical.Tags
                }
                
                index <- entry :: index
                saveIndex()
                
                return schemaId
            }

            member _.Refine () = async {
                // Phase 1: No-op
                return ()
            }
