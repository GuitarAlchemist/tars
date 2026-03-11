namespace Tars.Kernel

open System
open System.IO
open System.Text.Json
open Tars.Core
open Tars.Llm

module SemanticMemory =

    type MemoryIndexEntry =
        { Id: MemorySchemaId
          Embedding: float32 array
          TaskKind: string
          Tags: string list }

    type SemanticMemoryConfig = { StorageRoot: string; TopK: int }

    type SemanticMemoryService(config: SemanticMemoryConfig, embedder: Embedder, llm: ILlmService) =

        let indexFile = Path.Combine(config.StorageRoot, "index.json")
        let recordsDir = Path.Combine(config.StorageRoot, "records")

        let mutable index: MemoryIndexEntry list = []

        let loadIndex () =
            if File.Exists(indexFile) then
                try
                    let json = File.ReadAllText(indexFile)
                    JsonSerializer.Deserialize<MemoryIndexEntry list>(json)
                with _ ->
                    []
            else
                []

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
                with _ ->
                    None
            else
                None

        // Cosine similarity
        let similarity (v1: float32 array) (v2: float32 array) =
            let mutable dot = 0.0f
            let mutable n1 = 0.0f
            let mutable n2 = 0.0f

            if v1.Length <> v2.Length then
                0.0f
            else
                for i in 0 .. v1.Length - 1 do
                    let a = v1.[i]
                    let b = v2.[i]
                    dot <- dot + a * b
                    n1 <- n1 + a * a
                    n2 <- n2 + b * b

                if n1 = 0.0f || n2 = 0.0f then
                    0.0f
                else
                    dot / (sqrt n1 * sqrt n2)

        let summarizeEpisode (trace: MemoryTrace) (output: string) =
            task {
                let prompt =
                    $"""Analyze the following agent execution trace and generate a structured summary.

Task ID: %s{trace.TaskId}
Output: %s{output}

Trace Variables:
%A{trace.Variables |> Map.remove "trace"}

Please generate a JSON object with the following fields:
- problem_summary: A 1-sentence summary of the task goal.
- strategy_summary: A 2-3 sentence summary of the approach taken and tools used.
- outcome_label: "success", "failure", or "partial".

JSON:"""
                // Avoid dumping huge trace

                let req =
                    { ModelHint = None
                      Model = Some "qwen2.5-coder:1.5b"
                      SystemPrompt = Some "You are a memory consolidation assistant."
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Temperature = Some 0.1
                      MaxTokens = Some 512
                      Stop = []
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = None
                      Stream = false
                      JsonMode = true
                      Seed = None
                      ContextWindow = None }

                try
                    let! response = llm.CompleteAsync req
                    let content = response.Text.Trim()

                    let json =
                        if content.StartsWith("```json") then
                            content.Substring(7, content.Length - 10).Trim()
                        elif content.StartsWith("```") then
                            content.Substring(3, content.Length - 6).Trim()
                        else
                            content

                    let doc = JsonDocument.Parse(json)
                    let root = doc.RootElement

                    let problem =
                        match root.TryGetProperty("problem_summary") with
                        | true, prop -> prop.GetString()
                        | false, _ -> $"Task {trace.TaskId}"

                    let strategy =
                        match root.TryGetProperty("strategy_summary") with
                        | true, prop -> prop.GetString()
                        | false, _ -> "Unknown strategy"

                    let outcome =
                        match root.TryGetProperty("outcome_label") with
                        | true, prop -> prop.GetString()
                        | false, _ -> "unknown"

                    return (problem, strategy, outcome)
                with ex ->
                    return ($"Task {trace.TaskId}", $"Error summarizing: {ex.Message}", "unknown")
            }

        do
            if not (Directory.Exists(config.StorageRoot)) then
                Directory.CreateDirectory(config.StorageRoot) |> ignore

            if not (Directory.Exists(recordsDir)) then
                Directory.CreateDirectory(recordsDir) |> ignore

            index <- loadIndex ()

        interface ISemanticMemory with
            member _.Retrieve(query: MemoryQuery) =
                async {
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
                            if score > 0.0f then loadSchema entry.Id else None)

                    return results
                }

            member _.Grow(traceObj: obj, verifObj: obj) =
                async {
                    let schemaId = Guid.NewGuid().ToString("N")

                    let trace, output =
                        match traceObj with
                        | :? MemoryTrace as t ->
                            let out =
                                t.Variables
                                |> Map.tryFind "output"
                                |> Option.map string
                                |> Option.defaultValue ""

                            (Some t, out)
                        | _ -> (None, "")

                    let taskId =
                        trace |> Option.map (fun t -> t.TaskId) |> Option.defaultValue "Unknown Task"

                    let! (probSum, stratSum, outcome) =
                        match trace with
                        | Some t -> summarizeEpisode t output |> Async.AwaitTask
                        | None -> async { return ($"Memory for task: {taskId}", "No trace available", "unknown") }

                    let! embedding = embedder probSum

                    let logical =
                        { ProblemSummary = probSum
                          StrategySummary = stratSum
                          ErrorKinds = []
                          ErrorTags = []
                          OutcomeLabel = outcome
                          Score = None
                          CostTokens = None
                          Embedding = embedding
                          Tags = [ "auto-generated"; taskId; outcome ] }

                    let perceptual =
                        match trace with
                        | Some t ->
                            let cs =
                                t.Variables
                                |> Map.tryFind "code_structure"
                                |> Option.bind (fun o ->
                                    match o with
                                    | :? CodeStructure as c -> Some c
                                    | _ -> None)

                            match cs with
                            | Some c ->
                                Some
                                    { TouchedResources = []
                                      EnvFingerprint = "todo"
                                      ToolsUsed = []
                                      CodeStructure = Some c
                                      Embedding = Array.empty }
                            | None -> None
                        | None -> None

                    let schema =
                        { Id = schemaId
                          Logical = Some logical
                          Perceptual = perceptual
                          CreatedAt = DateTime.UtcNow
                          LastUsedAt = None
                          UsageCount = 0 }

                    saveSchema schema

                    let entry =
                        { Id = schemaId
                          Embedding = embedding
                          TaskKind = "metascript"
                          Tags = logical.Tags }

                    index <- entry :: index
                    saveIndex ()

                    return schemaId
                }

            member _.Refine() =
                async {
                    let threshold = 0.99f
                    let mutable toRemove = Set.empty

                    // Naive O(N^2) check
                    let entries = index |> List.toArray

                    for i in 0 .. entries.Length - 1 do
                        if not (toRemove.Contains entries.[i].Id) then
                            for j in i + 1 .. entries.Length - 1 do
                                if not (toRemove.Contains entries.[j].Id) then
                                    let sim = similarity entries.[i].Embedding entries.[j].Embedding

                                    if sim > threshold then
                                        // Keep i (newer), remove j (older)
                                        toRemove <- toRemove.Add entries.[j].Id

                    if not toRemove.IsEmpty then
                        index <- index |> List.filter (fun e -> not (toRemove.Contains e.Id))
                        saveIndex ()

                        for id in toRemove do
                            let path = Path.Combine(recordsDir, id + ".json")

                            if File.Exists path then
                                File.Delete path

                        printfn $"Refined memory: Removed %d{toRemove.Count} duplicates."
                    else
                        printfn "Refined memory: No duplicates found."
                }
