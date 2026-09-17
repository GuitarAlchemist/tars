namespace Tars.Tools.Standard

open System.Text.Json
open Tars.Tools
open Tars.Cortex

/// Tools for semantic code search using CodebaseRAG
module SemanticCodeTools =

    /// Set the codebase index, e.g. one a CLI command ingested with embeddings.
    let setCodebaseIndex (index: CodebaseRAG.CodebaseIndex) = CodebaseRAG.SharedIndex.set index

    /// Clear the codebase index.
    let clearCodebaseIndex () = CodebaseRAG.SharedIndex.clear ()

    /// The tree these tools search: TARS_CODEBASE_ROOT, or the working directory.
    let private codebaseRoot () =
        match System.Environment.GetEnvironmentVariable "TARS_CODEBASE_ROOT" with
        | null
        | "" -> System.IO.Directory.GetCurrentDirectory()
        | root -> root

    /// The shared index, quick-ingested on first use so these tools return results
    /// instead of "not initialized" — nothing ever called setCodebaseIndex (#241).
    let private index () =
        CodebaseRAG.SharedIndex.getOrQuickIngest (codebaseRoot ())

    [<TarsToolAttribute("search_codebase",
                        "Semantically searches the TARS codebase for relevant code. Input JSON: { \"query\": \"how to create an agent\" }")>]
    let searchCodebase (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement

                let mutable queryProp = Unchecked.defaultof<JsonElement>

                let query =
                    if root.TryGetProperty("query", &queryProp) then
                        queryProp.GetString()
                    else
                        args // Allow raw query string

                let mutable topKProp = Unchecked.defaultof<JsonElement>

                let topK =
                    if root.TryGetProperty("topK", &topKProp) then
                        topKProp.GetInt32()
                    else
                        5

                let! (index: CodebaseRAG.CodebaseIndex) = index ()
                printfn $"🔍 Searching codebase for: {query}"

                let! (results: CodebaseRAG.SearchResult list) = index.SearchAsync(query, topK)

                if results.IsEmpty then
                    return $"No relevant code found for: {query}"
                else
                    let resultText =
                        results
                        |> List.mapi (fun i result ->
                            let preview =
                                if result.Chunk.Content.Length > 300 then
                                    result.Chunk.Content.Substring(0, 300) + "..."
                                else
                                    result.Chunk.Content

                            $"\n[%d{i + 1}] %s{result.Chunk.FilePath} (lines %d{result.Chunk.StartLine}-%d{result.Chunk.EndLine}) [Score: %.2f{result.Score}]\n```fsharp\n%s{preview}\n```")
                        |> String.concat "\n"

                    return $"Found %d{results.Length} relevant code sections:\n%s{resultText}"
            with ex ->
                return $"search_codebase error: {ex.Message}"
        }

    [<TarsToolAttribute("find_similar_code",
                        "Finds code similar to a given snippet. Input JSON: { \"code\": \"let myFunc x = x + 1\" }")>]
    let findSimilarCode (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement

                let code = root.GetProperty("code").GetString()

                let! (index: CodebaseRAG.CodebaseIndex) = index ()
                printfn $"🔍 Finding similar code..."

                let! (results: CodebaseRAG.SearchResult list) = index.SearchAsync(code, 3)

                if results.IsEmpty then
                    return "No similar code found."
                else
                    let resultText =
                        results
                        |> List.mapi (fun i result ->
                            sprintf
                                "\n[%d] %s (lines %d-%d)\n```fsharp\n%s\n```"
                                (i + 1)
                                result.Chunk.FilePath
                                result.Chunk.StartLine
                                result.Chunk.EndLine
                                (if result.Chunk.Content.Length > 500 then
                                     result.Chunk.Content.Substring(0, 500) + "..."
                                 else
                                     result.Chunk.Content))
                        |> String.concat "\n"

                    return $"Found %d{results.Length} similar code sections:\n%s{resultText}"
            with ex ->
                return $"find_similar_code error: {ex.Message}"
        }

    [<TarsToolAttribute("get_codebase_context",
                        "Gets relevant codebase context for a task description. Input: task description string")>]
    let getCodebaseContext (taskDescription: string) =
        task {
            try
                let! index = index ()
                let! contextOpt = CodebaseRAG.buildTaskContext index taskDescription

                match contextOpt with
                | Some context -> return context
                | None -> return "No relevant codebase context found."
            with ex ->
                return $"get_codebase_context error: {ex.Message}"
        }

    [<TarsToolAttribute("codebase_stats", "Shows statistics about the indexed codebase. No input required.")>]
    let codebaseStats (_: string) =
        task {
            try
                let! index = index ()
                let stats = index.GetStats()

                return
                    $"Codebase Index Statistics:\n  Total Chunks: %d{stats.TotalChunks}\n  With Embeddings: %d{stats.WithEmbeddings}\n  Files Indexed: %d{stats.Files}"
            with ex ->
                return $"codebase_stats error: {ex.Message}"
        }
