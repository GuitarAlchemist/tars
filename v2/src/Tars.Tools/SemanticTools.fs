namespace Tars.Tools.Semantic

open System.IO
open System.Threading.Tasks
open Tars.Tools

module SemanticTools =

    /// Helper to format file content with line numbers
    let private formatWithLineNumbers (content: string) =
        content.Split('\n')
        |> Array.mapi (fun i line -> $"%4d{i + 1} | %s{line}")
        |> String.concat "\n"

    let private parseExploreArgs (args: string) =
        try
            let doc = System.Text.Json.JsonDocument.Parse(args)
            let root = doc.RootElement

            let mutable pathProp = Unchecked.defaultof<System.Text.Json.JsonElement>

            let path =
                if root.TryGetProperty("path", &pathProp) then
                    pathProp.GetString()
                else
                    "."

            let mutable depthProp = Unchecked.defaultof<System.Text.Json.JsonElement>

            let maxDepth =
                if root.TryGetProperty("depth", &depthProp) then
                    depthProp.GetInt32()
                else
                    2

            Ok(path, maxDepth)
        with ex ->
            Error ex.Message

    [<TarsToolAttribute("explore_project",
                        "Explores the project structure. Input JSON: { \"path\": \"root_path\", \"depth\": 2 }")>]
    let exploreProject (args: string) =
        match parseExploreArgs args with
        | Error msg -> Task.FromResult($"explore_project error: {msg}")
        | Ok(path, maxDepth) ->
            try
                let fullPath = Path.GetFullPath(path)

                if not (Directory.Exists fullPath) then
                    Task.FromResult($"Directory not found: {fullPath}")
                else
                    // Exclude common noise
                    let excludes = Set.ofList [ ".git"; "bin"; "obj"; ".vs"; ".idea"; "node_modules" ]

                    let rec walk dir depth =
                        if depth > maxDepth then
                            []
                        else
                            let dirInfo = DirectoryInfo(dir)

                            let files =
                                dirInfo.GetFiles()
                                |> Array.map (fun f -> $"%s{f.Name} (file)")
                                |> Array.toList

                            let subDirs =
                                dirInfo.GetDirectories()
                                |> Array.filter (fun d -> not (excludes.Contains d.Name))
                                |> Array.toList

                            let subDirOutput =
                                subDirs
                                |> List.collect (fun d ->
                                    let children = walk d.FullName (depth + 1)
                                    $"%s{d.Name}/ (dir)" :: (children |> List.map (fun c -> "  " + c)))

                            files @ subDirOutput

                    let tree = walk fullPath 0 |> String.concat "\n"
                    Task.FromResult $"Project Structure for %s{fullPath}:\n%s{tree}"
            with ex ->
                Task.FromResult($"explore_project error: {ex.Message}")

    let private parseReadCodeArgs (args: string) =
        try
            let doc = System.Text.Json.JsonDocument.Parse(args)
            let root = doc.RootElement

            let mutable pathProp = Unchecked.defaultof<System.Text.Json.JsonElement>

            let path =
                if root.TryGetProperty("path", &pathProp) then
                    pathProp.GetString()
                else
                    args

            Ok path
        with ex ->
            Error ex.Message

    [<TarsToolAttribute("read_code", "Reads a code file with line numbers. Input JSON: { \"path\": \"file.fs\" }")>]
    let readCode (args: string) =
        match parseReadCodeArgs args with
        | Error msg -> Task.FromResult($"read_code error: {msg}")
        | Ok path ->
            try
                let fullPath = Path.GetFullPath(path)

                if not (File.Exists fullPath) then
                    Task.FromResult($"File not found: {fullPath}")
                else
                    let content = File.ReadAllText(fullPath)

                    if content.Length > 50000 then
                        Task.FromResult(
                            $"File too large ({content.Length} chars). Use 'explore_project' or read specific sections."
                        )
                    else
                        let numbered = formatWithLineNumbers content
                        Task.FromResult(numbered)
            with ex ->
                Task.FromResult($"read_code error: {ex.Message}")

    let private parsePatchArgs (args: string) =
        try
            let doc = System.Text.Json.JsonDocument.Parse(args)
            let root = doc.RootElement

            let path = root.GetProperty("path").GetString()
            let original = root.GetProperty("original").GetString()
            let replacement = root.GetProperty("replacement").GetString()
            Ok(path, original, replacement)
        with ex ->
            Error ex.Message

    [<TarsToolAttribute("patch_code",
                        "Replaces a block of code. Input JSON: { \"path\": \"file.fs\", \"original\": \"...\", \"replacement\": \"...\" }")>]
    let patchCode (args: string) =
        match parsePatchArgs args with
        | Error msg -> Task.FromResult($"patch_code error: {msg}")
        | Ok(path, original, replacement) ->
            try
                let fullPath = Path.GetFullPath(path)

                if not (File.Exists fullPath) then
                    Task.FromResult($"File not found: {fullPath}")
                else
                    let content = File.ReadAllText(fullPath)

                    // Normalize line endings for matching
                    let normContent = content.Replace("\r\n", "\n")
                    let normOriginal = original.Replace("\r\n", "\n")

                    if normContent.Contains(normOriginal) then
                        let newContent = normContent.Replace(normOriginal, replacement)
                        File.WriteAllText(fullPath, newContent)
                        Task.FromResult("Successfully patched file.")
                    else
                        Task.FromResult(
                            "Error: Original code block not found in file. Please verify exact content (whitespace matters)."
                        )
            with ex ->
                Task.FromResult($"patch_code error: {ex.Message}")
