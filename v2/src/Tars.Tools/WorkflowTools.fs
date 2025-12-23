namespace Tars.Tools.Standard

open System
open System.IO
open System.Text.RegularExpressions
open Tars.Tools

module WorkflowTools =

    [<TarsToolAttribute("list_files",
                        "Lists files in a directory with optional filtering. Input JSON: { \"path\": \".\", \"pattern\": \"*.fs\", \"recursive\": true }")>]
    let listFiles (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let mutable pathProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let path =
                    if root.TryGetProperty("path", &pathProp) then
                        pathProp.GetString()
                    else
                        "."

                let mutable patternProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let pattern =
                    if root.TryGetProperty("pattern", &patternProp) then
                        patternProp.GetString()
                    else
                        "*"

                let mutable recursiveProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let recursive =
                    if root.TryGetProperty("recursive", &recursiveProp) then
                        recursiveProp.GetBoolean()
                    else
                        false

                let fullPath = Path.GetFullPath(path)

                if not (Directory.Exists fullPath) then
                    return "Directory not found: " + fullPath
                else
                    printfn $"Listing files in: %s{fullPath} (pattern: %s{pattern})"

                    let searchOption =
                        if recursive then
                            SearchOption.AllDirectories
                        else
                            SearchOption.TopDirectoryOnly

                    let files = Directory.GetFiles(fullPath, pattern, searchOption)

                    let maxFiles = 50

                    let fileList =
                        files
                        |> Array.take (Math.Min(files.Length, maxFiles))
                        |> Array.map (fun f ->
                            let relativePath = Path.GetRelativePath(fullPath, f)
                            let size = FileInfo(f).Length
                            $"  %s{relativePath} (%d{size} bytes)")
                        |> String.concat "\n"

                    let truncated =
                        if files.Length > maxFiles then
                            $"\n... and %d{files.Length - maxFiles} more files"
                        else
                            ""

                    return $"Found %d{files.Length} files:\n%s{fileList}%s{truncated}"
            with ex ->
                return "list_files error: " + ex.Message
        }

    [<TarsToolAttribute("search_code",
                        "Searches for a pattern in code files. Input JSON: { \"pattern\": \"TODO\", \"path\": \".\", \"filePattern\": \"*.fs\" }")>]
    let searchCode (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let pattern = root.GetProperty("pattern").GetString()

                let mutable pathProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let path =
                    if root.TryGetProperty("path", &pathProp) then
                        pathProp.GetString()
                    else
                        "."

                let mutable fileProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let filePattern =
                    if root.TryGetProperty("filePattern", &fileProp) then
                        fileProp.GetString()
                    else
                        "*.fs"

                let fullPath = Path.GetFullPath(path)
                printfn $"Searching for '%s{pattern}' in %s{filePattern} files..."

                let files = Directory.GetFiles(fullPath, filePattern, SearchOption.AllDirectories)
                let results = ResizeArray<string>()

                for file in files do
                    try
                        let content = File.ReadAllText(file)
                        let lines = content.Split('\n')

                        for i = 0 to lines.Length - 1 do
                            if lines.[i].Contains(pattern, StringComparison.OrdinalIgnoreCase) then
                                let relativePath = Path.GetRelativePath(fullPath, file)

                                let linePreview =
                                    let line = lines.[i].Trim()

                                    if line.Length > 80 then
                                        line.Substring(0, 80) + "..."
                                    else
                                        line

                                results.Add $"  %s{relativePath}:%d{i + 1}: %s{linePreview}"

                                if results.Count >= 30 then
                                    ()
                    with _ ->
                        ()

                if results.Count = 0 then
                    return $"No matches found for '%s{pattern}'"
                else
                    let resultList = String.concat "\n" results
                    return $"Found %d{results.Count} matches for '%s{pattern}':\n%s{resultList}"
            with ex ->
                return "search_code error: " + ex.Message
        }

    [<TarsToolAttribute("count_lines",
                        "Counts lines of code in files. Input JSON: { \"path\": \".\", \"pattern\": \"*.fs\" }")>]
    let countLines (args: string) =
        task {
            try
                let doc = System.Text.Json.JsonDocument.Parse(args)
                let root = doc.RootElement

                let mutable pathProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let path =
                    if root.TryGetProperty("path", &pathProp) then
                        pathProp.GetString()
                    else
                        "."

                let mutable patternProp = Unchecked.defaultof<System.Text.Json.JsonElement>

                let pattern =
                    if root.TryGetProperty("pattern", &patternProp) then
                        patternProp.GetString()
                    else
                        "*.fs"

                let fullPath = Path.GetFullPath(path)
                printfn $"Counting lines in %s{pattern} files..."

                let files = Directory.GetFiles(fullPath, pattern, SearchOption.AllDirectories)
                let mutable totalLines = 0
                let mutable totalCodeLines = 0
                let mutable totalFiles = 0

                let fileStats = ResizeArray<string * int * int>()

                for file in files do
                    try
                        let lines = File.ReadAllLines(file)

                        let codeLines =
                            lines
                            |> Array.filter (fun l ->
                                not (String.IsNullOrWhiteSpace(l)) && not (l.Trim().StartsWith("//")))
                            |> Array.length

                        totalLines <- totalLines + lines.Length
                        totalCodeLines <- totalCodeLines + codeLines
                        totalFiles <- totalFiles + 1
                        let relativePath = Path.GetRelativePath(fullPath, file)
                        fileStats.Add((relativePath, lines.Length, codeLines))
                    with _ ->
                        ()

                let topFiles =
                    fileStats
                    |> Seq.sortByDescending (fun (_, _, code) -> code)
                    |> Seq.take (Math.Min(10, fileStats.Count))
                    |> Seq.map (fun (name, total, code) -> $"  %s{name}: %d{total} lines (%d{code} code)")
                    |> String.concat "\n"

                return
                    $"Code Statistics:\n  Total files: %d{totalFiles}\n  Total lines: %d{totalLines}\n  Code lines: %d{totalCodeLines}\n\nTop 10 files by code lines:\n%s{topFiles}"
            with ex ->
                return "count_lines error: " + ex.Message
        }

    [<TarsToolAttribute("find_todos",
                        "Finds all TODO/FIXME/HACK comments in the codebase. Input: path (default: current directory)")>]
    let findTodos (path: string) =
        task {
            try
                let searchPath = if String.IsNullOrWhiteSpace(path) then "." else path
                let fullPath = Path.GetFullPath(searchPath)
                printfn $"Finding TODOs in: %s{fullPath}"

                let files = Directory.GetFiles(fullPath, "*.fs", SearchOption.AllDirectories)
                let results = ResizeArray<string>()

                let patterns = [| "TODO"; "FIXME"; "HACK"; "XXX"; "BUG" |]

                for file in files do
                    try
                        let lines = File.ReadAllLines(file)

                        for i = 0 to lines.Length - 1 do
                            let line = lines.[i]

                            for pattern in patterns do
                                if line.Contains(pattern, StringComparison.OrdinalIgnoreCase) then
                                    let relativePath = Path.GetRelativePath(fullPath, file)
                                    let linePreview = line.Trim()

                                    let preview =
                                        if linePreview.Length > 60 then
                                            linePreview.Substring(0, 60) + "..."
                                        else
                                            linePreview

                                    results.Add $"  [%s{pattern}] %s{relativePath}:%d{i + 1}: %s{preview}"
                    with _ ->
                        ()

                if results.Count = 0 then
                    return "No TODO/FIXME/HACK comments found."
                else
                    return sprintf "Found %d action items:\n%s" results.Count (String.concat "\n" results)
            with ex ->
                return "find_todos error: " + ex.Message
        }
