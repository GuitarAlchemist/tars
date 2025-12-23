namespace Tars.Tools.Standard

open System
open System.IO
open System.Text.RegularExpressions
open System.Text.Json
open Tars.Tools

/// Tools for deep code analysis, metrics, and understanding
module CodeAnalysisTools =

    /// Calculate cyclomatic complexity for a function
    let private calculateComplexity (code: string) =
        // Count decision points
        let patterns =
            [ @"\bif\b"
              @"\belif\b"
              @"\belse\b"
              @"\bwhile\b"
              @"\bfor\b"
              @"\bcase\b"
              @"\bcatch\b"
              @"\b\?\?"
              @"\b\|\|"
              @"\b&&"
              @"\bmatch\b"
              @"\bwith\b"
              @"\b->\b"
              @"\btry\b" ]

        let count =
            patterns
            |> List.sumBy (fun p -> Regex.Matches(code, p, RegexOptions.IgnoreCase).Count)

        1 + count // Base complexity is 1

    /// Extract function/method definitions from F# code
    let private extractFSharpFunctions (code: string) =
        let pattern = @"let\s+(rec\s+)?(\w+)(\s+\([^)]*\)|\s+\w+)*\s*="

        Regex.Matches(code, pattern)
        |> Seq.cast<Match>
        |> Seq.map (fun m -> m.Groups.[2].Value, m.Index)
        |> Seq.toList

    /// Extract type definitions from F# code
    let private extractFSharpTypes (code: string) =
        let typePattern = @"type\s+(\w+)(?:\s*<[^>]+>)?\s*(?:=|\()"
        let modulePattern = @"module\s+(\w+)\s*="

        let types =
            Regex.Matches(code, typePattern)
            |> Seq.cast<Match>
            |> Seq.map (fun m -> ("type", m.Groups.[1].Value, m.Index))

        let modules =
            Regex.Matches(code, modulePattern)
            |> Seq.cast<Match>
            |> Seq.map (fun m -> ("module", m.Groups.[1].Value, m.Index))

        Seq.append types modules |> Seq.toList

    /// Count lines of code (excluding blanks and comments)
    let private countLoc (code: string) =
        let lines = code.Split([| '\n'; '\r' |], StringSplitOptions.RemoveEmptyEntries)
        let total = lines.Length
        let blank = lines |> Array.filter String.IsNullOrWhiteSpace |> Array.length

        let comment =
            lines
            |> Array.filter (fun l ->
                let trimmed = l.Trim()
                trimmed.StartsWith("//") || trimmed.StartsWith("(*") || trimmed.StartsWith("*"))
            |> Array.length

        let code = total - blank - comment
        (total, blank, comment, code)

    [<TarsToolAttribute("analyze_file_complexity",
                        "Analyzes code complexity metrics for a file. Input JSON: { \"path\": \"file.fs\" }")>]
    let analyzeFileComplexity (args: string) =
        task {
            try
                let path = ToolHelpers.parseStringArg args "path"
                let fullPath = Path.GetFullPath(path)

                if not (File.Exists fullPath) then
                    return $"File not found: {fullPath}"
                else
                    printfn $"📊 Analyzing: {fullPath}"
                    let code = File.ReadAllText(fullPath)

                    let (total, blank, comment, loc) = countLoc code
                    let complexity = calculateComplexity code
                    let functions = extractFSharpFunctions code
                    let types = extractFSharpTypes code

                    let ext = Path.GetExtension(fullPath).ToLower()

                    let lang =
                        match ext with
                        | ".fs"
                        | ".fsx" -> "F#"
                        | ".cs" -> "C#"
                        | ".py" -> "Python"
                        | ".js"
                        | ".ts" -> "JavaScript/TypeScript"
                        | _ -> "Unknown"

                    let complexityStatus =
                        if complexity < 10 then "✅ Low"
                        elif complexity < 20 then "⚠️ Moderate"
                        else "❌ High"

                    let sizeStatus =
                        if loc < 200 then "✅ Small"
                        elif loc < 500 then "⚠️ Medium"
                        else "❌ Large"

                    let commentStatus =
                        if comment > 0 && float comment / float loc > 0.1 then
                            "✅ Well documented"
                        else
                            "⚠️ Needs documentation"

                    let funcList =
                        functions |> List.map (fun (name, _) -> $"- `{name}`") |> String.concat "\n"

                    let typeList =
                        types
                        |> List.map (fun (kind, name, _) -> $"- {kind} `{name}`")
                        |> String.concat "\n"

                    let report =
                        $"# Code Analysis: {Path.GetFileName(fullPath)}\n\n"
                        + $"## Metrics\n"
                        + $"| Metric | Value |\n|--------|-------|\n"
                        + $"| Language | {lang} |\n"
                        + $"| Total Lines | {total} |\n"
                        + $"| Blank Lines | {blank} |\n"
                        + $"| Comment Lines | {comment} |\n"
                        + $"| Lines of Code | {loc} |\n"
                        + $"| Cyclomatic Complexity | {complexity} |\n\n"
                        + $"## Structure\n"
                        + $"- **Functions/Methods**: {functions.Length}\n"
                        + $"- **Types/Modules**: {types.Length}\n\n"
                        + $"### Functions\n{funcList}\n\n"
                        + $"### Types\n{typeList}\n\n"
                        + $"## Health Assessment\n"
                        + $"- Complexity: {complexityStatus}\n"
                        + $"- Size: {sizeStatus}\n"
                        + $"- Comment Ratio: {commentStatus}"

                    return report

            with ex ->
                return $"analyze_file_complexity error: {ex.Message}"
        }

    [<TarsToolAttribute("find_code_smells",
                        "Finds potential code smells in a file. Input JSON: { \"path\": \"file.fs\" }")>]
    let findCodeSmells (args: string) =
        task {
            try
                let path = ToolHelpers.parseStringArg args "path"
                let fullPath = Path.GetFullPath(path)

                if not (File.Exists fullPath) then
                    return $"File not found: {fullPath}"
                else
                    printfn $"🔍 Checking for code smells: {fullPath}"
                    let code = File.ReadAllText(fullPath)
                    let lines = code.Split('\n')

                    let smells = ResizeArray<string>()

                    // Check for long lines
                    lines
                    |> Array.iteri (fun i line ->
                        if line.Length > 120 then
                            smells.Add($"Line {i + 1}: Long line ({line.Length} chars)"))

                    // Check for TODO/FIXME/HACK
                    lines
                    |> Array.iteri (fun i line ->
                        if Regex.IsMatch(line, @"\b(TODO|FIXME|HACK|XXX)\b", RegexOptions.IgnoreCase) then
                            smells.Add($"Line {i + 1}: Contains TODO/FIXME marker"))

                    // Check for magic numbers
                    let magicPattern = @"(?<![.\w])(\d{2,})(?![.\d])"

                    lines
                    |> Array.iteri (fun i line ->
                        if Regex.IsMatch(line, magicPattern) && not (line.Trim().StartsWith("//")) then
                            let numbers =
                                Regex.Matches(line, magicPattern)
                                |> Seq.cast<Match>
                                |> Seq.map (fun m -> m.Value)
                                |> Seq.distinct
                                |> String.concat ", "

                            if not (String.IsNullOrEmpty numbers) then
                                smells.Add($"Line {i + 1}: Magic numbers: {numbers}"))

                    // Check for empty catch blocks
                    if Regex.IsMatch(code, @"catch\s*\([^)]*\)\s*\{\s*\}|with\s*\|\s*_\s*->") then
                        smells.Add("Empty exception handler detected")

                    // Check for deep nesting (simplified)
                    let maxIndent =
                        lines
                        |> Array.filter (fun l -> not (String.IsNullOrWhiteSpace l))
                        |> Array.map (fun l -> l.Length - l.TrimStart().Length)
                        |> Array.max

                    if maxIndent > 20 then
                        smells.Add($"Deep nesting detected (max indent: {maxIndent} spaces)")

                    // Check for commented-out code
                    let commentedCode =
                        lines
                        |> Array.filter (fun l ->
                            let t = l.Trim()

                            t.StartsWith("//")
                            && (t.Contains("let ") || t.Contains("fun ") || t.Contains("type ")))
                        |> Array.length

                    if commentedCode > 3 then
                        smells.Add($"Commented-out code detected ({commentedCode} lines)")

                    if smells.Count = 0 then
                        return $"✅ No code smells detected in {Path.GetFileName(fullPath)}"
                    else
                        let report =
                            $"# Code Smells: {Path.GetFileName(fullPath)}\n\n"
                            + $"Found **{smells.Count}** potential issues:\n\n"
                            + (smells |> Seq.mapi (fun i s -> $"{i + 1}. {s}") |> String.concat "\n")

                        return report
            with ex ->
                return $"find_code_smells error: {ex.Message}"
        }

    [<TarsToolAttribute("extract_symbols",
                        "Extracts all symbols (functions, types, modules) from a code file. Input JSON: { \"path\": \"file.fs\" }")>]
    let extractSymbols (args: string) =
        task {
            try
                let path = ToolHelpers.parseStringArg args "path"
                let fullPath = Path.GetFullPath(path)

                if not (File.Exists fullPath) then
                    return $"File not found: {fullPath}"
                else
                    printfn $"📖 Extracting symbols from: {fullPath}"
                    let code = File.ReadAllText(fullPath)

                    let symbols = ResizeArray<string * string * int>()

                    // Modules
                    for m in Regex.Matches(code, @"^\s*module\s+(\w+)", RegexOptions.Multiline) do
                        symbols.Add("module", m.Groups.[1].Value, m.Index)

                    // Types
                    for m in Regex.Matches(code, @"^\s*type\s+(\w+)", RegexOptions.Multiline) do
                        symbols.Add("type", m.Groups.[1].Value, m.Index)

                    // Functions (let bindings)
                    for m in Regex.Matches(code, @"^\s*let\s+(rec\s+)?(\w+)", RegexOptions.Multiline) do
                        let name = m.Groups.[2].Value

                        if name <> "private" && name <> "mutable" then
                            symbols.Add("function", name, m.Index)

                    // Values (private let bindings are often values)
                    for m in Regex.Matches(code, @"^\s*let\s+private\s+(\w+)", RegexOptions.Multiline) do
                        symbols.Add("value", m.Groups.[1].Value, m.Index)

                    // Attributes
                    for m in Regex.Matches(code, @"\[\<(\w+)", RegexOptions.Multiline) do
                        symbols.Add("attribute", m.Groups.[1].Value, m.Index)

                    let grouped =
                        symbols
                        |> Seq.groupBy (fun (kind, _, _) -> kind)
                        |> Seq.map (fun (kind, items) ->
                            let names =
                                items
                                |> Seq.map (fun (_, name, _) -> $"  - {name}")
                                |> Seq.distinct
                                |> String.concat "\n"

                            $"## {kind}s ({Seq.length items})\n{names}")
                        |> String.concat "\n\n"

                    return $"# Symbols: {Path.GetFileName(fullPath)}\n\n{grouped}"
            with ex ->
                return $"extract_symbols error: {ex.Message}"
        }

    [<TarsToolAttribute("compare_files",
                        "Compares two files and shows differences. Input JSON: { \"file1\": \"path1.fs\", \"file2\": \"path2.fs\" }")>]
    let compareFiles (args: string) =
        task {
            try
                let doc = JsonDocument.Parse(args)
                let root = doc.RootElement
                let file1 = root.GetProperty("file1").GetString()
                let file2 = root.GetProperty("file2").GetString()

                let path1 = Path.GetFullPath(file1)
                let path2 = Path.GetFullPath(file2)

                if not (File.Exists path1) then
                    return $"File not found: {path1}"
                elif not (File.Exists path2) then
                    return $"File not found: {path2}"
                else
                    printfn $"⚖️ Comparing: {file1} vs {file2}"
                    let lines1 = File.ReadAllLines(path1)
                    let lines2 = File.ReadAllLines(path2)

                    let added = lines2 |> Array.except lines1 |> Array.length
                    let removed = lines1 |> Array.except lines2 |> Array.length

                    let common =
                        lines1 |> Array.filter (fun l -> lines2 |> Array.contains l) |> Array.length

                    let (_, _, _, loc1) = countLoc (File.ReadAllText(path1))
                    let (_, _, _, loc2) = countLoc (File.ReadAllText(path2))

                    let netChange = if lines2.Length > lines1.Length then "+" else ""

                    let report =
                        $"# File Comparison\n\n"
                        + $"| Metric | {Path.GetFileName(path1)} | {Path.GetFileName(path2)} |\n"
                        + $"|--------|---------------------------|---------------------------|\n"
                        + $"| Total Lines | {lines1.Length} | {lines2.Length} |\n"
                        + $"| Lines of Code | {loc1} | {loc2} |\n\n"
                        + $"## Changes\n"
                        + $"- Lines added: {added}\n"
                        + $"- Lines removed: {removed}\n"
                        + $"- Common lines: {common}\n"
                        + $"- Net change: {netChange}{lines2.Length - lines1.Length}"

                    return report

            with ex ->
                return $"compare_files error: {ex.Message}"
        }

    [<TarsToolAttribute("find_duplicates",
                        "Finds duplicate or similar code blocks in a directory. Input JSON: { \"path\": \"src/\", \"min_lines\": 5 }")>]
    let findDuplicates (args: string) =
        task {
            try
                let path = ToolHelpers.parseStringArg args "path"

                let minLines =
                    try
                        let doc = JsonDocument.Parse(args)
                        let mutable prop = Unchecked.defaultof<JsonElement>

                        if doc.RootElement.TryGetProperty("min_lines", &prop) then
                            prop.GetInt32()
                        else
                            5
                    with _ ->
                        5

                let fullPath = Path.GetFullPath(path)

                if not (Directory.Exists fullPath) then
                    return $"Directory not found: {fullPath}"
                else
                    printfn $"🔎 Searching for duplicates in: {fullPath}"

                    let files =
                        Directory.GetFiles(fullPath, "*.fs", SearchOption.AllDirectories)
                        |> Array.truncate 50 // Limit for performance

                    // Extract code blocks
                    let blocks = ResizeArray<string * string * int>() // file, block, lineNum

                    for file in files do
                        let lines = File.ReadAllLines(file)

                        for i in 0 .. lines.Length - minLines do
                            let block =
                                lines.[i .. min (i + minLines - 1) (lines.Length - 1)] |> String.concat "\n"

                            if not (String.IsNullOrWhiteSpace block) && block.Trim().Length > 50 then
                                blocks.Add(file, block.Trim(), i + 1)

                    // Find duplicates (simplified - exact match)
                    let duplicates =
                        blocks
                        |> Seq.groupBy (fun (_, block, _) -> block)
                        |> Seq.filter (fun (_, group) -> Seq.length group > 1)
                        |> Seq.map (fun (block, group) ->
                            let locations =
                                group
                                |> Seq.map (fun (file, _, line) ->
                                    $"  - {Path.GetRelativePath(fullPath, file)}:{line}")
                                |> String.concat "\n"

                            let preview =
                                if block.Length > 100 then
                                    block.Substring(0, 100) + "..."
                                else
                                    block

                            $"### Duplicate ({Seq.length group} occurrences)\n```\n{preview}\n```\nLocations:\n{locations}")
                        |> Seq.truncate 10
                        |> String.concat "\n\n"

                    if String.IsNullOrEmpty duplicates then
                        return $"✅ No duplicate code blocks found (checked {files.Length} files)"
                    else
                        return $"# Duplicate Code Analysis\n\n{duplicates}"
            with ex ->
                return $"find_duplicates error: {ex.Message}"
        }
