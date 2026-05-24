namespace TarsEngine.FSharp.Core.Metascript

open System
open System.IO
open System.Text.RegularExpressions

/// Representation of a metascript specification authored in Markdown.
type MetascriptSpec =
    { Id: string
      Title: string option
      Grammar: string
      Expectations: Map<string, string>
      SourcePath: string }

module MetascriptSpecLoader =

    let private grammarPattern = Regex("```(?:metascript|dsl|trsx)\\s*\\r?\\n([\\s\\S]*?)```", RegexOptions.IgnoreCase ||| RegexOptions.Compiled)
    let private expectationsPattern = Regex("```expectations\\s*\\r?\\n([\\s\\S]*?)```", RegexOptions.IgnoreCase ||| RegexOptions.Compiled)
    let private idPattern = Regex("^id:\\s*(.+)$", RegexOptions.Multiline ||| RegexOptions.Compiled)
    let private titlePattern = Regex("^#\\s*(.+)$", RegexOptions.Multiline ||| RegexOptions.Compiled)

    let private toAbsolutePath (path: string) =
        if Path.IsPathRooted(path) then path else Path.GetFullPath(Path.Combine(Directory.GetCurrentDirectory(), path))

    let private parseExpectations (block: string) =
        block.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        |> Array.choose (fun line ->
            let trimmed = line.Trim()
            if String.IsNullOrWhiteSpace(trimmed) || trimmed.StartsWith("#") then
                None
            else
                let parts = trimmed.Split([|'='|], 2)
                if parts.Length = 2 then
                    Some(parts.[0].Trim(), parts.[1].Trim())
                else
                    None)
        |> Map.ofArray

    let loadFromFile (path: string) =
        let fullPath = toAbsolutePath path
        if not (File.Exists(fullPath)) then
            invalidArg "path" $"Metascript spec not found: %s{fullPath}"

        let markdown = File.ReadAllText(fullPath)

        let grammarMatch = grammarPattern.Match(markdown)
        if not grammarMatch.Success then
            invalidOp $"No metascript code block found in %s{fullPath}"

        let id =
            let m = idPattern.Match(markdown)
            if m.Success then m.Groups.[1].Value.Trim() else Path.GetFileNameWithoutExtension(fullPath)

        let title =
            let m = titlePattern.Match(markdown)
            if m.Success then Some(m.Groups.[1].Value.Trim()) else None

        let expectations =
            let m = expectationsPattern.Match(markdown)
            if m.Success then parseExpectations m.Groups.[1].Value else Map.empty

        { Id = id
          Title = title
          Grammar = grammarMatch.Groups.[1].Value.Trim()
          Expectations = expectations
          SourcePath = fullPath }
