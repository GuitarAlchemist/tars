namespace Tars.Metascript

open System
open System.IO
open System.Collections.Generic
open Tars.Metascript.V1

module V1Parser =

    let parseBlockType (name: string) =
        match name.ToLowerInvariant() with
        | "text" -> MetascriptBlockType.Text
        | "code" -> MetascriptBlockType.Code
        | "fsharp"
        | "f#" -> MetascriptBlockType.FSharp
        | "csharp"
        | "c#" -> MetascriptBlockType.CSharp
        | "python" -> MetascriptBlockType.Python
        | "javascript"
        | "js" -> MetascriptBlockType.JavaScript
        | "sql" -> MetascriptBlockType.SQL
        | "markdown"
        | "md" -> MetascriptBlockType.Markdown
        | "html" -> MetascriptBlockType.HTML
        | "css" -> MetascriptBlockType.CSS
        | "json" -> MetascriptBlockType.JSON
        | "xml" -> MetascriptBlockType.XML
        | "yaml" -> MetascriptBlockType.YAML
        | "command"
        | "cmd"
        | "bash"
        | "sh" -> MetascriptBlockType.Command
        | "query"
        | "ask" -> MetascriptBlockType.Query
        | "transform" -> MetascriptBlockType.Transformation
        | "analyze" -> MetascriptBlockType.Analysis
        | "reflect" -> MetascriptBlockType.Reflection
        | "execute" -> MetascriptBlockType.Execution
        | "import" -> MetascriptBlockType.Import
        | "export" -> MetascriptBlockType.Export
        | "grammar" -> MetascriptBlockType.Grammar
        | "meta" -> MetascriptBlockType.Meta
        | _ -> MetascriptBlockType.Unknown

    let parseParameters (paramString: string) =
        let parameters = List<MetascriptBlockParameter>()

        if String.IsNullOrWhiteSpace(paramString) then
            []
        else
            // Simple space-separated name="value" parser
            let regex = System.Text.RegularExpressions.Regex("(\\w+)=\"([^\"]*)\"")
            let matches = regex.Matches(paramString)

            for m in matches do
                parameters.Add(
                    { Name = m.Groups.[1].Value
                      Value = m.Groups.[2].Value }
                )

            Seq.toList parameters

    let parseMetascript (text: string) (name: string) (filePath: string option) : Metascript =
        let lines = text.Split([| '\r'; '\n' |], StringSplitOptions.None)
        let mutable blocks = []
        let mutable currentBlock = None
        let mutable currentContent = []
        let mutable startLine = 0

        for i in 0 .. lines.Length - 1 do
            let line = lines.[i]
            let trimmed = line.Trim()

            match currentBlock with
            | None ->
                // Look for block start: TYPE { or TYPE(params) { or ```TYPE
                if trimmed.EndsWith("{") then
                    let head = trimmed.Substring(0, trimmed.Length - 1).Trim()
                    let parenIdx = head.IndexOf('(')

                    if parenIdx > 0 && head.EndsWith(")") then
                        let typeName = head.Substring(0, parenIdx).Trim()
                        let paramsPart = head.Substring(parenIdx + 1, head.Length - parenIdx - 2)
                        currentBlock <- Some(parseBlockType typeName, parseParameters paramsPart)
                    else
                        currentBlock <- Some(parseBlockType head, [])

                    startLine <- i + 1
                    currentContent <- []
                elif trimmed.StartsWith("```") then
                    let typeName = trimmed.Substring(3).Trim()
                    currentBlock <- Some(parseBlockType typeName, [])
                    startLine <- i + 1
                    currentContent <- []
            | Some(blockType, parameters) ->
                // Look for block end: } or ```
                if trimmed = "}" || (trimmed = "```" && i > startLine - 1) then
                    let content = String.Join(Environment.NewLine, List.rev currentContent)

                    let block =
                        { Type = blockType
                          Content = content
                          LineNumber = startLine
                          ColumnNumber = 1
                          Parameters = parameters
                          Id = Guid.NewGuid().ToString("N")
                          ParentId = None
                          Metadata = Map.empty }

                    blocks <- block :: blocks
                    currentBlock <- None
                else
                    currentContent <- line :: currentContent

        { Name = name
          Blocks = List.rev blocks
          FilePath = filePath
          Variables = Map.empty
          Metadata = Map.empty }

    /// Serialize a block type to its string representation
    let blockTypeToString (blockType: MetascriptBlockType) =
        match blockType with
        | MetascriptBlockType.Meta -> "meta"
        | MetascriptBlockType.Text -> "text"
        | MetascriptBlockType.Code -> "code"
        | MetascriptBlockType.FSharp -> "fsharp"
        | MetascriptBlockType.CSharp -> "csharp"
        | MetascriptBlockType.Python -> "python"
        | MetascriptBlockType.JavaScript -> "javascript"
        | MetascriptBlockType.SQL -> "sql"
        | MetascriptBlockType.Markdown -> "markdown"
        | MetascriptBlockType.HTML -> "html"
        | MetascriptBlockType.CSS -> "css"
        | MetascriptBlockType.JSON -> "json"
        | MetascriptBlockType.XML -> "xml"
        | MetascriptBlockType.YAML -> "yaml"
        | MetascriptBlockType.Command -> "command"
        | MetascriptBlockType.Query -> "query"
        | MetascriptBlockType.Transformation -> "transform"
        | MetascriptBlockType.Analysis -> "analyze"
        | MetascriptBlockType.Reflection -> "reflect"
        | MetascriptBlockType.Execution -> "execute"
        | MetascriptBlockType.Import -> "import"
        | MetascriptBlockType.Export -> "export"
        | MetascriptBlockType.Grammar -> "grammar"
        | MetascriptBlockType.Unknown -> "unknown"

    /// Serialize parameters to string format
    let parametersToString (parameters: MetascriptBlockParameter list) =
        if List.isEmpty parameters then
            ""
        else
            let paramStr =
                parameters
                |> List.map (fun p -> $"""{p.Name}="{p.Value}" """)
                |> String.concat ""

            $"({paramStr.TrimEnd()})"

    /// Serialize a single block to metascript text format
    let blockToString (block: MetascriptBlock) =
        let typeName = blockTypeToString block.Type
        let paramString = parametersToString block.Parameters
        let header = $"{typeName}{paramString} {{"
        let footer = "}"
        [ header; block.Content; footer ] |> String.concat Environment.NewLine

    /// Serialize a Metascript to text format
    let toMetascript (metascript: Metascript) : string =
        // Add meta block with name if not already present
        let metaBlock =
            $"""meta {{
name = "{metascript.Name}"
}}"""

        let blockTexts = metascript.Blocks |> List.map blockToString
        let allBlocks = metaBlock :: blockTexts
        String.concat (Environment.NewLine + Environment.NewLine) allBlocks
