namespace TarsEngine.DSL

open FParsec
open Ast
open System.Text.RegularExpressions

/// Module containing the parser for the TARS DSL
module Parser =
    /// Parse a string into a BlockType
    let parseBlockType (blockType: string) =
        match blockType.ToUpper() with
        | "CONFIG" -> BlockType.Config
        | "PROMPT" -> BlockType.Prompt
        | "ACTION" -> BlockType.Action
        | "TASK" -> BlockType.Task
        | "AGENT" -> BlockType.Agent
        | "AUTO_IMPROVE" -> BlockType.AutoImprove
        | "DESCRIBE" -> BlockType.Describe
        | "SPAWN_AGENT" -> BlockType.SpawnAgent
        | "MESSAGE" -> BlockType.Message
        | "SELF_IMPROVE" -> BlockType.SelfImprove
        | "TARS" -> BlockType.Tars
        | "COMMUNICATION" -> BlockType.Communication
        | "VARIABLE" -> BlockType.Variable
        | "IF" -> BlockType.If
        | "ELSE" -> BlockType.Else
        | "FOR" -> BlockType.For
        | "WHILE" -> BlockType.While
        | "FUNCTION" -> BlockType.Function
        | "RETURN" -> BlockType.Return
        | "IMPORT" -> BlockType.Import
        | "EXPORT" -> BlockType.Export
        | _ -> BlockType.Unknown blockType

    /// Parse properties from a block content string
    let parseBlockProperties (content: string) (blockType: BlockType) =
        // Simple property parser for demonstration
        // In a real implementation, this would be more sophisticated
        let properties = Map.empty<string, PropertyValue>

        // Split the content by lines and parse each line as a property
        let lines = content.Split('\n')

        let parseProperty (line: string) =
            let parts = line.Split(':')
            if parts.Length >= 2 then
                let key = parts.[0].Trim()
                let value = parts.[1].Trim()

                // Try to parse as different types
                let propertyValue =
                    match value with
                    | v when v.StartsWith("\"") && v.EndsWith("\"") ->
                        // String value
                        StringValue(v.Substring(1, v.Length - 2))
                    | v when System.Boolean.TryParse(v, ref false) ->
                        // Boolean value
                        BoolValue(System.Boolean.Parse(v))
                    | v when System.Double.TryParse(v, ref 0.0) ->
                        // Number value
                        NumberValue(System.Double.Parse(v))
                    | _ ->
                        // Default to string value
                        StringValue(value)

                Some(key, propertyValue)
            else
                None

        // Parse each line and add to properties
        let mutable result = properties
        for line in lines do
            match parseProperty line with
            | Some(key, value) -> result <- result.Add(key, value)
            | None -> ()

        result

    /// Parse a TARS program string into a structured TarsProgram
    let rec parse (code: string) =
        // Enhanced regex to capture block type, optional name, and content
        let blockRegex = Regex(@"(CONFIG|PROMPT|ACTION|TASK|AGENT|AUTO_IMPROVE|DESCRIBE|SPAWN_AGENT|MESSAGE|SELF_IMPROVE|TARS|COMMUNICATION|VARIABLE|IF|ELSE|FOR|WHILE|FUNCTION|RETURN|IMPORT|EXPORT)\s+(?:([a-zA-Z0-9_]+)\s+)?{([^{}]*(?:{[^{}]*}[^{}]*)*)}", RegexOptions.Singleline)

        let rec parseNestedBlocks (content: string) =
            // Find all nested blocks in the content
            let nestedBlockRegex = Regex(@"(CONFIG|PROMPT|ACTION|TASK|AGENT|AUTO_IMPROVE|DESCRIBE|SPAWN_AGENT|MESSAGE|SELF_IMPROVE|TARS|COMMUNICATION|VARIABLE|IF|ELSE|FOR|WHILE|FUNCTION|RETURN|IMPORT|EXPORT)\s+(?:([a-zA-Z0-9_]+)\s+)?{([^{}]*(?:{[^{}]*}[^{}]*)*)}", RegexOptions.Singleline)

            let nestedBlocks =
                nestedBlockRegex.Matches(content)
                |> Seq.cast<Match>
                |> Seq.map (fun m ->
                    let blockType = parseBlockType m.Groups.[1].Value
                    let blockName = if m.Groups.[2].Success then Some(m.Groups.[2].Value) else None
                    let blockContent = m.Groups.[3].Value.Trim()

                    // Parse nested blocks recursively
                    let nestedBlocks = parseNestedBlocks blockContent

                    // Remove nested blocks from content for property parsing
                    let contentWithoutNestedBlocks = nestedBlockRegex.Replace(blockContent, "")

                    let properties = parseBlockProperties contentWithoutNestedBlocks blockType

                    {
                        Type = blockType
                        Name = blockName
                        Content = blockContent
                        Properties = properties
                        NestedBlocks = nestedBlocks
                    }
                )
                |> Seq.toList

            nestedBlocks

        let blocks =
            blockRegex.Matches(code)
            |> Seq.cast<Match>
            |> Seq.map (fun m ->
                let blockType = parseBlockType m.Groups.[1].Value
                let blockName = if m.Groups.[2].Success then Some(m.Groups.[2].Value) else None
                let blockContent = m.Groups.[3].Value.Trim()

                // Parse nested blocks
                let nestedBlocks = parseNestedBlocks blockContent

                // Remove nested blocks from content for property parsing
                let contentWithoutNestedBlocks =
                    nestedBlocks
                    |> List.fold (fun (content: string) block ->
                        let blockRegex = Regex(sprintf "%A\s+(?:%s\s+)?{[^{}]*(?:{[^{}]*}[^{}]*)*}"
                                                block.Type
                                                (match block.Name with Some name -> name | None -> ""))
                        blockRegex.Replace(content, "")
                    ) blockContent

                let properties = parseBlockProperties contentWithoutNestedBlocks blockType

                {
                    Type = blockType
                    Name = blockName
                    Content = blockContent
                    Properties = properties
                    NestedBlocks = nestedBlocks
                }
            )
            |> Seq.toList

        { Blocks = blocks }

    /// Parse a TARS program from a file
    let parseFile (filePath: string) =
        let code = System.IO.File.ReadAllText(filePath)
        parse code
