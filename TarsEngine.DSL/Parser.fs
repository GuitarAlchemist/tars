namespace TarsEngine.DSL

open FParsec
open Ast

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
    let parse (code: string) =
        let blockRegex = System.Text.RegularExpressions.Regex(@"(CONFIG|PROMPT|ACTION|TASK|AGENT|AUTO_IMPROVE)\s*{([^}]*)}", System.Text.RegularExpressions.RegexOptions.Singleline)
        
        let blocks = 
            blockRegex.Matches(code)
            |> Seq.cast<System.Text.RegularExpressions.Match>
            |> Seq.map (fun m -> 
                let blockType = parseBlockType m.Groups.[1].Value
                let blockContent = m.Groups.[2].Value.Trim()
                
                let properties = parseBlockProperties blockContent blockType
                
                { Type = blockType; Content = blockContent; Properties = properties }
            )
            |> Seq.toList
            
        { Blocks = blocks }
    
    /// Parse a TARS program from a file
    let parseFile (filePath: string) =
        let code = System.IO.File.ReadAllText(filePath)
        parse code
