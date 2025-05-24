namespace TarsEngine.DSL

open FParsec
open Ast
open System

/// Module containing the FParsec-based parser for the TARS DSL
module FParsecParser =
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
        | "CALL" -> BlockType.Call
        | "RETURN" -> BlockType.Return
        | "IMPORT" -> BlockType.Import
        | "EXPORT" -> BlockType.Export
        | _ -> BlockType.Unknown blockType

    // Forward reference for recursive parsers
    let blockParser, blockParserRef = createParserForwardedToRef<TarsBlock, unit>()
    
    // Basic parsers
    let ws = spaces
    let str_ws s = pstring s >>. ws
    let identifier = many1Chars (letter <|> digit <|> pchar '_') .>> ws
    
    // Property value parsers
    let stringLiteral =
        between (pstring "\"") (pstring "\"") (manySatisfy (fun c -> c <> '"'))
        |>> StringValue
        
    let numberLiteral =
        pfloat |>> NumberValue
        
    let boolLiteral =
        (stringReturn "true" (BoolValue true)) <|>
        (stringReturn "false" (BoolValue false))
        
    let listLiteral, listLiteralRef = createParserForwardedToRef<PropertyValue, unit>()
    
    let objectLiteral, objectLiteralRef = createParserForwardedToRef<PropertyValue, unit>()
    
    let propertyValue =
        choice [
            attempt stringLiteral
            attempt numberLiteral
            attempt boolLiteral
            attempt listLiteral
            attempt objectLiteral
        ]
    
    // Define the list literal parser
    do listLiteralRef :=
        between (str_ws "[") (str_ws "]")
            (sepBy propertyValue (str_ws ","))
        |>> ListValue
    
    // Define the object literal parser
    let propertyPair =
        identifier .>> str_ws ":" .>>. propertyValue
        
    do objectLiteralRef :=
        between (str_ws "{") (str_ws "}")
            (sepBy propertyPair (str_ws ","))
        |>> (Map.ofList >> ObjectValue)
    
    // Property parser
    let property =
        identifier .>> str_ws ":" .>>. propertyValue .>> optional (str_ws ",")
        
    // Block parser
    let blockContent =
        let notBlockStart = 
            notFollowedBy (
                many1Chars (letter <|> digit <|> pchar '_') .>> ws .>> 
                optional (many1Chars (letter <|> digit <|> pchar '_') .>> ws) .>> 
                pstring "{"
            )
            
        let notBlockEnd = notFollowedBy (pstring "}")
        
        many (notBlockStart >>. notBlockEnd >>. anyChar)
        |>> (Array.ofList >> String)
    
    // Define the block parser
    do blockParserRef :=
        pipe4
            (many1Chars (letter <|> digit <|> pchar '_') .>> ws)
            (opt (identifier .>> ws))
            (between (pstring "{" .>> ws) (pstring "}" .>> ws)
                (many property .>>. many blockParser))
            (fun blockTypeStr blockName ((properties, nestedBlocks)) ->
                let blockType = parseBlockType blockTypeStr
                {
                    Type = blockType
                    Name = blockName
                    Content = ""  // Content is extracted from properties
                    Properties = Map.ofList properties
                    NestedBlocks = nestedBlocks
                }
            )
    
    // Program parser
    let programParser =
        ws >>. many blockParser .>> eof
        |>> (fun blocks -> { Blocks = blocks })
    
    /// Parse a TARS program string into a structured TarsProgram
    let parse (code: string) =
        match run programParser code with
        | Success(result, _, _) -> result
        | Failure(errorMsg, _, _) -> failwith errorMsg
    
    /// Parse a TARS program from a file
    let parseFile (filePath: string) =
        let code = System.IO.File.ReadAllText(filePath)
        parse code
