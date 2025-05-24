namespace TarsEngine.DSL

open System.IO
open Ast
open ParserConfiguration

/// <summary>
/// Module containing a unified parser for the TARS DSL that can use either the original parser or the FParsec-based parser
/// </summary>
module UnifiedParser =
    /// <summary>
    /// Parse a TARS program string into a structured TarsProgram
    /// </summary>
    /// <param name="code">The TARS program string to parse</param>
    /// <param name="config">The parser configuration to use</param>
    /// <returns>The parsed TarsProgram</returns>
    let parse (code: string) (config: ParserConfig option) =
        let config = defaultArg config ParserConfiguration.currentConfig
        
        match config.ParserType with
        | ParserType.Original -> Parser.parse code
        | ParserType.FParsec -> FParsecParser.parse code
    
    /// <summary>
    /// Parse a TARS program from a file
    /// </summary>
    /// <param name="filePath">The path to the file containing the TARS program</param>
    /// <param name="config">The parser configuration to use</param>
    /// <returns>The parsed TarsProgram</returns>
    let parseFile (filePath: string) (config: ParserConfig option) =
        let config = defaultArg config ParserConfiguration.currentConfig
        
        match config.ParserType with
        | ParserType.Original -> Parser.parseFile filePath
        | ParserType.FParsec -> FParsecParser.parseFile filePath
    
    /// <summary>
    /// Parse a TARS program string into a structured TarsProgram using the current configuration
    /// </summary>
    /// <param name="code">The TARS program string to parse</param>
    /// <returns>The parsed TarsProgram</returns>
    let parseWithCurrentConfig (code: string) =
        parse code None
    
    /// <summary>
    /// Parse a TARS program from a file using the current configuration
    /// </summary>
    /// <param name="filePath">The path to the file containing the TARS program</param>
    /// <returns>The parsed TarsProgram</returns>
    let parseFileWithCurrentConfig (filePath: string) =
        parseFile filePath None
