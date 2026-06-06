namespace TarsEngine.DSL

/// <summary>
/// Module containing configuration options for the TARS DSL parser
/// </summary>
module ParserConfiguration =
    /// <summary>
    /// The type of parser to use
    /// </summary>
    type ParserType =
        | Original
        | FParsec
    
    /// <summary>
    /// Configuration options for the TARS DSL parser
    /// </summary>
    type ParserConfig = {
        /// <summary>
        /// The type of parser to use
        /// </summary>
        ParserType: ParserType
        
        /// <summary>
        /// Whether to resolve imports and includes
        /// </summary>
        ResolveImportsAndIncludes: bool
        
        /// <summary>
        /// Whether to validate the parsed program
        /// </summary>
        ValidateProgram: bool
        
        /// <summary>
        /// Whether to optimize the parsed program
        /// </summary>
        OptimizeProgram: bool
    }
    
    /// <summary>
    /// Default configuration options for the TARS DSL parser
    /// </summary>
    let defaultConfig = {
        ParserType = ParserType.Original
        ResolveImportsAndIncludes = true
        ValidateProgram = true
        OptimizeProgram = false
    }
    
    /// <summary>
    /// Configuration options for the TARS DSL parser using the FParsec parser
    /// </summary>
    let fparsecConfig = {
        defaultConfig with
            ParserType = ParserType.FParsec
    }
    
    /// <summary>
    /// Current configuration options for the TARS DSL parser
    /// </summary>
    let mutable currentConfig = defaultConfig
    
    /// <summary>
    /// Set the parser type to use
    /// </summary>
    /// <param name="parserType">The parser type to use</param>
    let setParserType (parserType: ParserType) =
        currentConfig <- { currentConfig with ParserType = parserType }
    
    /// <summary>
    /// Set whether to resolve imports and includes
    /// </summary>
    /// <param name="resolveImportsAndIncludes">Whether to resolve imports and includes</param>
    let setResolveImportsAndIncludes (resolveImportsAndIncludes: bool) =
        currentConfig <- { currentConfig with ResolveImportsAndIncludes = resolveImportsAndIncludes }
    
    /// <summary>
    /// Set whether to validate the parsed program
    /// </summary>
    /// <param name="validateProgram">Whether to validate the parsed program</param>
    let setValidateProgram (validateProgram: bool) =
        currentConfig <- { currentConfig with ValidateProgram = validateProgram }
    
    /// <summary>
    /// Set whether to optimize the parsed program
    /// </summary>
    /// <param name="optimizeProgram">Whether to optimize the parsed program</param>
    let setOptimizeProgram (optimizeProgram: bool) =
        currentConfig <- { currentConfig with OptimizeProgram = optimizeProgram }
