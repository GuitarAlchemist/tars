namespace TarsEngine.FSharp.Metascript.Services

open System.Threading.Tasks
open TarsEngine.FSharp.Metascript

/// <summary>
/// Interface for metascript services.
/// </summary>
type IMetascriptService =
    /// <summary>
    /// Parses a metascript from text.
    /// </summary>
    /// <param name="text">The text to parse.</param>
    /// <param name="name">The name of the metascript.</param>
    /// <param name="filePath">The file path of the metascript.</param>
    /// <param name="config">The parser configuration.</param>
    /// <returns>The parsed metascript.</returns>
    abstract member ParseMetascriptAsync : text: string * ?name: string * ?filePath: string * ?config: MetascriptParserConfig -> Task<Metascript>
    
    /// <summary>
    /// Parses a metascript from a file.
    /// </summary>
    /// <param name="filePath">The file path to parse.</param>
    /// <param name="config">The parser configuration.</param>
    /// <returns>The parsed metascript.</returns>
    abstract member ParseMetascriptFileAsync : filePath: string * ?config: MetascriptParserConfig -> Task<Metascript>
    
    /// <summary>
    /// Executes a metascript.
    /// </summary>
    /// <param name="metascript">The metascript to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The execution result.</returns>
    abstract member ExecuteMetascriptAsync : metascript: Metascript * ?context: MetascriptContext -> Task<MetascriptExecutionResult>
    
    /// <summary>
    /// Executes a metascript from text.
    /// </summary>
    /// <param name="text">The text to execute.</param>
    /// <param name="name">The name of the metascript.</param>
    /// <param name="filePath">The file path of the metascript.</param>
    /// <param name="context">The execution context.</param>
    /// <param name="config">The parser configuration.</param>
    /// <returns>The execution result.</returns>
    abstract member ExecuteMetascriptTextAsync : text: string * ?name: string * ?filePath: string * ?context: MetascriptContext * ?config: MetascriptParserConfig -> Task<MetascriptExecutionResult>
    
    /// <summary>
    /// Executes a metascript from a file.
    /// </summary>
    /// <param name="filePath">The file path to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <param name="config">The parser configuration.</param>
    /// <returns>The execution result.</returns>
    abstract member ExecuteMetascriptFileAsync : filePath: string * ?context: MetascriptContext * ?config: MetascriptParserConfig -> Task<MetascriptExecutionResult>
    
    /// <summary>
    /// Creates a new metascript context.
    /// </summary>
    /// <param name="workingDirectory">The working directory.</param>
    /// <param name="variables">The initial variables.</param>
    /// <param name="parent">The parent context.</param>
    /// <returns>The new context.</returns>
    abstract member CreateContextAsync : ?workingDirectory: string * ?variables: Map<string, MetascriptVariable> * ?parent: MetascriptContext -> Task<MetascriptContext>
    
    /// <summary>
    /// Gets the default parser configuration.
    /// </summary>
    /// <returns>The default parser configuration.</returns>
    abstract member GetDefaultParserConfig : unit -> MetascriptParserConfig
    
    /// <summary>
    /// Validates a metascript.
    /// </summary>
    /// <param name="metascript">The metascript to validate.</param>
    /// <returns>Whether the metascript is valid and any validation errors.</returns>
    abstract member ValidateMetascriptAsync : metascript: Metascript -> Task<bool * string list>
