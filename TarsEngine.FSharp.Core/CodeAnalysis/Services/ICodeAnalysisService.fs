namespace TarsEngine.FSharp.Core.CodeAnalysis.Services

open System.Threading.Tasks
open TarsEngine.FSharp.Core.CodeAnalysis

/// <summary>
/// Interface for code analysis services.
/// </summary>
type ICodeAnalysisService =
    /// <summary>
    /// Analyzes code.
    /// </summary>
    /// <param name="code">The code to analyze.</param>
    /// <param name="language">The programming language of the code.</param>
    /// <param name="filePath">The file path of the code.</param>
    /// <returns>The result of analyzing the code.</returns>
    abstract member AnalyzeCodeAsync : code: string * language: ProgrammingLanguage * ?filePath: string -> Task<CodeAnalysisResult>
    
    /// <summary>
    /// Analyzes a file.
    /// </summary>
    /// <param name="filePath">The file path to analyze.</param>
    /// <returns>The result of analyzing the file.</returns>
    abstract member AnalyzeFileAsync : filePath: string -> Task<CodeAnalysisResult>
    
    /// <summary>
    /// Analyzes a directory.
    /// </summary>
    /// <param name="directoryPath">The directory path to analyze.</param>
    /// <param name="recursive">Whether to analyze subdirectories.</param>
    /// <param name="fileExtensions">The file extensions to analyze.</param>
    /// <returns>The results of analyzing the directory.</returns>
    abstract member AnalyzeDirectoryAsync : directoryPath: string * ?recursive: bool * ?fileExtensions: string list -> Task<CodeAnalysisResult list>
    
    /// <summary>
    /// Gets the supported programming languages.
    /// </summary>
    /// <returns>The supported programming languages.</returns>
    abstract member GetSupportedLanguages : unit -> ProgrammingLanguage list
    
    /// <summary>
    /// Gets the programming language for a file extension.
    /// </summary>
    /// <param name="fileExtension">The file extension.</param>
    /// <returns>The programming language for the file extension.</returns>
    abstract member GetLanguageForExtension : fileExtension: string -> ProgrammingLanguage
    
    /// <summary>
    /// Gets the file extensions for a programming language.
    /// </summary>
    /// <param name="language">The programming language.</param>
    /// <returns>The file extensions for the programming language.</returns>
    abstract member GetExtensionsForLanguage : language: ProgrammingLanguage -> string list
