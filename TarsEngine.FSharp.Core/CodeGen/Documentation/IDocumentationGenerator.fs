namespace TarsEngine.FSharp.Core.CodeGen.Documentation

open System
open System.Threading.Tasks

/// <summary>
/// Represents a documentation generation result.
/// </summary>
type DocumentationGenerationResult = {
    /// <summary>
    /// The generated documentation content.
    /// </summary>
    Content: string
    
    /// <summary>
    /// The format of the documentation.
    /// </summary>
    Format: string
    
    /// <summary>
    /// The path to the generated documentation file, if any.
    /// </summary>
    FilePath: string option
    
    /// <summary>
    /// Additional information about the documentation generation.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Interface for generating documentation.
/// </summary>
type IDocumentationGenerator =
    /// <summary>
    /// Gets the name of the documentation generator.
    /// </summary>
    abstract member Name : string
    
    /// <summary>
    /// Gets the supported documentation formats.
    /// </summary>
    abstract member SupportedFormats : string list
    
    /// <summary>
    /// Generates documentation for code.
    /// </summary>
    /// <param name="code">The code to generate documentation for.</param>
    /// <param name="format">The format of the documentation.</param>
    /// <returns>The documentation generation result.</returns>
    abstract member GenerateDocumentationForCodeAsync : code:string * format:string -> Task<DocumentationGenerationResult>
    
    /// <summary>
    /// Generates documentation for a file.
    /// </summary>
    /// <param name="filePath">The path to the file to generate documentation for.</param>
    /// <param name="format">The format of the documentation.</param>
    /// <returns>The documentation generation result.</returns>
    abstract member GenerateDocumentationForFileAsync : filePath:string * format:string -> Task<DocumentationGenerationResult>
    
    /// <summary>
    /// Generates documentation for a project.
    /// </summary>
    /// <param name="projectPath">The path to the project to generate documentation for.</param>
    /// <param name="format">The format of the documentation.</param>
    /// <param name="outputPath">The path to output the documentation.</param>
    /// <returns>The documentation generation result.</returns>
    abstract member GenerateDocumentationForProjectAsync : projectPath:string * format:string * outputPath:string -> Task<DocumentationGenerationResult>
    
    /// <summary>
    /// Extracts documentation from code.
    /// </summary>
    /// <param name="code">The code to extract documentation from.</param>
    /// <returns>The extracted documentation.</returns>
    abstract member ExtractDocumentationFromCodeAsync : code:string -> Task<string>
    
    /// <summary>
    /// Extracts documentation from a file.
    /// </summary>
    /// <param name="filePath">The path to the file to extract documentation from.</param>
    /// <returns>The extracted documentation.</returns>
    abstract member ExtractDocumentationFromFileAsync : filePath:string -> Task<string>
