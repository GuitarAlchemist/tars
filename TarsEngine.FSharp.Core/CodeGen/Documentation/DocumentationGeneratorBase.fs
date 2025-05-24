namespace TarsEngine.FSharp.Core.CodeGen.Documentation

open System
open System.IO
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Base class for documentation generators.
/// </summary>
[<AbstractClass>]
type DocumentationGeneratorBase(logger: ILogger) =
    
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
    member this.GenerateDocumentationForFileAsync(filePath: string, format: string) =
        task {
            try
                logger.LogInformation("Generating documentation for file: {FilePath}", filePath)
                
                // Read the file content
                let code = File.ReadAllText(filePath)
                
                // Generate documentation for the code
                return! this.GenerateDocumentationForCodeAsync(code, format)
            with
            | ex ->
                logger.LogError(ex, "Error generating documentation for file: {FilePath}", filePath)
                return {
                    Content = $"Error generating documentation: {ex.Message}"
                    Format = format
                    FilePath = None
                    AdditionalInfo = Map.empty
                }
        }
    
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
    member _.ExtractDocumentationFromCodeAsync(code: string) =
        task {
            try
                // Extract XML documentation comments
                let xmlCommentPattern = @"///\s*<summary>(.*?)</summary>"
                let xmlCommentMatches = Regex.Matches(code, xmlCommentPattern, RegexOptions.Singleline)
                
                let xmlComments = 
                    xmlCommentMatches
                    |> Seq.cast<Match>
                    |> Seq.map (fun m -> m.Groups.[1].Value.Trim())
                    |> Seq.toList
                
                // Extract triple-slash comments
                let tripleSlashPattern = @"///\s*(.*)"
                let tripleSlashMatches = Regex.Matches(code, tripleSlashPattern)
                
                let tripleSlashComments = 
                    tripleSlashMatches
                    |> Seq.cast<Match>
                    |> Seq.map (fun m -> m.Groups.[1].Value.Trim())
                    |> Seq.filter (fun c -> not (c.StartsWith("<") && c.EndsWith(">")))
                    |> Seq.toList
                
                // Extract XML doc comments
                let xmlDocPattern = @"<doc>(.*?)</doc>"
                let xmlDocMatches = Regex.Matches(code, xmlDocPattern, RegexOptions.Singleline)
                
                let xmlDocComments = 
                    xmlDocMatches
                    |> Seq.cast<Match>
                    |> Seq.map (fun m -> m.Groups.[1].Value.Trim())
                    |> Seq.toList
                
                // Extract F# doc comments
                let fsharpDocPattern = @"\(\*\*(.*?)\*\)"
                let fsharpDocMatches = Regex.Matches(code, fsharpDocPattern, RegexOptions.Singleline)
                
                let fsharpDocComments = 
                    fsharpDocMatches
                    |> Seq.cast<Match>
                    |> Seq.map (fun m -> m.Groups.[1].Value.Trim())
                    |> Seq.toList
                
                // Combine all comments
                let allComments = 
                    List.concat [
                        xmlComments
                        tripleSlashComments
                        xmlDocComments
                        fsharpDocComments
                    ]
                
                return String.Join(Environment.NewLine + Environment.NewLine, allComments)
            with
            | ex ->
                logger.LogError(ex, "Error extracting documentation from code")
                return ""
        }
    
    /// <summary>
    /// Extracts documentation from a file.
    /// </summary>
    /// <param name="filePath">The path to the file to extract documentation from.</param>
    /// <returns>The extracted documentation.</returns>
    member this.ExtractDocumentationFromFileAsync(filePath: string) =
        task {
            try
                logger.LogInformation("Extracting documentation from file: {FilePath}", filePath)
                
                // Read the file content
                let code = File.ReadAllText(filePath)
                
                // Extract documentation from the code
                return! this.ExtractDocumentationFromCodeAsync(code)
            with
            | ex ->
                logger.LogError(ex, "Error extracting documentation from file: {FilePath}", filePath)
                return ""
        }
    
    interface IDocumentationGenerator with
        member this.Name = this.Name
        member this.SupportedFormats = this.SupportedFormats
        member this.GenerateDocumentationForCodeAsync(code, format) = this.GenerateDocumentationForCodeAsync(code, format)
        member this.GenerateDocumentationForFileAsync(filePath, format) = this.GenerateDocumentationForFileAsync(filePath, format)
        member this.GenerateDocumentationForProjectAsync(projectPath, format, outputPath) = this.GenerateDocumentationForProjectAsync(projectPath, format, outputPath)
        member this.ExtractDocumentationFromCodeAsync(code) = this.ExtractDocumentationFromCodeAsync(code)
        member this.ExtractDocumentationFromFileAsync(filePath) = this.ExtractDocumentationFromFileAsync(filePath)
