namespace TarsEngine.FSharp.Core.CodeGen.Documentation

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Text.RegularExpressions
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.CodeGen.Testing

/// <summary>
/// Documentation generator for Markdown format.
/// </summary>
type MarkdownDocumentationGenerator(logger: ILogger<MarkdownDocumentationGenerator>, codeAnalyzer: TestCodeAnalyzer) =
    inherit DocumentationGeneratorBase(logger :> ILogger)
    
    /// <summary>
    /// Gets the name of the documentation generator.
    /// </summary>
    override _.Name = "Markdown"
    
    /// <summary>
    /// Gets the supported documentation formats.
    /// </summary>
    override _.SupportedFormats = ["markdown"; "md"]
    
    /// <summary>
    /// Generates documentation for code.
    /// </summary>
    /// <param name="code">The code to generate documentation for.</param>
    /// <param name="format">The format of the documentation.</param>
    /// <returns>The documentation generation result.</returns>
    override _.GenerateDocumentationForCodeAsync(code: string, format: string) =
        task {
            try
                logger.LogInformation("Generating {Format} documentation for code", format)
                
                // Determine the language
                let language = 
                    if code.Contains("namespace") && code.Contains("using") then
                        "csharp"
                    elif code.Contains("namespace") && code.Contains("open") then
                        "fsharp"
                    else
                        "unknown"
                
                // Extract methods and classes
                let methods, classes = 
                    match language with
                    | "csharp" -> codeAnalyzer.ExtractMethodsFromCSharp(code), codeAnalyzer.ExtractClassesFromCSharp(code)
                    | "fsharp" -> codeAnalyzer.ExtractMethodsFromFSharp(code), codeAnalyzer.ExtractClassesFromFSharp(code)
                    | _ -> [], []
                
                // Generate documentation
                let sb = StringBuilder()
                
                // Add title
                sb.AppendLine("# Code Documentation") |> ignore
                sb.AppendLine() |> ignore
                
                // Add language
                sb.AppendLine($"**Language:** {language}") |> ignore
                sb.AppendLine() |> ignore
                
                // Add classes
                if not (List.isEmpty classes) then
                    sb.AppendLine("## Classes") |> ignore
                    sb.AppendLine() |> ignore
                    
                    for class' in classes do
                        sb.AppendLine($"### {class'.Name}") |> ignore
                        sb.AppendLine() |> ignore
                        
                        // Add namespace
                        sb.AppendLine($"**Namespace:** {class'.Namespace}") |> ignore
                        sb.AppendLine() |> ignore
                        
                        // Add base class
                        match class'.BaseClass with
                        | Some baseClass -> 
                            sb.AppendLine($"**Base Class:** {baseClass}") |> ignore
                            sb.AppendLine() |> ignore
                        | None -> ()
                        
                        // Add interfaces
                        if not (List.isEmpty class'.Interfaces) then
                            sb.AppendLine("**Interfaces:**") |> ignore
                            sb.AppendLine() |> ignore
                            
                            for interface' in class'.Interfaces do
                                sb.AppendLine($"- {interface'}") |> ignore
                            
                            sb.AppendLine() |> ignore
                        
                        // Add properties
                        if not (List.isEmpty class'.Properties) then
                            sb.AppendLine("#### Properties") |> ignore
                            sb.AppendLine() |> ignore
                            
                            sb.AppendLine("| Name | Type |") |> ignore
                            sb.AppendLine("|------|------|") |> ignore
                            
                            for name, type' in class'.Properties do
                                sb.AppendLine($"| {name} | {type'} |") |> ignore
                            
                            sb.AppendLine() |> ignore
                        
                        // Add methods
                        let classMethods = methods |> List.filter (fun m -> m.ClassName = class'.Name)
                        
                        if not (List.isEmpty classMethods) then
                            sb.AppendLine("#### Methods") |> ignore
                            sb.AppendLine() |> ignore
                            
                            for method in classMethods do
                                sb.AppendLine($"##### {method.Name}") |> ignore
                                sb.AppendLine() |> ignore
                                
                                // Add return type
                                sb.AppendLine($"**Returns:** {method.ReturnType}") |> ignore
                                sb.AppendLine() |> ignore
                                
                                // Add parameters
                                if not (List.isEmpty method.Parameters) then
                                    sb.AppendLine("**Parameters:**") |> ignore
                                    sb.AppendLine() |> ignore
                                    
                                    sb.AppendLine("| Name | Type |") |> ignore
                                    sb.AppendLine("|------|------|") |> ignore
                                    
                                    for name, type' in method.Parameters do
                                        sb.AppendLine($"| {name} | {type'} |") |> ignore
                                    
                                    sb.AppendLine() |> ignore
                
                // Add methods that are not part of a class
                let standaloneMethods = methods |> List.filter (fun m -> not (classes |> List.exists (fun c -> c.Name = m.ClassName)))
                
                if not (List.isEmpty standaloneMethods) then
                    sb.AppendLine("## Methods") |> ignore
                    sb.AppendLine() |> ignore
                    
                    for method in standaloneMethods do
                        sb.AppendLine($"### {method.Name}") |> ignore
                        sb.AppendLine() |> ignore
                        
                        // Add namespace
                        sb.AppendLine($"**Namespace:** {method.Namespace}") |> ignore
                        sb.AppendLine() |> ignore
                        
                        // Add return type
                        sb.AppendLine($"**Returns:** {method.ReturnType}") |> ignore
                        sb.AppendLine() |> ignore
                        
                        // Add parameters
                        if not (List.isEmpty method.Parameters) then
                            sb.AppendLine("**Parameters:**") |> ignore
                            sb.AppendLine() |> ignore
                            
                            sb.AppendLine("| Name | Type |") |> ignore
                            sb.AppendLine("|------|------|") |> ignore
                            
                            for name, type' in method.Parameters do
                                sb.AppendLine($"| {name} | {type'} |") |> ignore
                            
                            sb.AppendLine() |> ignore
                
                return {
                    Content = sb.ToString()
                    Format = format
                    FilePath = None
                    AdditionalInfo = Map.empty
                }
            with
            | ex ->
                logger.LogError(ex, "Error generating {Format} documentation for code", format)
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
    override this.GenerateDocumentationForProjectAsync(projectPath: string, format: string, outputPath: string) =
        task {
            try
                logger.LogInformation("Generating {Format} documentation for project: {ProjectPath}", format, projectPath)
                
                // Create the output directory if it doesn't exist
                Directory.CreateDirectory(outputPath) |> ignore
                
                // Find all source files
                let sourceFiles = 
                    [
                        Directory.GetFiles(projectPath, "*.cs", SearchOption.AllDirectories)
                        Directory.GetFiles(projectPath, "*.fs", SearchOption.AllDirectories)
                    ]
                    |> Array.concat
                
                // Generate documentation for each file
                let fileResults = ResizeArray<DocumentationGenerationResult>()
                
                for file in sourceFiles do
                    let! result = this.GenerateDocumentationForFileAsync(file, format)
                    
                    // Save the documentation to a file
                    let relativePath = file.Substring(projectPath.Length).TrimStart(Path.DirectorySeparatorChar)
                    let outputFilePath = Path.Combine(outputPath, relativePath + ".md")
                    
                    // Create the directory if it doesn't exist
                    Directory.CreateDirectory(Path.GetDirectoryName(outputFilePath)) |> ignore
                    
                    // Write the documentation to the file
                    File.WriteAllText(outputFilePath, result.Content)
                    
                    // Add the result with the file path
                    fileResults.Add({ result with FilePath = Some outputFilePath })
                
                // Generate an index file
                let indexSb = StringBuilder()
                
                indexSb.AppendLine("# Project Documentation") |> ignore
                indexSb.AppendLine() |> ignore
                
                indexSb.AppendLine($"**Project:** {Path.GetFileName(projectPath)}") |> ignore
                indexSb.AppendLine() |> ignore
                
                indexSb.AppendLine("## Files") |> ignore
                indexSb.AppendLine() |> ignore
                
                for result in fileResults do
                    match result.FilePath with
                    | Some filePath ->
                        let relativePath = filePath.Substring(outputPath.Length).TrimStart(Path.DirectorySeparatorChar)
                        indexSb.AppendLine($"- [{relativePath}]({relativePath.Replace("\\", "/")})") |> ignore
                    | None -> ()
                
                // Write the index file
                let indexFilePath = Path.Combine(outputPath, "index.md")
                File.WriteAllText(indexFilePath, indexSb.ToString())
                
                return {
                    Content = indexSb.ToString()
                    Format = format
                    FilePath = Some indexFilePath
                    AdditionalInfo = Map.ofList [
                        "FileCount", sourceFiles.Length.ToString()
                    ]
                }
            with
            | ex ->
                logger.LogError(ex, "Error generating {Format} documentation for project: {ProjectPath}", format, projectPath)
                return {
                    Content = $"Error generating documentation: {ex.Message}"
                    Format = format
                    FilePath = None
                    AdditionalInfo = Map.empty
                }
        }
