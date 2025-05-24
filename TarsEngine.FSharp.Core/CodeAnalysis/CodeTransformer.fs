namespace TarsEngine.FSharp.Core.CodeAnalysis

/// Module for transforming code
module CodeTransformer =
    open System
    open System.IO
    open System.Text.RegularExpressions
    open Types
    
    /// Applies a transformation to a line of code
    let applyTransformationToLine (transformation: Transformation) (line: string) : string =
        let regex = Regex(transformation.Pattern)
        regex.Replace(line, transformation.Replacement)
    
    /// Applies a transformation to a file
    let applyTransformationToFile (transformation: Transformation) (filePath: string) : string =
        try
            let lines = File.ReadAllLines(filePath)
            let transformedLines = 
                lines
                |> Array.map (fun line -> applyTransformationToLine transformation line)
            
            String.Join(Environment.NewLine, transformedLines)
        with
        | ex -> 
            printfn "Error transforming file %s: %s" filePath ex.Message
            ""
    
    /// Applies transformations to a file
    let applyTransformationsToFile (transformations: Transformation list) (filePath: string) : (Transformation * string) list =
        transformations
        |> List.map (fun transformation ->
            let transformedContent = applyTransformationToFile transformation filePath
            (transformation, transformedContent))
    
    /// Applies transformations to files in a directory
    let applyTransformationsToDirectory (config: Configuration) (directoryPath: string) : (string * (Transformation * string) list) list =
        let isExcluded (path: string) =
            config.ExcludeDirectories
            |> List.exists (fun exclude -> path.Contains(exclude))
            || config.ExcludeFiles
               |> List.exists (fun exclude -> Path.GetFileName(path) = exclude)
        
        let isIncluded (path: string) =
            config.FileExtensions
            |> List.exists (fun ext -> Path.GetExtension(path).ToLower() = ext.ToLower())
        
        Directory.GetFiles(directoryPath, "*.*", SearchOption.AllDirectories)
        |> Array.filter (fun path -> not (isExcluded path) && isIncluded path)
        |> Array.map (fun path -> (path, applyTransformationsToFile config.Transformations path))
        |> Array.toList
    
    /// Saves transformed content to a file
    let saveTransformedContent (filePath: string) (content: string) : unit =
        try
            File.WriteAllText(filePath, content)
        with
        | ex -> 
            printfn "Error saving file %s: %s" filePath ex.Message
    
    /// Saves transformed content to files
    let saveTransformedFiles (transformedFiles: (string * (Transformation * string) list) list) : unit =
        transformedFiles
        |> List.iter (fun (filePath, transformations) ->
            transformations
            |> List.iter (fun (transformation, content) ->
                let outputPath = filePath + ".transformed"
                saveTransformedContent outputPath content))
    
    /// Filters transformations by language
    let filterTransformationsByLanguage (language: string) (transformations: Transformation list) : Transformation list =
        transformations
        |> List.filter (fun t -> t.Language = language)
    
    /// Creates a summary of the transformations
    let createTransformationSummary (transformedFiles: (string * (Transformation * string) list) list) : string =
        let transformationGroups =
            transformedFiles
            |> List.collect (fun (filePath, transformations) ->
                transformations
                |> List.map (fun (transformation, _) -> (filePath, transformation)))
            |> List.groupBy (fun (_, transformation) -> transformation.Name)
        
        transformationGroups
        |> List.map (fun (transformationName, fileTransformations) ->
            sprintf "Transformation '%s': %d files" transformationName fileTransformations.Length)
        |> String.concat "\n"
