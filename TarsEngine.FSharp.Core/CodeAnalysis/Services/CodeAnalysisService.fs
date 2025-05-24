namespace TarsEngine.FSharp.Core.CodeAnalysis.Services

open System
open System.IO
open System.Threading.Tasks
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.CodeAnalysis

/// <summary>
/// Implementation of the ICodeAnalysisService interface.
/// </summary>
type CodeAnalysisService(logger: ILogger<CodeAnalysisService>) =
    let languageExtensionMap = 
        Map.ofList [
            (ProgrammingLanguage.CSharp, [".cs"])
            (ProgrammingLanguage.FSharp, [".fs"; ".fsx"; ".fsi"])
            (ProgrammingLanguage.JavaScript, [".js"; ".jsx"])
            (ProgrammingLanguage.TypeScript, [".ts"; ".tsx"])
            (ProgrammingLanguage.Python, [".py"; ".pyw"])
            (ProgrammingLanguage.Java, [".java"])
            (ProgrammingLanguage.Cpp, [".cpp"; ".h"; ".hpp"; ".cc"])
        ]
    
    let extensionLanguageMap =
        languageExtensionMap
        |> Map.toSeq
        |> Seq.collect (fun (lang, exts) -> exts |> List.map (fun ext -> (ext, lang)))
        |> Map.ofSeq
    
    let codeAnalyzer = CodeAnalyzer()
    
    let getLanguageForFile (filePath: string) =
        let extension = Path.GetExtension(filePath).ToLowerInvariant()
        match extensionLanguageMap.TryGetValue(extension) with
        | true, language -> language
        | false, _ -> ProgrammingLanguage.Unknown
    
    let analyzeCodeInternal (code: string) (language: ProgrammingLanguage) (filePath: string option) =
        // Create a new analysis result ID
        let analysisId = Guid.NewGuid().ToString()
        
        // Get the file path
        let filePath = defaultArg filePath "unknown"
        
        // Analyze the code
        let result = 
            match language with
            | ProgrammingLanguage.FSharp ->
                let complexityResult = codeAnalyzer.Analyze(code)
                {
                    Id = analysisId
                    FilePath = filePath
                    Language = language
                    AnalyzedAt = DateTime.UtcNow
                    Issues = []
                    Metrics = []
                    Structures = []
                    IsSuccessful = true
                    Errors = []
                    Metadata = Map.empty
                    ComplexityMetrics = Some complexityResult.ComplexityMetrics
                    Patterns = complexityResult.Patterns
                    Namespaces = []
                    Classes = []
                    Interfaces = []
                    Methods = []
                    Functions = []
                    Types = []
                    Modules = []
                    Imports = []
                    Dependencies = []
                }
            | _ ->
                {
                    Id = analysisId
                    FilePath = filePath
                    Language = language
                    AnalyzedAt = DateTime.UtcNow
                    Issues = []
                    Metrics = []
                    Structures = []
                    IsSuccessful = false
                    Errors = ["Unsupported language: " + language.ToString()]
                    Metadata = Map.empty
                    ComplexityMetrics = None
                    Patterns = []
                    Namespaces = []
                    Classes = []
                    Interfaces = []
                    Methods = []
                    Functions = []
                    Types = []
                    Modules = []
                    Imports = []
                    Dependencies = []
                }
        
        result
    
    /// <summary>
    /// Analyzes code.
    /// </summary>
    /// <param name="code">The code to analyze.</param>
    /// <param name="language">The programming language of the code.</param>
    /// <param name="filePath">The file path of the code.</param>
    /// <returns>The result of analyzing the code.</returns>
    member this.AnalyzeCodeAsync(code: string, language: ProgrammingLanguage, ?filePath: string) =
        task {
            try
                logger.LogInformation("Analyzing code: {Language}", language)
                return analyzeCodeInternal code language filePath
            with
            | ex ->
                logger.LogError(ex, "Error analyzing code: {Language}", language)
                return {
                    Id = Guid.NewGuid().ToString()
                    FilePath = defaultArg filePath "unknown"
                    Language = language
                    AnalyzedAt = DateTime.UtcNow
                    Issues = []
                    Metrics = []
                    Structures = []
                    IsSuccessful = false
                    Errors = [ex.Message]
                    Metadata = Map.empty
                    ComplexityMetrics = None
                    Patterns = []
                    Namespaces = []
                    Classes = []
                    Interfaces = []
                    Methods = []
                    Functions = []
                    Types = []
                    Modules = []
                    Imports = []
                    Dependencies = []
                }
        }
    
    /// <summary>
    /// Analyzes a file.
    /// </summary>
    /// <param name="filePath">The file path to analyze.</param>
    /// <returns>The result of analyzing the file.</returns>
    member this.AnalyzeFileAsync(filePath: string) =
        task {
            try
                logger.LogInformation("Analyzing file: {FilePath}", filePath)
                
                // Check if the file exists
                if not (File.Exists(filePath)) then
                    return {
                        Id = Guid.NewGuid().ToString()
                        FilePath = filePath
                        Language = ProgrammingLanguage.Unknown
                        AnalyzedAt = DateTime.UtcNow
                        Issues = []
                        Metrics = []
                        Structures = []
                        IsSuccessful = false
                        Errors = ["File not found: " + filePath]
                        Metadata = Map.empty
                        ComplexityMetrics = None
                        Patterns = []
                        Namespaces = []
                        Classes = []
                        Interfaces = []
                        Methods = []
                        Functions = []
                        Types = []
                        Modules = []
                        Imports = []
                        Dependencies = []
                    }
                
                // Get the language for the file
                let language = getLanguageForFile filePath
                
                // Read the file
                let! code = File.ReadAllTextAsync(filePath)
                
                // Analyze the code
                return analyzeCodeInternal code language (Some filePath)
            with
            | ex ->
                logger.LogError(ex, "Error analyzing file: {FilePath}", filePath)
                return {
                    Id = Guid.NewGuid().ToString()
                    FilePath = filePath
                    Language = ProgrammingLanguage.Unknown
                    AnalyzedAt = DateTime.UtcNow
                    Issues = []
                    Metrics = []
                    Structures = []
                    IsSuccessful = false
                    Errors = [ex.Message]
                    Metadata = Map.empty
                    ComplexityMetrics = None
                    Patterns = []
                    Namespaces = []
                    Classes = []
                    Interfaces = []
                    Methods = []
                    Functions = []
                    Types = []
                    Modules = []
                    Imports = []
                    Dependencies = []
                }
        }
    
    /// <summary>
    /// Analyzes a directory.
    /// </summary>
    /// <param name="directoryPath">The directory path to analyze.</param>
    /// <param name="recursive">Whether to analyze subdirectories.</param>
    /// <param name="fileExtensions">The file extensions to analyze.</param>
    /// <returns>The results of analyzing the directory.</returns>
    member this.AnalyzeDirectoryAsync(directoryPath: string, ?recursive: bool, ?fileExtensions: string list) =
        task {
            try
                logger.LogInformation("Analyzing directory: {DirectoryPath}", directoryPath)
                
                // Check if the directory exists
                if not (Directory.Exists(directoryPath)) then
                    return []
                
                // Get the search option
                let searchOption = 
                    if defaultArg recursive false then
                        SearchOption.AllDirectories
                    else
                        SearchOption.TopDirectoryOnly
                
                // Get the file extensions
                let fileExtensions = 
                    match fileExtensions with
                    | Some exts -> exts
                    | None -> 
                        languageExtensionMap
                        |> Map.toSeq
                        |> Seq.collect (fun (_, exts) -> exts)
                        |> Seq.toList
                
                // Get the files
                let files = 
                    fileExtensions
                    |> Seq.collect (fun ext -> Directory.GetFiles(directoryPath, "*" + ext, searchOption))
                    |> Seq.toList
                
                // Analyze each file
                let! results = 
                    files
                    |> Seq.map (fun file -> this.AnalyzeFileAsync(file))
                    |> Task.WhenAll
                
                return results |> Array.toList
            with
            | ex ->
                logger.LogError(ex, "Error analyzing directory: {DirectoryPath}", directoryPath)
                return []
        }
    
    /// <summary>
    /// Gets the supported programming languages.
    /// </summary>
    /// <returns>The supported programming languages.</returns>
    member this.GetSupportedLanguages() =
        languageExtensionMap
        |> Map.toSeq
        |> Seq.map fst
        |> Seq.toList
    
    /// <summary>
    /// Gets the programming language for a file extension.
    /// </summary>
    /// <param name="fileExtension">The file extension.</param>
    /// <returns>The programming language for the file extension.</returns>
    member this.GetLanguageForExtension(fileExtension: string) =
        let fileExtension = 
            if fileExtension.StartsWith(".") then
                fileExtension
            else
                "." + fileExtension
        
        match extensionLanguageMap.TryGetValue(fileExtension.ToLowerInvariant()) with
        | true, language -> language
        | false, _ -> ProgrammingLanguage.Unknown
    
    /// <summary>
    /// Gets the file extensions for a programming language.
    /// </summary>
    /// <param name="language">The programming language.</param>
    /// <returns>The file extensions for the programming language.</returns>
    member this.GetExtensionsForLanguage(language: ProgrammingLanguage) =
        match languageExtensionMap.TryGetValue(language) with
        | true, extensions -> extensions
        | false, _ -> []
    
    interface ICodeAnalysisService with
        member this.AnalyzeCodeAsync(code, language, ?filePath) = 
            this.AnalyzeCodeAsync(code, language, ?filePath = filePath)
        
        member this.AnalyzeFileAsync(filePath) = 
            this.AnalyzeFileAsync(filePath)
        
        member this.AnalyzeDirectoryAsync(directoryPath, ?recursive, ?fileExtensions) = 
            this.AnalyzeDirectoryAsync(directoryPath, ?recursive = recursive, ?fileExtensions = fileExtensions)
        
        member this.GetSupportedLanguages() = 
            this.GetSupportedLanguages()
        
        member this.GetLanguageForExtension(fileExtension) = 
            this.GetLanguageForExtension(fileExtension)
        
        member this.GetExtensionsForLanguage(language) = 
            this.GetExtensionsForLanguage(language)
