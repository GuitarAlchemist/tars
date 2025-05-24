namespace TarsEngine.FSharp.Core.Compilation

open System
open System.Collections.Generic
open System.IO
open System.Reflection
open System.Text
open System.Threading.Tasks
open Microsoft.CodeAnalysis.Scripting
open Microsoft.Extensions.Logging

/// <summary>
/// Adapter that implements the C# IFSharpCompiler interface using the F# FSharpCompiler.
/// This provides compatibility with existing C# code while using the F# implementation.
/// </summary>
type FSharpCompilerAdapter(logger: ILogger<FSharpCompilerAdapter>) =
    let fsharpCompiler = FSharpCompiler()
    let tempDirectory = Path.Combine(Path.GetTempPath(), "TarsEngine", "FSharpCompiler")
    
    do
        // Ensure the temporary directory exists
        if not (Directory.Exists(tempDirectory)) then
            Directory.CreateDirectory(tempDirectory) |> ignore
    
    /// <summary>
    /// Converts a C# CompilationResult to an F# CompilationResult.
    /// </summary>
    let convertToFSharpResult (result: TarsEngine.Services.Compilation.CompilationResult) =
        let errors = 
            if result.Errors <> null then
                result.Errors 
                |> Seq.map (fun error -> 
                    {
                        Message = error.Message
                        Line = None
                        Column = None
                        FilePath = None
                    })
                |> Seq.toList
            else
                []
        
        let diagnostics = 
            if result.Diagnostics <> null then
                result.Diagnostics 
                |> Seq.map (fun diag -> 
                    {
                        Severity = "Error"
                        Message = diag.Message
                        Line = diag.Line
                        Column = diag.Column
                        FilePath = diag.FilePath
                    })
                |> Seq.toList
            else
                []
        
        {
            Success = result.Success
            CompiledAssembly = if result.CompiledAssembly <> null then Some result.CompiledAssembly else None
            Errors = errors
            Diagnostics = diagnostics
        }
    
    /// <summary>
    /// Converts an F# CompilationResult to a C# CompilationResult.
    /// </summary>
    let convertToCSharpResult (result: CompilationResult) =
        let errors = 
            result.Errors 
            |> Seq.map (fun error -> 
                new TarsEngine.Services.Compilation.CompilationError(
                    Message = error.Message
                ))
            |> Seq.toList
        
        let diagnostics = 
            result.Diagnostics 
            |> Seq.map (fun diag -> 
                new TarsEngine.Services.Compilation.CompilationDiagnostic(
                    Severity = diag.Severity,
                    Message = diag.Message,
                    Line = diag.Line,
                    Column = diag.Column,
                    FilePath = diag.FilePath
                ))
            |> Seq.toList
        
        let compilationResult = new TarsEngine.Services.Compilation.CompilationResult()
        compilationResult.Success <- result.Success
        compilationResult.CompiledAssembly <- match result.CompiledAssembly with Some asm -> asm | None -> null
        compilationResult.Errors <- errors
        compilationResult.Diagnostics <- diagnostics
        compilationResult
    
    /// <summary>
    /// Creates compilation options from the given parameters.
    /// </summary>
    let createCompilationOptions outputPath references =
        {
            OutputPath = if String.IsNullOrEmpty(outputPath) then None else Some outputPath
            References = if references = null then [] else references |> Seq.toList
            Defines = []
            GenerateExecutable = false
            SourceFiles = []
            Resources = []
            OtherFlags = []
        }
    
    /// <summary>
    /// Compiles F# code with script options.
    /// </summary>
    member _.CompileAsync(code: string, options: ScriptOptions) =
        task {
            try
                logger.LogInformation("Compiling F# code with script options")
                
                // Extract references from script options
                let references = 
                    if options <> null && options.MetadataReferences <> null then
                        options.MetadataReferences
                        |> Seq.map (fun ref -> ref.Display)
                        |> Seq.toArray
                    else
                        [||]
                
                // Create a temporary file for the F# code
                let tempFilePath = Path.Combine(tempDirectory, $"{Guid.NewGuid()}.fs")
                do! File.WriteAllTextAsync(tempFilePath, code)
                
                // Output assembly path
                let outputPath = Path.Combine(tempDirectory, $"{Guid.NewGuid()}.dll")
                
                // Create compilation options
                let compilationOptions = createCompilationOptions outputPath references
                
                // Compile the code
                let! result = fsharpCompiler.CompileToAssemblyAsync(code, compilationOptions)
                
                // Convert the result to C# format
                return convertToCSharpResult result
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# code")
                let result = new TarsEngine.Services.Compilation.CompilationResult()
                result.Success <- false
                result.Errors <- [new TarsEngine.Services.Compilation.CompilationError(Message = ex.Message)]
                return result
        }
    
    /// <summary>
    /// Compiles F# code.
    /// </summary>
    member this.CompileAsync(code: string) =
        task {
            try
                logger.LogInformation("Compiling F# code")
                
                // Output assembly path
                let outputPath = Path.Combine(tempDirectory, $"{Guid.NewGuid()}.dll")
                
                // Create compilation options
                let compilationOptions = createCompilationOptions outputPath [||]
                
                // Compile the code
                let! result = fsharpCompiler.CompileToAssemblyAsync(code, compilationOptions)
                
                // Convert the result to C# format
                return convertToCSharpResult result
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# code")
                let result = new TarsEngine.Services.Compilation.CompilationResult()
                result.Success <- false
                result.Errors <- [new TarsEngine.Services.Compilation.CompilationError(Message = ex.Message)]
                return result
        }
    
    /// <summary>
    /// Compiles F# code with references.
    /// </summary>
    member this.CompileAsync(code: string, references: string[]) =
        task {
            try
                logger.LogInformation("Compiling F# code with references")
                
                // Output assembly path
                let outputPath = Path.Combine(tempDirectory, $"{Guid.NewGuid()}.dll")
                
                // Create compilation options
                let compilationOptions = createCompilationOptions outputPath references
                
                // Compile the code
                let! result = fsharpCompiler.CompileToAssemblyAsync(code, compilationOptions)
                
                // Convert the result to C# format
                return convertToCSharpResult result
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# code")
                let result = new TarsEngine.Services.Compilation.CompilationResult()
                result.Success <- false
                result.Errors <- [new TarsEngine.Services.Compilation.CompilationError(Message = ex.Message)]
                return result
        }
    
    /// <summary>
    /// Compiles F# code with references and output path.
    /// </summary>
    member this.CompileAsync(code: string, references: string[], outputPath: string) =
        task {
            try
                logger.LogInformation("Compiling F# code with references and output path")
                
                // Create compilation options
                let compilationOptions = createCompilationOptions outputPath references
                
                // Compile the code
                let! result = fsharpCompiler.CompileToAssemblyAsync(code, compilationOptions)
                
                // Convert the result to C# format
                return convertToCSharpResult result
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# code")
                let result = new TarsEngine.Services.Compilation.CompilationResult()
                result.Success <- false
                result.Errors <- [new TarsEngine.Services.Compilation.CompilationError(Message = ex.Message)]
                return result
        }
    
    /// <summary>
    /// Compiles F# code with references, output path, and executable flag.
    /// </summary>
    member this.CompileAsync(code: string, references: string[], outputPath: string, generateExecutable: bool) =
        task {
            try
                logger.LogInformation("Compiling F# code with references, output path, and executable flag")
                
                // Create compilation options
                let compilationOptions = 
                    {
                        OutputPath = Some outputPath
                        References = if references = null then [] else references |> Seq.toList
                        Defines = []
                        GenerateExecutable = generateExecutable
                        SourceFiles = []
                        Resources = []
                        OtherFlags = []
                    }
                
                // Compile the code
                let! result = 
                    if generateExecutable then
                        fsharpCompiler.CompileToExeAsync(code, compilationOptions)
                    else
                        fsharpCompiler.CompileToDllAsync(code, compilationOptions)
                
                // Convert the result to C# format
                return convertToCSharpResult result
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# code")
                let result = new TarsEngine.Services.Compilation.CompilationResult()
                result.Success <- false
                result.Errors <- [new TarsEngine.Services.Compilation.CompilationError(Message = ex.Message)]
                return result
        }
    
    /// <summary>
    /// Compiles F# code with references, output path, executable flag, and defines.
    /// </summary>
    member this.CompileAsync(code: string, references: string[], outputPath: string, generateExecutable: bool, defines: string[]) =
        task {
            try
                logger.LogInformation("Compiling F# code with references, output path, executable flag, and defines")
                
                // Create compilation options
                let compilationOptions = 
                    {
                        OutputPath = Some outputPath
                        References = if references = null then [] else references |> Seq.toList
                        Defines = if defines = null then [] else defines |> Seq.toList
                        GenerateExecutable = generateExecutable
                        SourceFiles = []
                        Resources = []
                        OtherFlags = []
                    }
                
                // Compile the code
                let! result = 
                    if generateExecutable then
                        fsharpCompiler.CompileToExeAsync(code, compilationOptions)
                    else
                        fsharpCompiler.CompileToDllAsync(code, compilationOptions)
                
                // Convert the result to C# format
                return convertToCSharpResult result
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# code")
                let result = new TarsEngine.Services.Compilation.CompilationResult()
                result.Success <- false
                result.Errors <- [new TarsEngine.Services.Compilation.CompilationError(Message = ex.Message)]
                return result
        }
    
    /// <summary>
    /// Compiles F# code with references, output path, executable flag, defines, and source files.
    /// </summary>
    member this.CompileAsync(code: string, references: string[], outputPath: string, generateExecutable: bool, defines: string[], sourceFiles: string[]) =
        task {
            try
                logger.LogInformation("Compiling F# code with references, output path, executable flag, defines, and source files")
                
                // Create compilation options
                let compilationOptions = 
                    {
                        OutputPath = Some outputPath
                        References = if references = null then [] else references |> Seq.toList
                        Defines = if defines = null then [] else defines |> Seq.toList
                        GenerateExecutable = generateExecutable
                        SourceFiles = if sourceFiles = null then [] else sourceFiles |> Seq.toList
                        Resources = []
                        OtherFlags = []
                    }
                
                // Compile the code
                let! result = 
                    if generateExecutable then
                        fsharpCompiler.CompileToExeAsync(code, compilationOptions)
                    else
                        fsharpCompiler.CompileToDllAsync(code, compilationOptions)
                
                // Convert the result to C# format
                return convertToCSharpResult result
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# code")
                let result = new TarsEngine.Services.Compilation.CompilationResult()
                result.Success <- false
                result.Errors <- [new TarsEngine.Services.Compilation.CompilationError(Message = ex.Message)]
                return result
        }
    
    /// <summary>
    /// Compiles F# code with references, output path, executable flag, defines, source files, and resources.
    /// </summary>
    member this.CompileAsync(code: string, references: string[], outputPath: string, generateExecutable: bool, defines: string[], sourceFiles: string[], resources: string[]) =
        task {
            try
                logger.LogInformation("Compiling F# code with references, output path, executable flag, defines, source files, and resources")
                
                // Create compilation options
                let compilationOptions = 
                    {
                        OutputPath = Some outputPath
                        References = if references = null then [] else references |> Seq.toList
                        Defines = if defines = null then [] else defines |> Seq.toList
                        GenerateExecutable = generateExecutable
                        SourceFiles = if sourceFiles = null then [] else sourceFiles |> Seq.toList
                        Resources = if resources = null then [] else resources |> Seq.toList
                        OtherFlags = []
                    }
                
                // Compile the code
                let! result = 
                    if generateExecutable then
                        fsharpCompiler.CompileToExeAsync(code, compilationOptions)
                    else
                        fsharpCompiler.CompileToDllAsync(code, compilationOptions)
                
                // Convert the result to C# format
                return convertToCSharpResult result
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# code")
                let result = new TarsEngine.Services.Compilation.CompilationResult()
                result.Success <- false
                result.Errors <- [new TarsEngine.Services.Compilation.CompilationError(Message = ex.Message)]
                return result
        }
    
    /// <summary>
    /// Compiles F# code with references, output path, executable flag, defines, source files, resources, and other flags.
    /// </summary>
    member this.CompileAsync(code: string, references: string[], outputPath: string, generateExecutable: bool, defines: string[], sourceFiles: string[], resources: string[], otherFlags: string[]) =
        task {
            try
                logger.LogInformation("Compiling F# code with references, output path, executable flag, defines, source files, resources, and other flags")
                
                // Create compilation options
                let compilationOptions = 
                    {
                        OutputPath = Some outputPath
                        References = if references = null then [] else references |> Seq.toList
                        Defines = if defines = null then [] else defines |> Seq.toList
                        GenerateExecutable = generateExecutable
                        SourceFiles = if sourceFiles = null then [] else sourceFiles |> Seq.toList
                        Resources = if resources = null then [] else resources |> Seq.toList
                        OtherFlags = if otherFlags = null then [] else otherFlags |> Seq.toList
                    }
                
                // Compile the code
                let! result = 
                    if generateExecutable then
                        fsharpCompiler.CompileToExeAsync(code, compilationOptions)
                    else
                        fsharpCompiler.CompileToDllAsync(code, compilationOptions)
                
                // Convert the result to C# format
                return convertToCSharpResult result
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# code")
                let result = new TarsEngine.Services.Compilation.CompilationResult()
                result.Success <- false
                result.Errors <- [new TarsEngine.Services.Compilation.CompilationError(Message = ex.Message)]
                return result
        }
    
    /// <summary>
    /// Implements the C# IFSharpCompiler interface.
    /// </summary>
    interface TarsEngine.Services.Compilation.IFSharpCompiler with
        member this.CompileAsync(code, options) = this.CompileAsync(code, options)
        member this.CompileAsync(code) = this.CompileAsync(code)
        member this.CompileAsync(code, references) = this.CompileAsync(code, references)
        member this.CompileAsync(code, references, outputPath) = this.CompileAsync(code, references, outputPath)
        member this.CompileAsync(code, references, outputPath, generateExecutable) = this.CompileAsync(code, references, outputPath, generateExecutable)
        member this.CompileAsync(code, references, outputPath, generateExecutable, defines) = this.CompileAsync(code, references, outputPath, generateExecutable, defines)
        member this.CompileAsync(code, references, outputPath, generateExecutable, defines, sourceFiles) = this.CompileAsync(code, references, outputPath, generateExecutable, defines, sourceFiles)
        member this.CompileAsync(code, references, outputPath, generateExecutable, defines, sourceFiles, resources) = this.CompileAsync(code, references, outputPath, generateExecutable, defines, sourceFiles, resources)
        member this.CompileAsync(code, references, outputPath, generateExecutable, defines, sourceFiles, resources, otherFlags) = this.CompileAsync(code, references, outputPath, generateExecutable, defines, sourceFiles, resources, otherFlags)
