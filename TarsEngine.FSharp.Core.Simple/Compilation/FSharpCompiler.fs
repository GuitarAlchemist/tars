namespace TarsEngine.FSharp.Core.Simple.Compilation

open System
open System.Diagnostics
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Interface for F# compilation.
/// </summary>
type IFSharpCompiler =
    /// <summary>
    /// Compiles F# source code.
    /// </summary>
    abstract member CompileAsync: sourceCode: string * options: CompilationOptions -> Task<CompilationResult>
    
    /// <summary>
    /// Compiles F# source files.
    /// </summary>
    abstract member CompileFilesAsync: sourceFiles: string list * options: CompilationOptions -> Task<CompilationResult>

/// <summary>
/// Simple F# compiler implementation.
/// </summary>
type FSharpCompiler(logger: ILogger<FSharpCompiler>) =
    
    /// <summary>
    /// Compiles F# source code.
    /// </summary>
    member _.CompileAsync(sourceCode: string, options: CompilationOptions) =
        task {
            let stopwatch = Stopwatch.StartNew()
            let mutable errors = []
            let mutable warnings = []
            let mutable success = false
            let mutable outputPath = None
            
            try
                logger.LogInformation("Compiling F# source code")
                
                // Create a temporary file for the source code
                let tempFile = Path.GetTempFileName() + ".fs"
                File.WriteAllText(tempFile, sourceCode)
                
                try
                    // Compile using dotnet fsc
                    let! result = compileWithFsc [tempFile] options
                    success <- result.Success
                    errors <- result.Errors
                    warnings <- result.Warnings
                    outputPath <- result.OutputPath
                finally
                    // Clean up temporary file
                    if File.Exists(tempFile) then
                        File.Delete(tempFile)
                
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# source code")
                errors <- [ex.Message]
                success <- false
            
            stopwatch.Stop()
            
            return {
                Success = success
                OutputPath = outputPath
                Errors = errors
                Warnings = warnings
                CompilationTime = stopwatch.Elapsed
            }
        }
    
    /// <summary>
    /// Compiles F# source files.
    /// </summary>
    member _.CompileFilesAsync(sourceFiles: string list, options: CompilationOptions) =
        task {
            let stopwatch = Stopwatch.StartNew()
            let mutable errors = []
            let mutable warnings = []
            let mutable success = false
            let mutable outputPath = None
            
            try
                logger.LogInformation($"Compiling {sourceFiles.Length} F# source files")
                
                // Validate that all files exist
                for file in sourceFiles do
                    if not (File.Exists(file)) then
                        errors <- $"Source file not found: {file}" :: errors
                
                if errors.IsEmpty then
                    let! result = compileWithFsc sourceFiles options
                    success <- result.Success
                    errors <- result.Errors
                    warnings <- result.Warnings
                    outputPath <- result.OutputPath
                
            with
            | ex ->
                logger.LogError(ex, "Error compiling F# source files")
                errors <- [ex.Message]
                success <- false
            
            stopwatch.Stop()
            
            return {
                Success = success
                OutputPath = outputPath
                Errors = errors
                Warnings = warnings
                CompilationTime = stopwatch.Elapsed
            }
        }
    
    /// <summary>
    /// Compiles using the F# compiler (fsc).
    /// </summary>
    member private _.compileWithFsc(sourceFiles: string list, options: CompilationOptions) =
        task {
            try
                // Build fsc arguments
                let args = System.Text.StringBuilder()
                
                // Add source files
                for file in sourceFiles do
                    args.Append($"\"{file}\" ") |> ignore
                
                // Add target
                match options.Target with
                | Library -> args.Append("--target:library ") |> ignore
                | Executable -> args.Append("--target:exe ") |> ignore
                | Module -> args.Append("--target:module ") |> ignore
                
                // Add output path
                match options.OutputPath with
                | Some path -> args.Append($"--out:\"{path}\" ") |> ignore
                | None -> ()
                
                // Add optimization
                if options.Optimize then
                    args.Append("--optimize+ ") |> ignore
                else
                    args.Append("--optimize- ") |> ignore
                
                // Add debug info
                if options.Debug then
                    args.Append("--debug+ ") |> ignore
                else
                    args.Append("--debug- ") |> ignore
                
                // Add references
                for reference in options.References do
                    args.Append($"--reference:\"{reference}\" ") |> ignore
                
                // Execute fsc
                let processInfo = ProcessStartInfo()
                processInfo.FileName <- "dotnet"
                processInfo.Arguments <- $"fsc {args.ToString().Trim()}"
                processInfo.UseShellExecute <- false
                processInfo.RedirectStandardOutput <- true
                processInfo.RedirectStandardError <- true
                processInfo.CreateNoWindow <- true
                
                use process = Process.Start(processInfo)
                let! output = process.StandardOutput.ReadToEndAsync()
                let! errorOutput = process.StandardError.ReadToEndAsync()
                
                process.WaitForExit()
                
                let success = process.ExitCode = 0
                let errors = if String.IsNullOrEmpty(errorOutput) then [] else [errorOutput]
                let warnings = [] // TODO: Parse warnings from output
                let outputPath = options.OutputPath
                
                return {
                    Success = success
                    OutputPath = outputPath
                    Errors = errors
                    Warnings = warnings
                    CompilationTime = TimeSpan.Zero // Will be set by caller
                }
            with
            | ex ->
                return {
                    Success = false
                    OutputPath = None
                    Errors = [ex.Message]
                    Warnings = []
                    CompilationTime = TimeSpan.Zero
                }
        }
    
    interface IFSharpCompiler with
        member this.CompileAsync(sourceCode, options) = this.CompileAsync(sourceCode, options)
        member this.CompileFilesAsync(sourceFiles, options) = this.CompileFilesAsync(sourceFiles, options)
