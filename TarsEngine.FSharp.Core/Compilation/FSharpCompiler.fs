namespace TarsEngine.FSharp.Core.Compilation

open System
open System.IO
open System.Reflection
open System.Threading.Tasks
open System.Diagnostics
open System.Collections.Generic
open System.Text

/// <summary>
/// Implementation of the IFSharpCompiler interface.
/// </summary>
type FSharpCompiler() =
    /// <summary>
    /// Compiles F# code to an assembly.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    member _.CompileToAssemblyAsync(code: string, options: CompilationOptions) =
        Task.Run(fun () ->
            try
                // Create a temporary file for the code
                let tempFile = Path.GetTempFileName() |> Path.ChangeExtension |> (fun p -> p + ".fs")
                File.WriteAllText(tempFile, code)
                
                // Determine the output path
                let outputPath = 
                    match options.OutputPath with
                    | Some path -> path
                    | None -> Path.ChangeExtension(tempFile, ".dll")
                
                // Build the compiler arguments
                let args = StringBuilder()
                args.Append($"fsc.exe --target:library --out:{outputPath} ") |> ignore
                
                // Add references
                for reference in options.References do
                    args.Append($"--reference:{reference} ") |> ignore
                
                // Add optimization flag
                if options.Optimize then
                    args.Append("--optimize+ ") |> ignore
                else
                    args.Append("--optimize- ") |> ignore
                
                // Add treat warnings as errors flag
                if options.TreatWarningsAsErrors then
                    args.Append("--warnaserror+ ") |> ignore
                
                // Add warning level
                args.Append($"--warn:{options.WarningLevel} ") |> ignore
                
                // Add debug type
                match options.DebugType with
                | Some debugType -> args.Append($"--debug:{debugType} ") |> ignore
                | None -> ()
                
                // Add target framework
                match options.TargetFramework with
                | Some targetFramework -> args.Append($"--targetprofile:{targetFramework} ") |> ignore
                | None -> ()
                
                // Add language version
                match options.LanguageVersion with
                | Some languageVersion -> args.Append($"--langversion:{languageVersion} ") |> ignore
                | None -> ()
                
                // Add additional options
                for additionalOption in options.AdditionalOptions do
                    args.Append($"{additionalOption} ") |> ignore
                
                // Add the source file
                args.Append(tempFile) |> ignore
                
                // Create the process
                let psi = ProcessStartInfo()
                psi.FileName <- "dotnet"
                psi.Arguments <- args.ToString()
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true
                
                // Start the process
                use process = Process.Start(psi)
                let output = process.StandardOutput.ReadToEnd()
                let error = process.StandardError.ReadToEnd()
                process.WaitForExit()
                
                // Check if the compilation was successful
                if process.ExitCode = 0 then
                    // Load the assembly
                    let assembly = Assembly.LoadFrom(outputPath)
                    
                    // Return the result
                    CompilationResult.createSuccess output (Some assembly) (Some outputPath) []
                else
                    // Parse the errors and warnings
                    let errors = []
                    let warnings = []
                    
                    // Return the result
                    CompilationResult.createFailure errors warnings error
            with
            | ex ->
                // Return the result
                CompilationResult.createFailure [ex.Message] [] ""
        )
    
    /// <summary>
    /// Compiles F# code to an assembly in memory.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    member this.CompileToAssemblyInMemoryAsync(code: string, options: CompilationOptions) =
        Task.Run(fun () ->
            try
                // Create a temporary file for the code
                let tempFile = Path.GetTempFileName() |> Path.ChangeExtension |> (fun p -> p + ".fs")
                File.WriteAllText(tempFile, code)
                
                // Determine the output path
                let outputPath = Path.GetTempFileName() |> Path.ChangeExtension |> (fun p -> p + ".dll")
                
                // Compile to a temporary assembly
                let result = this.CompileToAssemblyAsync(code, { options with OutputPath = Some outputPath }).Result
                
                // Check if the compilation was successful
                if result.Success then
                    // Load the assembly into memory
                    let assemblyBytes = File.ReadAllBytes(outputPath)
                    let assembly = Assembly.Load(assemblyBytes)
                    
                    // Delete the temporary files
                    File.Delete(tempFile)
                    File.Delete(outputPath)
                    
                    // Return the result
                    CompilationResult.createSuccess result.Output (Some assembly) None result.Warnings
                else
                    // Delete the temporary file
                    File.Delete(tempFile)
                    
                    // Return the result
                    result
            with
            | ex ->
                // Return the result
                CompilationResult.createFailure [ex.Message] [] ""
        )
    
    /// <summary>
    /// Compiles and executes F# code.
    /// </summary>
    /// <param name="code">The F# code to compile and execute.</param>
    /// <param name="options">The script execution options.</param>
    /// <returns>The script execution result.</returns>
    member _.CompileAndExecuteAsync(code: string, options: ScriptExecutionOptions) =
        Task.Run(fun () ->
            try
                // Create a temporary file for the code
                let tempFile = Path.GetTempFileName() |> Path.ChangeExtension |> (fun p -> p + ".fsx")
                File.WriteAllText(tempFile, code)
                
                // Build the compiler arguments
                let args = StringBuilder()
                args.Append($"fsi.exe ") |> ignore
                
                // Add references
                for reference in options.References do
                    args.Append($"--reference:{reference} ") |> ignore
                
                // Add the source file
                args.Append(tempFile) |> ignore
                
                // Add the arguments
                for argument in options.Arguments do
                    args.Append($" {argument}") |> ignore
                
                // Create the process
                let psi = ProcessStartInfo()
                psi.FileName <- "dotnet"
                psi.Arguments <- args.ToString()
                psi.RedirectStandardOutput <- options.CaptureOutput
                psi.RedirectStandardError <- options.CaptureOutput
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true
                
                // Set the working directory
                match options.WorkingDirectory with
                | Some workingDirectory -> psi.WorkingDirectory <- workingDirectory
                | None -> ()
                
                // Set the environment variables
                match options.EnvironmentVariables with
                | Some environmentVariables ->
                    for KeyValue(key, value) in environmentVariables do
                        psi.Environment.[key] <- value
                | None -> ()
                
                // Start the process
                let stopwatch = Stopwatch.StartNew()
                use process = Process.Start(psi)
                
                // Set the timeout
                match options.Timeout with
                | Some timeout ->
                    if not (process.WaitForExit(int timeout.TotalMilliseconds)) then
                        process.Kill()
                        stopwatch.Stop()
                        ScriptExecutionResult.createFailure ["Script execution timed out."] [] "" stopwatch.Elapsed
                    else
                        let output = if options.CaptureOutput then process.StandardOutput.ReadToEnd() else ""
                        let error = if options.CaptureOutput then process.StandardError.ReadToEnd() else ""
                        stopwatch.Stop()
                        
                        // Check if the execution was successful
                        if process.ExitCode = 0 then
                            ScriptExecutionResult.createSuccess output None stopwatch.Elapsed []
                        else
                            ScriptExecutionResult.createFailure [error] [] output stopwatch.Elapsed
                | None ->
                    process.WaitForExit()
                    let output = if options.CaptureOutput then process.StandardOutput.ReadToEnd() else ""
                    let error = if options.CaptureOutput then process.StandardError.ReadToEnd() else ""
                    stopwatch.Stop()
                    
                    // Check if the execution was successful
                    if process.ExitCode = 0 then
                        ScriptExecutionResult.createSuccess output None stopwatch.Elapsed []
                    else
                        ScriptExecutionResult.createFailure [error] [] output stopwatch.Elapsed
            with
            | ex ->
                // Return the result
                ScriptExecutionResult.createFailure [ex.Message] [] "" TimeSpan.Zero
        )
    
    /// <summary>
    /// Compiles F# code to a DLL.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    member this.CompileToDllAsync(code: string, options: CompilationOptions) =
        this.CompileToAssemblyAsync(code, options)
    
    /// <summary>
    /// Compiles F# code to an executable.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    member _.CompileToExeAsync(code: string, options: CompilationOptions) =
        Task.Run(fun () ->
            try
                // Create a temporary file for the code
                let tempFile = Path.GetTempFileName() |> Path.ChangeExtension |> (fun p -> p + ".fs")
                File.WriteAllText(tempFile, code)
                
                // Determine the output path
                let outputPath = 
                    match options.OutputPath with
                    | Some path -> path
                    | None -> Path.ChangeExtension(tempFile, ".exe")
                
                // Build the compiler arguments
                let args = StringBuilder()
                args.Append($"fsc.exe --target:exe --out:{outputPath} ") |> ignore
                
                // Add references
                for reference in options.References do
                    args.Append($"--reference:{reference} ") |> ignore
                
                // Add optimization flag
                if options.Optimize then
                    args.Append("--optimize+ ") |> ignore
                else
                    args.Append("--optimize- ") |> ignore
                
                // Add treat warnings as errors flag
                if options.TreatWarningsAsErrors then
                    args.Append("--warnaserror+ ") |> ignore
                
                // Add warning level
                args.Append($"--warn:{options.WarningLevel} ") |> ignore
                
                // Add debug type
                match options.DebugType with
                | Some debugType -> args.Append($"--debug:{debugType} ") |> ignore
                | None -> ()
                
                // Add target framework
                match options.TargetFramework with
                | Some targetFramework -> args.Append($"--targetprofile:{targetFramework} ") |> ignore
                | None -> ()
                
                // Add language version
                match options.LanguageVersion with
                | Some languageVersion -> args.Append($"--langversion:{languageVersion} ") |> ignore
                | None -> ()
                
                // Add additional options
                for additionalOption in options.AdditionalOptions do
                    args.Append($"{additionalOption} ") |> ignore
                
                // Add the source file
                args.Append(tempFile) |> ignore
                
                // Create the process
                let psi = ProcessStartInfo()
                psi.FileName <- "dotnet"
                psi.Arguments <- args.ToString()
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true
                
                // Start the process
                use process = Process.Start(psi)
                let output = process.StandardOutput.ReadToEnd()
                let error = process.StandardError.ReadToEnd()
                process.WaitForExit()
                
                // Check if the compilation was successful
                if process.ExitCode = 0 then
                    // Return the result
                    CompilationResult.createSuccess output None (Some outputPath) []
                else
                    // Parse the errors and warnings
                    let errors = []
                    let warnings = []
                    
                    // Return the result
                    CompilationResult.createFailure errors warnings error
            with
            | ex ->
                // Return the result
                CompilationResult.createFailure [ex.Message] [] ""
        )
    
    /// <summary>
    /// Compiles F# code to a script.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    member _.CompileToScriptAsync(code: string, options: CompilationOptions) =
        Task.Run(fun () ->
            try
                // Determine the output path
                let outputPath = 
                    match options.OutputPath with
                    | Some path -> path
                    | None -> Path.GetTempFileName() |> Path.ChangeExtension |> (fun p -> p + ".fsx")
                
                // Write the code to the output file
                File.WriteAllText(outputPath, code)
                
                // Return the result
                CompilationResult.createSuccess "" None (Some outputPath) []
            with
            | ex ->
                // Return the result
                CompilationResult.createFailure [ex.Message] [] ""
        )
    
    /// <summary>
    /// Compiles F# code to a NuGet package.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <param name="packageName">The name of the NuGet package.</param>
    /// <param name="packageVersion">The version of the NuGet package.</param>
    /// <returns>The compilation result.</returns>
    member this.CompileToNuGetAsync(code: string, options: CompilationOptions, packageName: string, packageVersion: string) =
        Task.Run(fun () ->
            try
                // Create a temporary directory for the project
                let tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString())
                Directory.CreateDirectory(tempDir) |> ignore
                
                // Create the project file
                let projectFile = Path.Combine(tempDir, $"{packageName}.fsproj")
                let projectContent = $"""
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>{options.TargetFramework.Value}</TargetFramework>
    <PackageId>{packageName}</PackageId>
    <Version>{packageVersion}</Version>
    <Authors>TARS</Authors>
    <Company>TARS</Company>
    <Description>Generated by TARS</Description>
  </PropertyGroup>
  <ItemGroup>
    <Compile Include="Library.fs" />
  </ItemGroup>
</Project>
"""
                File.WriteAllText(projectFile, projectContent)
                
                // Create the source file
                let sourceFile = Path.Combine(tempDir, "Library.fs")
                File.WriteAllText(sourceFile, code)
                
                // Determine the output path
                let outputPath = 
                    match options.OutputPath with
                    | Some path -> path
                    | None -> Path.Combine(tempDir, "bin", "Debug", options.TargetFramework.Value, $"{packageName}.{packageVersion}.nupkg")
                
                // Build the package
                let args = $"pack {projectFile} -o {Path.GetDirectoryName(outputPath)}"
                
                // Create the process
                let psi = ProcessStartInfo()
                psi.FileName <- "dotnet"
                psi.Arguments <- args
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true
                psi.WorkingDirectory <- tempDir
                
                // Start the process
                use process = Process.Start(psi)
                let output = process.StandardOutput.ReadToEnd()
                let error = process.StandardError.ReadToEnd()
                process.WaitForExit()
                
                // Check if the compilation was successful
                if process.ExitCode = 0 then
                    // Return the result
                    CompilationResult.createSuccess output None (Some outputPath) []
                else
                    // Parse the errors and warnings
                    let errors = []
                    let warnings = []
                    
                    // Return the result
                    CompilationResult.createFailure errors warnings error
            with
            | ex ->
                // Return the result
                CompilationResult.createFailure [ex.Message] [] ""
        )
    
    /// <summary>
    /// Compiles F# code to a JavaScript file.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    member _.CompileToJavaScriptAsync(code: string, options: CompilationOptions) =
        Task.Run(fun () ->
            // This is a placeholder implementation
            // In a real implementation, we would use Fable to compile F# to JavaScript
            CompilationResult.createFailure ["Not implemented"] [] ""
        )
    
    /// <summary>
    /// Compiles F# code to a TypeScript file.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    member _.CompileToTypeScriptAsync(code: string, options: CompilationOptions) =
        Task.Run(fun () ->
            // This is a placeholder implementation
            // In a real implementation, we would use Fable to compile F# to TypeScript
            CompilationResult.createFailure ["Not implemented"] [] ""
        )

/// <summary>
/// Implementation of the IFSharpCompiler interface.
/// </summary>
type FSharpCompilerImpl() =
    let compiler = FSharpCompiler()
    
    interface IFSharpCompiler with
        member this.CompileToAssemblyAsync(code, options) =
            compiler.CompileToAssemblyAsync(code, options)
        
        member this.CompileToAssemblyInMemoryAsync(code, options) =
            compiler.CompileToAssemblyInMemoryAsync(code, options)
        
        member this.CompileAndExecuteAsync(code, options) =
            compiler.CompileAndExecuteAsync(code, options)
        
        member this.CompileToDllAsync(code, options) =
            compiler.CompileToDllAsync(code, options)
        
        member this.CompileToExeAsync(code, options) =
            compiler.CompileToExeAsync(code, options)
        
        member this.CompileToScriptAsync(code, options) =
            compiler.CompileToScriptAsync(code, options)
        
        member this.CompileToNuGetAsync(code, options, packageName, packageVersion) =
            compiler.CompileToNuGetAsync(code, options, packageName, packageVersion)
        
        member this.CompileToJavaScriptAsync(code, options) =
            compiler.CompileToJavaScriptAsync(code, options)
        
        member this.CompileToTypeScriptAsync(code, options) =
            compiler.CompileToTypeScriptAsync(code, options)
