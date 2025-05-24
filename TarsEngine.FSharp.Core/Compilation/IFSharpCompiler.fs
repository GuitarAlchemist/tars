namespace TarsEngine.FSharp.Core.Compilation

open System.Threading.Tasks

/// <summary>
/// Interface for F# compiler services.
/// </summary>
type IFSharpCompiler =
    /// <summary>
    /// Compiles F# code to an assembly.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    abstract member CompileToAssemblyAsync : code:string * options:CompilationOptions -> Task<CompilationResult>
    
    /// <summary>
    /// Compiles F# code to an assembly in memory.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    abstract member CompileToAssemblyInMemoryAsync : code:string * options:CompilationOptions -> Task<CompilationResult>
    
    /// <summary>
    /// Compiles and executes F# code.
    /// </summary>
    /// <param name="code">The F# code to compile and execute.</param>
    /// <param name="options">The script execution options.</param>
    /// <returns>The script execution result.</returns>
    abstract member CompileAndExecuteAsync : code:string * options:ScriptExecutionOptions -> Task<ScriptExecutionResult>
    
    /// <summary>
    /// Compiles F# code to a DLL.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    abstract member CompileToDllAsync : code:string * options:CompilationOptions -> Task<CompilationResult>
    
    /// <summary>
    /// Compiles F# code to an executable.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    abstract member CompileToExeAsync : code:string * options:CompilationOptions -> Task<CompilationResult>
    
    /// <summary>
    /// Compiles F# code to a script.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    abstract member CompileToScriptAsync : code:string * options:CompilationOptions -> Task<CompilationResult>
    
    /// <summary>
    /// Compiles F# code to a NuGet package.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <param name="packageName">The name of the NuGet package.</param>
    /// <param name="packageVersion">The version of the NuGet package.</param>
    /// <returns>The compilation result.</returns>
    abstract member CompileToNuGetAsync : code:string * options:CompilationOptions * packageName:string * packageVersion:string -> Task<CompilationResult>
    
    /// <summary>
    /// Compiles F# code to a JavaScript file.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    abstract member CompileToJavaScriptAsync : code:string * options:CompilationOptions -> Task<CompilationResult>
    
    /// <summary>
    /// Compiles F# code to a TypeScript file.
    /// </summary>
    /// <param name="code">The F# code to compile.</param>
    /// <param name="options">The compilation options.</param>
    /// <returns>The compilation result.</returns>
    abstract member CompileToTypeScriptAsync : code:string * options:CompilationOptions -> Task<CompilationResult>
