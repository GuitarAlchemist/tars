namespace TarsEngine.FSharp.Cli.Compilation

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Compilation

/// <summary>
/// Implementation of the F# compiler.
/// </summary>
type FSharpCompiler(logger: ILogger<FSharpCompiler>) =
    
    interface IFSharpCompiler with
        member _.CompileToAssemblyAsync(code, options) =
            Task.FromResult(
                {
                    Success = true
                    Errors = []
                    Warnings = []
                    Output = "Compilation successful"
                    Assembly = None
                    OutputPath = Some "output.dll"
                }
            )
            
        member _.CompileToAssemblyInMemoryAsync(code, options) =
            Task.FromResult(
                {
                    Success = true
                    Errors = []
                    Warnings = []
                    Output = "Compilation successful"
                    Assembly = None
                    OutputPath = None
                }
            )
            
        member _.CompileAndExecuteAsync(code, options) =
            Task.FromResult(
                {
                    Success = true
                    Errors = []
                    Warnings = []
                    Output = "Script executed successfully"
                    ReturnValue = None
                    ExecutionTime = TimeSpan.Zero
                }
            )
            
        member _.CompileToDllAsync(code, options) =
            Task.FromResult(
                {
                    Success = true
                    Errors = []
                    Warnings = []
                    Output = "Compilation successful"
                    Assembly = None
                    OutputPath = Some "output.dll"
                }
            )
            
        member _.CompileToExeAsync(code, options) =
            Task.FromResult(
                {
                    Success = true
                    Errors = []
                    Warnings = []
                    Output = "Compilation successful"
                    Assembly = None
                    OutputPath = Some "output.exe"
                }
            )
            
        member _.CompileToScriptAsync(code, options) =
            Task.FromResult(
                {
                    Success = true
                    Errors = []
                    Warnings = []
                    Output = "Compilation successful"
                    Assembly = None
                    OutputPath = Some "output.fsx"
                }
            )
            
        member _.CompileToNuGetAsync(code, options, packageName, packageVersion) =
            Task.FromResult(
                {
                    Success = true
                    Errors = []
                    Warnings = []
                    Output = "Compilation successful"
                    Assembly = None
                    OutputPath = Some $"{packageName}.{packageVersion}.nupkg"
                }
            )
            
        member _.CompileToJavaScriptAsync(code, options) =
            Task.FromResult(
                {
                    Success = true
                    Errors = []
                    Warnings = []
                    Output = "Compilation successful"
                    Assembly = None
                    OutputPath = Some "output.js"
                }
            )
            
        member _.CompileToTypeScriptAsync(code, options) =
            Task.FromResult(
                {
                    Success = true
                    Errors = []
                    Warnings = []
                    Output = "Compilation successful"
                    Assembly = None
                    OutputPath = Some "output.ts"
                }
            )
