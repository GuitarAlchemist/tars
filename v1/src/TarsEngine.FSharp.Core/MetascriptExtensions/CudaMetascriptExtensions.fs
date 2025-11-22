namespace TarsEngine.FSharp.Core.MetascriptExtensions

open System
open System.IO
open TarsEngine.FSharp.Core.CudaTranspilation.CudaTranspiler

/// CUDA Transpilation Extensions for TARS Metascripts
/// Enables seamless F# to CUDA transpilation within metascript blocks
module CudaMetascriptExtensions =

    // ============================================================================
    // METASCRIPT CUDA BLOCKS
    // ============================================================================

    /// CUDA transpilation block for metascripts
    type CudaBlock = {
        Code: string
        KernelName: string
        Backend: CudaBackend
        Options: CudaTranspilationOptions
        mutable Result: CudaTranspilationResult option
    }

    /// CUDA execution context for metascripts
    type CudaExecutionContext = {
        WorkingDirectory: string
        DefaultBackend: CudaBackend
        DefaultOptions: CudaTranspilationOptions
        mutable CompiledKernels: Map<string, CudaTranspilationResult>
        mutable ExecutionLog: string list
    }

    // ============================================================================
    // CUDA METASCRIPT DSL
    // ============================================================================

    /// CUDA computation builder for metascripts
    type CudaBuilder(context: CudaExecutionContext) =
        member _.Return(value: 'T) = value
        member _.ReturnFrom(value: 'T) = value
        
        member _.Bind(cudaBlock: CudaBlock, f: CudaTranspilationResult -> 'T) = 
            // Transpile and compile the CUDA code
            let result = transpileAndCompile cudaBlock.Code cudaBlock.KernelName cudaBlock.Options
            cudaBlock.Result <- Some result
            
            // Update context
            context.CompiledKernels <- context.CompiledKernels.Add(cudaBlock.KernelName, result)
            context.ExecutionLog <- sprintf "Compiled kernel '%s': %s" cudaBlock.KernelName (if result.Success then "SUCCESS" else "FAILED") :: context.ExecutionLog
            
            f result
        
        member _.Zero() = ()
        member _.Combine(a: unit, b: unit) = ()
        member _.Delay(f: unit -> 'T) = f()

    /// Create CUDA block from F# code
    let cudaKernel (kernelName: string) (fsharpCode: string) (backend: CudaBackend) (options: CudaTranspilationOptions) : CudaBlock =
        {
            Code = fsharpCode
            KernelName = kernelName
            Backend = backend
            Options = options
            Result = None
        }

    /// Quick CUDA kernel creation with default options
    let quickCudaKernel (kernelName: string) (fsharpCode: string) (context: CudaExecutionContext) : CudaBlock =
        cudaKernel kernelName fsharpCode context.DefaultBackend context.DefaultOptions

    // ============================================================================
    // METASCRIPT CUDA FUNCTIONS
    // ============================================================================

    /// Transpile F# function to CUDA kernel
    let transpileToCuda (fsharpCode: string) (kernelName: string) (backend: string) : string =
        let outputDir = Path.Combine(Path.GetTempPath(), "tars_cuda_transpile")
        let result = cudaTranspile fsharpCode kernelName backend outputDir
        
        if result.Success then
            match result.OutputPath with
            | Some path when File.Exists(path) -> File.ReadAllText(path)
            | _ -> "// CUDA code generated but file not found"
        else
            sprintf "// CUDA transpilation failed: %s" (String.concat "; " result.Errors)

    /// Compile F# code to CUDA executable
    let compileToExecutable (fsharpCode: string) (kernelName: string) (backend: string) (outputDir: string) : CudaTranspilationResult =
        cudaTranspile fsharpCode kernelName backend outputDir

    /// Execute CUDA kernel (if compiled successfully)
    let executeCudaKernel (result: CudaTranspilationResult) (args: string) : string =
        match result.ExecutablePath with
        | Some exePath when File.Exists(exePath) ->
            try
                let psi = System.Diagnostics.ProcessStartInfo()
                psi.FileName <- exePath
                psi.Arguments <- args
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true

                use process = System.Diagnostics.Process.Start(psi)
                let stdout = process.StandardOutput.ReadToEnd()
                let stderr = process.StandardError.ReadToEnd()
                process.WaitForExit()
                
                if process.ExitCode = 0 then
                    stdout
                else
                    sprintf "Execution failed (exit code %d): %s" process.ExitCode stderr
            with
            | ex -> sprintf "Execution error: %s" ex.Message
        | _ -> "No executable available to run"

    // ============================================================================
    // METASCRIPT INTEGRATION HELPERS
    // ============================================================================

    /// Create default CUDA execution context
    let createCudaContext (workingDir: string) : CudaExecutionContext =
        let defaultOptions = {
            Backend = WSLCompilation("nvcc")
            OptimizationLevel = 2
            Architecture = "sm_75"
            DebugInfo = false
            FastMath = true
            OutputDirectory = workingDir
            IncludePaths = []
            Libraries = []
        }

        {
            WorkingDirectory = workingDir
            DefaultBackend = WSLCompilation("nvcc")
            DefaultOptions = defaultOptions
            CompiledKernels = Map.empty
            ExecutionLog = []
        }

    /// CUDA builder instance for metascripts
    let cuda (context: CudaExecutionContext) = CudaBuilder(context)

    // ============================================================================
    // ADVANCED CUDA OPERATIONS
    // ============================================================================

    /// Batch compile multiple CUDA kernels
    let batchCompileCuda (kernels: (string * string) list) (backend: string) (outputDir: string) : Map<string, CudaTranspilationResult> =
        kernels
        |> List.map (fun (name, code) -> 
            let result = cudaTranspile code name backend outputDir
            (name, result))
        |> Map.ofList

    /// Generate CUDA kernel from mathematical expression
    let mathToCuda (expression: string) (kernelName: string) (inputType: string) : string =
        let fsharpCode = sprintf """
let %s (input: %s array) (output: %s array) (n: int) =
    for i in 0 .. n-1 do
        output.[i] <- %s input.[i]
""" kernelName inputType inputType expression

        transpileToCuda fsharpCode kernelName "wsl"

    /// Create CUDA kernel for vector operations
    let vectorOpToCuda (operation: string) (kernelName: string) : string =
        let fsharpCode = sprintf """
let %s (a: float32 array) (b: float32 array) (result: float32 array) (n: int) =
    for i in 0 .. n-1 do
        result.[i] <- a.[i] %s b.[i]
""" kernelName operation

        transpileToCuda fsharpCode kernelName "wsl"

    /// Create CUDA kernel for matrix operations
    let matrixOpToCuda (operation: string) (kernelName: string) : string =
        let fsharpCode = sprintf """
let %s (a: float32 array) (b: float32 array) (result: float32 array) (rows: int) (cols: int) =
    for i in 0 .. rows-1 do
        for j in 0 .. cols-1 do
            let idx = i * cols + j
            result.[idx] <- a.[idx] %s b.[idx]
""" kernelName operation

        transpileToCuda fsharpCode kernelName "wsl"

    // ============================================================================
    // METASCRIPT CUDA SYNTAX EXTENSIONS
    // ============================================================================

    /// Inline CUDA compilation syntax for metascripts
    let inline (|CUDA|) (fsharpCode: string) (kernelName: string) (backend: string) : CudaTranspilationResult =
        let outputDir = Path.Combine(Path.GetTempPath(), "tars_inline_cuda")
        cudaTranspile fsharpCode kernelName backend outputDir

    /// CUDA execution syntax for metascripts
    let inline (|>CUDA) (result: CudaTranspilationResult) (args: string) : string =
        executeCudaKernel result args

    /// CUDA transpilation only (no compilation)
    let inline (|>TRANSPILE) (fsharpCode: string) (kernelName: string) : string =
        transpileToCuda fsharpCode kernelName "wsl"

    // ============================================================================
    // CUDA PERFORMANCE MONITORING
    // ============================================================================

    type CudaPerformanceMetrics = {
        KernelName: string
        CompilationTime: TimeSpan
        ExecutionTime: TimeSpan option
        BinarySize: int64 option
        Backend: string
        Success: bool
        Errors: string list
    }

    let collectPerformanceMetrics (result: CudaTranspilationResult) (kernelName: string) (backend: string) : CudaPerformanceMetrics =
        {
            KernelName = kernelName
            CompilationTime = result.CompilationTime
            ExecutionTime = None  // Would need to measure execution separately
            BinarySize = result.BinarySize
            Backend = backend
            Success = result.Success
            Errors = result.Errors
        }

    /// Generate performance report for CUDA operations
    let generatePerformanceReport (metrics: CudaPerformanceMetrics list) : string =
        let successful = metrics |> List.filter (fun m -> m.Success)
        let failed = metrics |> List.filter (fun m -> not m.Success)
        
        let avgCompilationTime = 
            if successful.IsEmpty then 0.0
            else successful |> List.averageBy (fun m -> m.CompilationTime.TotalMilliseconds)
        
        let totalBinarySize = 
            successful 
            |> List.choose (fun m -> m.BinarySize)
            |> List.sum

        sprintf """
CUDA Performance Report
=======================
Total Kernels: %d
Successful: %d
Failed: %d
Average Compilation Time: %.2f ms
Total Binary Size: %d bytes
Success Rate: %.1f%%

Failed Kernels:
%s
""" 
            metrics.Length
            successful.Length
            failed.Length
            avgCompilationTime
            totalBinarySize
            (float successful.Length / float metrics.Length * 100.0)
            (failed |> List.map (fun m -> sprintf "- %s: %s" m.KernelName (String.concat "; " m.Errors)) |> String.concat "\n")

    // ============================================================================
    // CUDA DEBUGGING SUPPORT
    // ============================================================================

    /// Add debug information to CUDA kernel
    let addDebugInfo (cudaCode: string) (kernelName: string) : string =
        sprintf """
#ifdef DEBUG
#include <stdio.h>
#define DEBUG_PRINT(fmt, ...) printf("[%s] " fmt "\\n", "%s", ##__VA_ARGS__)
#else
#define DEBUG_PRINT(fmt, ...)
#endif

%s
""" kernelName cudaCode

    /// Validate CUDA code syntax (basic check)
    let validateCudaCode (cudaCode: string) : string list =
        let errors = ResizeArray<string>()
        
        if not (cudaCode.Contains("__global__")) then
            errors.Add("Missing __global__ kernel declaration")
        
        if not (cudaCode.Contains("blockIdx") || cudaCode.Contains("threadIdx")) then
            errors.Add("Kernel doesn't use CUDA thread indexing")
        
        let braceCount = cudaCode.ToCharArray() |> Array.fold (fun acc c -> 
            match c with 
            | '{' -> acc + 1 
            | '}' -> acc - 1 
            | _ -> acc) 0
        
        if braceCount <> 0 then
            errors.Add("Mismatched braces in CUDA code")
        
        errors |> Seq.toList
