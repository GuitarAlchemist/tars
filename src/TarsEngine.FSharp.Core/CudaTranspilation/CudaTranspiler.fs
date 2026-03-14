namespace TarsEngine.FSharp.Core.CudaTranspilation

open System
open System.IO
open System.Text
open System.Diagnostics
open System.Text.RegularExpressions
open System.Collections.Generic

/// CUDA Transpilation Engine for TARS Metascripts
/// Supports F# to CUDA transpilation with multiple compilation backends
module CudaTranspiler =

    // ============================================================================
    // CUDA TRANSPILATION TYPES
    // ============================================================================

    type CudaBackend =
        | ManagedCuda of libraryPath: string
        | WSLCompilation of cudaPath: string
        | DockerContainer of imageName: string
        | NativeCuda of nvccPath: string

    type CudaKernelSpec = {
        KernelName: string
        BlockSize: int * int * int
        GridSize: int * int * int
        SharedMemory: int
        Parameters: (string * string) list  // (name, type)
        ReturnType: string
        Body: string
    }

    type CudaTranspilationOptions = {
        Backend: CudaBackend
        OptimizationLevel: int  // 0-3
        Architecture: string    // sm_75, sm_80, etc.
        DebugInfo: bool
        FastMath: bool
        OutputDirectory: string
        IncludePaths: string list
        Libraries: string list
    }

    type CudaTranspilationResult = {
        Success: bool
        OutputPath: string option
        ExecutablePath: string option
        CompilationLog: string
        Errors: string list
        Warnings: string list
        CompilationTime: TimeSpan
        BinarySize: int64 option
    }

    // ============================================================================
    // F# TO CUDA TRANSPILATION ENGINE
    // ============================================================================

    module FSharpToCuda =
        
        let private cudaTypeMap = Map.ofList [
            ("int", "int")
            ("int32", "int")
            ("int64", "long long")
            ("float", "float")
            ("float32", "float")
            ("double", "double")
            ("bool", "bool")
            ("byte", "unsigned char")
            ("uint32", "unsigned int")
            ("uint64", "unsigned long long")
            ("string", "char*")
        ]

        let private transpileFSharpType (fsharpType: string) : string =
            cudaTypeMap.TryFind(fsharpType.ToLower()) |> Option.defaultValue "void"

        let private transpileFSharpExpression (expr: string) : string =
            expr
                .Replace("let ", "")
                .Replace(" = ", " = ")
                .Replace("Array.length", "length")
                .Replace("Array.sum", "sum")
                .Replace("sqrt", "sqrtf")
                .Replace("sin", "sinf")
                .Replace("cos", "cosf")
                .Replace("exp", "expf")
                .Replace("log", "logf")

        let transpileFSharpFunction (functionCode: string) (kernelName: string) : CudaKernelSpec =
            let lines = functionCode.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
            
            // Extract function signature
            let signatureLine = lines |> Array.find (fun line -> line.Contains("let ") && line.Contains("="))
            let bodyLines = lines |> Array.skip 1 |> Array.map transpileFSharpExpression
            
            // Parse parameters (simplified)
            let paramPattern = @"(\w+)\s*:\s*(\w+(?:\[\])?)"
            let matches = Regex.Matches(signatureLine, paramPattern)
            let parameters = [
                for m in matches ->
                    let name = m.Groups.[1].Value
                    let fsharpType = m.Groups.[2].Value
                    let cudaType = transpileFSharpType fsharpType
                    (name, cudaType)
            ]

            {
                KernelName = kernelName
                BlockSize = (256, 1, 1)
                GridSize = (1, 1, 1)
                SharedMemory = 0
                Parameters = parameters
                ReturnType = "void"
                Body = String.Join("\n    ", bodyLines)
            }

        let generateCudaKernel (spec: CudaKernelSpec) : string =
            let paramList = spec.Parameters |> List.map (fun (name, cudaType) -> sprintf "%s %s" cudaType name) |> String.concat ", "
            
            sprintf """
__global__ void %s(%s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    %s
}
""" spec.KernelName paramList spec.Body

    // ============================================================================
    // CUDA COMPILATION BACKENDS
    // ============================================================================

    module CompilationBackends =
        
        let private runProcess (fileName: string) (arguments: string) (workingDir: string) : string * string * int =
            let psi = ProcessStartInfo()
            psi.FileName <- fileName
            psi.Arguments <- arguments
            psi.WorkingDirectory <- workingDir
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.UseShellExecute <- false
            psi.CreateNoWindow <- true

            use proc = Process.Start(psi)
            let stdout = proc.StandardOutput.ReadToEnd()
            let stderr = proc.StandardError.ReadToEnd()
            proc.WaitForExit()

            (stdout, stderr, proc.ExitCode)

        let compileManagedCuda (cudaCode: string) (options: CudaTranspilationOptions) : CudaTranspilationResult =
            let startTime = DateTime.UtcNow

            try
                // For managed CUDA, we generate C# wrapper code
                let wrapperCode = sprintf """
using System;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace TarsGenerated {
    public class CudaKernel {
        private CudaContext context;
        private CudaKernel kernel;

        public CudaKernel() {
            context = new CudaContext();
            // Load and compile CUDA code
        }

        public void Execute(params object[] parameters) {
            // Execute kernel with parameters
        }
    }
}

// Embedded CUDA code:
/*
%s
*/
""" cudaCode

                let outputPath = Path.Combine(options.OutputDirectory, "generated_kernel.cs")
                File.WriteAllText(outputPath, wrapperCode)

                {
                    Success = true
                    OutputPath = Some outputPath
                    ExecutablePath = None
                    CompilationLog = "Managed CUDA wrapper generated"
                    Errors = []
                    Warnings = []
                    CompilationTime = DateTime.UtcNow - startTime
                    BinarySize = Some (FileInfo(outputPath).Length)
                }
            with
            | ex ->
                {
                    Success = false
                    OutputPath = None
                    ExecutablePath = None
                    CompilationLog = ex.Message
                    Errors = [ex.Message]
                    Warnings = []
                    CompilationTime = DateTime.UtcNow - startTime
                    BinarySize = None
                }

        let compileWSL (cudaCode: string) (options: CudaTranspilationOptions) : CudaTranspilationResult =
            let startTime = DateTime.UtcNow
            
            try
                let sourceFile = Path.Combine(options.OutputDirectory, "kernel.cu")
                let outputFile = Path.Combine(options.OutputDirectory, "kernel")
                
                // Write CUDA source
                File.WriteAllText(sourceFile, cudaCode)
                
                // Compile using WSL
                let nvccPath = match options.Backend with
                                | WSLCompilation path -> path
                                | _ -> "nvcc"
                
                let compileArgs = sprintf "-O%d -arch=%s %s -o %s %s" 
                    options.OptimizationLevel 
                    options.Architecture
                    (if options.FastMath then "--use_fast_math" else "")
                    outputFile
                    sourceFile

                let wslCommand = sprintf "wsl %s %s" nvccPath compileArgs
                let (stdout, stderr, exitCode) = runProcess "cmd" ("/c " + wslCommand) options.OutputDirectory
                
                let success = exitCode = 0
                let errors = if stderr.Length > 0 then [stderr] else []
                
                {
                    Success = success
                    OutputPath = Some sourceFile
                    ExecutablePath = if success then Some outputFile else None
                    CompilationLog = stdout + "\n" + stderr
                    Errors = errors
                    Warnings = []
                    CompilationTime = DateTime.UtcNow - startTime
                    BinarySize = if success && File.Exists(outputFile) then Some (FileInfo(outputFile).Length) else None
                }
            with
            | ex ->
                {
                    Success = false
                    OutputPath = None
                    ExecutablePath = None
                    CompilationLog = ex.Message
                    Errors = [ex.Message]
                    Warnings = []
                    CompilationTime = DateTime.UtcNow - startTime
                    BinarySize = None
                }

        let compileDocker (cudaCode: string) (options: CudaTranspilationOptions) : CudaTranspilationResult =
            let startTime = DateTime.UtcNow
            
            try
                let sourceFile = Path.Combine(options.OutputDirectory, "kernel.cu")
                let outputFile = Path.Combine(options.OutputDirectory, "kernel")
                
                // Write CUDA source
                File.WriteAllText(sourceFile, cudaCode)
                
                // Create Dockerfile
                let dockerfile = sprintf """
FROM nvidia/cuda:11.8-devel-ubuntu20.04

WORKDIR /app
COPY kernel.cu .

RUN nvcc -O%d -arch=%s %s -o kernel kernel.cu

CMD ["./kernel"]
""" options.OptimizationLevel options.Architecture (if options.FastMath then "--use_fast_math" else "")

                let dockerfilePath = Path.Combine(options.OutputDirectory, "Dockerfile")
                File.WriteAllText(dockerfilePath, dockerfile)
                
                // Build Docker image
                let imageName = match options.Backend with
                                | DockerContainer name -> name
                                | _ -> "tars-cuda-kernel"
                
                let buildArgs = sprintf "build -t %s %s" imageName options.OutputDirectory
                let (stdout, stderr, exitCode) = runProcess "docker" buildArgs options.OutputDirectory
                
                let success = exitCode = 0
                let errors = if stderr.Length > 0 then [stderr] else []
                
                {
                    Success = success
                    OutputPath = Some sourceFile
                    ExecutablePath = if success then Some imageName else None
                    CompilationLog = stdout + "\n" + stderr
                    Errors = errors
                    Warnings = []
                    CompilationTime = DateTime.UtcNow - startTime
                    BinarySize = None  // Docker image size would need separate query
                }
            with
            | ex ->
                {
                    Success = false
                    OutputPath = None
                    ExecutablePath = None
                    CompilationLog = ex.Message
                    Errors = [ex.Message]
                    Warnings = []
                    CompilationTime = DateTime.UtcNow - startTime
                    BinarySize = None
                }

        let compileNative (cudaCode: string) (options: CudaTranspilationOptions) : CudaTranspilationResult =
            let startTime = DateTime.UtcNow
            
            try
                let sourceFile = Path.Combine(options.OutputDirectory, "kernel.cu")
                let outputFile = Path.Combine(options.OutputDirectory, "kernel.exe")
                
                // Write CUDA source
                File.WriteAllText(sourceFile, cudaCode)
                
                // Compile using native nvcc
                let nvccPath = match options.Backend with
                                | NativeCuda path -> path
                                | _ -> "nvcc"
                
                let compileArgs = sprintf "-O%d -arch=%s %s -o %s %s" 
                    options.OptimizationLevel 
                    options.Architecture
                    (if options.FastMath then "--use_fast_math" else "")
                    outputFile
                    sourceFile

                let (stdout, stderr, exitCode) = runProcess nvccPath compileArgs options.OutputDirectory
                
                let success = exitCode = 0
                let errors = if stderr.Length > 0 then [stderr] else []
                
                {
                    Success = success
                    OutputPath = Some sourceFile
                    ExecutablePath = if success then Some outputFile else None
                    CompilationLog = stdout + "\n" + stderr
                    Errors = errors
                    Warnings = []
                    CompilationTime = DateTime.UtcNow - startTime
                    BinarySize = if success && File.Exists(outputFile) then Some (FileInfo(outputFile).Length) else None
                }
            with
            | ex ->
                {
                    Success = false
                    OutputPath = None
                    ExecutablePath = None
                    CompilationLog = ex.Message
                    Errors = [ex.Message]
                    Warnings = []
                    CompilationTime = DateTime.UtcNow - startTime
                    BinarySize = None
                }

    // ============================================================================
    // MAIN TRANSPILATION API
    // ============================================================================

    let transpileAndCompile (fsharpCode: string) (kernelName: string) (options: CudaTranspilationOptions) : CudaTranspilationResult =
        try
            // Ensure output directory exists
            if not (Directory.Exists(options.OutputDirectory)) then
                Directory.CreateDirectory(options.OutputDirectory) |> ignore

            // Transpile F# to CUDA
            let kernelSpec = FSharpToCuda.transpileFSharpFunction fsharpCode kernelName
            let cudaCode = FSharpToCuda.generateCudaKernel kernelSpec

            // Add CUDA headers and main function
            let completeCudaCode = sprintf """
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

%s

int main() {
    printf("CUDA kernel compiled successfully\\n");
    return 0;
}
""" cudaCode

            // Compile based on backend
            match options.Backend with
            | ManagedCuda _ -> CompilationBackends.compileManagedCuda completeCudaCode options
            | WSLCompilation _ -> CompilationBackends.compileWSL completeCudaCode options
            | DockerContainer _ -> CompilationBackends.compileDocker completeCudaCode options
            | NativeCuda _ -> CompilationBackends.compileNative completeCudaCode options

        with
        | ex ->
            {
                Success = false
                OutputPath = None
                ExecutablePath = None
                CompilationLog = sprintf "Transpilation failed: %s" ex.Message
                Errors = [ex.Message]
                Warnings = []
                CompilationTime = TimeSpan.Zero
                BinarySize = None
            }

    // ============================================================================
    // METASCRIPT INTEGRATION
    // ============================================================================

    /// CUDA transpilation function for use in metascripts
    let cudaTranspile (code: string) (kernelName: string) (backend: string) (outputDir: string) : CudaTranspilationResult =
        let cudaBackend = 
            match backend.ToLower() with
            | "managed" -> ManagedCuda("")
            | "wsl" -> WSLCompilation("nvcc")
            | "docker" -> DockerContainer("tars-cuda")
            | "native" -> NativeCuda("nvcc")
            | _ -> WSLCompilation("nvcc")  // Default to WSL

        let options = {
            Backend = cudaBackend
            OptimizationLevel = 2
            Architecture = "sm_75"
            DebugInfo = false
            FastMath = true
            OutputDirectory = outputDir
            IncludePaths = []
            Libraries = []
        }

        transpileAndCompile code kernelName options

    /// Quick CUDA compilation for metascripts
    let quickCudaCompile (fsharpCode: string) : string =
        let tempDir = Path.Combine(Path.GetTempPath(), "tars_cuda_" + Guid.NewGuid().ToString("N").[..7])
        let result = cudaTranspile fsharpCode "generated_kernel" "wsl" tempDir
        
        if result.Success then
            sprintf "✅ CUDA compilation successful in %.2f ms\n📁 Output: %s\n🚀 Executable: %s" 
                result.CompilationTime.TotalMilliseconds
                (result.OutputPath |> Option.defaultValue "N/A")
                (result.ExecutablePath |> Option.defaultValue "N/A")
        else
            sprintf "❌ CUDA compilation failed:\n%s\nErrors: %s" 
                result.CompilationLog
                (String.concat "\n" result.Errors)
