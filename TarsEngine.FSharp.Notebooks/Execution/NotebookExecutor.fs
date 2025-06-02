namespace TarsEngine.FSharp.Notebooks.Execution

open System
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Notebooks.Types

/// <summary>
/// Notebook execution engine
/// </summary>

/// Execution options
type ExecutionOptions = {
    Timeout: TimeSpan option
    AllowErrors: bool
    CaptureOutput: bool
    WorkingDirectory: string option
    Environment: Map<string, string>
    KernelSpec: SupportedKernel option
}

/// Cell execution result
type CellExecutionResult = {
    CellIndex: int
    Success: bool
    ExecutionTime: TimeSpan
    Output: string list
    Error: string option
    Data: Map<string, obj>
    Metadata: Map<string, obj>
}

/// Notebook execution result
type NotebookExecutionResult = {
    Notebook: JupyterNotebook
    Success: bool
    TotalExecutionTime: TimeSpan
    CellResults: CellExecutionResult list
    Errors: string list
    KernelId: string option
}

/// Notebook executor
type NotebookExecutor(kernelManager: IKernelManager, logger: ILogger<NotebookExecutor>) =
    
    /// Execute entire notebook
    member _.ExecuteNotebookAsync(notebook: JupyterNotebook, options: ExecutionOptions) : Async<NotebookExecutionResult> = async {
        let startTime = DateTime.UtcNow
        let cellResults = ResizeArray<CellExecutionResult>()
        let errors = ResizeArray<string>()
        let mutable kernelId = None
        let mutable overallSuccess = true
        
        try
            // Determine kernel to use
            let kernelSpec = 
                match options.KernelSpec with
                | Some spec -> spec
                | None ->
                    match notebook.Metadata.KernelSpec with
                    | Some ks -> 
                        // Convert kernel spec to supported kernel (simplified)
                        match ks.Language with
                        | Some "python" -> Python { Version = "3.9"; Packages = []; VirtualEnv = None }
                        | Some "fsharp" -> FSharp { DotNetVersion = "9.0"; Packages = []; References = [] }
                        | Some "csharp" -> CSharp { DotNetVersion = "9.0"; Packages = []; References = [] }
                        | _ -> Python { Version = "3.9"; Packages = []; VirtualEnv = None }
                    | None -> Python { Version = "3.9"; Packages = []; VirtualEnv = None }
            
            // Start kernel
            logger.LogInformation("Starting kernel for notebook execution")
            let! kernel = kernelManager.StartKernelAsync(kernelSpec)
            kernelId <- Some kernel.KernelId
            
            // Execute each code cell
            let codeCells = 
                notebook.Cells 
                |> List.mapi (fun i cell -> (i, cell))
                |> List.choose (fun (i, cell) -> 
                    match cell with 
                    | CodeCell codeData -> Some (i, codeData)
                    | _ -> None)
            
            for (cellIndex, codeData) in codeCells do
                try
                    logger.LogInformation("Executing cell {CellIndex}", cellIndex)
                    let! result = this.ExecuteCellAsync(kernel.KernelId, cellIndex, codeData, options)
                    cellResults.Add(result)
                    
                    if not result.Success then
                        overallSuccess <- false
                        if not options.AllowErrors then
                            logger.LogWarning("Cell execution failed and AllowErrors is false, stopping execution")
                            break
                        
                with
                | ex ->
                    logger.LogError(ex, "Error executing cell {CellIndex}", cellIndex)
                    errors.Add($"Cell {cellIndex}: {ex.Message}")
                    overallSuccess <- false
                    
                    let errorResult = {
                        CellIndex = cellIndex
                        Success = false
                        ExecutionTime = TimeSpan.Zero
                        Output = []
                        Error = Some ex.Message
                        Data = Map.empty
                        Metadata = Map.empty
                    }
                    cellResults.Add(errorResult)
                    
                    if not options.AllowErrors then
                        break
            
            // Stop kernel
            match kernelId with
            | Some kid ->
                let! stopped = kernelManager.StopKernelAsync(kid)
                if not stopped then
                    logger.LogWarning("Failed to stop kernel {KernelId}", kid)
            | None -> ()
            
            let totalTime = DateTime.UtcNow - startTime
            
            return {
                Notebook = notebook
                Success = overallSuccess
                TotalExecutionTime = totalTime
                CellResults = cellResults |> List.ofSeq
                Errors = errors |> List.ofSeq
                KernelId = kernelId
            }
            
        with
        | ex ->
            logger.LogError(ex, "Failed to execute notebook")
            
            // Try to stop kernel if it was started
            match kernelId with
            | Some kid ->
                try
                    let! stopped = kernelManager.StopKernelAsync(kid)
                    ()
                with
                | _ -> ()
            | None -> ()
            
            let totalTime = DateTime.UtcNow - startTime
            
            return {
                Notebook = notebook
                Success = false
                TotalExecutionTime = totalTime
                CellResults = cellResults |> List.ofSeq
                Errors = [ex.Message]
                KernelId = kernelId
            }
    }
    
    /// Execute single cell
    member private _.ExecuteCellAsync(kernelId: string, cellIndex: int, codeData: CodeCellData, options: ExecutionOptions) : Async<CellExecutionResult> = async {
        let startTime = DateTime.UtcNow
        
        try
            let code = String.Join("\n", codeData.Source)
            
            if String.IsNullOrWhiteSpace(code) then
                return {
                    CellIndex = cellIndex
                    Success = true
                    ExecutionTime = TimeSpan.Zero
                    Output = []
                    Error = None
                    Data = Map.empty
                    Metadata = Map.empty
                }
            else
                let request = KernelUtils.createExecutionRequest code
                
                let! result = kernelManager.ExecuteAsync(kernelId, request)
                let executionTime = DateTime.UtcNow - startTime
                
                return {
                    CellIndex = cellIndex
                    Success = result.Success
                    ExecutionTime = executionTime
                    Output = if String.IsNullOrEmpty(result.Output) then [] else [result.Output]
                    Error = result.Error
                    Data = result.Data
                    Metadata = result.Metadata
                }
                
        with
        | ex ->
            let executionTime = DateTime.UtcNow - startTime
            return {
                CellIndex = cellIndex
                Success = false
                ExecutionTime = executionTime
                Output = []
                Error = Some ex.Message
                Data = Map.empty
                Metadata = Map.empty
            }
    }
    
    /// Execute notebook and update with results
    member this.ExecuteAndUpdateNotebookAsync(notebook: JupyterNotebook, options: ExecutionOptions) : Async<JupyterNotebook * NotebookExecutionResult> = async {
        let! result = this.ExecuteNotebookAsync(notebook, options)
        
        // Update notebook with execution results
        let updatedCells = 
            notebook.Cells
            |> List.mapi (fun i cell ->
                match cell with
                | CodeCell codeData ->
                    // Find corresponding execution result
                    let cellResult = 
                        result.CellResults 
                        |> List.tryFind (fun r -> r.CellIndex = i)
                    
                    match cellResult with
                    | Some res ->
                        let outputs = 
                            if res.Success && not res.Output.IsEmpty then
                                Some [
                                    {
                                        OutputType = "stream"
                                        Name = Some "stdout"
                                        Text = res.Output
                                        Data = Map.empty
                                        Metadata = Map.empty
                                        ExecutionCount = None
                                    }
                                ]
                            elif not res.Success && res.Error.IsSome then
                                Some [
                                    {
                                        OutputType = "error"
                                        Name = Some "stderr"
                                        Text = [res.Error.Value]
                                        Data = Map.empty
                                        Metadata = Map.empty
                                        ExecutionCount = None
                                    }
                                ]
                            else
                                codeData.Outputs
                        
                        CodeCell { codeData with Outputs = outputs }
                    | None ->
                        cell
                | _ -> cell
            )
        
        let updatedNotebook = { notebook with Cells = updatedCells }
        
        return (updatedNotebook, result)
    }

/// Execution utilities
module ExecutionUtils =
    
    /// Create default execution options
    let createDefaultOptions() = {
        Timeout = Some (TimeSpan.FromMinutes(5.0))
        AllowErrors = false
        CaptureOutput = true
        WorkingDirectory = None
        Environment = Map.empty
        KernelSpec = None
    }
    
    /// Create execution options with timeout
    let createOptionsWithTimeout (timeout: TimeSpan) = {
        createDefaultOptions() with Timeout = Some timeout
    }
    
    /// Create execution options allowing errors
    let createOptionsAllowingErrors() = {
        createDefaultOptions() with AllowErrors = true
    }
    
    /// Format execution result
    let formatExecutionResult (result: NotebookExecutionResult) : string =
        let sb = System.Text.StringBuilder()
        
        sb.AppendLine($"📊 Notebook Execution Result") |> ignore
        sb.AppendLine($"Success: {if result.Success then "✅" else "❌"}") |> ignore
        sb.AppendLine($"Total Time: {result.TotalExecutionTime.TotalSeconds:F1}s") |> ignore
        sb.AppendLine($"Cells Executed: {result.CellResults.Length}") |> ignore
        
        let successfulCells = result.CellResults |> List.filter (fun r -> r.Success) |> List.length
        let failedCells = result.CellResults.Length - successfulCells
        
        sb.AppendLine($"Successful: {successfulCells}") |> ignore
        sb.AppendLine($"Failed: {failedCells}") |> ignore
        
        if not result.Errors.IsEmpty then
            sb.AppendLine() |> ignore
            sb.AppendLine("❌ Errors:") |> ignore
            for error in result.Errors do
                sb.AppendLine($"  • {error}") |> ignore
        
        if failedCells > 0 then
            sb.AppendLine() |> ignore
            sb.AppendLine("📋 Failed Cells:") |> ignore
            for cellResult in result.CellResults do
                if not cellResult.Success then
                    sb.AppendLine($"  Cell {cellResult.CellIndex}: {cellResult.Error |> Option.defaultValue "Unknown error"}") |> ignore
        
        sb.ToString()
    
    /// Get execution summary
    let getExecutionSummary (result: NotebookExecutionResult) : string =
        let successfulCells = result.CellResults |> List.filter (fun r -> r.Success) |> List.length
        let totalCells = result.CellResults.Length
        let successRate = if totalCells > 0 then float successfulCells / float totalCells * 100.0 else 0.0
        
        if result.Success then
            $"✅ {successfulCells}/{totalCells} cells executed successfully ({successRate:F0}%) in {result.TotalExecutionTime.TotalSeconds:F1}s"
        else
            $"❌ {successfulCells}/{totalCells} cells executed successfully ({successRate:F0}%) in {result.TotalExecutionTime.TotalSeconds:F1}s"
    
    /// Check if notebook can be executed
    let canExecuteNotebook (notebook: JupyterNotebook) : bool * string list =
        let issues = ResizeArray<string>()
        
        // Check if notebook has code cells
        let codeCells = notebook.Cells |> List.choose (function | CodeCell cd -> Some cd | _ -> None)
        if codeCells.IsEmpty then
            issues.Add("Notebook contains no code cells")
        
        // Check if kernel spec is available
        match notebook.Metadata.KernelSpec with
        | Some kernelSpec ->
            match kernelSpec.Language with
            | Some "python" ->
                if not (KernelUtils.isKernelAvailable (Python { Version = "3.9"; Packages = []; VirtualEnv = None })) then
                    issues.Add("Python kernel not available")
            | Some "fsharp" ->
                if not (KernelUtils.isKernelAvailable (FSharp { DotNetVersion = "9.0"; Packages = []; References = [] })) then
                    issues.Add("F# kernel not available")
            | Some lang ->
                issues.Add($"Unsupported kernel language: {lang}")
            | None ->
                issues.Add("Kernel language not specified")
        | None ->
            issues.Add("No kernel specification found")
        
        let canExecute = issues.Count = 0
        (canExecute, issues |> List.ofSeq)
