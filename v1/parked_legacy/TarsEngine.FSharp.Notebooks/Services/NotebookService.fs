namespace TarsEngine.FSharp.Notebooks.Services

open System
open System.IO
open System.Net.Http
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Notebooks.Types
open TarsEngine.FSharp.Notebooks.Generation
open TarsEngine.FSharp.Notebooks.Serialization
open TarsEngine.FSharp.Notebooks.Discovery
open TarsEngine.FSharp.Notebooks.Execution

/// <summary>
/// High-level service for notebook operations
/// </summary>

/// Notebook service configuration
type NotebookServiceConfig = {
    DefaultKernel: SupportedKernel
    WorkingDirectory: string
    EnableExecution: bool
    EnableDiscovery: bool
    CacheDirectory: string option
    MaxConcurrentExecutions: int
}

/// Notebook service
type NotebookService(
    config: NotebookServiceConfig,
    httpClient: HttpClient,
    logger: ILogger<NotebookService>) =
    
    let kernelManager = KernelManager(logger.CreateLogger<KernelManager>())
    let executor = NotebookExecutor(kernelManager, logger.CreateLogger<NotebookExecutor>())
    let discoveryService = NotebookDiscoveryService(httpClient)
    let qualityAssessor = NotebookQualityAssessor()
    
    /// Create a new notebook from template
    member _.CreateNotebookAsync(name: string, strategy: NotebookGenerationStrategy) : Async<JupyterNotebook> = async {
        try
            logger.LogInformation("Creating notebook: {Name} with strategy: {Strategy}", name, strategy)
            
            // Create basic analysis for template
            let analysis = {
                FilePath = $"{name}.trsx"
                Agents = []
                Variables = []
                Actions = []
                DataSources = []
                Dependencies = { Nodes = []; Edges = []; Levels = Map.empty }
                Narrative = {
                    Title = name
                    Objective = "Template-based notebook for " + strategy.ToString()
                    Sections = []
                    Conclusion = None
                }
                Complexity = {
                    AgentCount = 0
                    ActionCount = 0
                    VariableCount = 0
                    DependencyDepth = 0
                    EstimatedExecutionTime = TimeSpan.FromMinutes(5.0)
                    ComplexityScore = 10.0
                }
            }
            
            // Generate notebook
            let! notebook = NotebookGenerator.generateNotebook analysis strategy
            
            logger.LogInformation("Notebook created successfully with {CellCount} cells", notebook.Cells.Length)
            return notebook
            
        with
        | ex ->
            logger.LogError(ex, "Failed to create notebook: {Name}", name)
            return failwith $"Failed to create notebook: {ex.Message}"
    }
    
    /// Generate notebook from metascript
    member _.GenerateFromMetascriptAsync(metascriptPath: string, strategy: NotebookGenerationStrategy) : Async<JupyterNotebook> = async {
        try
            if not (File.Exists(metascriptPath)) then
                return failwith $"Metascript file not found: {metascriptPath}"
            
            logger.LogInformation("Generating notebook from metascript: {MetascriptPath}", metascriptPath)
            
            // Analyze metascript
            let! analysis = MetascriptAnalyzer.analyzeMetascript metascriptPath
            
            logger.LogInformation("Metascript analysis complete - Complexity: {ComplexityScore:F1}/100", analysis.Complexity.ComplexityScore)
            
            // Generate notebook
            let! notebook = NotebookGenerator.generateNotebook analysis strategy
            
            logger.LogInformation("Notebook generated successfully with {CellCount} cells", notebook.Cells.Length)
            return notebook
            
        with
        | ex ->
            logger.LogError(ex, "Failed to generate notebook from metascript: {MetascriptPath}", metascriptPath)
            return failwith $"Failed to generate notebook: {ex.Message}"
    }
    
    /// Load notebook from file
    member _.LoadNotebookAsync(filePath: string) : Async<JupyterNotebook> = async {
        try
            if not (File.Exists(filePath)) then
                return failwith $"Notebook file not found: {filePath}"
            
            logger.LogInformation("Loading notebook from: {FilePath}", filePath)
            
            let json = File.ReadAllText(filePath)
            match NotebookSerialization.deserializeFromJson json with
            | Ok notebook ->
                logger.LogInformation("Notebook loaded successfully with {CellCount} cells", notebook.Cells.Length)
                return notebook
            | Error error ->
                logger.LogError("Failed to deserialize notebook: {Error}", error)
                return failwith $"Failed to load notebook: {error}"
                
        with
        | ex ->
            logger.LogError(ex, "Failed to load notebook: {FilePath}", filePath)
            return failwith $"Failed to load notebook: {ex.Message}"
    }
    
    /// Save notebook to file
    member _.SaveNotebookAsync(notebook: JupyterNotebook, filePath: string) : Async<unit> = async {
        try
            logger.LogInformation("Saving notebook to: {FilePath}", filePath)
            
            let json = NotebookSerialization.serializeToJson notebook
            File.WriteAllText(filePath, json)
            
            logger.LogInformation("Notebook saved successfully")
            
        with
        | ex ->
            logger.LogError(ex, "Failed to save notebook: {FilePath}", filePath)
            return failwith $"Failed to save notebook: {ex.Message}"
    }
    
    /// Execute notebook
    member _.ExecuteNotebookAsync(notebook: JupyterNotebook, options: ExecutionOptions option) : Async<NotebookExecutionResult> = async {
        try
            if not config.EnableExecution then
                return failwith "Notebook execution is disabled in configuration"
            
            logger.LogInformation("Executing notebook with {CellCount} cells", notebook.Cells.Length)
            
            let execOptions = options |> Option.defaultValue (ExecutionUtils.createDefaultOptions())
            let! result = executor.ExecuteNotebookAsync(notebook, execOptions)
            
            logger.LogInformation("Notebook execution completed - Success: {Success}, Time: {Time:F1}s", 
                result.Success, result.TotalExecutionTime.TotalSeconds)
            
            return result
            
        with
        | ex ->
            logger.LogError(ex, "Failed to execute notebook")
            return failwith $"Failed to execute notebook: {ex.Message}"
    }
    
    /// Execute notebook and save results
    member this.ExecuteAndSaveNotebookAsync(inputPath: string, outputPath: string option, options: ExecutionOptions option) : Async<NotebookExecutionResult> = async {
        try
            // Load notebook
            let! notebook = this.LoadNotebookAsync(inputPath)
            
            // Execute notebook
            let! (updatedNotebook, result) = executor.ExecuteAndUpdateNotebookAsync(notebook, options |> Option.defaultValue (ExecutionUtils.createDefaultOptions()))
            
            // Save updated notebook
            let savePath = outputPath |> Option.defaultValue inputPath
            do! this.SaveNotebookAsync(updatedNotebook, savePath)
            
            logger.LogInformation("Notebook executed and saved to: {OutputPath}", savePath)
            return result
            
        with
        | ex ->
            logger.LogError(ex, "Failed to execute and save notebook")
            return failwith $"Failed to execute and save notebook: {ex.Message}"
    }
    
    /// Search for notebooks
    member _.SearchNotebooksAsync(criteria: SearchCriteria) : Async<SearchResult> = async {
        try
            if not config.EnableDiscovery then
                return failwith "Notebook discovery is disabled in configuration"
            
            logger.LogInformation("Searching for notebooks: {Query} on {Source}", criteria.Query, criteria.Source)
            
            let! result = discoveryService.SearchAsync(criteria)
            
            logger.LogInformation("Search completed - Found {Count} results in {Time:F0}ms", 
                result.TotalCount, result.SearchTime.TotalMilliseconds)
            
            return result
            
        with
        | ex ->
            logger.LogError(ex, "Failed to search notebooks")
            return failwith $"Failed to search notebooks: {ex.Message}"
    }
    
    /// Download notebook from URL
    member _.DownloadNotebookAsync(url: string, outputPath: string) : Async<bool> = async {
        try
            logger.LogInformation("Downloading notebook from: {Url}", url)
            
            let! success = discoveryService.DownloadNotebookAsync(url, outputPath)
            
            if success then
                logger.LogInformation("Notebook downloaded successfully to: {OutputPath}", outputPath)
            else
                logger.LogWarning("Failed to download notebook from: {Url}", url)
            
            return success
            
        with
        | ex ->
            logger.LogError(ex, "Failed to download notebook from: {Url}", url)
            return false
    }
    
    /// Assess notebook quality
    member _.AssessQualityAsync(notebook: JupyterNotebook) : Async<QualityAssessment> = async {
        try
            logger.LogInformation("Assessing notebook quality")
            
            let assessment = qualityAssessor.AssessQuality(notebook)
            
            logger.LogInformation("Quality assessment completed - Score: {Score:F1}/100 ({Grade})", 
                assessment.Metrics.OverallScore, assessment.Grade)
            
            return assessment
            
        with
        | ex ->
            logger.LogError(ex, "Failed to assess notebook quality")
            return failwith $"Failed to assess notebook quality: {ex.Message}"
    }
    
    /// Validate notebook
    member _.ValidateNotebookAsync(notebook: JupyterNotebook) : Async<ValidationResult> = async {
        try
            logger.LogInformation("Validating notebook")
            
            let validator = NotebookValidator()
            let result = validator.ValidateNotebook(notebook)
            
            logger.LogInformation("Validation completed - Valid: {IsValid}, Errors: {ErrorCount}, Warnings: {WarningCount}", 
                result.IsValid, result.Errors.Length, result.Warnings.Length)
            
            return result
            
        with
        | ex ->
            logger.LogError(ex, "Failed to validate notebook")
            return failwith $"Failed to validate notebook: {ex.Message}"
    }
    
    /// Convert notebook to different format
    member _.ConvertNotebookAsync(notebook: JupyterNotebook, format: OutputFormat, outputPath: string) : Async<OutputGenerationResult> = async {
        try
            logger.LogInformation("Converting notebook to {Format}", format)
            
            let outputService = OutputGenerationService()
            let request = {
                Notebook = notebook
                Format = format
                OutputPath = Some outputPath
                Options = outputService.CreateDefaultOptions()
            }
            
            let! result = outputService.GenerateOutputAsync(request)
            
            if result.Success then
                logger.LogInformation("Notebook converted successfully to: {OutputPath}", result.OutputPath)
            else
                logger.LogWarning("Notebook conversion failed with {ErrorCount} errors", result.Errors.Length)
            
            return result
            
        with
        | ex ->
            logger.LogError(ex, "Failed to convert notebook")
            return failwith $"Failed to convert notebook: {ex.Message}"
    }
    
    /// Get notebook statistics
    member _.GetNotebookStatistics(notebook: JupyterNotebook) : NotebookStatistics =
        let codeCells = notebook.Cells |> List.choose (function | CodeCell cd -> Some cd | _ -> None)
        let markdownCells = notebook.Cells |> List.choose (function | MarkdownCell md -> Some md | _ -> None)
        let rawCells = notebook.Cells |> List.choose (function | RawCell rd -> Some rd | _ -> None)
        
        let totalLines = 
            notebook.Cells 
            |> List.sumBy (function
                | CodeCell cd -> cd.Source.Length
                | MarkdownCell md -> md.Source.Length
                | RawCell rd -> rd.Source.Length)
        
        let codeLines = codeCells |> List.sumBy (fun cd -> cd.Source.Length)
        let markdownLines = markdownCells |> List.sumBy (fun md -> md.Source.Length)
        
        {
            TotalCells = notebook.Cells.Length
            CodeCells = codeCells.Length
            MarkdownCells = markdownCells.Length
            RawCells = rawCells.Length
            TotalLines = totalLines
            CodeLines = codeLines
            MarkdownLines = markdownLines
            HasOutputs = codeCells |> List.exists (fun cd -> cd.Outputs.IsSome && not cd.Outputs.Value.IsEmpty)
            EstimatedReadingTime = TimeSpan.FromMinutes(float markdownLines * 0.1)
            EstimatedExecutionTime = TimeSpan.FromMinutes(float codeLines * 0.05)
        }

/// Notebook statistics
and NotebookStatistics = {
    TotalCells: int
    CodeCells: int
    MarkdownCells: int
    RawCells: int
    TotalLines: int
    CodeLines: int
    MarkdownLines: int
    HasOutputs: bool
    EstimatedReadingTime: TimeSpan
    EstimatedExecutionTime: TimeSpan
}

/// Service utilities
module ServiceUtils =
    
    /// Create default service configuration
    let createDefaultConfig() = {
        DefaultKernel = Python { Version = "3.9"; Packages = []; VirtualEnv = None }
        WorkingDirectory = Environment.CurrentDirectory
        EnableExecution = true
        EnableDiscovery = true
        CacheDirectory = None
        MaxConcurrentExecutions = 1
    }
    
    /// Create service configuration with custom settings
    let createConfig defaultKernel workingDir enableExecution enableDiscovery = {
        DefaultKernel = defaultKernel
        WorkingDirectory = workingDir
        EnableExecution = enableExecution
        EnableDiscovery = enableDiscovery
        CacheDirectory = None
        MaxConcurrentExecutions = 1
    }
    
    /// Format notebook statistics
    let formatStatistics (stats: NotebookStatistics) : string =
        let sb = System.Text.StringBuilder()
        
        sb.AppendLine($"📊 Notebook Statistics") |> ignore
        sb.AppendLine($"Total Cells: {stats.TotalCells}") |> ignore
        sb.AppendLine($"  Code: {stats.CodeCells}") |> ignore
        sb.AppendLine($"  Markdown: {stats.MarkdownCells}") |> ignore
        sb.AppendLine($"  Raw: {stats.RawCells}") |> ignore
        sb.AppendLine($"Total Lines: {stats.TotalLines}") |> ignore
        sb.AppendLine($"  Code: {stats.CodeLines}") |> ignore
        sb.AppendLine($"  Markdown: {stats.MarkdownLines}") |> ignore
        sb.AppendLine($"Has Outputs: {if stats.HasOutputs then "Yes" else "No"}") |> ignore
        sb.AppendLine($"Est. Reading Time: {stats.EstimatedReadingTime.TotalMinutes:F1} min") |> ignore
        sb.AppendLine($"Est. Execution Time: {stats.EstimatedExecutionTime.TotalMinutes:F1} min") |> ignore
        
        sb.ToString()
