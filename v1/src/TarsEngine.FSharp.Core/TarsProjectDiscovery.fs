namespace TarsEngine.FSharp.Core

open System
open System.IO
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Enhanced TARS project discovery for auto-improvement system
/// Fixes the critical issue where 0 projects were found in evolution experiments
module TarsProjectDiscovery =

    /// TARS project information with detailed metrics
    type TarsProjectInfo = {
        Name: string
        ProjectPath: string
        ProjectType: string
        SourceFiles: string array
        LineCount: int
        ComplexityScore: int
        HasTests: bool
        BuildSuccess: bool option
        LastModified: DateTime
        Dependencies: string array
    }

    /// Discovery result with comprehensive information
    type TarsDiscoveryResult = {
        Success: bool
        TarsRootPath: string
        ProjectsFound: TarsProjectInfo array
        TotalSourceFiles: int
        TotalLineCount: int
        ErrorMessage: string option
        SearchPaths: string array
        DiscoveryTimeMs: int64
    }

    /// Enhanced TARS project discovery service
    type TarsProjectDiscoveryService(logger: ILogger<TarsProjectDiscoveryService>) =

        /// Find TARS root directory using multiple search strategies
        member this.FindTarsRoot(startPath: string) : string option =
            let searchPaths = [
                startPath
                Path.GetDirectoryName(startPath)
                Path.Combine(startPath, "..")
                Path.Combine(startPath, "../..")
                Path.Combine(startPath, "../../..")
                @"C:\Users\spare\source\repos\tars"
                Environment.CurrentDirectory
                Path.Combine(Environment.CurrentDirectory, "..")
            ]

            let tarsIndicators = [
                "TarsEngine.FSharp.Cli.fsproj"
                "TarsEngine.FSharp.Core.fsproj"
                "TARS.sln"
                "tars.sln"
            ]

            logger.LogInformation("Starting TARS root discovery...")

            let mutable result = None
            for searchPath in searchPaths do
                if result.IsNone then
                    try
                        if Directory.Exists(searchPath) then
                            logger.LogDebug($"Searching in: {searchPath}")

                            for indicator in tarsIndicators do
                                if result.IsNone then
                                    let foundFiles = Directory.GetFiles(searchPath, indicator, SearchOption.AllDirectories)
                                    if foundFiles.Length > 0 then
                                        let tarsRoot = Path.GetDirectoryName(foundFiles.[0])
                                        logger.LogInformation($"✅ Found TARS root: {tarsRoot}")
                                        result <- Some tarsRoot
                    with
                    | ex ->
                        logger.LogDebug(ex, $"Error searching path: {searchPath}")

            if result.IsNone then
                logger.LogWarning("❌ TARS root not found in any search path")
            result

        /// Analyze F# project file for detailed information
        member this.AnalyzeProject(projectPath: string) : TarsProjectInfo option =
            try
                if not (File.Exists(projectPath)) then
                    None
                else
                    let projectDir = Path.GetDirectoryName(projectPath)
                    let projectName = Path.GetFileNameWithoutExtension(projectPath)
                    
                    // Find F# source files
                    let sourceFiles = 
                        Directory.GetFiles(projectDir, "*.fs", SearchOption.AllDirectories)
                        |> Array.filter (fun f -> 
                            not (f.Contains("bin")) && 
                            not (f.Contains("obj")) &&
                            not (f.Contains("packages")))

                    // Calculate metrics
                    let mutable totalLines = 0
                    let mutable complexityScore = 0

                    for sourceFile in sourceFiles do
                        try
                            let lines = File.ReadAllLines(sourceFile)
                            totalLines <- totalLines + lines.Length
                            
                            // Simple complexity calculation
                            for line in lines do
                                let trimmed = line.Trim()
                                if trimmed.Contains("match") || trimmed.Contains("if") || 
                                   trimmed.Contains("try") || trimmed.Contains("async") ||
                                   trimmed.Contains("let rec") then
                                    complexityScore <- complexityScore + 1
                        with
                        | ex -> logger.LogDebug(ex, $"Error analyzing file: {sourceFile}")

                    // Check for tests
                    let hasTests = 
                        projectName.Contains("Test") || 
                        sourceFiles |> Array.exists (fun f -> f.Contains("Test"))

                    // Read project file for dependencies
                    let projectContent = File.ReadAllText(projectPath)
                    let dependencies = 
                        projectContent.Split('\n')
                        |> Array.filter (fun line -> line.Contains("PackageReference"))
                        |> Array.map (fun line -> 
                            let parts = line.Split('"')
                            if parts.Length > 1 then parts.[1] else "Unknown")

                    let projectInfo = {
                        Name = projectName
                        ProjectPath = projectPath
                        ProjectType = "F# Project"
                        SourceFiles = sourceFiles
                        LineCount = totalLines
                        ComplexityScore = complexityScore
                        HasTests = hasTests
                        BuildSuccess = None // Will be determined later
                        LastModified = File.GetLastWriteTime(projectPath)
                        Dependencies = dependencies
                    }

                    logger.LogDebug($"Analyzed project: {projectName} ({totalLines} lines, {sourceFiles.Length} files)")
                    Some projectInfo
            with
            | ex ->
                logger.LogWarning(ex, $"Failed to analyze project: {projectPath}")
                None

        /// Discover all TARS projects with comprehensive analysis
        member this.DiscoverTarsProjects(startPath: string) : Async<TarsDiscoveryResult> = async {
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            
            try
                logger.LogInformation("🔍 Starting comprehensive TARS project discovery...")
                
                match this.FindTarsRoot(startPath) with
                | None ->
                    stopwatch.Stop()
                    return {
                        Success = false
                        TarsRootPath = ""
                        ProjectsFound = [||]
                        TotalSourceFiles = 0
                        TotalLineCount = 0
                        ErrorMessage = Some "TARS root directory not found"
                        SearchPaths = [| startPath |]
                        DiscoveryTimeMs = stopwatch.ElapsedMilliseconds
                    }
                | Some tarsRoot ->
                    logger.LogInformation($"📁 Analyzing TARS projects in: {tarsRoot}")
                    
                    // Find all F# project files
                    let projectFiles = Directory.GetFiles(tarsRoot, "*.fsproj", SearchOption.AllDirectories)
                    logger.LogInformation($"Found {projectFiles.Length} project files")
                    
                    // Analyze each project
                    let projects = ResizeArray<TarsProjectInfo>()
                    
                    for projectFile in projectFiles do
                        match this.AnalyzeProject(projectFile) with
                        | Some projectInfo -> projects.Add(projectInfo)
                        | None -> ()
                    
                    let projectArray = projects.ToArray()
                    let totalSourceFiles = projectArray |> Array.sumBy (fun p -> p.SourceFiles.Length)
                    let totalLineCount = projectArray |> Array.sumBy (fun p -> p.LineCount)
                    
                    stopwatch.Stop()
                    
                    logger.LogInformation($"✅ Discovery completed: {projectArray.Length} projects, {totalSourceFiles} source files, {totalLineCount} lines")
                    
                    return {
                        Success = true
                        TarsRootPath = tarsRoot
                        ProjectsFound = projectArray
                        TotalSourceFiles = totalSourceFiles
                        TotalLineCount = totalLineCount
                        ErrorMessage = None
                        SearchPaths = [| startPath; tarsRoot |]
                        DiscoveryTimeMs = stopwatch.ElapsedMilliseconds
                    }
            with
            | ex ->
                stopwatch.Stop()
                logger.LogError(ex, "❌ TARS project discovery failed")
                return {
                    Success = false
                    TarsRootPath = ""
                    ProjectsFound = [||]
                    TotalSourceFiles = 0
                    TotalLineCount = 0
                    ErrorMessage = Some ex.Message
                    SearchPaths = [| startPath |]
                    DiscoveryTimeMs = stopwatch.ElapsedMilliseconds
                }
        }

        /// Quick discovery for performance-critical scenarios
        member this.QuickDiscovery(startPath: string) : TarsDiscoveryResult =
            try
                match this.FindTarsRoot(startPath) with
                | None -> 
                    {
                        Success = false
                        TarsRootPath = ""
                        ProjectsFound = [||]
                        TotalSourceFiles = 0
                        TotalLineCount = 0
                        ErrorMessage = Some "TARS root not found"
                        SearchPaths = [| startPath |]
                        DiscoveryTimeMs = 0L
                    }
                | Some tarsRoot ->
                    let projectFiles = Directory.GetFiles(tarsRoot, "*.fsproj", SearchOption.AllDirectories)
                    let sourceFiles = Directory.GetFiles(tarsRoot, "*.fs", SearchOption.AllDirectories)
                                     |> Array.filter (fun f -> not (f.Contains("bin")) && not (f.Contains("obj")))
                    
                    {
                        Success = true
                        TarsRootPath = tarsRoot
                        ProjectsFound = [||] // Quick mode doesn't analyze projects
                        TotalSourceFiles = sourceFiles.Length
                        TotalLineCount = 0 // Quick mode doesn't count lines
                        ErrorMessage = None
                        SearchPaths = [| startPath; tarsRoot |]
                        DiscoveryTimeMs = 0L
                    }
            with
            | ex ->
                logger.LogError(ex, "Quick discovery failed")
                {
                    Success = false
                    TarsRootPath = ""
                    ProjectsFound = [||]
                    TotalSourceFiles = 0
                    TotalLineCount = 0
                    ErrorMessage = Some ex.Message
                    SearchPaths = [| startPath |]
                    DiscoveryTimeMs = 0L
                }

        /// Test if a path contains TARS projects
        member this.IsTarsRepository(path: string) : bool =
            try
                if not (Directory.Exists(path)) then false
                else
                    let indicators = [
                        "TarsEngine.FSharp.Cli.fsproj"
                        "TarsEngine.FSharp.Core.fsproj"
                        "TARS.sln"
                    ]
                    
                    indicators |> List.exists (fun indicator ->
                        Directory.GetFiles(path, indicator, SearchOption.AllDirectories).Length > 0)
            with
            | _ -> false

    /// Static helper functions for quick access
    module TarsDiscoveryHelpers =
        
        /// Quick check if current directory contains TARS
        let isCurrentDirectoryTars (logger: ILogger<_>) =
            let discovery = TarsProjectDiscoveryService(logger)
            discovery.IsTarsRepository(Environment.CurrentDirectory)

        /// Find TARS root from current directory
        let findTarsRootFromCurrent (logger: ILogger<_>) =
            let discovery = TarsProjectDiscoveryService(logger)
            discovery.FindTarsRoot(Environment.CurrentDirectory)

        /// Get quick project count
        let getQuickProjectCount (logger: ILogger<_>) =
            let discovery = TarsProjectDiscoveryService(logger)
            let result = discovery.QuickDiscovery(Environment.CurrentDirectory)
            if result.Success then result.TotalSourceFiles else 0
