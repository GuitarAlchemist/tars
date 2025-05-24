namespace TarsEngine.FSharp.Core.CodeGen.Testing.Coverage

open System
open System.Collections.Generic
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Represents a coverage result for a file.
/// </summary>
type FileCoverageResult = {
    /// <summary>
    /// The path to the file.
    /// </summary>
    FilePath: string
    
    /// <summary>
    /// The total number of lines in the file.
    /// </summary>
    TotalLines: int
    
    /// <summary>
    /// The number of covered lines in the file.
    /// </summary>
    CoveredLines: int
    
    /// <summary>
    /// The coverage percentage for the file.
    /// </summary>
    CoveragePercentage: float
    
    /// <summary>
    /// The coverage status for each line in the file.
    /// </summary>
    LineStatus: Map<int, bool>
    
    /// <summary>
    /// Additional information about the coverage result.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Represents a coverage result for a project.
/// </summary>
type ProjectCoverageResult = {
    /// <summary>
    /// The path to the project.
    /// </summary>
    ProjectPath: string
    
    /// <summary>
    /// The coverage results for each file in the project.
    /// </summary>
    FileCoverageResults: FileCoverageResult list
    
    /// <summary>
    /// The total number of lines in the project.
    /// </summary>
    TotalLines: int
    
    /// <summary>
    /// The number of covered lines in the project.
    /// </summary>
    CoveredLines: int
    
    /// <summary>
    /// The coverage percentage for the project.
    /// </summary>
    CoveragePercentage: float
    
    /// <summary>
    /// Additional information about the coverage result.
    /// </summary>
    AdditionalInfo: Map<string, string>
}

/// <summary>
/// Interface for analyzing test coverage.
/// </summary>
type ITestCoverageAnalyzer =
    /// <summary>
    /// Gets the name of the coverage analyzer.
    /// </summary>
    abstract member Name : string
    
    /// <summary>
    /// Analyzes coverage for a test run.
    /// </summary>
    /// <param name="testResultsPath">The path to the test results.</param>
    /// <param name="sourcePath">The path to the source code.</param>
    /// <returns>The coverage result.</returns>
    abstract member AnalyzeCoverageAsync : testResultsPath:string * sourcePath:string -> Task<ProjectCoverageResult>
    
    /// <summary>
    /// Generates a coverage report.
    /// </summary>
    /// <param name="coverageResult">The coverage result.</param>
    /// <param name="outputPath">The path to output the report.</param>
    /// <returns>The path to the generated report.</returns>
    abstract member GenerateReportAsync : coverageResult:ProjectCoverageResult * outputPath:string -> Task<string>

/// <summary>
/// Base class for test coverage analyzers.
/// </summary>
[<AbstractClass>]
type TestCoverageAnalyzerBase(logger: ILogger) =
    
    /// <summary>
    /// Gets the name of the coverage analyzer.
    /// </summary>
    abstract member Name : string
    
    /// <summary>
    /// Analyzes coverage for a test run.
    /// </summary>
    /// <param name="testResultsPath">The path to the test results.</param>
    /// <param name="sourcePath">The path to the source code.</param>
    /// <returns>The coverage result.</returns>
    abstract member AnalyzeCoverageAsync : testResultsPath:string * sourcePath:string -> Task<ProjectCoverageResult>
    
    /// <summary>
    /// Generates a coverage report.
    /// </summary>
    /// <param name="coverageResult">The coverage result.</param>
    /// <param name="outputPath">The path to output the report.</param>
    /// <returns>The path to the generated report.</returns>
    abstract member GenerateReportAsync : coverageResult:ProjectCoverageResult * outputPath:string -> Task<string>
    
    /// <summary>
    /// Calculates coverage statistics for a project.
    /// </summary>
    /// <param name="fileCoverageResults">The coverage results for each file in the project.</param>
    /// <param name="projectPath">The path to the project.</param>
    /// <returns>The project coverage result.</returns>
    member _.CalculateProjectCoverage(fileCoverageResults: FileCoverageResult list, projectPath: string) =
        let totalLines = fileCoverageResults |> List.sumBy (fun r -> r.TotalLines)
        let coveredLines = fileCoverageResults |> List.sumBy (fun r -> r.CoveredLines)
        
        let coveragePercentage = 
            if totalLines > 0 then
                (float coveredLines) / (float totalLines) * 100.0
            else
                0.0
        
        {
            ProjectPath = projectPath
            FileCoverageResults = fileCoverageResults
            TotalLines = totalLines
            CoveredLines = coveredLines
            CoveragePercentage = coveragePercentage
            AdditionalInfo = Map.empty
        }
    
    interface ITestCoverageAnalyzer with
        member this.Name = this.Name
        member this.AnalyzeCoverageAsync(testResultsPath, sourcePath) = this.AnalyzeCoverageAsync(testResultsPath, sourcePath)
        member this.GenerateReportAsync(coverageResult, outputPath) = this.GenerateReportAsync(coverageResult, outputPath)
