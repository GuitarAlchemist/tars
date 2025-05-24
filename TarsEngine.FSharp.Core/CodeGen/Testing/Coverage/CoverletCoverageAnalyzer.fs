namespace TarsEngine.FSharp.Core.CodeGen.Testing.Coverage

open System
open System.Collections.Generic
open System.IO
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Coverage analyzer for Coverlet.
/// </summary>
type CoverletCoverageAnalyzer(logger: ILogger<CoverletCoverageAnalyzer>) =
    inherit TestCoverageAnalyzerBase(logger :> ILogger)
    
    /// <summary>
    /// Gets the name of the coverage analyzer.
    /// </summary>
    override _.Name = "Coverlet"
    
    /// <summary>
    /// Analyzes coverage for a test run.
    /// </summary>
    /// <param name="testResultsPath">The path to the test results.</param>
    /// <param name="sourcePath">The path to the source code.</param>
    /// <returns>The coverage result.</returns>
    override this.AnalyzeCoverageAsync(testResultsPath: string, sourcePath: string) =
        task {
            try
                logger.LogInformation("Analyzing coverage for test results: {TestResultsPath}", testResultsPath)
                
                // Find the coverage.json file
                let coverageFile = 
                    if File.Exists(testResultsPath) then
                        testResultsPath
                    elif Directory.Exists(testResultsPath) then
                        let coverageFiles = Directory.GetFiles(testResultsPath, "coverage.json", SearchOption.AllDirectories)
                        
                        if coverageFiles.Length > 0 then
                            coverageFiles.[0]
                        else
                            raise (FileNotFoundException("Coverage file not found"))
                    else
                        raise (FileNotFoundException("Test results path not found"))
                
                // Read the coverage file
                let coverageJson = File.ReadAllText(coverageFile)
                
                // Parse the JSON
                let coverageData = JsonDocument.Parse(coverageJson)
                
                // Extract file coverage results
                let fileCoverageResults = ResizeArray<FileCoverageResult>()
                
                for assemblyElement in coverageData.RootElement.EnumerateObject() do
                    let assemblyName = assemblyElement.Name
                    
                    for moduleElement in assemblyElement.Value.GetProperty("Modules").EnumerateObject() do
                        let moduleName = moduleElement.Name
                        
                        for classElement in moduleElement.Value.GetProperty("Classes").EnumerateObject() do
                            let className = classElement.Name
                            
                            for methodElement in classElement.Value.GetProperty("Methods").EnumerateObject() do
                                let methodName = methodElement.Name
                                
                                // Get the file path
                                let filePath = methodElement.Value.GetProperty("FileRef").GetString()
                                
                                // Get the line coverage
                                let lineCoverage = methodElement.Value.GetProperty("Lines")
                                
                                // Create a map of line status
                                let lineStatus = Dictionary<int, bool>()
                                
                                for lineElement in lineCoverage.EnumerateObject() do
                                    let lineNumber = Int32.Parse(lineElement.Name)
                                    let hits = lineElement.Value.GetInt32()
                                    
                                    lineStatus.[lineNumber] <- hits > 0
                                
                                // Find or create the file coverage result
                                let fileCoverageResult = 
                                    match fileCoverageResults |> Seq.tryFind (fun r -> r.FilePath = filePath) with
                                    | Some result -> result
                                    | None ->
                                        // Count the total lines in the file
                                        let totalLines = 
                                            if File.Exists(filePath) then
                                                File.ReadAllLines(filePath).Length
                                            else
                                                lineStatus.Keys |> Seq.max
                                        
                                        // Create a new file coverage result
                                        {
                                            FilePath = filePath
                                            TotalLines = totalLines
                                            CoveredLines = 0
                                            CoveragePercentage = 0.0
                                            LineStatus = Map.empty
                                            AdditionalInfo = Map.empty
                                        }
                                
                                // Update the line status
                                let updatedLineStatus = 
                                    lineStatus
                                    |> Seq.fold (fun map kvp -> Map.add kvp.Key kvp.Value map) fileCoverageResult.LineStatus
                                
                                // Calculate the covered lines
                                let coveredLines = updatedLineStatus |> Map.filter (fun _ v -> v) |> Map.count
                                
                                // Calculate the coverage percentage
                                let coveragePercentage = 
                                    if fileCoverageResult.TotalLines > 0 then
                                        (float coveredLines) / (float fileCoverageResult.TotalLines) * 100.0
                                    else
                                        0.0
                                
                                // Update the file coverage result
                                let updatedFileCoverageResult = {
                                    fileCoverageResult with
                                        LineStatus = updatedLineStatus
                                        CoveredLines = coveredLines
                                        CoveragePercentage = coveragePercentage
                                }
                                
                                // Add or update the file coverage result
                                match fileCoverageResults |> Seq.tryFindIndex (fun r -> r.FilePath = filePath) with
                                | Some index -> fileCoverageResults.[index] <- updatedFileCoverageResult
                                | None -> fileCoverageResults.Add(updatedFileCoverageResult)
                
                // Calculate the project coverage
                return this.CalculateProjectCoverage(fileCoverageResults |> Seq.toList, sourcePath)
            with
            | ex ->
                logger.LogError(ex, "Error analyzing coverage for test results: {TestResultsPath}", testResultsPath)
                return {
                    ProjectPath = sourcePath
                    FileCoverageResults = []
                    TotalLines = 0
                    CoveredLines = 0
                    CoveragePercentage = 0.0
                    AdditionalInfo = Map.ofList [
                        "Error", ex.Message
                    ]
                }
        }
    
    /// <summary>
    /// Generates a coverage report.
    /// </summary>
    /// <param name="coverageResult">The coverage result.</param>
    /// <param name="outputPath">The path to output the report.</param>
    /// <returns>The path to the generated report.</returns>
    override _.GenerateReportAsync(coverageResult: ProjectCoverageResult, outputPath: string) =
        task {
            try
                logger.LogInformation("Generating coverage report for project: {ProjectPath}", coverageResult.ProjectPath)
                
                // Create the output directory if it doesn't exist
                Directory.CreateDirectory(outputPath) |> ignore
                
                // Generate an HTML report
                let reportPath = Path.Combine(outputPath, "coverage.html")
                
                // Create the HTML content
                let html = StringBuilder()
                
                html.AppendLine("<!DOCTYPE html>") |> ignore
                html.AppendLine("<html>") |> ignore
                html.AppendLine("<head>") |> ignore
                html.AppendLine("    <title>Coverage Report</title>") |> ignore
                html.AppendLine("    <style>") |> ignore
                html.AppendLine("        body { font-family: Arial, sans-serif; margin: 20px; }") |> ignore
                html.AppendLine("        h1 { color: #333; }") |> ignore
                html.AppendLine("        table { border-collapse: collapse; width: 100%; }") |> ignore
                html.AppendLine("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }") |> ignore
                html.AppendLine("        th { background-color: #f2f2f2; }") |> ignore
                html.AppendLine("        tr:nth-child(even) { background-color: #f9f9f9; }") |> ignore
                html.AppendLine("        .covered { background-color: #dff0d8; }") |> ignore
                html.AppendLine("        .not-covered { background-color: #f2dede; }") |> ignore
                html.AppendLine("        .progress { height: 20px; background-color: #f2f2f2; border-radius: 4px; }") |> ignore
                html.AppendLine("        .progress-bar { height: 100%; background-color: #5cb85c; border-radius: 4px; }") |> ignore
                html.AppendLine("    </style>") |> ignore
                html.AppendLine("</head>") |> ignore
                html.AppendLine("<body>") |> ignore
                
                // Add the project summary
                html.AppendLine($"    <h1>Coverage Report for {coverageResult.ProjectPath}</h1>") |> ignore
                html.AppendLine("    <div>") |> ignore
                html.AppendLine($"        <p>Total Lines: {coverageResult.TotalLines}</p>") |> ignore
                html.AppendLine($"        <p>Covered Lines: {coverageResult.CoveredLines}</p>") |> ignore
                html.AppendLine($"        <p>Coverage Percentage: {coverageResult.CoveragePercentage:F2}%</p>") |> ignore
                html.AppendLine("        <div class=\"progress\">") |> ignore
                html.AppendLine($"            <div class=\"progress-bar\" style=\"width: {coverageResult.CoveragePercentage}%\"></div>") |> ignore
                html.AppendLine("        </div>") |> ignore
                html.AppendLine("    </div>") |> ignore
                
                // Add the file summary table
                html.AppendLine("    <h2>File Summary</h2>") |> ignore
                html.AppendLine("    <table>") |> ignore
                html.AppendLine("        <tr>") |> ignore
                html.AppendLine("            <th>File</th>") |> ignore
                html.AppendLine("            <th>Total Lines</th>") |> ignore
                html.AppendLine("            <th>Covered Lines</th>") |> ignore
                html.AppendLine("            <th>Coverage Percentage</th>") |> ignore
                html.AppendLine("        </tr>") |> ignore
                
                for fileCoverageResult in coverageResult.FileCoverageResults do
                    html.AppendLine("        <tr>") |> ignore
                    html.AppendLine($"            <td>{fileCoverageResult.FilePath}</td>") |> ignore
                    html.AppendLine($"            <td>{fileCoverageResult.TotalLines}</td>") |> ignore
                    html.AppendLine($"            <td>{fileCoverageResult.CoveredLines}</td>") |> ignore
                    html.AppendLine($"            <td>{fileCoverageResult.CoveragePercentage:F2}%</td>") |> ignore
                    html.AppendLine("        </tr>") |> ignore
                
                html.AppendLine("    </table>") |> ignore
                
                // Add the file details
                html.AppendLine("    <h2>File Details</h2>") |> ignore
                
                for fileCoverageResult in coverageResult.FileCoverageResults do
                    html.AppendLine($"    <h3>{fileCoverageResult.FilePath}</h3>") |> ignore
                    html.AppendLine("    <div class=\"progress\">") |> ignore
                    html.AppendLine($"        <div class=\"progress-bar\" style=\"width: {fileCoverageResult.CoveragePercentage}%\"></div>") |> ignore
                    html.AppendLine("    </div>") |> ignore
                    
                    // Add the line coverage table
                    html.AppendLine("    <table>") |> ignore
                    html.AppendLine("        <tr>") |> ignore
                    html.AppendLine("            <th>Line</th>") |> ignore
                    html.AppendLine("            <th>Covered</th>") |> ignore
                    html.AppendLine("        </tr>") |> ignore
                    
                    for lineNumber in 1 .. fileCoverageResult.TotalLines do
                        let covered = 
                            match fileCoverageResult.LineStatus.TryFind(lineNumber) with
                            | Some status -> status
                            | None -> false
                        
                        let rowClass = if covered then "covered" else "not-covered"
                        
                        html.AppendLine($"        <tr class=\"{rowClass}\">") |> ignore
                        html.AppendLine($"            <td>{lineNumber}</td>") |> ignore
                        html.AppendLine($"            <td>{covered}</td>") |> ignore
                        html.AppendLine("        </tr>") |> ignore
                    
                    html.AppendLine("    </table>") |> ignore
                
                html.AppendLine("</body>") |> ignore
                html.AppendLine("</html>") |> ignore
                
                // Write the HTML to the file
                File.WriteAllText(reportPath, html.ToString())
                
                return reportPath
            with
            | ex ->
                logger.LogError(ex, "Error generating coverage report for project: {ProjectPath}", coverageResult.ProjectPath)
                return ""
        }
