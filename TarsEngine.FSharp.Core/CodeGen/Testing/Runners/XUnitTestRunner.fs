namespace TarsEngine.FSharp.Core.CodeGen.Testing.Runners

open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open System.Xml.Linq
open Microsoft.Extensions.Logging

/// <summary>
/// Test runner for xUnit.
/// </summary>
type XUnitTestRunner(logger: ILogger<XUnitTestRunner>) =
    inherit TestRunnerBase(logger :> ILogger)
    
    /// <summary>
    /// Gets the name of the test runner.
    /// </summary>
    override _.Name = "xUnit"
    
    /// <summary>
    /// Gets the supported test frameworks.
    /// </summary>
    override _.SupportedFrameworks = ["xunit"]
    
    /// <summary>
    /// Gets the path to the xUnit console runner.
    /// </summary>
    /// <returns>The path to the xUnit console runner.</returns>
    member private _.GetXUnitConsoleRunnerPath() =
        // Look for xunit.console.exe in common locations
        let locations = [
            // NuGet package location
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".nuget", "packages", "xunit.runner.console")
            // Global tool location
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotnet", "tools")
        ]
        
        let mutable runnerPath = null
        
        for location in locations do
            if Directory.Exists(location) then
                let versionDirs = Directory.GetDirectories(location)
                
                if versionDirs.Length > 0 then
                    // Use the latest version
                    let latestVersionDir = versionDirs |> Array.sortDescending |> Array.head
                    
                    let possiblePaths = [
                        Path.Combine(latestVersionDir, "tools", "net472", "xunit.console.exe")
                        Path.Combine(latestVersionDir, "tools", "net471", "xunit.console.exe")
                        Path.Combine(latestVersionDir, "tools", "net47", "xunit.console.exe")
                        Path.Combine(latestVersionDir, "tools", "net462", "xunit.console.exe")
                        Path.Combine(latestVersionDir, "tools", "net461", "xunit.console.exe")
                        Path.Combine(latestVersionDir, "tools", "net46", "xunit.console.exe")
                        Path.Combine(latestVersionDir, "tools", "net45", "xunit.console.exe")
                    ]
                    
                    for path in possiblePaths do
                        if File.Exists(path) then
                            runnerPath <- path
                            break
                
                if runnerPath <> null then
                    break
        
        // If not found, use dotnet xunit
        if runnerPath = null then
            "dotnet"
        else
            runnerPath
    
    /// <summary>
    /// Gets the command to run tests.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <returns>The command to run tests.</returns>
    override this.GetTestCommand(assemblyPath: string, ?filter: string) =
        let runnerPath = this.GetXUnitConsoleRunnerPath()
        
        let arguments = ResizeArray<string>()
        
        if runnerPath = "dotnet" then
            arguments.Add("test")
            arguments.Add(assemblyPath)
            arguments.Add("--logger:trx")
            
            match filter with
            | Some f -> arguments.Add($"--filter \"{f}\"")
            | None -> ()
        else
            arguments.Add(assemblyPath)
            arguments.Add("-xml")
            arguments.Add("TestResults.xml")
            
            match filter with
            | Some f -> arguments.Add($"-method \"{f}\"")
            | None -> ()
        
        (runnerPath, arguments |> Seq.toList)
    
    /// <summary>
    /// Gets the command to discover tests.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <returns>The command to discover tests.</returns>
    override this.GetTestCommand(assemblyPath: string, ?filter: string) =
        let runnerPath = this.GetXUnitConsoleRunnerPath()
        
        let arguments = ResizeArray<string>()
        
        if runnerPath = "dotnet" then
            arguments.Add("test")
            arguments.Add(assemblyPath)
            arguments.Add("--list-tests")
            
            match filter with
            | Some f -> arguments.Add($"--filter \"{f}\"")
            | None -> ()
        else
            arguments.Add(assemblyPath)
            arguments.Add("-nologo")
            arguments.Add("-nocolor")
            arguments.Add("-noheader")
            arguments.Add("-noerror")
            arguments.Add("-noautoreporters")
            arguments.Add("-list")
            
            match filter with
            | Some f -> arguments.Add($"-method \"{f}\"")
            | None -> ()
        
        (runnerPath, arguments |> Seq.toList)
    
    /// <summary>
    /// Parses the output of a test run.
    /// </summary>
    /// <param name="output">The output of the test run.</param>
    /// <param name="error">The error output of the test run.</param>
    /// <param name="exitCode">The exit code of the test run.</param>
    /// <param name="durationMs">The duration of the test run in milliseconds.</param>
    /// <returns>The test run result.</returns>
    override _.ParseTestOutput(output: string, error: string, exitCode: int, durationMs: float) =
        try
            // Check if the XML results file exists
            if File.Exists("TestResults.xml") then
                // Parse the XML results
                let doc = XDocument.Load("TestResults.xml")
                let root = doc.Root
                
                // Get the test results
                let testResults = ResizeArray<TestResult>()
                
                for testElement in root.Elements(XName.Get("test")) do
                    let testName = testElement.Attribute(XName.Get("name")).Value
                    let result = testElement.Attribute(XName.Get("result")).Value
                    let time = testElement.Attribute(XName.Get("time")).Value |> Double.Parse
                    
                    let passed = result = "Pass"
                    
                    let errorMessage, stackTrace =
                        if not passed then
                            let failureElement = testElement.Element(XName.Get("failure"))
                            
                            if failureElement <> null then
                                let message = failureElement.Element(XName.Get("message"))?.Value
                                let stackTraceElement = failureElement.Element(XName.Get("stack-trace"))?.Value
                                
                                (Some message, Some stackTraceElement)
                            else
                                (None, None)
                        else
                            (None, None)
                    
                    let testResult = {
                        TestName = testName
                        Passed = passed
                        ErrorMessage = errorMessage
                        StackTrace = stackTrace
                        DurationMs = time * 1000.0
                        Output = None
                        AdditionalInfo = Map.empty
                    }
                    
                    testResults.Add(testResult)
                
                // Calculate summary
                let totalTests = testResults.Count
                let passedTests = testResults |> Seq.filter (fun r -> r.Passed) |> Seq.length
                let failedTests = testResults |> Seq.filter (fun r -> not r.Passed) |> Seq.length
                let skippedTests = 0 // xUnit doesn't report skipped tests in the XML
                
                {
                    Results = testResults |> Seq.toList
                    TotalTests = totalTests
                    PassedTests = passedTests
                    FailedTests = failedTests
                    SkippedTests = skippedTests
                    TotalDurationMs = durationMs
                    AdditionalInfo = Map.empty
                }
            else
                // Parse the console output
                let testResults = ResizeArray<TestResult>()
                
                // Parse test results from the output
                let testPattern = @"^\s*([^:]+): (Passed|Failed|Skipped)(?:\s+\[([0-9\.]+)s\])?(?:\s+(.+))?$"
                let matches = Regex.Matches(output, testPattern, RegexOptions.Multiline)
                
                for m in matches do
                    let testName = m.Groups.[1].Value
                    let result = m.Groups.[2].Value
                    let time = 
                        if m.Groups.[3].Success then
                            Double.Parse(m.Groups.[3].Value) * 1000.0
                        else
                            0.0
                    
                    let passed = result = "Passed"
                    
                    let errorMessage = 
                        if not passed && m.Groups.[4].Success then
                            Some m.Groups.[4].Value
                        else
                            None
                    
                    let testResult = {
                        TestName = testName
                        Passed = passed
                        ErrorMessage = errorMessage
                        StackTrace = None
                        DurationMs = time
                        Output = None
                        AdditionalInfo = Map.empty
                    }
                    
                    testResults.Add(testResult)
                
                // Calculate summary
                let totalTests = testResults.Count
                let passedTests = testResults |> Seq.filter (fun r -> r.Passed) |> Seq.length
                let failedTests = testResults |> Seq.filter (fun r -> not r.Passed && r.ErrorMessage.IsSome) |> Seq.length
                let skippedTests = testResults |> Seq.filter (fun r -> not r.Passed && r.ErrorMessage.IsNone) |> Seq.length
                
                {
                    Results = testResults |> Seq.toList
                    TotalTests = totalTests
                    PassedTests = passedTests
                    FailedTests = failedTests
                    SkippedTests = skippedTests
                    TotalDurationMs = durationMs
                    AdditionalInfo = Map.empty
                }
        with
        | ex ->
            // Return a basic result
            {
                Results = []
                TotalTests = 0
                PassedTests = 0
                FailedTests = 0
                SkippedTests = 0
                TotalDurationMs = durationMs
                AdditionalInfo = Map.ofList [
                    "Error", ex.Message
                    "Output", output
                    "Error Output", error
                ]
            }
    
    /// <summary>
    /// Parses the output of a test discovery.
    /// </summary>
    /// <param name="output">The output of the test discovery.</param>
    /// <param name="error">The error output of the test discovery.</param>
    /// <param name="exitCode">The exit code of the test discovery.</param>
    /// <returns>The list of discovered tests.</returns>
    override _.ParseDiscoverOutput(output: string, error: string, exitCode: int) =
        try
            // Parse test names from the output
            let testPattern = @"^\s*([^:]+)$"
            let matches = Regex.Matches(output, testPattern, RegexOptions.Multiline)
            
            matches
            |> Seq.cast<Match>
            |> Seq.map (fun m -> m.Groups.[1].Value)
            |> Seq.toList
        with
        | ex ->
            // Return an empty list
            []
