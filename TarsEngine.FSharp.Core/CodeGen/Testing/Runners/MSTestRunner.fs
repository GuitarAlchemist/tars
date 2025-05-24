namespace TarsEngine.FSharp.Core.CodeGen.Testing.Runners

open System
open System.Collections.Generic
open System.IO
open System.Text.RegularExpressions
open System.Xml.Linq
open Microsoft.Extensions.Logging

/// <summary>
/// Test runner for MSTest.
/// </summary>
type MSTestRunner(logger: ILogger<MSTestRunner>) =
    inherit TestRunnerBase(logger :> ILogger)
    
    /// <summary>
    /// Gets the name of the test runner.
    /// </summary>
    override _.Name = "MSTest"
    
    /// <summary>
    /// Gets the supported test frameworks.
    /// </summary>
    override _.SupportedFrameworks = ["mstest"]
    
    /// <summary>
    /// Gets the path to the MSTest console runner.
    /// </summary>
    /// <returns>The path to the MSTest console runner.</returns>
    member private _.GetMSTestConsoleRunnerPath() =
        // Look for vstest.console.exe in common locations
        let locations = [
            // Visual Studio location
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft Visual Studio", "2022", "Enterprise", "Common7", "IDE", "CommonExtensions", "Microsoft", "TestWindow")
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft Visual Studio", "2022", "Professional", "Common7", "IDE", "CommonExtensions", "Microsoft", "TestWindow")
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft Visual Studio", "2022", "Community", "Common7", "IDE", "CommonExtensions", "Microsoft", "TestWindow")
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft Visual Studio", "2019", "Enterprise", "Common7", "IDE", "CommonExtensions", "Microsoft", "TestWindow")
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft Visual Studio", "2019", "Professional", "Common7", "IDE", "CommonExtensions", "Microsoft", "TestWindow")
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft Visual Studio", "2019", "Community", "Common7", "IDE", "CommonExtensions", "Microsoft", "TestWindow")
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft Visual Studio", "2017", "Enterprise", "Common7", "IDE", "CommonExtensions", "Microsoft", "TestWindow")
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft Visual Studio", "2017", "Professional", "Common7", "IDE", "CommonExtensions", "Microsoft", "TestWindow")
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86), "Microsoft Visual Studio", "2017", "Community", "Common7", "IDE", "CommonExtensions", "Microsoft", "TestWindow")
        ]
        
        let mutable runnerPath = null
        
        for location in locations do
            let path = Path.Combine(location, "vstest.console.exe")
            if File.Exists(path) then
                runnerPath <- path
                break
        
        // If not found, use dotnet test
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
        let runnerPath = this.GetMSTestConsoleRunnerPath()
        
        let arguments = ResizeArray<string>()
        
        if runnerPath = "dotnet" then
            arguments.Add("test")
            arguments.Add(assemblyPath)
            arguments.Add("--logger:trx")
            arguments.Add("--results-directory:TestResults")
            
            match filter with
            | Some f -> arguments.Add($"--filter FullyQualifiedName~{f}")
            | None -> ()
        else
            arguments.Add(assemblyPath)
            arguments.Add("/logger:trx")
            arguments.Add("/ResultsDirectory:TestResults")
            
            match filter with
            | Some f -> arguments.Add($"/TestCaseFilter:FullyQualifiedName~{f}")
            | None -> ()
        
        (runnerPath, arguments |> Seq.toList)
    
    /// <summary>
    /// Gets the command to discover tests.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <returns>The command to discover tests.</returns>
    override this.GetDiscoverCommand(assemblyPath: string, ?filter: string) =
        let runnerPath = this.GetMSTestConsoleRunnerPath()
        
        let arguments = ResizeArray<string>()
        
        if runnerPath = "dotnet" then
            arguments.Add("test")
            arguments.Add(assemblyPath)
            arguments.Add("--list-tests")
            
            match filter with
            | Some f -> arguments.Add($"--filter FullyQualifiedName~{f}")
            | None -> ()
        else
            arguments.Add(assemblyPath)
            arguments.Add("/ListTests")
            
            match filter with
            | Some f -> arguments.Add($"/TestCaseFilter:FullyQualifiedName~{f}")
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
            // Check if the TRX results file exists
            let trxFiles = 
                if Directory.Exists("TestResults") then
                    Directory.GetFiles("TestResults", "*.trx")
                else
                    [||]
            
            if trxFiles.Length > 0 then
                // Parse the TRX results
                let doc = XDocument.Load(trxFiles.[0])
                let root = doc.Root
                
                // Get the test results
                let testResults = ResizeArray<TestResult>()
                
                // TRX file format
                let ns = XNamespace.Get("http://microsoft.com/schemas/VisualStudio/TeamTest/2010")
                
                for testElement in root.Descendants(ns + "UnitTestResult") do
                    let testName = testElement.Attribute(XName.Get("testName")).Value
                    let outcome = testElement.Attribute(XName.Get("outcome")).Value
                    let durationAttribute = testElement.Attribute(XName.Get("duration"))
                    let duration = 
                        if durationAttribute <> null then
                            let timeSpan = TimeSpan.Parse(durationAttribute.Value)
                            timeSpan.TotalMilliseconds
                        else
                            0.0
                    
                    let passed = outcome = "Passed"
                    
                    let errorMessage, stackTrace =
                        if not passed then
                            let outputElement = testElement.Element(ns + "Output")
                            
                            if outputElement <> null then
                                let errorInfoElement = outputElement.Element(ns + "ErrorInfo")
                                
                                if errorInfoElement <> null then
                                    let message = errorInfoElement.Element(ns + "Message")?.Value
                                    let stackTraceElement = errorInfoElement.Element(ns + "StackTrace")?.Value
                                    
                                    (Some message, Some stackTraceElement)
                                else
                                    (None, None)
                            else
                                (None, None)
                        else
                            (None, None)
                    
                    let testResult = {
                        TestName = testName
                        Passed = passed
                        ErrorMessage = errorMessage
                        StackTrace = stackTrace
                        DurationMs = duration
                        Output = None
                        AdditionalInfo = Map.empty
                    }
                    
                    testResults.Add(testResult)
                
                // Calculate summary
                let totalTests = testResults.Count
                let passedTests = testResults |> Seq.filter (fun r -> r.Passed) |> Seq.length
                let failedTests = testResults |> Seq.filter (fun r -> not r.Passed) |> Seq.length
                let skippedTests = 0 // Not available in the TRX
                
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
            let testPattern = @"^\s*([^\s]+)\s*$"
            let matches = Regex.Matches(output, testPattern, RegexOptions.Multiline)
            
            matches
            |> Seq.cast<Match>
            |> Seq.map (fun m -> m.Groups.[1].Value)
            |> Seq.toList
        with
        | ex ->
            // Return an empty list
            []
