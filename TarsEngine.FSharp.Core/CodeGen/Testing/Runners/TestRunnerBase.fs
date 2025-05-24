namespace TarsEngine.FSharp.Core.CodeGen.Testing.Runners

open System
open System.Collections.Generic
open System.Diagnostics
open System.IO
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// <summary>
/// Base class for test runners.
/// </summary>
[<AbstractClass>]
type TestRunnerBase(logger: ILogger) =
    
    /// <summary>
    /// Gets the name of the test runner.
    /// </summary>
    abstract member Name : string
    
    /// <summary>
    /// Gets the supported test frameworks.
    /// </summary>
    abstract member SupportedFrameworks : string list
    
    /// <summary>
    /// Gets the command to run tests.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <returns>The command to run tests.</returns>
    abstract member GetTestCommand : assemblyPath:string * ?filter:string -> string * string list
    
    /// <summary>
    /// Gets the command to discover tests.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <returns>The command to discover tests.</returns>
    abstract member GetDiscoverCommand : assemblyPath:string * ?filter:string -> string * string list
    
    /// <summary>
    /// Parses the output of a test run.
    /// </summary>
    /// <param name="output">The output of the test run.</param>
    /// <param name="error">The error output of the test run.</param>
    /// <param name="exitCode">The exit code of the test run.</param>
    /// <param name="durationMs">The duration of the test run in milliseconds.</param>
    /// <returns>The test run result.</returns>
    abstract member ParseTestOutput : output:string * error:string * exitCode:int * durationMs:float -> TestRunResult
    
    /// <summary>
    /// Parses the output of a test discovery.
    /// </summary>
    /// <param name="output">The output of the test discovery.</param>
    /// <param name="error">The error output of the test discovery.</param>
    /// <param name="exitCode">The exit code of the test discovery.</param>
    /// <returns>The list of discovered tests.</returns>
    abstract member ParseDiscoverOutput : output:string * error:string * exitCode:int -> string list
    
    /// <summary>
    /// Runs a command and returns the output.
    /// </summary>
    /// <param name="command">The command to run.</param>
    /// <param name="arguments">The arguments for the command.</param>
    /// <param name="workingDirectory">The working directory for the command.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The output, error output, exit code, and duration of the command.</returns>
    member private _.RunCommandAsync(command: string, arguments: string list, workingDirectory: string, ?cancellationToken: CancellationToken) =
        task {
            let ct = defaultArg cancellationToken CancellationToken.None
            
            // Create the process start info
            let psi = ProcessStartInfo()
            psi.FileName <- command
            psi.Arguments <- String.Join(" ", arguments)
            psi.WorkingDirectory <- workingDirectory
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.UseShellExecute <- false
            psi.CreateNoWindow <- true
            
            // Start the process
            use process = new Process()
            process.StartInfo <- psi
            
            // Create tasks for reading output and error
            let outputTask = TaskCompletionSource<string>()
            let errorTask = TaskCompletionSource<string>()
            
            // Set up output and error handlers
            process.OutputDataReceived.Add(fun args ->
                if args.Data = null then
                    outputTask.SetResult(process.StandardOutput.ReadToEnd())
                else
                    ()
            )
            
            process.ErrorDataReceived.Add(fun args ->
                if args.Data = null then
                    errorTask.SetResult(process.StandardError.ReadToEnd())
                else
                    ()
            )
            
            // Start the process and begin reading output and error
            let stopwatch = Stopwatch.StartNew()
            process.Start() |> ignore
            process.BeginOutputReadLine()
            process.BeginErrorReadLine()
            
            // Create a task that completes when the process exits
            let processTask = Task.Run(fun () ->
                process.WaitForExit()
                process.ExitCode
            )
            
            // Wait for the process to exit or cancellation
            let! completed = Task.WhenAny(processTask, Task.Delay(-1, ct))
            
            // If cancellation was requested, kill the process
            if completed <> processTask && ct.IsCancellationRequested then
                try
                    process.Kill()
                with
                | _ -> ()
                
                ct.ThrowIfCancellationRequested()
            
            // Get the exit code
            let! exitCode = processTask
            
            // Get the output and error
            let! output = outputTask.Task
            let! error = errorTask.Task
            
            // Stop the stopwatch
            stopwatch.Stop()
            
            // Return the output, error, exit code, and duration
            return (output, error, exitCode, stopwatch.Elapsed.TotalMilliseconds)
        }
    
    /// <summary>
    /// Runs tests in a directory.
    /// </summary>
    /// <param name="directoryPath">The path to the directory containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The test run result.</returns>
    member this.RunTestsInDirectoryAsync(directoryPath: string, ?filter: string, ?cancellationToken: CancellationToken) =
        task {
            try
                logger.LogInformation("Running tests in directory: {DirectoryPath}", directoryPath)
                
                // Find test assemblies
                let testAssemblies = Directory.GetFiles(directoryPath, "*.dll", SearchOption.AllDirectories)
                
                // Run tests in each assembly
                let results = ResizeArray<TestRunResult>()
                
                for assembly in testAssemblies do
                    try
                        let! result = this.RunTestsInAssemblyAsync(assembly, ?filter = filter, ?cancellationToken = cancellationToken)
                        results.Add(result)
                    with
                    | ex ->
                        logger.LogError(ex, "Error running tests in assembly: {AssemblyPath}", assembly)
                
                // Combine results
                let totalTests = results |> Seq.sumBy (fun r -> r.TotalTests)
                let passedTests = results |> Seq.sumBy (fun r -> r.PassedTests)
                let failedTests = results |> Seq.sumBy (fun r -> r.FailedTests)
                let skippedTests = results |> Seq.sumBy (fun r -> r.SkippedTests)
                let totalDurationMs = results |> Seq.sumBy (fun r -> r.TotalDurationMs)
                
                let combinedResults = results |> Seq.collect (fun r -> r.Results) |> Seq.toList
                
                return {
                    Results = combinedResults
                    TotalTests = totalTests
                    PassedTests = passedTests
                    FailedTests = failedTests
                    SkippedTests = skippedTests
                    TotalDurationMs = totalDurationMs
                    AdditionalInfo = Map.empty
                }
            with
            | ex ->
                logger.LogError(ex, "Error running tests in directory: {DirectoryPath}", directoryPath)
                return {
                    Results = []
                    TotalTests = 0
                    PassedTests = 0
                    FailedTests = 0
                    SkippedTests = 0
                    TotalDurationMs = 0.0
                    AdditionalInfo = Map.empty
                }
        }
    
    /// <summary>
    /// Runs tests in an assembly.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The test run result.</returns>
    member this.RunTestsInAssemblyAsync(assemblyPath: string, ?filter: string, ?cancellationToken: CancellationToken) =
        task {
            try
                logger.LogInformation("Running tests in assembly: {AssemblyPath}", assemblyPath)
                
                // Get the command to run tests
                let command, arguments = this.GetTestCommand(assemblyPath, ?filter = filter)
                
                // Run the command
                let workingDirectory = Path.GetDirectoryName(assemblyPath)
                let! output, error, exitCode, durationMs = this.RunCommandAsync(command, arguments, workingDirectory, ?cancellationToken = cancellationToken)
                
                // Parse the output
                return this.ParseTestOutput(output, error, exitCode, durationMs)
            with
            | ex ->
                logger.LogError(ex, "Error running tests in assembly: {AssemblyPath}", assemblyPath)
                return {
                    Results = []
                    TotalTests = 0
                    PassedTests = 0
                    FailedTests = 0
                    SkippedTests = 0
                    TotalDurationMs = 0.0
                    AdditionalInfo = Map.empty
                }
        }
    
    /// <summary>
    /// Runs a specific test.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the test.</param>
    /// <param name="testName">The name of the test to run.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The test result.</returns>
    member this.RunTestAsync(assemblyPath: string, testName: string, ?cancellationToken: CancellationToken) =
        task {
            try
                logger.LogInformation("Running test: {TestName} in assembly: {AssemblyPath}", testName, assemblyPath)
                
                // Run tests with a filter for the specific test
                let! result = this.RunTestsInAssemblyAsync(assemblyPath, filter = testName, ?cancellationToken = cancellationToken)
                
                // Find the test result
                match result.Results |> List.tryFind (fun r -> r.TestName = testName) with
                | Some testResult -> return testResult
                | None ->
                    logger.LogWarning("Test not found: {TestName} in assembly: {AssemblyPath}", testName, assemblyPath)
                    return {
                        TestName = testName
                        Passed = false
                        ErrorMessage = Some "Test not found"
                        StackTrace = None
                        DurationMs = 0.0
                        Output = None
                        AdditionalInfo = Map.empty
                    }
            with
            | ex ->
                logger.LogError(ex, "Error running test: {TestName} in assembly: {AssemblyPath}", testName, assemblyPath)
                return {
                    TestName = testName
                    Passed = false
                    ErrorMessage = Some $"Error running test: {ex.Message}"
                    StackTrace = Some (ex.StackTrace |> Option.ofObj |> Option.defaultValue "")
                    DurationMs = 0.0
                    Output = None
                    AdditionalInfo = Map.empty
                }
        }
    
    /// <summary>
    /// Discovers tests in a directory.
    /// </summary>
    /// <param name="directoryPath">The path to the directory containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The list of discovered tests.</returns>
    member this.DiscoverTestsInDirectoryAsync(directoryPath: string, ?filter: string, ?cancellationToken: CancellationToken) =
        task {
            try
                logger.LogInformation("Discovering tests in directory: {DirectoryPath}", directoryPath)
                
                // Find test assemblies
                let testAssemblies = Directory.GetFiles(directoryPath, "*.dll", SearchOption.AllDirectories)
                
                // Discover tests in each assembly
                let tests = ResizeArray<string>()
                
                for assembly in testAssemblies do
                    try
                        let! assemblyTests = this.DiscoverTestsInAssemblyAsync(assembly, ?filter = filter, ?cancellationToken = cancellationToken)
                        tests.AddRange(assemblyTests)
                    with
                    | ex ->
                        logger.LogError(ex, "Error discovering tests in assembly: {AssemblyPath}", assembly)
                
                return tests |> Seq.toList
            with
            | ex ->
                logger.LogError(ex, "Error discovering tests in directory: {DirectoryPath}", directoryPath)
                return []
        }
    
    /// <summary>
    /// Discovers tests in an assembly.
    /// </summary>
    /// <param name="assemblyPath">The path to the assembly containing the tests.</param>
    /// <param name="filter">Optional filter for tests.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>The list of discovered tests.</returns>
    member this.DiscoverTestsInAssemblyAsync(assemblyPath: string, ?filter: string, ?cancellationToken: CancellationToken) =
        task {
            try
                logger.LogInformation("Discovering tests in assembly: {AssemblyPath}", assemblyPath)
                
                // Get the command to discover tests
                let command, arguments = this.GetDiscoverCommand(assemblyPath, ?filter = filter)
                
                // Run the command
                let workingDirectory = Path.GetDirectoryName(assemblyPath)
                let! output, error, exitCode, _ = this.RunCommandAsync(command, arguments, workingDirectory, ?cancellationToken = cancellationToken)
                
                // Parse the output
                return this.ParseDiscoverOutput(output, error, exitCode)
            with
            | ex ->
                logger.LogError(ex, "Error discovering tests in assembly: {AssemblyPath}", assemblyPath)
                return []
        }
    
    interface ITestRunner with
        member this.Name = this.Name
        member this.SupportedFrameworks = this.SupportedFrameworks
        member this.RunTestsInDirectoryAsync(directoryPath, ?filter, ?cancellationToken) = this.RunTestsInDirectoryAsync(directoryPath, ?filter = filter, ?cancellationToken = cancellationToken)
        member this.RunTestsInAssemblyAsync(assemblyPath, ?filter, ?cancellationToken) = this.RunTestsInAssemblyAsync(assemblyPath, ?filter = filter, ?cancellationToken = cancellationToken)
        member this.RunTestAsync(assemblyPath, testName, ?cancellationToken) = this.RunTestAsync(assemblyPath, testName, ?cancellationToken = cancellationToken)
        member this.DiscoverTestsInDirectoryAsync(directoryPath, ?filter, ?cancellationToken) = this.DiscoverTestsInDirectoryAsync(directoryPath, ?filter = filter, ?cancellationToken = cancellationToken)
        member this.DiscoverTestsInAssemblyAsync(assemblyPath, ?filter, ?cancellationToken) = this.DiscoverTestsInAssemblyAsync(assemblyPath, ?filter = filter, ?cancellationToken = cancellationToken)
