namespace TarsEngine.FSharp.Requirements.Validation

open System
open System.Diagnostics
open System.Threading.Tasks
open System.IO
open System.Text
open TarsEngine.FSharp.Requirements.Models

/// <summary>
/// Test execution environment configuration
/// </summary>
type TestEnvironment = {
    Name: string
    WorkingDirectory: string
    EnvironmentVariables: Map<string, string>
    Timeout: TimeSpan
    MaxMemory: int64 option
}

/// <summary>
/// Test execution context
/// </summary>
type TestExecutionContext = {
    TestCase: TestCase
    Environment: TestEnvironment
    StartTime: DateTime
    ProcessId: int option
}

/// <summary>
/// Test executor for running requirement validation tests
/// Real implementation with actual test execution
/// </summary>
type TestExecutor() =
    
    /// <summary>
    /// Default test environment
    /// </summary>
    let defaultEnvironment = {
        Name = "default"
        WorkingDirectory = Directory.GetCurrentDirectory()
        EnvironmentVariables = Map.empty
        Timeout = TimeSpan.FromMinutes(5.0)
        MaxMemory = None
    }
    
    /// <summary>
    /// Execute F# script test
    /// </summary>
    let executeFSharpScript (testCase: TestCase) (environment: TestEnvironment) = task {
        let tempFile = Path.GetTempFileName() + ".fsx"
        try
            // Write test code to temporary file
            let fullCode = $"""
{testCase.SetupCode |> Option.defaultValue ""}

{testCase.TestCode}

{testCase.TeardownCode |> Option.defaultValue ""}
"""
            do! File.WriteAllTextAsync(tempFile, fullCode)
            
            // Execute F# script
            let startInfo = ProcessStartInfo()
            startInfo.FileName <- "dotnet"
            startInfo.Arguments <- $"fsi \"{tempFile}\""
            startInfo.WorkingDirectory <- environment.WorkingDirectory
            startInfo.UseShellExecute <- false
            startInfo.RedirectStandardOutput <- true
            startInfo.RedirectStandardError <- true
            startInfo.CreateNoWindow <- true
            
            // Add environment variables
            for kvp in environment.EnvironmentVariables do
                startInfo.EnvironmentVariables.[kvp.Key] <- kvp.Value
            
            let startTime = DateTime.UtcNow
            use process = new Process()
            process.StartInfo <- startInfo
            
            let outputBuilder = StringBuilder()
            let errorBuilder = StringBuilder()
            
            process.OutputDataReceived.Add(fun args ->
                if not (String.IsNullOrEmpty(args.Data)) then
                    outputBuilder.AppendLine(args.Data) |> ignore
            )
            
            process.ErrorDataReceived.Add(fun args ->
                if not (String.IsNullOrEmpty(args.Data)) then
                    errorBuilder.AppendLine(args.Data) |> ignore
            )
            
            process.Start() |> ignore
            process.BeginOutputReadLine()
            process.BeginErrorReadLine()
            
            let! exited = Task.Run(fun () ->
                process.WaitForExit(int environment.Timeout.TotalMilliseconds)
            )
            
            let endTime = DateTime.UtcNow
            
            if not exited then
                try process.Kill() with | _ -> ()
                return TestCaseHelpers.createFailedResult 
                    testCase.Id 
                    "Test execution timed out" 
                    $"Test exceeded timeout of {environment.Timeout.TotalMinutes} minutes"
                    None
                    startTime 
                    endTime 
                    environment.Name
            else
                let output = outputBuilder.ToString()
                let error = errorBuilder.ToString()
                
                if process.ExitCode = 0 then
                    return TestCaseHelpers.createExecutionResult 
                        testCase.Id 
                        TestStatus.Passed 
                        output 
                        startTime 
                        endTime 
                        environment.Name
                else
                    return TestCaseHelpers.createFailedResult 
                        testCase.Id 
                        output 
                        error 
                        None
                        startTime 
                        endTime 
                        environment.Name
        finally
            try File.Delete(tempFile) with | _ -> ()
    }
    
    /// <summary>
    /// Execute PowerShell test
    /// </summary>
    let executePowerShellScript (testCase: TestCase) (environment: TestEnvironment) = task {
        let tempFile = Path.GetTempFileName() + ".ps1"
        try
            // Write test code to temporary file
            let fullCode = $"""
{testCase.SetupCode |> Option.defaultValue ""}

{testCase.TestCode}

{testCase.TeardownCode |> Option.defaultValue ""}
"""
            do! File.WriteAllTextAsync(tempFile, fullCode)
            
            // Execute PowerShell script
            let startInfo = ProcessStartInfo()
            startInfo.FileName <- "powershell.exe"
            startInfo.Arguments <- $"-ExecutionPolicy Bypass -File \"{tempFile}\""
            startInfo.WorkingDirectory <- environment.WorkingDirectory
            startInfo.UseShellExecute <- false
            startInfo.RedirectStandardOutput <- true
            startInfo.RedirectStandardError <- true
            startInfo.CreateNoWindow <- true
            
            let startTime = DateTime.UtcNow
            use process = new Process()
            process.StartInfo <- startInfo
            
            process.Start() |> ignore
            let! output = process.StandardOutput.ReadToEndAsync()
            let! error = process.StandardError.ReadToEndAsync()
            
            let! exited = Task.Run(fun () ->
                process.WaitForExit(int environment.Timeout.TotalMilliseconds)
            )
            
            let endTime = DateTime.UtcNow
            
            if not exited then
                try process.Kill() with | _ -> ()
                return TestCaseHelpers.createFailedResult 
                    testCase.Id 
                    "Test execution timed out" 
                    $"Test exceeded timeout of {environment.Timeout.TotalMinutes} minutes"
                    None
                    startTime 
                    endTime 
                    environment.Name
            else
                if process.ExitCode = 0 then
                    return TestCaseHelpers.createExecutionResult 
                        testCase.Id 
                        TestStatus.Passed 
                        output 
                        startTime 
                        endTime 
                        environment.Name
                else
                    return TestCaseHelpers.createFailedResult 
                        testCase.Id 
                        output 
                        error 
                        None
                        startTime 
                        endTime 
                        environment.Name
        finally
            try File.Delete(tempFile) with | _ -> ()
    }
    
    /// <summary>
    /// Execute batch/cmd test
    /// </summary>
    let executeBatchScript (testCase: TestCase) (environment: TestEnvironment) = task {
        let tempFile = Path.GetTempFileName() + ".cmd"
        try
            // Write test code to temporary file
            let fullCode = $"""
@echo off
{testCase.SetupCode |> Option.defaultValue ""}

{testCase.TestCode}

{testCase.TeardownCode |> Option.defaultValue ""}
"""
            do! File.WriteAllTextAsync(tempFile, fullCode)
            
            // Execute batch script
            let startInfo = ProcessStartInfo()
            startInfo.FileName <- tempFile
            startInfo.WorkingDirectory <- environment.WorkingDirectory
            startInfo.UseShellExecute <- false
            startInfo.RedirectStandardOutput <- true
            startInfo.RedirectStandardError <- true
            startInfo.CreateNoWindow <- true
            
            let startTime = DateTime.UtcNow
            use process = new Process()
            process.StartInfo <- startInfo
            
            process.Start() |> ignore
            let! output = process.StandardOutput.ReadToEndAsync()
            let! error = process.StandardError.ReadToEndAsync()
            
            let! exited = Task.Run(fun () ->
                process.WaitForExit(int environment.Timeout.TotalMilliseconds)
            )
            
            let endTime = DateTime.UtcNow
            
            if not exited then
                try process.Kill() with | _ -> ()
                return TestCaseHelpers.createFailedResult 
                    testCase.Id 
                    "Test execution timed out" 
                    $"Test exceeded timeout of {environment.Timeout.TotalMinutes} minutes"
                    None
                    startTime 
                    endTime 
                    environment.Name
            else
                if process.ExitCode = 0 then
                    return TestCaseHelpers.createExecutionResult 
                        testCase.Id 
                        TestStatus.Passed 
                        output 
                        startTime 
                        endTime 
                        environment.Name
                else
                    return TestCaseHelpers.createFailedResult 
                        testCase.Id 
                        output 
                        error 
                        None
                        startTime 
                        endTime 
                        environment.Name
        finally
            try File.Delete(tempFile) with | _ -> ()
    }
    
    /// <summary>
    /// Execute test case based on language
    /// </summary>
    member this.ExecuteTestAsync(testCase: TestCase, ?environment: TestEnvironment) = task {
        let env = environment |> Option.defaultValue defaultEnvironment
        
        try
            match testCase.Language.ToLowerInvariant() with
            | "fsharp" | "f#" | "fs" ->
                return! executeFSharpScript testCase env
            | "powershell" | "ps1" ->
                return! executePowerShellScript testCase env
            | "batch" | "cmd" ->
                return! executeBatchScript testCase env
            | _ ->
                let startTime = DateTime.UtcNow
                let endTime = DateTime.UtcNow
                return TestCaseHelpers.createFailedResult 
                    testCase.Id 
                    "Unsupported language" 
                    $"Language '{testCase.Language}' is not supported"
                    None
                    startTime 
                    endTime 
                    env.Name
        with
        | ex ->
            let startTime = DateTime.UtcNow
            let endTime = DateTime.UtcNow
            return TestCaseHelpers.createFailedResult 
                testCase.Id 
                "Execution error" 
                ex.Message
                (Some ex.StackTrace)
                startTime 
                endTime 
                env.Name
    }
    
    /// <summary>
    /// Execute multiple test cases
    /// </summary>
    member this.ExecuteTestsAsync(testCases: TestCase list, ?environment: TestEnvironment) = task {
        let env = environment |> Option.defaultValue defaultEnvironment
        let results = ResizeArray<TestExecutionResult>()
        
        for testCase in testCases do
            let! result = this.ExecuteTestAsync(testCase, env)
            results.Add(result)
        
        return results |> List.ofSeq
    }
    
    /// <summary>
    /// Execute test cases in parallel
    /// </summary>
    member this.ExecuteTestsParallelAsync(testCases: TestCase list, ?environment: TestEnvironment, ?maxConcurrency: int) = task {
        let env = environment |> Option.defaultValue defaultEnvironment
        let concurrency = maxConcurrency |> Option.defaultValue Environment.ProcessorCount
        
        let semaphore = new System.Threading.SemaphoreSlim(concurrency)
        
        let executeWithSemaphore (testCase: TestCase) = task {
            do! semaphore.WaitAsync()
            try
                return! this.ExecuteTestAsync(testCase, env)
            finally
                semaphore.Release() |> ignore
        }
        
        let tasks = testCases |> List.map executeWithSemaphore
        let! results = Task.WhenAll(tasks)
        
        return results |> List.ofArray
    }
    
    /// <summary>
    /// Create custom test environment
    /// </summary>
    member this.CreateEnvironment(name: string, workingDirectory: string, ?environmentVariables: Map<string, string>, ?timeout: TimeSpan) =
        {
            Name = name
            WorkingDirectory = workingDirectory
            EnvironmentVariables = environmentVariables |> Option.defaultValue Map.empty
            Timeout = timeout |> Option.defaultValue (TimeSpan.FromMinutes(5.0))
            MaxMemory = None
        }

module TestExecutorHelpers =
    
    /// <summary>
    /// Create a simple F# test case
    /// </summary>
    let createFSharpTest (requirementId: string) (name: string) (testCode: string) (createdBy: string) =
        TestCaseHelpers.create requirementId name $"F# test for {name}" testCode "fsharp" createdBy
    
    /// <summary>
    /// Create a PowerShell test case
    /// </summary>
    let createPowerShellTest (requirementId: string) (name: string) (testCode: string) (createdBy: string) =
        TestCaseHelpers.create requirementId name $"PowerShell test for {name}" testCode "powershell" createdBy
    
    /// <summary>
    /// Create a batch test case
    /// </summary>
    let createBatchTest (requirementId: string) (name: string) (testCode: string) (createdBy: string) =
        TestCaseHelpers.create requirementId name $"Batch test for {name}" testCode "batch" createdBy
