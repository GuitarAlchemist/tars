namespace TarsEngine.FSharp.Cli.Tests

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Commands

/// Test utilities and helpers for CLI testing
module TestHelpers =
    
    /// Mock logger for testing
    type MockLogger<'T>() =
        let mutable logs = []
        
        member _.Logs = logs |> List.rev
        
        interface ILogger<'T> with
            member _.BeginScope(state) = null
            member _.IsEnabled(logLevel) = true
            member _.Log(logLevel, eventId, state, ex, formatter) =
                let message = formatter.Invoke(state, ex)
                logs <- (logLevel, message) :: logs
    
    /// Create a mock logger
    let createMockLogger<'T>() = new MockLogger<'T>()
    
    /// Mock service provider for dependency injection
    type MockServiceProvider() =
        let services = System.Collections.Generic.Dictionary<Type, obj>()
        
        member this.AddService<'T>(service: 'T) =
            services.[typeof<'T>] <- box service
        
        interface IServiceProvider with
            member _.GetService(serviceType) =
                match services.TryGetValue(serviceType) with
                | true, service -> service
                | false, _ -> null
    
    /// Create test command options
    let createCommandOptions (args: string list) =
        {
            Arguments = args
            Flags = Map.empty
            GlobalOptions = Map.empty
        }
    
    /// Create test command options with flags
    let createCommandOptionsWithFlags (args: string list) (flags: (string * string) list) =
        {
            Arguments = args
            Flags = flags |> Map.ofList
            GlobalOptions = Map.empty
        }
    
    /// Temporary directory helper
    type TempDirectory() =
        let tempPath = Path.Combine(Path.GetTempPath(), "tars-cli-tests", Guid.NewGuid().ToString("N")[..7])
        
        do
            Directory.CreateDirectory(tempPath) |> ignore
        
        member _.Path = tempPath
        
        interface IDisposable with
            member _.Dispose() =
                try
                    if Directory.Exists(tempPath) then
                        Directory.Delete(tempPath, true)
                with
                | _ -> () // Ignore cleanup errors in tests
    
    /// Create a temporary directory for testing
    let createTempDirectory() = new TempDirectory()
    
    /// Mock MixtralService for testing
    type MockMixtralService() =
        interface IMixtralService with
            member _.QueryAsync(query: string) =
                task {
                    return Ok {
                        Content = $"Mock response for: {query}"
                        Confidence = 0.85
                        RoutingDecision = {
                            SelectedExpert = { Name = "MockExpert"; Confidence = 0.85 }
                            Reasoning = "Mock routing decision"
                        }
                    }
                }
    
    /// Create a mock MixtralService
    let createMockMixtralService() = new MockMixtralService()
    
    /// Test assertion helpers
    module Assertions =
        
        /// Assert that a command result is successful
        let assertSuccess (result: CommandResult) =
            match result with
            | { IsSuccess = true } -> ()
            | { IsSuccess = false; ErrorMessage = Some msg } -> 
                failwith $"Expected success but got error: {msg}"
            | { IsSuccess = false; ErrorMessage = None } -> 
                failwith "Expected success but got failure with no error message"
        
        /// Assert that a command result is a failure
        let assertFailure (result: CommandResult) =
            match result with
            | { IsSuccess = false } -> ()
            | { IsSuccess = true } -> 
                failwith "Expected failure but got success"
        
        /// Assert that a command result has a specific error message
        let assertErrorMessage (expectedMessage: string) (result: CommandResult) =
            match result with
            | { IsSuccess = false; ErrorMessage = Some msg } when msg.Contains(expectedMessage) -> ()
            | { IsSuccess = false; ErrorMessage = Some msg } -> 
                failwith $"Expected error message containing '{expectedMessage}' but got '{msg}'"
            | { IsSuccess = false; ErrorMessage = None } -> 
                failwith $"Expected error message containing '{expectedMessage}' but got no error message"
            | { IsSuccess = true } -> 
                failwith $"Expected error message but got success"
        
        /// Assert that a file exists
        let assertFileExists (filePath: string) =
            if not (File.Exists(filePath)) then
                failwith $"Expected file to exist: {filePath}"
        
        /// Assert that a directory exists
        let assertDirectoryExists (dirPath: string) =
            if not (Directory.Exists(dirPath)) then
                failwith $"Expected directory to exist: {dirPath}"
        
        /// Assert that a string contains a substring
        let assertContains (substring: string) (text: string) =
            if not (text.Contains(substring)) then
                failwith $"Expected text to contain '{substring}' but got: {text}"
        
        /// Assert that a logger contains a specific log message
        let assertLogContains (expectedMessage: string) (logger: MockLogger<'T>) =
            let logs = logger.Logs |> List.map snd
            if not (logs |> List.exists (fun msg -> msg.Contains(expectedMessage))) then
                let allLogs = String.Join("\n", logs)
                failwith $"Expected log to contain '{expectedMessage}' but logs were:\n{allLogs}"
    
    /// Performance testing helpers
    module Performance =
        
        /// Measure execution time of a function
        let measureTime (action: unit -> 'T) : 'T * TimeSpan =
            let stopwatch = System.Diagnostics.Stopwatch.StartNew()
            let result = action()
            stopwatch.Stop()
            (result, stopwatch.Elapsed)
        
        /// Measure async execution time
        let measureTimeAsync (action: unit -> Task<'T>) : Task<'T * TimeSpan> =
            task {
                let stopwatch = System.Diagnostics.Stopwatch.StartNew()
                let! result = action()
                stopwatch.Stop()
                return (result, stopwatch.Elapsed)
            }
        
        /// Assert that execution time is within expected bounds
        let assertExecutionTime (maxDuration: TimeSpan) (action: unit -> 'T) : 'T =
            let (result, duration) = measureTime action
            if duration > maxDuration then
                failwith $"Execution took {duration.TotalMilliseconds}ms, expected less than {maxDuration.TotalMilliseconds}ms"
            result
        
        /// Assert that async execution time is within expected bounds
        let assertExecutionTimeAsync (maxDuration: TimeSpan) (action: unit -> Task<'T>) : Task<'T> =
            task {
                let! (result, duration) = measureTimeAsync action
                if duration > maxDuration then
                    failwith $"Execution took {duration.TotalMilliseconds}ms, expected less than {maxDuration.TotalMilliseconds}ms"
                return result
            }
    
    /// Memory testing helpers
    module Memory =
        
        /// Get current memory usage
        let getCurrentMemoryUsage() =
            GC.Collect()
            GC.WaitForPendingFinalizers()
            GC.Collect()
            GC.GetTotalMemory(false)
        
        /// Measure memory usage of a function
        let measureMemory (action: unit -> 'T) : 'T * int64 =
            let beforeMemory = getCurrentMemoryUsage()
            let result = action()
            let afterMemory = getCurrentMemoryUsage()
            (result, afterMemory - beforeMemory)
        
        /// Assert that memory usage is within expected bounds
        let assertMemoryUsage (maxMemoryBytes: int64) (action: unit -> 'T) : 'T =
            let (result, memoryUsed) = measureMemory action
            if memoryUsed > maxMemoryBytes then
                failwith $"Memory usage was {memoryUsed} bytes, expected less than {maxMemoryBytes} bytes"
            result
    
    /// Integration testing helpers
    module Integration =
        
        /// Run a CLI command and capture output
        let runCliCommand (args: string list) : Task<int * string * string> =
            task {
                let startInfo = System.Diagnostics.ProcessStartInfo()
                startInfo.FileName <- "dotnet"
                startInfo.Arguments <- $"run --project TarsEngine.FSharp.Cli -- {String.concat " " args}"
                startInfo.UseShellExecute <- false
                startInfo.RedirectStandardOutput <- true
                startInfo.RedirectStandardError <- true
                startInfo.CreateNoWindow <- true
                
                use proc = System.Diagnostics.Process.Start(startInfo)
                let! stdout = proc.StandardOutput.ReadToEndAsync()
                let! stderr = proc.StandardError.ReadToEndAsync()
                proc.WaitForExit()
                
                return (proc.ExitCode, stdout, stderr)
            }
        
        /// Assert that a CLI command succeeds
        let assertCliCommandSucceeds (args: string list) : Task<string> =
            task {
                let! (exitCode, stdout, stderr) = runCliCommand args
                if exitCode <> 0 then
                    failwith $"CLI command failed with exit code {exitCode}. Stderr: {stderr}"
                return stdout
            }
        
        /// Assert that a CLI command fails
        let assertCliCommandFails (args: string list) : Task<string> =
            task {
                let! (exitCode, stdout, stderr) = runCliCommand args
                if exitCode = 0 then
                    failwith $"Expected CLI command to fail but it succeeded. Stdout: {stdout}"
                return stderr
            }
