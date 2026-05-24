namespace TarsEngine.SelfImprovement.Tests

open System
open System.IO
open System.Net.Http
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.SelfImprovement
open TarsEngine.FSharp.SelfImprovement.ImprovementTypes
open TarsEngine.FSharp.SelfImprovement.ExecutionHarness

module SelfImprovementServiceTests =

    let private writeTempFile (suffix: string) (content: string) =
        let dir = Path.Combine(Path.GetTempPath(), "tars_self_improve_tests")
        Directory.CreateDirectory(dir) |> ignore
        let path = Path.Combine(dir, (Guid.NewGuid().ToString("N") + suffix))
        File.WriteAllText(path, content)
        path

    [<Fact>]
    let ``ApplyImprovementsAsync inserts throw in empty C# catch`` () =
        // Arrange
        let cs = """
using System;
class A {
    void M(){
        try { Console.WriteLine(1); }
        catch { }
    }
}
"""
        let tempPath = writeTempFile ".cs" cs
        use http = new HttpClient()
        let logger = NullLogger<SelfImprovementService>.Instance
        let svc = SelfImprovementService(http, logger)

        let pattern = {
            Name = "Empty catch blocks"
            Description = "Exception handling with empty catch blocks"
            PatternType = PatternType.Security
            Severity = Severity.High
            Example = None
            Recommendation = "Insert rethrow or proper logging"
        }

        // Act
        let results = svc.ApplyImprovementsAsync(tempPath, [pattern]) |> Async.RunSynchronously
        let updated = File.ReadAllText(tempPath)
        let backupExists = File.Exists(tempPath + ".bak")

        // Assert
        Assert.Contains("catch { throw; }", updated)
        Assert.NotEmpty(results)
        Assert.True(results |> List.exists (fun r -> r.Success))
        Assert.True(backupExists)

    [<Fact>]
    let ``ApplyImprovementsAsync normalizes TODO comments`` () =
        // Arrange
        let fs = """
module Sample

// TODO fix this logic later
let add x y = x + y
"""
        let tempPath = writeTempFile ".fs" fs
        use http = new HttpClient()
        let logger = NullLogger<SelfImprovementService>.Instance
        let svc = SelfImprovementService(http, logger)

        let pattern = {
            Name = "TODO comments"
            Description = "TODO, FIXME, HACK comments indicating incomplete work"
            PatternType = PatternType.Documentation
            Severity = Severity.Low
            Example = None
            Recommendation = "Track TODOs with consistent format"
        }

        // Act
        let results = svc.ApplyImprovementsAsync(tempPath, [pattern]) |> Async.RunSynchronously
        let updated = File.ReadAllText(tempPath)
        let backup = File.ReadAllText(tempPath + ".bak")
        let history = svc.GetImprovementHistoryAsync(Some tempPath) |> Async.RunSynchronously

        // Assert
        Assert.Contains("TODO[tracked-", updated)
        Assert.NotEmpty(results)
        Assert.True(results |> List.exists (fun r -> r.Success))
        Assert.True(File.Exists(tempPath + ".bak"))
        Assert.Contains("TODO fix this logic later", backup)
        Assert.Equal(results.Length, history.Length)
        Assert.DoesNotContain("Simulated", updated)

    [<Fact>]
    let ``ApplyImprovementsAsync skips unsupported pattern`` () =
        // Arrange
        let fs = """
module SampleTwo

let area radius = 3.14 * radius * radius
"""
        let tempPath = writeTempFile ".fs" fs
        use http = new HttpClient()
        let logger = NullLogger<SelfImprovementService>.Instance
        let svc = SelfImprovementService(http, logger)

        let unsupported = {
            Name = "Magic numbers"
            Description = "Hardcoded numeric values without explanation"
            PatternType = PatternType.Maintainability
            Severity = Severity.Medium
            Example = None
            Recommendation = "Replace with named constants"
        }

        // Act
        let results = svc.ApplyImprovementsAsync(tempPath, [unsupported]) |> Async.RunSynchronously
        let updated = File.ReadAllText(tempPath)
        let backupExists = File.Exists(tempPath + ".bak")

        // Assert
        Assert.Empty(results)
        Assert.Equal(fs.ReplaceLineEndings(), updated.ReplaceLineEndings())
        Assert.False(backupExists)

    [<Fact>]
    let ``RunExecutionHarnessAsync executes provided commands`` () =
        // Arrange
        use http = new HttpClient()
        let logger = NullLogger<SelfImprovementService>.Instance
        let svc = SelfImprovementService(http, logger)

        let commandSpec = {
            Name = "unit-tests"
            Executable = "dotnet"
            Arguments = "test"
            WorkingDirectory = None
            Timeout = None
            Environment = Map.empty
        }

        let now = DateTime.UtcNow
        let commandResult = {
            Command = commandSpec
            ExitCode = 0
            Duration = TimeSpan.FromSeconds(1.0)
            StandardOutput = "tests passed"
            StandardError = ""
            StartedAt = now
            CompletedAt = now.AddSeconds(1.0)
        }

        let executor =
            { new ICommandExecutor with
                member _.RunCommandAsync _ = async { return commandResult } }

        let config = {
            Description = "harness smoke test"
            PreValidation = []
            Validation = [commandSpec]
            Benchmarks = []
            Rollback = None
            StopOnFailure = true
            CaptureLogs = true
        }

        // Act
        let report = svc.RunExecutionHarnessAsync(config, executor = executor) |> Async.RunSynchronously

        // Assert
        match report.Outcome with
        | AllPassed commands ->
            Assert.Single(commands) |> ignore
            Assert.Equal("tests passed", commands.Head.StandardOutput)
        | Failed _ -> Assert.True(false, "Expected harness to succeed")
