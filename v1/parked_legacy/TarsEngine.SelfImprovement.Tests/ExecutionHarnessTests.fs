namespace TarsEngine.SelfImprovement.Tests

open System
open System.Collections.Generic
open System.Threading.Tasks
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.SelfImprovement.ExecutionHarness

module ExecutionHarnessTests =

    type private FakeCommandExecutor(results: IDictionary<string, CommandResult>) =
        interface ICommandExecutor with
            member _.RunCommandAsync(spec: CommandSpec) =
                async {
                    match results.TryGetValue(spec.Name) with
                    | true, value -> return { value with Command = spec }
                    | _ -> return failwithf $"No fake result registered for command '%s{spec.Name}'"
                }

    type private RecordingRollback() =
        let mutable invoked = false
        interface IRollbackHandler with
            member _.RollbackAsync() =
                async {
                    invoked <- true
                }
        member _.Invoked = invoked

    let private command name exitCode =
        let spec = {
            Name = name
            Executable = "dotnet"
            Arguments = $"%s{name}-args"
            WorkingDirectory = None
            Timeout = None
            Environment = Map.empty
        }
        let now = DateTime.UtcNow
        spec,
        {
            Command = spec
            ExitCode = exitCode
            Duration = TimeSpan.FromMilliseconds(10.0)
            StandardOutput = "ok"
            StandardError = ""
            StartedAt = now
            CompletedAt = now.AddMilliseconds(10.0)
        }

    [<Fact>]
    let ``Harness reports success when all commands pass`` () =
        // Arrange
        let preSpec, preResult = command "restore" 0
        let testSpec, testResult = command "test" 0
        let benchSpec, benchResult = command "bench" 0

        let results : IDictionary<_, _> = dict [
            preSpec.Name, preResult;
            testSpec.Name, testResult;
            benchSpec.Name, benchResult
        ]

        let executor : ICommandExecutor = upcast FakeCommandExecutor(results)
        let logger = NullLoggerFactory.Instance.CreateLogger("execution-harness-test")
        let harness = AutonomousExecutionHarness(logger, executor)

        let config = {
            Description = "all pass"
            PreValidation = [preSpec]
            Validation = [testSpec]
            Benchmarks = [benchSpec]
            Rollback = None
            StopOnFailure = true
            CaptureLogs = true
        }

        // Act
        let report = harness.RunAsync(config) |> Async.RunSynchronously

        // Assert
        match report.Outcome with
        | AllPassed commands ->
            Assert.Equal(3, commands.Length)
            Assert.All(commands, fun c -> Assert.Equal(0, c.ExitCode))
        | Failed _ -> Assert.True(false, "Expected harness to succeed")

    [<Fact>]
    let ``Harness triggers rollback when validation fails`` () =
        // Arrange
        let testSpec, testResult = command "test" 1
        let failureResults : IDictionary<_, _> = dict [ testSpec.Name, testResult ]
        let executor : ICommandExecutor = upcast FakeCommandExecutor(failureResults)

        let logger = NullLoggerFactory.Instance.CreateLogger("execution-harness-test")
        let harness = AutonomousExecutionHarness(logger, executor)

        let rollback = RecordingRollback()

        let config = {
            Description = "validation failure"
            PreValidation = []
            Validation = [testSpec]
            Benchmarks = []
            Rollback = Some (rollback :> IRollbackHandler)
            StopOnFailure = true
            CaptureLogs = true
        }

        // Act
        let report = harness.RunAsync(config) |> Async.RunSynchronously

        // Assert
        match report.Outcome with
        | Failed (_, reason) -> Assert.Contains("test", reason)
        | AllPassed _ -> Assert.True(false, "Expected failure outcome")

        Assert.True(rollback.Invoked)

    [<Fact>]
    let ``Harness continues when StopOnFailure is false`` () =
        // Arrange
        let preSpec, preResult = command "pre" 1
        let testSpec, testResult = command "test" 0
        let benchSpec, benchResult = command "bench" 0

        let continueResults : IDictionary<_, _> = dict [
            preSpec.Name, preResult;
            testSpec.Name, testResult;
            benchSpec.Name, benchResult
        ]
        let executor : ICommandExecutor = upcast FakeCommandExecutor(continueResults)

        let logger = NullLoggerFactory.Instance.CreateLogger("execution-harness-test")
        let harness = AutonomousExecutionHarness(logger, executor)

        let config = {
            Description = "continue after pre failure"
            PreValidation = [preSpec]
            Validation = [testSpec]
            Benchmarks = [benchSpec]
            Rollback = None
            StopOnFailure = false
            CaptureLogs = true
        }

        // Act
        let report = harness.RunAsync(config) |> Async.RunSynchronously

        // Assert
        match report.Outcome with
        | Failed (commands, _) ->
            // Even though pre-validation failed, the remaining commands should have executed.
            Assert.Equal(3, commands.Length)
        | AllPassed _ -> Assert.True(false, "Expected overall failure due to pre-validation exit code")
