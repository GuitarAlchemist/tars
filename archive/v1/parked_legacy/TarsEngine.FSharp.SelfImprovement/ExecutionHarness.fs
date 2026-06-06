namespace TarsEngine.FSharp.SelfImprovement

open System
open System.Diagnostics
open System.IO
open System.Text
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Logging

/// Execution harness that applies autonomous modifications, validates them, and rolls back on failure.
module ExecutionHarness =

    /// Describes a single command the harness should run.
    type CommandSpec = {
        Name: string
        Executable: string
        Arguments: string
        WorkingDirectory: string option
        Timeout: TimeSpan option
        Environment: Map<string, string>
    }

    /// Result of running a command.
    type CommandResult = {
        Command: CommandSpec
        ExitCode: int
        Duration: TimeSpan
        StandardOutput: string
        StandardError: string
        StartedAt: DateTime
        CompletedAt: DateTime
    }

    /// Summary outcome of the harness execution.
    type HarnessOutcome =
        | AllPassed of CommandResult list
        | Failed of CommandResult list * string

    /// Interface that can revert changes when validation fails.
    type IRollbackHandler =
        abstract member RollbackAsync : unit -> Async<unit>

    /// Abstraction over process execution to enable deterministic testing.
    type ICommandExecutor =
        abstract member RunCommandAsync : CommandSpec -> Async<CommandResult>

    /// Real process executor that shells out to the operating system.
    type ProcessCommandExecutor(logger: ILogger) =

        interface ICommandExecutor with
            member _.RunCommandAsync(command: CommandSpec) =
                async {
                    let workingDirectory = command.WorkingDirectory |> Option.defaultValue (Directory.GetCurrentDirectory())

                    if not (Directory.Exists(workingDirectory)) then
                        return raise (DirectoryNotFoundException $"Working directory not found: %s{workingDirectory}")

                    let startInfo = ProcessStartInfo()
                    startInfo.FileName <- command.Executable
                    startInfo.Arguments <- command.Arguments
                    startInfo.WorkingDirectory <- workingDirectory
                    startInfo.RedirectStandardOutput <- true
                    startInfo.RedirectStandardError <- true
                    startInfo.UseShellExecute <- false
                    startInfo.CreateNoWindow <- true

                    command.Environment
                    |> Map.iter (fun key value ->
                        if startInfo.Environment.ContainsKey(key) then
                            startInfo.Environment.[key] <- value
                        else
                            startInfo.Environment.Add(key, value))

                    use proc = new Process()
                    proc.StartInfo <- startInfo

                    let outputBuilder = StringBuilder()
                    let errorBuilder = StringBuilder()

                    let startedAt = DateTime.UtcNow

                    if not (proc.Start()) then
                        return raise (InvalidOperationException $"Failed to start command '%s{command.Name}'")

                    use cts = new CancellationTokenSource()
                    command.Timeout |> Option.iter cts.CancelAfter

                    let outputTask = proc.StandardOutput.ReadToEndAsync()
                    let errorTask = proc.StandardError.ReadToEndAsync()

                    try
                        do! proc.WaitForExitAsync(cts.Token) |> Async.AwaitTask
                    with
                    | :? TaskCanceledException as ex ->
                        try
                            if not proc.HasExited then proc.Kill(true)
                        with
                        | :? InvalidOperationException -> ()
                        raise (TimeoutException(
                                   $"Command '%s{command.Name}' timed out after {command.Timeout |> Option.defaultValue TimeSpan.Zero}", ex))

                    let! outputText = outputTask |> Async.AwaitTask
                    let! errorText = errorTask |> Async.AwaitTask
                    outputBuilder.Append(outputText) |> ignore
                    errorBuilder.Append(errorText) |> ignore

                    let completedAt = DateTime.UtcNow
                    let duration = completedAt - startedAt

                    logger.LogInformation("[{Command}] completed with code {ExitCode} in {Duration}.", command.Name, proc.ExitCode, duration)

                    return {
                        Command = command
                        ExitCode = proc.ExitCode
                        Duration = duration
                        StandardOutput = outputBuilder.ToString()
                        StandardError = errorBuilder.ToString()
                        StartedAt = startedAt
                        CompletedAt = completedAt
                    }
                }

    /// Configuration for the harness run.
    type HarnessConfig = {
        Description: string
        PreValidation: CommandSpec list
        Validation: CommandSpec list
        Benchmarks: CommandSpec list
        Rollback: IRollbackHandler option
        StopOnFailure: bool
        CaptureLogs: bool
    }

    /// Report produced by each harness run.
    type HarnessReport = {
        Config: HarnessConfig
        Commands: CommandResult list
        StartedAt: DateTime
        CompletedAt: DateTime
        Outcome: HarnessOutcome
    }

    /// Executes commands in the prescribed order and manages rollback.
    type AutonomousExecutionHarness(logger: ILogger, executor: ICommandExecutor) =

        let runCommands commands =
            async {
                let! results =
                    commands
                    |> List.fold
                        (fun acc command ->
                            async {
                                let! accResults = acc
                                match accResults with
                                | Choice2Of2 failure -> return Choice2Of2 failure
                                | Choice1Of2 results ->
                                    let! result = executor.RunCommandAsync(command)
                                    if result.ExitCode = 0 then
                                        return Choice1Of2(result :: results)
                                    else
                                        let failureMessage = $"Command '%s{command.Name}' failed with exit code %d{result.ExitCode}"
                                        return Choice2Of2(failureMessage, result :: results)
                            })
                        (async { return Choice1Of2 [] })

                match results with
                | Choice1Of2 results -> return Choice1Of2(List.rev results)
                | Choice2Of2 (message, results) -> return Choice2Of2(message, List.rev results)
            }

        member _.RunAsync(config: HarnessConfig) =
            async {
                if config.PreValidation.IsEmpty && config.Validation.IsEmpty && config.Benchmarks.IsEmpty then
                    return raise (ArgumentException("Harness configuration must include at least one command."))

                logger.LogInformation("Starting execution harness: {Description}", config.Description)
                let startedAt = DateTime.UtcNow

                let! preValidationResult = runCommands config.PreValidation

                let! (preResults, shouldContinue, preFailure) =
                    async {
                        match preValidationResult with
                        | Choice1Of2 res -> return (res, true, None)
                        | Choice2Of2 (msg, res) ->
                            logger.LogError("Pre-validation failed: {Message}", msg)
                            return (res, not config.StopOnFailure, Some msg)
                    }

                let! validationResult = if shouldContinue then runCommands config.Validation else async { return Choice1Of2 [] }

                let! (validationResults, validationContinue, validationFailure) =
                    async {
                        match validationResult with
                        | Choice1Of2 res -> return (res, true, None)
                        | Choice2Of2 (msg, res) ->
                            logger.LogError("Validation failed: {Message}", msg)
                            return (res, not config.StopOnFailure, Some msg)
                    }

                let! benchmarkResult = if validationContinue then runCommands config.Benchmarks else async { return Choice1Of2 [] }

                let! (benchmarkResults, benchmarkFailure) =
                    async {
                        match benchmarkResult with
                        | Choice1Of2 res -> return (res, None)
                        | Choice2Of2 (msg, res) ->
                            logger.LogError("Benchmark failed: {Message}", msg)
                            return (res, Some msg)
                    }

                let finalResults = preResults @ validationResults @ benchmarkResults
                let failureReasons =
                    [ preFailure; validationFailure; benchmarkFailure ]
                    |> List.choose id

                if List.isEmpty failureReasons then
                    logger.LogInformation("Execution harness completed successfully: {Description}", config.Description)
                    let completedAt = DateTime.UtcNow
                    return {
                        Config = config
                        Commands = finalResults
                        StartedAt = startedAt
                        CompletedAt = completedAt
                        Outcome = AllPassed finalResults
                    }
                else
                    let combinedReason = String.concat "; " failureReasons
                    logger.LogWarning("Execution harness failed: {Reason}", combinedReason)
                    match config.Rollback with
                    | Some rollback ->
                        logger.LogInformation("Initiating rollback handler...")
                        do! rollback.RollbackAsync()
                        logger.LogInformation("Rollback handler completed.")
                    | None ->
                        logger.LogInformation("No rollback handler configured.")

                    let completedAt = DateTime.UtcNow
                    return {
                        Config = config
                        Commands = finalResults
                        StartedAt = startedAt
                        CompletedAt = completedAt
                        Outcome = Failed(finalResults, combinedReason)
                    }
            }

    /// Factory helper that instantiates the harness with a real process executor.
    let createDefaultHarness (loggerFactory: ILoggerFactory) =
        let logger = loggerFactory.CreateLogger<AutonomousExecutionHarness>()
        let executor = new ProcessCommandExecutor(logger) :> ICommandExecutor
        new AutonomousExecutionHarness(logger, executor)
