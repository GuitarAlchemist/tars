namespace TarsEngine.FSharp.SelfImprovement

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.SelfImprovement.ExecutionHarness

/// <summary>
/// Executes the build/test validation harness as part of the autonomous workflow.
/// </summary>
module ExecutionValidationStep =

    let private createCommand name arguments workingDir timeoutMinutes =
        {
            Name = name
            Executable = "dotnet"
            Arguments = arguments
            WorkingDirectory = workingDir
            Timeout = timeoutMinutes |> Option.map (fun minutes -> TimeSpan.FromMinutes(minutes : double))
            Environment = Map.empty
        }

    type private LoggingRollbackHandler(logger: ILogger, rootPath: string) =
        interface IRollbackHandler with
            member _.RollbackAsync() =
                async {
                    logger.LogWarning("Validation harness failed — manual review required. Pending changes located under {RootPath}.", rootPath)
                }

    /// <summary>
    /// Workflow handler that runs the execution harness for validation.
    /// </summary>
    let getHandler (logger: ILogger) : WorkflowState -> Task<StepResult> =
        fun state ->
            task {
                logger.LogInformation("Running execution validation harness")

                let solutionDir = Directory.GetCurrentDirectory()

                let preValidation = [
                    createCommand "restore" "restore Tars.sln" (Some solutionDir) (Some 10.0)
                    createCommand "build" "build Tars.sln -c Release" (Some solutionDir) (Some 15.0)
                ]

                let validation = [
                    createCommand "test" "test TarsEngine.SelfImprovement.Tests/TarsEngine.SelfImprovement.Tests.fsproj -c Release" (Some solutionDir) (Some 10.0)
                ]

                let benchmark = []

                let rollback = LoggingRollbackHandler(logger, solutionDir) :> IRollbackHandler
                let config = {
                    Description = "Autonomous execution validation"
                    PreValidation = preValidation
                    Validation = validation
                    Benchmarks = benchmark
                    Rollback = Some rollback
                    StopOnFailure = true
                    CaptureLogs = true
                }

                let executor = ProcessCommandExecutor(logger) :> ICommandExecutor
                let harness = AutonomousExecutionHarness(logger, executor)
                let! report = harness.RunAsync(config)

                let commandSummary =
                    report.Commands
                    |> List.map (fun result ->
                        $"%s{result.Command.Name}:%d{result.ExitCode}")
                    |> String.concat "; "

                match report.Outcome with
                | AllPassed _ ->
                    let data =
                        [
                            "HarnessDescription", report.Config.Description
                            "HarnessStartedAt", report.StartedAt.ToString("o")
                            "HarnessCompletedAt", report.CompletedAt.ToString("o")
                            "CommandSummary", commandSummary
                        ]
                        |> Map.ofList

                    return Ok data
                | Failed (_, reason) ->
                    return Error $"Execution harness failed: %s{reason}"
            }
