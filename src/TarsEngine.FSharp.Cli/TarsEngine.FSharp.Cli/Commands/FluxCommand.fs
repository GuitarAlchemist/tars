namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Text
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Core

/// CLI command that exposes the Codex-inspired Flux workflow.
type FluxCommand(logger: ILogger<FluxCommand>) =

    let printPlan (plan: FluxPlan) =
        printfn ""
        printfn "Flux Plan for branch %s" plan.BranchName
        printfn "--------------------------------------------"
        plan.Steps
        |> List.iter (fun step ->
            printfn "- [%s] %s" (step.Kind.ToString()) step.Title
            if not (String.IsNullOrWhiteSpace(step.Details)) then
                printfn "    %s" step.Details
            match step.Command with
            | Some command ->
                let args = command.Arguments |> String.concat " "
                printfn "    Command: %s %s" command.Executable args
            | None -> ())
        printfn ""

    let printResults (result: FluxRunResult) =
        printfn ""
        printfn "Flux Execution Summary"
        printfn "--------------------------------------------"
        printfn "Run ID      : %s" result.RunId
        printfn "Artifacts   : %s" result.ArtifactDirectory
        printfn "PR Draft    : %s" (result.DraftPrPath |> Option.defaultValue "not generated")
        result.StepResults
        |> List.iter (fun step ->
            match step.Outcome with
            | FluxStepOutcome.Completed(Some commandResult) ->
                printfn "- %s: completed (exit %d, %.0f ms)" step.Step.Title commandResult.ExitCode commandResult.Duration.TotalMilliseconds
            | FluxStepOutcome.Completed None ->
                printfn "- %s: completed (manual)" step.Step.Title
            | FluxStepOutcome.Skipped reason ->
                printfn "- %s: skipped (%s)" step.Step.Title reason
            | FluxStepOutcome.Failed reason ->
                printfn "- %s: FAILED (%s)" step.Step.Title reason)
        if result.Messages |> List.isEmpty |> not then
            printfn ""
            printfn "Notes:"
            result.Messages |> List.iter (printfn "- %s")
        printfn ""

    interface ICommand with
        member _.Name = "flux"

        member _.Description = "Execute Codex-inspired Flux task workflow"

        member _.Usage = "tars flux run \"<task-description>\" [--base-branch=<branch>] [--skip-build] [--skip-tests] [--no-pr]"

        member _.Examples = [
            "tars flux run \"Add debounce to search and update tests\""
            "tars flux run \"Improve benchmark logging\" --skip-tests --no-pr"
        ]

        member _.ValidateOptions(options) =
            match options.Arguments with
            | "run" :: tail -> tail.Length > 0
            | _ -> false

        member _.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | "run" :: taskSegments ->
                        let taskDescription = String.Join(" ", taskSegments)
                        let repoRoot = Directory.GetCurrentDirectory()
                        let config = FluxConfigLoader.load repoRoot

                        if not config.Enabled then
                            let message = "Flux workflow disabled. Set flux.enabled=true in tars.config.json."
                            logger.LogWarning("{Message}", message)
                            return CommandResult.failure(message)
                        else
                            let baseBranch = options.Options.TryFind("base") |> Option.orElse (options.Options.TryFind("base-branch")) |> Option.defaultValue config.BaseBranch
                            let skipBuild = options.Options.ContainsKey("skip-build")
                            let skipTests = options.Options.ContainsKey("skip-tests")
                            let enablePr = not (options.Options.ContainsKey("no-pr"))

                            let parameters = {
                                Task = taskDescription
                                RepoRoot = repoRoot
                                BaseBranch = baseBranch
                                EnablePullRequest = enablePr
                                SkipBuild = skipBuild
                                SkipTests = skipTests
                            }

                            let sandboxRunner: ISandboxRunner =
                                match config.Runner.ToLowerInvariant() with
                                | "local" -> new LocalSandboxRunner(repoRoot, config.AllowedCommands) :> ISandboxRunner
                                | _ -> new LocalSandboxRunner(repoRoot, config.AllowedCommands) :> ISandboxRunner

                            let planner = new DeterministicModelInvoker() :> IModelInvoker
                            let publisher = new FilePrPublisher() :> IPrPublisher
                            let runner = FluxRunner(config, sandboxRunner, planner, publisher)

                            logger.LogInformation("Starting Flux run for task: {Task}", taskDescription)
                            let! runResult = runner.Run(parameters)
                            printPlan runResult.Plan
                            printResults runResult

                            if runResult.Success then
                                logger.LogInformation("Flux run {RunId} completed successfully.", runResult.RunId)
                                return CommandResult.success("Flux run completed successfully.")
                            else
                                logger.LogWarning("Flux run {RunId} failed.", runResult.RunId)
                                return CommandResult.failure("Flux run encountered errors. Inspect artifacts for details.")
                    | _ ->
                        let message = "Usage: tars flux run \"<task-description>\""
                        logger.LogWarning("{Message}", message)
                        return CommandResult.failure(message)
                with
                | ex ->
                    logger.LogError(ex, "Flux command execution failed.")
                    return CommandResult.failure(sprintf "Flux command failed: %s" ex.Message)
            }
