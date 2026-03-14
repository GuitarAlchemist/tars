namespace TarsEngine.FSharp.Cli.Core

open System
open System.Diagnostics
open System.IO
open System.Text
open System.Text.Json
open System.Threading.Tasks

/// Internal helpers shared by the Flux Codex runtime.
module private FluxCodexHelpers =

    let sanitizeBranchName (prefix: string) (task: string) =
        let safe =
            task.ToLowerInvariant()
            |> Seq.map (fun ch ->
                if Char.IsLetterOrDigit(ch) then ch
                elif ch = ' ' || ch = '-' || ch = '_' then '-'
                else '-')
            |> Seq.toArray
            |> fun arr -> String(arr)
        let trimmed =
            safe.Trim([| '-'; '_' |])
        let truncated =
            if trimmed.Length > 32 then trimmed.Substring(0, 32) else trimmed
        sprintf "%s%s" prefix truncated

    let ensureDirectory (path: string) =
        Directory.CreateDirectory(path) |> ignore

    let toArguments (args: string list) =
        args
        |> List.map (fun arg ->
            if String.IsNullOrWhiteSpace(arg) then "\"\""
            elif arg.Contains(" ") || arg.Contains("\"") then
                "\"" + arg.Replace("\"", "\\\"") + "\""
            else
                arg)
        |> String.concat " "

    let runProcess (workingDirectory: string) (fileName: string) (arguments: string list) (environment: Map<string, string>) =
        let psi = ProcessStartInfo()
        psi.FileName <- fileName
        psi.Arguments <- toArguments arguments
        psi.WorkingDirectory <- workingDirectory
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.UseShellExecute <- false
        psi.CreateNoWindow <- true

        for KeyValue(key, value) in environment do
            psi.Environment[key] <- value

        use proc = new Process()
        proc.StartInfo <- psi
        let startTime = DateTime.UtcNow
        proc.Start() |> ignore
        let stdoutTask = proc.StandardOutput.ReadToEndAsync()
        let stderrTask = proc.StandardError.ReadToEndAsync()
        proc.WaitForExit()
        let endTime = DateTime.UtcNow
        let stdout = stdoutTask.Result.TrimEnd()
        let stderr = stderrTask.Result.TrimEnd()

        {
            ExitCode = proc.ExitCode
            StandardOutput = stdout
            StandardError = stderr
            Duration = endTime - startTime
            StartedAt = startTime
            CompletedAt = endTime
            Command = {
                Executable = fileName
                Arguments = arguments
                WorkingDirectory = Some workingDirectory
                Environment = environment
            }
        }

/// Local sandbox runner using git clone for isolated workspaces.
type LocalSandboxRunner(repoRoot: string, allowedCommands: string list) =
    let allowed =
        allowedCommands
        |> List.map (fun cmd -> cmd.ToLowerInvariant())
        |> Set.ofList

    let workspacesRoot = Path.Combine(repoRoot, ".tars", "flux", "workspaces")

    interface ISandboxRunner with
        member _.Name = "local"

        member _.CreateWorkspace(runId, baseBranch) =
            try
                FluxCodexHelpers.ensureDirectory workspacesRoot
                let workspacePath = Path.Combine(workspacesRoot, runId)
                if Directory.Exists(workspacePath) then
                    Directory.Delete(workspacePath, true)

                let cloneArgs = [
                    "clone"
                    "--depth"; "1"
                    "--branch"; baseBranch
                    "."
                    workspacePath
                ]

                let result = FluxCodexHelpers.runProcess repoRoot "git" cloneArgs Map.empty
                if result.ExitCode = 0 then
                    Ok workspacePath
                else
                    Error (sprintf "Failed to clone repository for workspace: %s" result.StandardError)
            with
            | ex -> Error (sprintf "Workspace creation failed: %s" ex.Message)

        member _.Run(workspace, command) =
            let executable = command.Executable.ToLowerInvariant()
            if not (allowed.Contains executable) then
                Error (sprintf "Command '%s' is not allowed by Flux security policy." command.Executable)
            else
                try
                    let workingDirectory =
                        command.WorkingDirectory
                        |> Option.defaultValue workspace
                    let result =
                        FluxCodexHelpers.runProcess workingDirectory command.Executable command.Arguments command.Environment
                    Ok result
                with
                | ex -> Error (sprintf "Command execution failed: %s" ex.Message)

        member _.CleanupWorkspace(workspace) =
            try
                if Directory.Exists(workspace) then
                    Directory.Delete(workspace, true)
            with
            | _ -> ()

/// Draft pull request publisher writing markdown summaries locally.
type FilePrPublisher() =
    interface IPrPublisher with
        member _.PublishDraft(_repoRoot, _workspace, plan, stepResults, artifactDirectory) =
            try
                FluxCodexHelpers.ensureDirectory artifactDirectory
                let path = Path.Combine(artifactDirectory, "pull_request.md")
                use writer = new StreamWriter(path, false, Encoding.UTF8)

                writer.WriteLine("# Draft Pull Request")
                writer.WriteLine(sprintf "Run ID: %s" plan.BranchName)
                writer.WriteLine(sprintf "Task: %s" plan.TaskDescription)
                writer.WriteLine("")
                writer.WriteLine("## Execution Summary")

                for result in stepResults do
                    let status =
                        match result.Outcome with
                        | FluxStepOutcome.Completed _ -> "completed"
                        | FluxStepOutcome.Skipped reason -> sprintf "skipped: %s" reason
                        | FluxStepOutcome.Failed reason -> sprintf "failed: %s" reason

                    writer.WriteLine(sprintf "- %s %s" result.Step.Title status)

                writer.WriteLine("")
                writer.WriteLine("Generated by the Flux Codex workflow.")
                writer.Flush()
                Ok (Some path)
            with
            | ex -> Error (sprintf "Failed to create draft PR summary: %s" ex.Message)

/// Deterministic fallback planner used when no LLM integration is configured.
type DeterministicModelInvoker() =
    interface IModelInvoker with
        member _.GeneratePlan(context: FluxPlanningContext) =
            let branchName =
                let timestamp = context.Timestamp.ToString("yyyyMMdd-HHmmss")
                FluxCodexHelpers.sanitizeBranchName "flux/" (context.Task) + "-" + timestamp

            let steps =
                [
                    {
                        Id = "plan-1"
                        Title = "Assess repository state"
                        Details = "Gather repository metadata and confirm base branch status."
                        Kind = FluxPlanStepKind.Analysis
                        Command = Some {
                            Executable = "git"
                            Arguments = [ "status"; "--short" ]
                            WorkingDirectory = None
                            Environment = Map.empty
                        }
                        AllowFailure = false
                    }
                    {
                        Id = "plan-2"
                        Title = "Create task branch"
                        Details = sprintf "Create dedicated branch '%s' from '%s'." branchName context.BaseBranch
                        Kind = FluxPlanStepKind.Branching
                        Command = Some {
                            Executable = "git"
                            Arguments = [ "checkout"; "-b"; branchName ]
                            WorkingDirectory = None
                            Environment = Map.empty
                        }
                        AllowFailure = false
                    }
                    {
                        Id = "plan-3"
                        Title = "Restore dependencies"
                        Details = "Restore solution dependencies for reproducible builds."
                        Kind = FluxPlanStepKind.EnvironmentSetup
                        Command = Some {
                            Executable = "dotnet"
                            Arguments = [ "restore" ]
                            WorkingDirectory = None
                            Environment = Map.empty
                        }
                        AllowFailure = false
                    }
                    {
                        Id = "plan-4"
                        Title = "Apply requested changes"
                        Details = "Implement the requested modifications. Manual step to allow human edits or downstream automation."
                        Kind = FluxPlanStepKind.Manual
                        Command = None
                        AllowFailure = false
                    }
                    {
                        Id = "plan-5"
                        Title = "Build solution"
                        Details = "Build the solution to ensure compilation succeeds."
                        Kind = FluxPlanStepKind.Build
                        Command = Some {
                            Executable = "dotnet"
                            Arguments = [ "build"; "Tars.sln"; "-c"; "Release" ]
                            WorkingDirectory = None
                            Environment = Map.empty
                        }
                        AllowFailure = false
                    }
                    {
                        Id = "plan-6"
                        Title = "Run automated tests"
                        Details = "Execute the solution test suite."
                        Kind = FluxPlanStepKind.Test
                        Command = Some {
                            Executable = "dotnet"
                            Arguments = [ "test"; "Tars.sln"; "-c"; "Release"; "--no-build" ]
                            WorkingDirectory = None
                            Environment = Map.empty
                        }
                        AllowFailure = false
                    }
                    {
                        Id = "plan-7"
                        Title = "Summarize changes"
                        Details = "Collect diff summary for reporting."
                        Kind = FluxPlanStepKind.Diff
                        Command = Some {
                            Executable = "git"
                            Arguments = [ "status"; "--short" ]
                            WorkingDirectory = None
                            Environment = Map.empty
                        }
                        AllowFailure = false
                    }
                    {
                        Id = "plan-8"
                        Title = "Prepare draft pull request"
                        Details = "Generate a local draft PR summary for review."
                        Kind = FluxPlanStepKind.Publish
                        Command = None
                        AllowFailure = true
                    }
                ]

            {
                TaskDescription = context.Task
                BranchName = branchName
                Steps = steps
            }

/// Flux runner executing plans and creating artifacts.
type FluxRunner(config: FluxConfig, sandboxRunner: ISandboxRunner, planner: IModelInvoker, publisher: IPrPublisher) =

    let writeLog (artifactDir: string) (builder: StringBuilder) =
        let logPath = Path.Combine(artifactDir, "log.txt")
        File.WriteAllText(logPath, builder.ToString())

    let writeSummary (artifactDir: string) (summary: FluxRunSummary) =
        let summaryPath = Path.Combine(artifactDir, "run.json")
        use stream = File.Create(summaryPath)
        use writer = new Utf8JsonWriter(stream, JsonWriterOptions(Indented = true))

        writer.WriteStartObject()
        writer.WriteString("runId", summary.RunId)
        writer.WriteString("task", summary.Task)
        writer.WriteBoolean("success", summary.Success)
        writer.WriteString("baseBranch", summary.BaseBranch)
        writer.WriteString("branchName", summary.BranchName)
        match summary.DraftPrPath with
        | Some path -> writer.WriteString("draftPrPath", path)
        | None -> writer.WriteNull("draftPrPath")
        writer.WriteString("startedAt", summary.StartedAt.ToString("o"))
        writer.WriteString("completedAt", summary.CompletedAt.ToString("o"))
        writer.WritePropertyName("steps")
        writer.WriteStartArray()
        for step in summary.Steps do
            writer.WriteStartObject()
            writer.WriteString("id", step.Id)
            writer.WriteString("title", step.Title)
            writer.WriteString("kind", step.Kind)
            writer.WriteString("outcome", step.Outcome)
            match step.ExitCode with
            | Some exitCode -> writer.WriteNumber("exitCode", exitCode)
            | None -> writer.WriteNull("exitCode")
            writer.WriteNumber("durationMs", step.DurationMs)
            match step.Notes with
            | Some notes -> writer.WriteString("notes", notes)
            | None -> writer.WriteNull("notes")
            writer.WriteEndObject()
        writer.WriteEndArray()
        writer.WriteEndObject()
        writer.Flush()

    let writeDiffArtifacts (workspace: string) (artifactDir: string) =
        let runGit args =
            FluxCodexHelpers.runProcess workspace "git" args Map.empty
        let diff = runGit [ "diff"; "--stat" ]
        File.WriteAllText(Path.Combine(artifactDir, "diff_stat.txt"), diff.StandardOutput + Environment.NewLine + diff.StandardError)
        let fullDiff = runGit [ "diff" ]
        File.WriteAllText(Path.Combine(artifactDir, "diff.txt"), fullDiff.StandardOutput + Environment.NewLine + fullDiff.StandardError)

    member _.Run(parameters: FluxRunParameters) =
        task {
            let startTime = DateTime.UtcNow
            let runId = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss")
            let planningContext = {
                Task = parameters.Task
                BaseBranch = parameters.BaseBranch
                RunId = runId
                Timestamp = startTime
                RepoRoot = parameters.RepoRoot
                Config = config
            }

            let plan = planner.GeneratePlan(planningContext)
            let artifactDir = Path.Combine(parameters.RepoRoot, "output", "versions", runId)
            FluxCodexHelpers.ensureDirectory artifactDir

            let logBuilder = StringBuilder()
            logBuilder.AppendLine(sprintf "[Flux] Run %s started at %O" runId startTime) |> ignore
            logBuilder.AppendLine(sprintf "[Flux] Task: %s" parameters.Task) |> ignore

            match sandboxRunner.CreateWorkspace(runId, parameters.BaseBranch) with
            | Error error ->
                logBuilder.AppendLine(sprintf "[Flux] Workspace creation failed: %s" error) |> ignore
                writeLog artifactDir logBuilder
                return {
                    RunId = runId
                    WorkspacePath = None
                    ArtifactDirectory = artifactDir
                    Plan = plan
                    StepResults = []
                    DraftPrPath = None
                    StartedAt = startTime
                    CompletedAt = DateTime.UtcNow
                    TaskDescription = parameters.Task
                    Success = false
                    Messages = [ error ]
                }
            | Ok workspace ->
                try
                    logBuilder.AppendLine(sprintf "[Flux] Workspace: %s" workspace) |> ignore
                    let stepResults = ResizeArray<FluxStepExecution>()
                    let mutable continueExecution = true

                    for step in plan.Steps do
                        if continueExecution then
                            let stepStart = DateTime.UtcNow
                            logBuilder.AppendLine(sprintf "[Flux] Step %s - %s" step.Id step.Title) |> ignore

                            let outcome =
                                match step.Kind, step.Command with
                                | FluxPlanStepKind.Build, _ when parameters.SkipBuild ->
                                    FluxStepOutcome.Skipped "Build step disabled via --skip-build."
                                | FluxPlanStepKind.Test, _ when parameters.SkipTests ->
                                    FluxStepOutcome.Skipped "Test step disabled via --skip-tests."
                                | _, Some command ->
                                    match sandboxRunner.Run(workspace, command) with
                                    | Ok result ->
                                        logBuilder.AppendLine(result.StandardOutput) |> ignore
                                        if not (String.IsNullOrWhiteSpace(result.StandardError)) then
                                            logBuilder.AppendLine(result.StandardError) |> ignore
                                        if result.ExitCode = 0 || step.AllowFailure then
                                            FluxStepOutcome.Completed(Some result)
                                        else
                                            FluxStepOutcome.Failed (sprintf "Command exited with %d" result.ExitCode)
                                    | Error err ->
                                        logBuilder.AppendLine(sprintf "[Flux] Command failed: %s" err) |> ignore
                                        FluxStepOutcome.Failed err
                                | _, None ->
                                    FluxStepOutcome.Completed None

                            let stepEnd = DateTime.UtcNow
                            let execution = {
                                Step = step
                                Outcome = outcome
                                StartedAt = stepStart
                                CompletedAt = stepEnd
                            }
                            stepResults.Add(execution)

                            match outcome with
                            | FluxStepOutcome.Failed reason ->
                                logBuilder.AppendLine(sprintf "[Flux] Step %s failed: %s" step.Id reason) |> ignore
                                continueExecution <- false
                            | _ -> ()

                    if continueExecution then
                        try
                            writeDiffArtifacts workspace artifactDir
                            logBuilder.AppendLine("[Flux] Diff artifacts recorded.") |> ignore
                        with ex ->
                            logBuilder.AppendLine(sprintf "[Flux] Failed to record diff artifacts: %s" ex.Message) |> ignore

                    let stepResultsList = stepResults |> Seq.toList
                    let success =
                        stepResultsList
                        |> List.forall (fun r -> match r.Outcome with | FluxStepOutcome.Failed _ -> false | _ -> true)

                    let prResult =
                        if success && parameters.EnablePullRequest then
                            publisher.PublishDraft(parameters.RepoRoot, workspace, plan, stepResultsList, artifactDir)
                        else
                            Ok None

                    let draftPath, messages =
                        match prResult with
                        | Ok path -> path, []
                        | Error err -> None, [ err ]

                    let summarySteps =
                        stepResultsList
                        |> List.map (fun r ->
                            let outcomeText, exitCode, notes =
                                match r.Outcome with
                                | FluxStepOutcome.Completed(Some result) ->
                                    "completed", Some result.ExitCode, if String.IsNullOrWhiteSpace(result.StandardError) then None else Some result.StandardError
                                | FluxStepOutcome.Completed None -> "completed", None, None
                                | FluxStepOutcome.Skipped reason -> "skipped", None, Some reason
                                | FluxStepOutcome.Failed reason -> "failed", None, Some reason
                            {
                                Id = r.Step.Id
                                Title = r.Step.Title
                                Kind = r.Step.Kind.ToString()
                                Outcome = outcomeText
                                ExitCode = exitCode
                                DurationMs = (r.CompletedAt - r.StartedAt).TotalMilliseconds
                                Notes = notes
                            })

                    let endTime = DateTime.UtcNow
                    let summary: FluxRunSummary = {
                        RunId = runId
                        Task = parameters.Task
                        Success = success
                        BaseBranch = parameters.BaseBranch
                        BranchName = plan.BranchName
                        DraftPrPath = draftPath
                        StartedAt = startTime
                        CompletedAt = endTime
                        Steps = summarySteps
                    }

                    writeLog artifactDir logBuilder
                    writeSummary artifactDir summary

                    return {
                        RunId = runId
                        WorkspacePath = Some workspace
                        ArtifactDirectory = artifactDir
                        Plan = plan
                        StepResults = stepResultsList
                        DraftPrPath = draftPath
                        StartedAt = startTime
                        CompletedAt = endTime
                        TaskDescription = parameters.Task
                        Success = success
                        Messages = messages
                    }
                finally
                    sandboxRunner.CleanupWorkspace(workspace)
        }
