namespace TarsEngine.FSharp.Cli.Tests

open System
open System.IO
open System.Diagnostics
open System.Collections.Generic
open System.Threading.Tasks
open Xunit
open TarsEngine.FSharp.Cli.Core

type private FakeSandboxRunner(repoRoot: string) =
    let executed = List<SandboxCommand>()

    let rec copyDirectory (source: string) (destination: string) =
        if not (Directory.Exists source) then
            raise (DirectoryNotFoundException(sprintf "Source directory not found: %s" source))

        Directory.CreateDirectory(destination) |> ignore

        for file in Directory.GetFiles(source, "*", SearchOption.TopDirectoryOnly) do
            let targetFile = Path.Combine(destination, Path.GetFileName(file))
            File.Copy(file, targetFile, true)

        for directory in Directory.GetDirectories(source, "*", SearchOption.TopDirectoryOnly) do
            let targetDir = Path.Combine(destination, Path.GetFileName(directory))
            copyDirectory directory targetDir

    interface ISandboxRunner with
        member _.Name = "fake"

        member _.CreateWorkspace(runId: string, _baseBranch: string) =
            try
                let workspace = Path.Combine(Path.GetTempPath(), "flux-integration-" + runId)
                if Directory.Exists workspace then
                    Directory.Delete(workspace, true)

                copyDirectory repoRoot workspace
                Ok workspace
            with
            | ex -> Error ex.Message

        member _.Run(_workspace: string, command: SandboxCommand) =
            executed.Add(command)
            let startTime = DateTime.UtcNow
            Ok {
                ExitCode = 0
                StandardOutput = sprintf "[mock] %s %s" command.Executable (String.concat " " command.Arguments)
                StandardError = ""
                Duration = TimeSpan.FromMilliseconds 1.0
                StartedAt = startTime
                CompletedAt = startTime.AddMilliseconds 1.0
                Command = command
            }

        member _.CleanupWorkspace(workspace: string) =
            if Directory.Exists workspace then
                Directory.Delete(workspace, true)

    member _.ExecutedCommands = executed |> Seq.toList

type private FakePublisher() =
    interface IPrPublisher with
        member _.PublishDraft(_repoRoot, _workspace, _plan, _stepResults, _artifactDirectory) =
            Ok None

module FluxTestHelpers =

    let runGit (workingDirectory: string) (arguments: string list) =
        let psi = ProcessStartInfo()
        psi.FileName <- "git"
        psi.Arguments <- String.concat " " arguments
        psi.WorkingDirectory <- workingDirectory
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.UseShellExecute <- false
        use proc = new Process()
        proc.StartInfo <- psi
        proc.Start() |> ignore
        proc.WaitForExit()
        if proc.ExitCode <> 0 then
            let error = proc.StandardError.ReadToEnd()
            failwithf "git %s failed: %s" psi.Arguments error

    let createTempRepository () =
        let repoRoot = Path.Combine(Path.GetTempPath(), "flux-runner-repo-" + Guid.NewGuid().ToString("N"))
        Directory.CreateDirectory(repoRoot) |> ignore
        runGit repoRoot ["init"]
        runGit repoRoot ["config"; "--local"; "user.email"; "flux-integration@example.com"]
        runGit repoRoot ["config"; "--local"; "user.name"; "Flux Integration Test"]
        File.WriteAllText(Path.Combine(repoRoot, "README.md"), "Flux Codex integration test repository")
        runGit repoRoot ["add"; "."]
        runGit repoRoot ["commit"; "-m"; "\"Initial commit\""]
        try
            runGit repoRoot ["branch"; "-M"; "main"]
        with _ -> ()
        repoRoot

module FluxCodexIntegrationTests =

    [<Fact>]
    let ``FluxRunner executes Codex plan and emits artifacts`` () : Task =
        task {
            let repoRoot = FluxTestHelpers.createTempRepository()
            try
                let sandbox = FakeSandboxRunner(repoRoot)
                let publisher = FakePublisher()
                let config = {
                    FluxConfigLoader.defaultConfig with
                        BaseBranch = "main"
                        Runner = "fake"
                        AllowedCommands = [ "git"; "dotnet" ]
                }

                let fluxRunner =
                    FluxRunner(
                        config,
                        sandbox :> ISandboxRunner,
                        DeterministicModelInvoker() :> IModelInvoker,
                        publisher :> IPrPublisher)

                let parameters = {
                    Task = "Add integration test coverage"
                    RepoRoot = repoRoot
                    BaseBranch = "main"
                    EnablePullRequest = false
                    SkipBuild = false
                    SkipTests = false
                }

                let! result = fluxRunner.Run(parameters)

                Assert.True(result.Success, "Flux run should succeed")
                Assert.True(Directory.Exists(result.ArtifactDirectory), "Artifacts directory should exist")
                Assert.True(File.Exists(Path.Combine(result.ArtifactDirectory, "log.txt")), "Log file should exist")
                Assert.True(File.Exists(Path.Combine(result.ArtifactDirectory, "run.json")), "Summary file should exist")
                Assert.True(File.Exists(Path.Combine(result.ArtifactDirectory, "diff.txt")), "Diff file should exist")

                let commandsWithProcess =
                    result.Plan.Steps |> List.filter (fun step -> step.Command.IsSome) |> List.length
                Assert.Equal(commandsWithProcess, sandbox.ExecutedCommands.Length)
            finally
                if Directory.Exists(repoRoot) then
                    Directory.Delete(repoRoot, true)
        }
