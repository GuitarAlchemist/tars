namespace TarsEngine.FSharp.Cli.Tests

open System
open System.IO
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Core

module FluxCodexTests =

    [<Fact>]
    let ``FluxConfigLoader returns defaults when config missing`` () =
        let tempDir = Path.Combine(Path.GetTempPath(), "flux-config-test-" + Guid.NewGuid().ToString("N"))
        try
            Directory.CreateDirectory(tempDir) |> ignore
            let config = FluxConfigLoader.load tempDir
            config.BaseBranch |> should equal FluxConfigLoader.defaultConfig.BaseBranch
            config.AllowedCommands |> should not' Empty
        finally
            if Directory.Exists(tempDir) then
                Directory.Delete(tempDir, true)

    [<Fact>]
    let ``Deterministic planner creates branching step`` () =
        let planner = DeterministicModelInvoker() :> IModelInvoker
        let timestamp = DateTime.UtcNow
        let ctx: FluxPlanningContext = {
            Task = "Add debounce to search"
            BaseBranch = FluxConfigLoader.defaultConfig.BaseBranch
            RunId = "test-run"
            Timestamp = timestamp
            RepoRoot = Environment.CurrentDirectory
            Config = FluxConfigLoader.defaultConfig
        }

        let plan = planner.GeneratePlan(ctx)

        plan.TaskDescription |> should equal ctx.Task
        plan.Steps |> should not' Empty
        plan.Steps |> List.exists (fun step -> step.Kind = FluxPlanStepKind.Branching) |> should equal true
