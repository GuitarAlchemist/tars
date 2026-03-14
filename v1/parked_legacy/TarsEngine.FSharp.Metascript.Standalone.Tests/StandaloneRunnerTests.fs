module TarsEngine.FSharp.Metascript.Standalone.Tests.StandaloneRunnerTests

open System
open System.IO
open Microsoft.Extensions.DependencyInjection
open TarsEngine.FSharp.Metascript
open TarsEngine.FSharp.Metascript.StandaloneApp.Program
open Xunit

let private sampleMetaPath () =
    Path.Combine(__SOURCE_DIRECTORY__, "..", "TarsEngine.FSharp.Metascript.Standalone", "simple_test.meta")
    |> Path.GetFullPath

[<Fact>]
let ``configureServices registers metascript dependencies`` () =
    let services = ServiceCollection()
    let provider = configureServices services |> fun svc -> svc.BuildServiceProvider()
    registerHandlers provider |> ignore

    let blockHandlers = provider.GetServices<TarsEngine.FSharp.Metascript.BlockHandlers.IBlockHandler>()
    Assert.NotEmpty(blockHandlers)

    let metascriptService = provider.GetService<TarsEngine.FSharp.Metascript.Services.IMetascriptService>()
    Assert.NotNull(metascriptService)

[<Fact>]
let ``runMetascriptFromFile executes bundled sample metascript`` () =
    let metaPath = sampleMetaPath()
    Assert.True(File.Exists(metaPath), $"Expected metascript at {metaPath}.")

    let services = ServiceCollection()
    use provider =
        configureServices services
        |> fun svc -> svc.BuildServiceProvider()

    registerHandlers provider |> ignore

    let result =
        runMetascriptFromFile provider metaPath
        |> Async.AwaitTask
        |> Async.RunSynchronously

    Assert.NotEqual(TarsEngine.FSharp.Metascript.MetascriptExecutionStatus.Failure, result.Status)

    let blockTypes = result.BlockResults |> List.map (fun br -> br.Block.Type)

    Assert.Contains(MetascriptBlockType.Command, blockTypes)
    Assert.Contains(MetascriptBlockType.Text, blockTypes)

    result.BlockResults
    |> List.iter (fun br ->
        Assert.NotEqual(TarsEngine.FSharp.Metascript.MetascriptExecutionStatus.Failure, br.Status))

    Assert.Contains(
        result.BlockResults,
        fun br -> br.Status = TarsEngine.FSharp.Metascript.MetascriptExecutionStatus.Success)
