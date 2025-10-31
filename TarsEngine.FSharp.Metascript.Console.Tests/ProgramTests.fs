module TarsEngine.FSharp.Metascript.Console.Tests.ProgramTests

open System
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript
open TarsEngine.FSharp.Metascript.BlockHandlers
open TarsEngine.FSharp.Metascript.ConsoleApp.Program
open Xunit

let private buildServiceProvider () =
    let services = ServiceCollection()
    let provider = configureServices services |> fun svc -> svc.BuildServiceProvider()

    // Register block handlers with the registry just like the console does.
    let registry = provider.GetRequiredService<BlockHandlerRegistry>()
    for handler in provider.GetServices<IBlockHandler>() do
        registry.RegisterHandler(handler)

    provider

[<Fact>]
let ``createSimpleMetascript yields three executable blocks`` () =
    let script = createSimpleMetascript()

    Assert.Equal("Simple Test", script.Name)
    Assert.Equal(3, script.Blocks.Length)

    let blockTypes = script.Blocks |> List.map (fun block -> block.Type)
    Assert.Equal<MetascriptBlockType list>([MetascriptBlockType.FSharp; MetascriptBlockType.Command; MetascriptBlockType.Text], blockTypes)

[<Fact>]
let ``runMetascript executes blocks and returns success`` () =
    use provider = buildServiceProvider()
    let script = createSimpleMetascript()

    let result =
        runMetascript provider script
        |> Async.AwaitTask
        |> Async.RunSynchronously

    Assert.Equal(MetascriptExecutionStatus.Success, result.Status)
    Assert.True(result.ExecutionTimeMs >= 0.0)

    Assert.Contains("Hello, World!", result.Output)
    Assert.Contains("Hello from the command line!", result.Output, StringComparison.OrdinalIgnoreCase)
    Assert.Contains("This is a text block.", result.Output)
