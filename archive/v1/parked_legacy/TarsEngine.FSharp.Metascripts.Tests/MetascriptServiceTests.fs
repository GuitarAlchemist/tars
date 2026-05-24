module TarsEngine.FSharp.Metascripts.Tests.MetascriptServiceTests

open System
open System.IO
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.Metascripts.Core
open TarsEngine.FSharp.Metascripts.Core.MetascriptHelpers
open TarsEngine.FSharp.Metascripts.Discovery
open TarsEngine.FSharp.Metascripts.Services
open Xunit

let private createService () =
    let registry = MetascriptRegistry()
    let manager = MetascriptManager(registry, NullLogger<MetascriptManager>.Instance)
    let discovery = MetascriptDiscovery(registry, manager, NullLogger<MetascriptDiscovery>.Instance)
    let service = MetascriptService(registry, manager, discovery, NullLogger<MetascriptService>.Instance)
    service, registry

[<Fact>]
let ``ExecuteMetascriptAsync runs command sections and returns context variables`` () =
    let tempDirectory = Path.Combine(Path.GetTempPath(), $"metascripts-tests-{Guid.NewGuid():N}")
    Directory.CreateDirectory(tempDirectory) |> ignore

    let content = """
```yaml
greeting: hello-world
```

```command
echo metascript-command-output
```

Normal markdown content.
"""

    let metadata = createMetadata "sample" "Test metascript" "TestSuite"
    let source = createMetascriptSource "sample" content (Path.Combine(tempDirectory, "sample.tars")) metadata

    let service, registry = createService ()
    let registered = registry.RegisterMetascript(source)

    let executionResult =
        (service :> IMetascriptService).ExecuteMetascriptAsync(registered.Source.Name)
        |> Async.AwaitTask
        |> Async.RunSynchronously

    Directory.Delete(tempDirectory, true)

    match executionResult with
    | Ok result ->
        Assert.Equal(MetascriptExecutionStatus.Completed, result.Status)
        Assert.True(result.Output.Contains("COMMAND_EXECUTION"))
        Assert.Equal("hello-world", result.Variables.["greeting"] :?> string)
    | Error error ->
        failwithf $"Metascript execution failed: %s{error}"

[<Fact>]
let ``ListMetascriptsAsync surfaces registered entries`` () =
    let service, registry = createService ()
    let metadata = createMetadata "simple" "Simple metascript" "Test"
    let source = createMetascriptSource "simple" "```command\necho ok\n```" "" metadata
    registry.RegisterMetascript(source) |> ignore

    let listResult =
        (service :> IMetascriptService).ListMetascriptsAsync()
        |> Async.AwaitTask
        |> Async.RunSynchronously

    match listResult with
    | Ok metascripts ->
        Assert.Single(metascripts |> List.filter (fun m -> m.Source.Name = "simple"))
    | Error error ->
        failwithf $"Listing metascripts failed: %s{error}"
