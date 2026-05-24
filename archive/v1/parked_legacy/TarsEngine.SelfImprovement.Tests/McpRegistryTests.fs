namespace TarsEngine.SelfImprovement.Tests

open System
open System.IO
open Xunit
open TarsEngine.FSharp.Core.Mcp

type McpRegistryTests() =
    let originalConfig = Environment.GetEnvironmentVariable("TARS_MCP_CONFIG")
    let tempFiles = ResizeArray<string>()

    let registerTempFile path =
        tempFiles.Add(path)
        path

    let writeConfig (contents: string) =
        let path = Path.Combine(Path.GetTempPath(), $"tars-mcp-{Guid.NewGuid():N}.yaml")
        File.WriteAllText(path, contents)
        registerTempFile path

    let loadFrom contents =
        let path = writeConfig contents
        Environment.SetEnvironmentVariable("TARS_MCP_CONFIG", path)
        McpRegistry.Refresh()
        path

    interface IDisposable with
        member _.Dispose() =
            for path in tempFiles do
                try File.Delete(path) with _ -> ()
            Environment.SetEnvironmentVariable("TARS_MCP_CONFIG", originalConfig)
            McpRegistry.Refresh()

    [<Fact>]
    member _.``TryGetServer returns registered server`` () =
        ignore (
            loadFrom """
servers:
  - name: augment-local
    url: http://localhost:9000/
    description: Local Augment endpoint
    headers:
      Authorization: bearer demo-token
"""
        )
        match McpRegistry.TryGetServer "augment-local" with
        | None -> Assert.True(false, "Expected server to be found.")
        | Some server ->
            Assert.Equal("augment-local", server.Name)
            Assert.Equal("http://localhost:9000/", server.Url)
            Assert.Equal("Local Augment endpoint", server.Description |> Option.defaultValue "")
            Assert.Equal("bearer demo-token", server.Headers.["Authorization"])

    [<Fact>]
    member _.``TryGetServer trims lookup values`` () =
        ignore (
            loadFrom """
servers:
  - name: Demo
    url: http://localhost:7100
"""
        )

        Assert.True(McpRegistry.TryGetServer(" demo  ").IsSome)

    [<Fact>]
    member _.``Refresh reloads configuration`` () =
        let first = loadFrom """
servers:
  - name: alpha
    url: http://alpha/
"""
        let second = writeConfig """
servers:
  - name: beta
    url: http://beta/
"""

        Environment.SetEnvironmentVariable("TARS_MCP_CONFIG", second)
        McpRegistry.Refresh()

        Assert.True(McpRegistry.TryGetServer("beta").IsSome)
        Assert.True(McpRegistry.TryGetServer("alpha").IsNone)

    [<Fact>]
    member _.``ConfigSearchPaths include environment override`` () =
        let path = writeConfig "servers: []"
        Environment.SetEnvironmentVariable("TARS_MCP_CONFIG", path)

        let searchPaths = McpRegistry.ConfigSearchPaths()
        Assert.Contains(path, searchPaths)
