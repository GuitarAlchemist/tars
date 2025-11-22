module TarsEngine.SelfImprovement.Tests.McpRequestParserTests

open Xunit
open TarsEngine.FSharp.Core.Mcp

type McpRequestParserTests() =

    [<Fact>]
    member _.``parse accepts JSON body`` () =
        let content = """
{
  "method": "tools/list",
  "params": {}
}
"""

        let result = McpRequestParser.parse content

        match result with
        | Ok request ->
            Assert.Contains("method", request.Body)
            Assert.Empty(request.Headers)
        | Error err ->
            Assert.True(false, err)

    [<Fact>]
    member _.``parse accepts YAML body and headers`` () =
        let content = """
headers:
  X-Test: test
body:
  method: tools/call
  params:
    tool: demo
"""

        let result = McpRequestParser.parse content

        match result with
        | Ok request ->
            Assert.Equal("test", request.Headers.["X-Test"])
            Assert.Contains("tools/call", request.Body)
        | Error err ->
            Assert.True(false, err)
