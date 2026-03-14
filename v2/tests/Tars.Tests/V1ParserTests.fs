namespace Tars.Tests

open System
open Xunit
open Tars.Metascript.V1
open Tars.Metascript

/// Tests for V1Parser round-trip serialization (parse -> serialize -> parse)
type V1ParserTests() =

    [<Fact>]
    member _.``parseMetascript parses simple text block``() =
        let input =
            """
text {
Hello, World!
}
"""

        let result = V1Parser.parseMetascript input "test" None

        Assert.Equal("test", result.Name)
        Assert.Equal(1, result.Blocks.Length)
        Assert.Equal(MetascriptBlockType.Text, result.Blocks.[0].Type)
        Assert.Contains("Hello, World!", result.Blocks.[0].Content)

    [<Fact>]
    member _.``parseMetascript parses block with parameters``() =
        let input =
            """
query(model="gpt-4" temperature="0.7") {
What is the meaning of life?
}
"""

        let result = V1Parser.parseMetascript input "param-test" None

        Assert.Equal(1, result.Blocks.Length)
        Assert.Equal(MetascriptBlockType.Query, result.Blocks.[0].Type)
        Assert.Equal(2, result.Blocks.[0].Parameters.Length)

        let modelParam =
            result.Blocks.[0].Parameters |> List.find (fun p -> p.Name = "model")

        Assert.Equal("gpt-4", modelParam.Value)

    [<Fact>]
    member _.``parseMetascript parses multiple blocks``() =
        let input =
            """
meta {
name = "multi-block"
}

fsharp {
let x = 42
printfn "%d" x
}

command {
echo "Hello"
}
"""

        let result = V1Parser.parseMetascript input "multi" None

        Assert.Equal(3, result.Blocks.Length)
        Assert.Equal(MetascriptBlockType.Meta, result.Blocks.[0].Type)
        Assert.Equal(MetascriptBlockType.FSharp, result.Blocks.[1].Type)
        Assert.Equal(MetascriptBlockType.Command, result.Blocks.[2].Type)

    [<Fact>]
    member _.``blockTypeToString returns correct string for all types``() =
        Assert.Equal("text", V1Parser.blockTypeToString MetascriptBlockType.Text)
        Assert.Equal("fsharp", V1Parser.blockTypeToString MetascriptBlockType.FSharp)
        Assert.Equal("command", V1Parser.blockTypeToString MetascriptBlockType.Command)
        Assert.Equal("query", V1Parser.blockTypeToString MetascriptBlockType.Query)
        Assert.Equal("meta", V1Parser.blockTypeToString MetascriptBlockType.Meta)

    [<Fact>]
    member _.``toMetascript serializes simple block correctly``() =
        let metascript =
            { Name = "test"
              Blocks =
                [ { Type = MetascriptBlockType.Text
                    Content = "Hello, World!"
                    LineNumber = 1
                    ColumnNumber = 1
                    Parameters = []
                    Id = Guid.NewGuid().ToString("N")
                    ParentId = None
                    Metadata = Map.empty } ]
              FilePath = None
              Variables = Map.empty
              Metadata = Map.empty }

        let output = V1Parser.toMetascript metascript

        Assert.Contains("meta {", output)
        Assert.Contains("name = \"test\"", output)
        Assert.Contains("text {", output)
        Assert.Contains("Hello, World!", output)

    [<Fact>]
    member _.``toMetascript serializes parameters correctly``() =
        let metascript =
            { Name = "param-test"
              Blocks =
                [ { Type = MetascriptBlockType.Query
                    Content = "What is 2+2?"
                    LineNumber = 1
                    ColumnNumber = 1
                    Parameters = [ { Name = "model"; Value = "gpt-4" }; { Name = "temperature"; Value = "0.5" } ]
                    Id = Guid.NewGuid().ToString("N")
                    ParentId = None
                    Metadata = Map.empty } ]
              FilePath = None
              Variables = Map.empty
              Metadata = Map.empty }

        let output = V1Parser.toMetascript metascript

        Assert.Contains("query(model=\"gpt-4\" temperature=\"0.5\")", output)
        Assert.Contains("What is 2+2?", output)

    [<Fact>]
    member _.``Round-trip parse then serialize preserves block types``() =
        let original =
            """
fsharp {
let add x y = x + y
add 1 2
}

command {
dotnet build
}
"""
        // Parse original
        let parsed = V1Parser.parseMetascript original "roundtrip" None

        // Serialize back
        let serialized = V1Parser.toMetascript parsed

        // Parse again
        let reparsed = V1Parser.parseMetascript serialized "roundtrip" None

        // Original blocks should be present (plus meta block added)
        let fsharpBlock =
            reparsed.Blocks |> List.tryFind (fun b -> b.Type = MetascriptBlockType.FSharp)

        let commandBlock =
            reparsed.Blocks |> List.tryFind (fun b -> b.Type = MetascriptBlockType.Command)

        Assert.True(fsharpBlock.IsSome, "FSharp block should exist")
        Assert.True(commandBlock.IsSome, "Command block should exist")
        Assert.Contains("add x y", fsharpBlock.Value.Content)
        Assert.Contains("dotnet build", commandBlock.Value.Content)

    [<Fact>]
    member _.``Round-trip preserves block content``() =
        let content = "This is some multi-line content!"

        let metascript =
            { Name = "content-test"
              Blocks =
                [ { Type = MetascriptBlockType.Text
                    Content = content
                    LineNumber = 1
                    ColumnNumber = 1
                    Parameters = []
                    Id = Guid.NewGuid().ToString("N")
                    ParentId = None
                    Metadata = Map.empty } ]
              FilePath = None
              Variables = Map.empty
              Metadata = Map.empty }

        let serialized = V1Parser.toMetascript metascript
        let reparsed = V1Parser.parseMetascript serialized "content-test" None

        let textBlock =
            reparsed.Blocks |> List.find (fun b -> b.Type = MetascriptBlockType.Text)

        Assert.Contains("This is some", textBlock.Content)
        Assert.Contains("multi-line content!", textBlock.Content)
