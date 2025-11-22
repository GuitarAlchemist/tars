namespace Tars.Engine.Grammar.Tests

open System
open Xunit
open Tars.Engine.Grammar

/// Tests covering RFC-backed grammar extraction
module RfcGrammarTests =

    let private loadRule (rfcId: string, ruleName: string) =
        let source = GrammarSource.EmbeddedRFC(rfcId, ruleName)
        Assert.True(GrammarSource.exists source, $"Expected grammar {rfcId}:{ruleName} to exist")
        GrammarSource.getContent source

    [<Theory>]
    [<InlineData("RFC5234", "ALPHA", "%x41-5A")>]
    [<InlineData("RFC5234", "DIGIT", "%x30-39")>]
    [<InlineData("RFC5234", "WSP", "; white space")>]
    [<InlineData("RFC3986", "scheme", "ALPHA")>]
    [<InlineData("RFC3986", "authority", "userinfo")>]
    [<InlineData("RFC9110", "field-name", "token")>]
    let ``Simple RFC productions load with expected fragments`` (rfcId: string, ruleName: string, expectedFragment: string) =
        let content = loadRule (rfcId, ruleName)
        Assert.Contains("=", content)
        Assert.Contains(expectedFragment, content)

    [<Fact>]
    let ``URI rule includes composite references`` () =
        // Act
        let content = loadRule ("RFC3986", "URI")
        let lines =
            content.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries)

        // Assert
        Assert.True(lines.Length >= 1, "URI grammar should contain multiple components")
        Assert.Contains("hier-part", content)
        Assert.Contains("fragment", content)

    [<Fact>]
    let ``Hier-part rule preserves multi-line layout`` () =
        // Act
        let content = loadRule ("RFC3986", "hier-part")
        let lines =
            content.Split(Environment.NewLine, StringSplitOptions.RemoveEmptyEntries)

        // Assert
        Assert.True(lines.Length >= 3, "hier-part rule should span multiple lines")
        Assert.StartsWith("      hier-part", lines[0])
        Assert.Matches(@".*path-absolute$", lines[1])
        Assert.Matches(@".*path-rootless$", lines[2])

    [<Fact>]
    let ``Missing rule surfaces descriptive error`` () =
        let ex = Assert.Throws<Exception>(fun () ->
            GrammarSource.EmbeddedRFC("RFC9110", "HTTP-message")
            |> GrammarSource.getContent
            |> ignore)

        Assert.Contains("Unable to locate rule 'HTTP-message'", ex.Message)
