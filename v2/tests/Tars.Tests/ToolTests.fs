namespace Tars.Tests

open Xunit

type ToolTests() =

    [<Fact(Skip = "Requires Docker with tars-sandbox image")>]
    member _.``runCommand executes in sandbox``() =
        task {
            let! result = Tars.Tools.Standard.StandardTools.runCommand "echo hello from sandbox"

            Assert.Equal("hello from sandbox", result)
        }

    [<Fact(Skip = "Requires Docker with tars-sandbox image")>]
    member _.``runCommand runs in isolated OS``() =
        task {
            let! result = Tars.Tools.Standard.StandardTools.runCommand "cat /etc/os-release"

            Assert.Contains("PRETTY_NAME=\"Debian", result)
            Assert.DoesNotContain("Microsoft Windows", result)
        }
