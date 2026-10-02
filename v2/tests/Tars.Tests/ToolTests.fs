namespace Tars.Tests

open System
open System.IO
open Xunit

type ToolTests() =

    let newWorkspace () =
        let dir = Path.Combine(Path.GetTempPath(), "tars-ws-" + Guid.NewGuid().ToString("N"))
        Directory.CreateDirectory dir |> ignore
        dir

    let writeArgs (path: string) =
        "{\"path\": " + Text.Json.JsonSerializer.Serialize(path) + ", \"content\": \"let x = 1\"}"

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

    // Evolve's executor wrote Fibonacci.fs into v2/, the repository it runs from.
    // Evolve now gives it a workspace of its own.
    [<Fact>]
    member _.``write_code writes inside the workspace when one is set``() =
        task {
            if not (TestHelpers.requireTools ()) then () else
            let ws = newWorkspace ()
            let name = $"ws-{Guid.NewGuid():N}.fs"
            Tars.Tools.ToolWorkspace.set (Some ws)

            try
                let! result = Tars.Tools.Standard.GitTools.writeCode (writeArgs name)

                Assert.Contains("Successfully wrote", result)
                Assert.True(File.Exists(Path.Combine(ws, name)))
                Assert.False(File.Exists(Path.GetFullPath name))
            finally
                Tars.Tools.ToolWorkspace.set None
                Directory.Delete(ws, true)
        }

    [<Fact>]
    member _.``write_code refuses a path that leaves the workspace``() =
        task {
            if not (TestHelpers.requireTools ()) then () else
            let ws = newWorkspace ()
            let outside = Path.Combine(Path.GetTempPath(), $"ws-outside-{Guid.NewGuid():N}.fs")
            Tars.Tools.ToolWorkspace.set (Some ws)

            try
                let! absolute = Tars.Tools.Standard.GitTools.writeCode (writeArgs outside)
                let! escaping = Tars.Tools.Standard.GitTools.writeCode (writeArgs $"../ws-escape-{Guid.NewGuid():N}.fs")

                Assert.StartsWith("write_code error", absolute)
                Assert.StartsWith("write_code error", escaping)
                Assert.False(File.Exists outside)
                Assert.Empty(Directory.GetFiles(Path.GetDirectoryName(ws), "ws-escape-*.fs"))
            finally
                Tars.Tools.ToolWorkspace.set None
                Directory.Delete(ws, true)
        }

    [<Fact>]
    member _.``patch_code patches a copy in the workspace and leaves the original alone``() =
        task {
            if not (TestHelpers.requireTools ()) then () else
            let ws = newWorkspace ()
            let name = $"ws-source-{Guid.NewGuid():N}.fs"
            File.WriteAllText(name, "let answer = 41")
            Tars.Tools.ToolWorkspace.set (Some ws)

            try
                let args =
                    "{\"path\": " + Text.Json.JsonSerializer.Serialize(name)
                    + ", \"original\": \"41\", \"replacement\": \"42\"}"

                let! result = Tars.Tools.Semantic.SemanticTools.patchCode args

                Assert.Equal("Successfully patched file.", result)
                Assert.Equal("let answer = 42", File.ReadAllText(Path.Combine(ws, name)))
                Assert.Equal("let answer = 41", File.ReadAllText name)
            finally
                Tars.Tools.ToolWorkspace.set None
                Directory.Delete(ws, true)
                File.Delete name
        }
