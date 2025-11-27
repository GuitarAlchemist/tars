namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Tools.Standard

type ToolTests() =

    [<Fact>]
    member _.``runCommand executes in sandbox``() =
        task {
            // This test requires Docker to be running and tars-sandbox image to exist
            // We assume the environment is set up correctly (checked via 'docker images' earlier)

            let! result = StandardTools.runCommand "echo hello from sandbox"

            Assert.Equal("hello from sandbox", result)
        }

    [<Fact>]
    member _.``runCommand runs in isolated OS``() =
        task {
            // Check OS release info
            let! result = StandardTools.runCommand "cat /etc/os-release"

            // The sandbox is Debian-based (python:3.11-slim)
            Assert.Contains("PRETTY_NAME=\"Debian", result)
            // It should NOT contain Windows
            Assert.DoesNotContain("Microsoft Windows", result)
        }
