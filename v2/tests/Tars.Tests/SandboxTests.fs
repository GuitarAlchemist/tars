namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Sandbox

type SandboxTests(output: Xunit.Abstractions.ITestOutputHelper) =

    [<Fact>]
    member this.``Can run python script in sandbox``() =
        task {
            use client = DockerClient.createClient ()

            // Simple python script to print hello
            let cmd = [ "python"; "-c"; "print('Hello from Sandbox')" ]

            let! result = DockerClient.runContainer client "tars-sandbox" cmd

            match result with
            | Ok(stdout, stderr, exitCode) ->
                output.WriteLine($"Stdout: {stdout}")
                output.WriteLine($"Stderr: {stderr}")
                output.WriteLine($"ExitCode: {exitCode}")

                Assert.Equal(0L, exitCode)
                Assert.Contains("Hello from Sandbox", stdout)
            | Error e -> Assert.Fail($"Failed to run container: {e}")
        }

    [<Fact>]
    member this.``Sandbox has no internet access``() =
        task {
            use client = DockerClient.createClient ()

            // Try to connect to google.com
            let cmd =
                [ "python"
                  "-c"
                  "import urllib.request; urllib.request.urlopen('http://google.com')" ]

            let! result = DockerClient.runContainer client "tars-sandbox" cmd

            match result with
            | Ok(stdout, stderr, exitCode) ->
                output.WriteLine($"Stdout: {stdout}")
                output.WriteLine($"Stderr: {stderr}")
                output.WriteLine($"ExitCode: {exitCode}")

                // Should fail
                Assert.NotEqual(0L, exitCode)
            | Error e ->
                // If docker fails to start, that's different, but here we expect the script to fail
                Assert.Fail($"Failed to run container: {e}")
        }
