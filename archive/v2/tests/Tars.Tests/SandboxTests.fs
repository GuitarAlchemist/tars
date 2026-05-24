namespace Tars.Tests

open Xunit
open Tars.Sandbox

type SandboxTests(output: Xunit.Abstractions.ITestOutputHelper) =

    [<Fact(Skip = "Requires Docker with tars-sandbox image")>]
    member this.``Can run python script in sandbox``() =
        task {
            use client = DockerClient.createClient ()

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

    [<Fact(Skip = "Requires Docker with tars-sandbox image")>]
    member this.``Sandbox has no internet access``() =
        task {
            use client = DockerClient.createClient ()

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

                Assert.NotEqual(0L, exitCode)
            | Error e ->
                Assert.Fail($"Failed to run container: {e}")
        }
