namespace Tars.Tests

open System
open System.Diagnostics
open System.IO
open Xunit
open Xunit.Abstractions
open System.Text

type GoldenRun(output: ITestOutputHelper) =

    [<Fact>]
    member _.``Golden Run: Demo Ping``() =
        output.WriteLine("Starting Golden Run: Demo Ping")
        // Locate the CLI project file
        let currentDir = Directory.GetCurrentDirectory()
        output.WriteLine($"Current directory: {currentDir}")
        // Navigate up from bin/Debug/net10.0
        let repoRoot =
            let d = DirectoryInfo(currentDir)
            // Depending on runner, might be in .../tests/Tars.Tests/bin/Debug/net10.0
            // traverse up until we find Tars.sln
            let rec findRoot (dir: DirectoryInfo) =
                if isNull dir then
                    failwith "Could not find repo root"

                if dir.GetFiles("Tars.sln").Length > 0 then
                    dir
                else
                    findRoot dir.Parent

            findRoot d

        output.WriteLine($"Repo root: {repoRoot.FullName}")

        let cliProject =
            Path.Combine(repoRoot.FullName, "src", "Tars.Interface.Cli", "Tars.Interface.Cli.fsproj")

        Assert.True(File.Exists(cliProject), $"CLI project not found at {cliProject}")

        // Run dotnet run
        let startInfo = ProcessStartInfo()
        startInfo.FileName <- "dotnet"
        startInfo.Arguments <- $"run --project \"{cliProject}\" -- demo-ping"
        startInfo.RedirectStandardOutput <- true
        startInfo.RedirectStandardError <- true
        startInfo.UseShellExecute <- false
        startInfo.CreateNoWindow <- true

        let outputSb = StringBuilder()
        let errorSb = StringBuilder()

        use proc = new Process()
        proc.StartInfo <- startInfo

        proc.OutputDataReceived.Add(fun args ->
            if not (isNull args.Data) then
                outputSb.AppendLine(args.Data) |> ignore)

        proc.ErrorDataReceived.Add(fun args ->
            if not (isNull args.Data) then
                errorSb.AppendLine(args.Data) |> ignore)

        output.WriteLine("Starting process...")
        proc.Start() |> ignore
        proc.BeginOutputReadLine()
        proc.BeginErrorReadLine()

        // Wait for a reasonable time (e.g., 10 seconds)
        let finished = proc.WaitForExit(15000)

        let outputStr = outputSb.ToString()
        let errorStr = errorSb.ToString()

        output.WriteLine("Process Output:")
        output.WriteLine(outputStr)

        if errorStr.Length > 0 then
            output.WriteLine("Process Error:")
            output.WriteLine(errorStr)

        if not finished then
            proc.Kill()
            failwith $"Process timed out. Output:\n{outputStr}\nError:\n{errorStr}"

        Assert.Equal(0, proc.ExitCode)

        // Assertions on output
        Assert.Contains("Starting TARS v2 Demo Ping", outputStr)

        if not (outputStr.Contains("DemoAgent received: PING")) then
            failwith $"Expected 'DemoAgent received: PING' but got:\n{outputStr}\nError:\n{errorStr}"

        Assert.Contains("Ping sent", outputStr)
