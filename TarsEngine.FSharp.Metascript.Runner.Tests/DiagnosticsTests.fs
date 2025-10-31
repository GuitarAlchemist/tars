module TarsEngine.FSharp.Metascript.Runner.Tests.DiagnosticsTests

open System
open System.IO
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.Metascript.DependencyInjection
open TarsEngine.FSharp.Metascript.Runner
open Xunit

let private withTemporaryWorkingDirectory (action: string -> unit) =
    let originalDirectory = Directory.GetCurrentDirectory()
    let tempDirectory = Directory.CreateTempSubdirectory()

    Directory.SetCurrentDirectory(tempDirectory.FullName)

    try
        action tempDirectory.FullName
    finally
        Directory.SetCurrentDirectory(originalDirectory)
        Directory.Delete(tempDirectory.FullName, true)

[<Fact>]
let ``initializeTarsSystem completes without errors`` () =
    withTemporaryWorkingDirectory (fun _ ->
        let logger = NullLogger.Instance :> ILogger
        let result = Program.initializeTarsSystem logger |> Async.RunSynchronously
        Assert.True(result))

[<Fact>]
let ``runComprehensiveDiagnostics produces report with passing tests`` () =
    withTemporaryWorkingDirectory (fun workingDirectory ->
        let services = ServiceCollection()
        services.AddLogging(fun builder ->
            builder.SetMinimumLevel(LogLevel.Information) |> ignore) |> ignore

        ServiceCollectionExtensions.addTarsEngineFSharpMetascript(services) |> ignore

        use provider = services.BuildServiceProvider()

        let reportName = "runner-diagnostics.md"
        let (passedTests, testsRun, elapsedSeconds) =
            Program.runComprehensiveDiagnostics provider reportName
            |> Async.RunSynchronously

        Assert.Equal(testsRun, passedTests)
        Assert.True(elapsedSeconds >= 0.0)

        let reportPath = Path.Combine(workingDirectory, "output", reportName)
        Assert.True(File.Exists(reportPath), $"Expected diagnostic report at %s{reportPath}.")

        let reportContent = File.ReadAllText(reportPath)
        Assert.Contains("# TARS System Diagnostic Report", reportContent)
        Assert.Contains("ALL TESTS PASSED", reportContent))
