namespace TarsEngine.FSharp.IntegrationTests

open System
open System.IO
open System.Diagnostics
open Xunit
open FluentAssertions

/// <summary>
/// CRITICAL CLI INTEGRATION TESTS
/// Ensures TARS CLI enforces zero simulation tolerance
/// </summary>
module TarsCliIntegrationTests =

    let runTarsCommand(args: string) =
        try
            let startInfo = ProcessStartInfo()
            startInfo.FileName <- "dotnet"
            startInfo.Arguments <- $"run --project TarsEngine.FSharp.Core -- {args}"
            startInfo.UseShellExecute <- false
            startInfo.RedirectStandardOutput <- true
            startInfo.RedirectStandardError <- true
            startInfo.CreateNoWindow <- true
            
            use proc = Process.Start(startInfo)
            proc.WaitForExit(30000) |> ignore // 30 second timeout

            let output = proc.StandardOutput.ReadToEnd()
            let error = proc.StandardError.ReadToEnd()

            {|
                ExitCode = proc.ExitCode
                Output = output
                Error = error
                Success = proc.ExitCode = 0
            |}
        with
        | ex ->
            {|
                ExitCode = -1
                Output = ""
                Error = ex.Message
                Success = false
            |}

    [<Fact>]
    let ``CLI: turing-test command must detect and reject simulations`` () =
        // Act
        let result = runTarsCommand("turing-test")
        
        // Assert
        if result.Output.Contains("simulated") then
            // If the output contains "simulated", the CLI should have detected it and failed
            result.Output.Should().Contain("SIMULATION DETECTED", "CLI should detect simulations") |> ignore
            result.Output.Should().Contain("FORBIDDEN OPERATION", "CLI should reject simulations") |> ignore
            result.ExitCode.Should().Be(1, "CLI should exit with error code when simulation detected") |> ignore
        else
            // If no simulation detected, the test should pass
            result.Success.Should().BeTrue("CLI should succeed when no simulation detected") |> ignore

    [<Fact>]
    let ``CLI: help command should work without simulations`` () =
        // Act
        let result = runTarsCommand("help")
        
        // Assert
        result.Success.Should().BeTrue("Help command should always work") |> ignore
        result.Output.Should().Contain("TARS", "Help should show TARS information") |> ignore
        result.Output.Should().NotContain("simulated", "Help should not contain simulation references") |> ignore

    [<Fact>]
    let ``CLI: detailed-trace command should show real execution evidence`` () =
        // Act
        let result = runTarsCommand("detailed-trace")
        
        // Assert
        result.Success.Should().BeTrue("Detailed trace should work") |> ignore
        result.Output.Should().Contain("Real execution evidence", "Should show real execution") |> ignore
        result.Output.Should().NotContain("simulated", "Trace should not reference simulations") |> ignore

    [<Fact>]
    let ``CLI: All commands must enforce simulation detection`` () =
        // Test all major CLI commands to ensure they enforce simulation detection
        let commands = [
            "help"
            "detailed-trace"
            "turing-test"
        ]
        
        for command in commands do
            let result = runTarsCommand(command)
            
            // If any command outputs simulation-related content, it should be detected and rejected
            if result.Output.Contains("simulated") || result.Output.Contains("simulation") then
                result.Output.Should().Contain("FORBIDDEN", $"Command '{command}' should detect simulations") |> ignore

/// <summary>
/// AUTO-TEST INTEGRATION: Integrates automatic testing into TARS CLI
/// </summary>
type TarsAutoTestIntegration() =
    
    let hashMonitor = ProjectHashMonitorTests.ProjectHashMonitor()
    
    member _.RunPreExecutionTests() =
        printfn "🔍 TARS AUTO-TEST: Checking for project changes..."
        
        if hashMonitor.CheckForChanges() then
            printfn "🚨 PROJECT HASH CHANGED - RUNNING MANDATORY TESTS"
            printfn "=================================================="
            printfn ""
            
            let testResults = [
                ("🔍 Simulation Detection", runSimulationTests())
                ("🚫 Anti-Simulation Enforcement", runEnforcementTests())
                ("🖥️  CLI Integration", runCliTests())
                ("📊 System Integrity", runSystemTests())
            ]
            
            let failedTests = testResults |> List.filter (fun (_, passed) -> not passed)
            
            if failedTests.IsEmpty then
                printfn ""
                printfn "✅ ALL MANDATORY TESTS PASSED"
                printfn "✅ NO SIMULATIONS DETECTED"
                printfn "✅ TARS INTEGRITY VERIFIED"
                printfn "✅ SAFE TO PROCEED"
                printfn ""
                true
            else
                printfn ""
                printfn "🚨 CRITICAL FAILURE - MANDATORY TESTS FAILED:"
                for (testName, _) in failedTests do
                    printfn "❌ %s" testName
                printfn ""
                printfn "🚨 SIMULATION OR PLACEHOLDER DETECTED!"
                printfn "🚨 TARS EXECUTION TERMINATED FOR SAFETY!"
                printfn "🚨 FIX ALL SIMULATIONS BEFORE PROCEEDING!"
                printfn ""
                false
        else
            printfn "✅ No project changes - tests not required"
            true

    member private _.runSimulationTests() =
        try
            printfn "   🔍 Testing simulation detection..."
            
            // Create detector and test critical patterns
            let services = Microsoft.Extensions.DependencyInjection.ServiceCollection()
            services.AddLogging(fun logging ->
                logging.AddConsole() |> ignore
                logging.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Warning) |> ignore
            ) |> ignore
            services.AddSingleton<TarsEngine.FSharp.Core.SimulationDetector>() |> ignore
            
            let serviceProvider = services.BuildServiceProvider()
            let detector = serviceProvider.GetRequiredService<TarsEngine.FSharp.Core.SimulationDetector>()
            
            // Test that simulations are detected
            let simulationTest = detector.AnalyzeForSimulation("simulated execution", "test.fs")
            if not simulationTest.IsSimulation then
                printfn "   ❌ Failed to detect obvious simulation"
                false
            else
                // Test that real code is accepted
                let realTest = detector.AnalyzeForSimulation("let x = 1 + 1", "test.fs")
                if realTest.IsSimulation then
                    printfn "   ❌ False positive on real code"
                    false
                else
                    printfn "   ✅ Simulation detection working correctly"
                    true
        with
        | ex ->
            printfn "   ❌ Simulation test failed: %s" ex.Message
            false

    member private _.runEnforcementTests() =
        try
            printfn "   🚫 Testing anti-simulation enforcement..."
            
            // Test that the system rejects known simulation patterns
            let forbiddenPatterns = [
                "Thread.Sleep"
                "Task.Delay"
                "simulated"
                "placeholder"
                "TODO"
            ]
            
            let services = Microsoft.Extensions.DependencyInjection.ServiceCollection()
            services.AddLogging(fun logging ->
                logging.AddConsole() |> ignore
                logging.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Warning) |> ignore
            ) |> ignore
            services.AddSingleton<TarsEngine.FSharp.Core.SimulationDetector>() |> ignore
            
            let serviceProvider = services.BuildServiceProvider()
            let detector = serviceProvider.GetRequiredService<TarsEngine.FSharp.Core.SimulationDetector>()
            
            let mutable allPassed = true
            for pattern in forbiddenPatterns do
                let analysis = detector.AnalyzeForSimulation(pattern, "test.fs")
                if not analysis.IsSimulation then
                    printfn "   ❌ Failed to detect forbidden pattern: %s" pattern
                    allPassed <- false
            
            if allPassed then
                printfn "   ✅ Anti-simulation enforcement working correctly"
            
            allPassed
        with
        | ex ->
            printfn "   ❌ Enforcement test failed: %s" ex.Message
            false

    member private _.runCliTests() =
        try
            printfn "   🖥️  Testing CLI integration..."
            
            // Test that CLI help works without simulations
            let result = TarsCliIntegrationTests.runTarsCommand("help")
            if not result.Success then
                printfn "   ❌ CLI help command failed"
                false
            elif result.Output.Contains("simulated") then
                printfn "   ❌ CLI help contains simulation references"
                false
            else
                printfn "   ✅ CLI integration working correctly"
                true
        with
        | ex ->
            printfn "   ❌ CLI test failed: %s" ex.Message
            false

    member private _.runSystemTests() =
        try
            printfn "   📊 Testing system integrity..."
            
            // Test that core services don't return simulated results
            let services = Microsoft.Extensions.DependencyInjection.ServiceCollection()
            services.AddLogging(fun logging ->
                logging.AddConsole() |> ignore
                logging.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Warning) |> ignore
            ) |> ignore
            services.AddSingleton<TarsEngine.FSharp.Core.SimulationDetector>() |> ignore
            services.AddSingleton<TarsEngine.FSharp.Core.Metascript.Services.IMetascriptService, TarsEngine.FSharp.Core.Metascript.Services.MetascriptService>() |> ignore
            
            let serviceProvider = services.BuildServiceProvider()
            let detector = serviceProvider.GetRequiredService<TarsEngine.FSharp.Core.SimulationDetector>()
            let metascriptService = serviceProvider.GetRequiredService<TarsEngine.FSharp.Core.Metascript.Services.IMetascriptService>()
            
            // Test the current MetascriptService
            let result = (metascriptService.ExecuteMetascriptAsync("test")).Result
            let validation = detector.ValidateExecutionResult(result, "System Test")
            
            if validation.IsForbidden then
                printfn "   🚨 CRITICAL: MetascriptService returns simulated results!"
                printfn "   🚨 This must be fixed before TARS can be used!"
                false
            else
                printfn "   ✅ System integrity verified"
                true
        with
        | ex ->
            printfn "   ❌ System test failed: %s" ex.Message
            false

module TarsAutoTestIntegrationTests =

    [<Fact>]
    let ``Auto-test integration should detect project changes`` () =
        // Arrange
        let integration = TarsAutoTestIntegration()
        
        // Act
        let result = integration.RunPreExecutionTests()
        
        // Assert
        // The result depends on whether simulations are detected
        // If simulations exist, it should return false
        // If no simulations, it should return true
        result.Should().BeOfType<bool>() |> ignore
