namespace TarsEngine.FSharp.IntegrationTests

open System
open System.IO
open System.Diagnostics
open Xunit
open FluentAssertions
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core

/// <summary>
/// CRITICAL ENFORCEMENT TESTS: Anti-Simulation System
/// These tests ensure TARS NEVER tolerates simulations or placeholders
/// ZERO TOLERANCE POLICY - Any simulation = IMMEDIATE FAILURE
/// </summary>
module AntiSimulationEnforcementTests =

    let createTestServices() =
        let services = ServiceCollection()
        services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
            logging.SetMinimumLevel(LogLevel.Information) |> ignore
        ) |> ignore
        services.AddSingleton<SimulationDetector>() |> ignore
        services.BuildServiceProvider()

    [<Fact>]
    let ``CRITICAL: Must REJECT any code containing Thread.Sleep`` () =
        // Arrange
        let services = createTestServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let sleepCode = """
            let simulateWork() =
                Thread.Sleep(1000) // This is simulation!
                "work complete"
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(sleepCode, "test.fs")
        
        // Assert - ZERO TOLERANCE
        analysis.IsSimulation.Should().BeTrue("Thread.Sleep is FORBIDDEN simulation") |> ignore
        analysis.ConfidenceScore.Should().BeGreaterThan(0.8) |> ignore
        analysis.Verdict.Should().Be("FORBIDDEN - SIMULATION DETECTED") |> ignore

    [<Fact>]
    let ``CRITICAL: Must REJECT any code containing Task.Delay`` () =
        // Arrange
        let services = createTestServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let delayCode = """
            let asyncWork() = async {
                do! Task.Delay(2000) |> Async.AwaitTask // FORBIDDEN!
                return "fake async work"
            }
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(delayCode, "test.fs")
        
        // Assert - ZERO TOLERANCE
        analysis.IsSimulation.Should().BeTrue("Task.Delay is FORBIDDEN simulation") |> ignore
        analysis.DetectedPatterns.Should().NotBeEmpty() |> ignore

    [<Fact>]
    let ``CRITICAL: Must REJECT any return statement with 'simulated'`` () =
        // Arrange
        let services = createTestServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let simulatedReturn = """
            let executeTask() =
                // Some processing...
                "simulated execution result"
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(simulatedReturn, "test.fs")
        
        // Assert - ZERO TOLERANCE
        analysis.IsSimulation.Should().BeTrue("Simulated return values are FORBIDDEN") |> ignore
        analysis.DetectedKeywords.Should().Contain("simulated") |> ignore

    [<Fact>]
    let ``CRITICAL: Must REJECT any TODO or FIXME comments`` () =
        // Arrange
        let services = createTestServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let todoCode = """
            let processData() =
                // TODO: Implement real data processing
                // FIXME: This is just a placeholder
                "temporary result"
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(todoCode, "test.fs")
        
        // Assert - ZERO TOLERANCE
        analysis.IsSimulation.Should().BeTrue("TODO/FIXME indicates incomplete implementation") |> ignore
        analysis.DetectedKeywords.Should().Contain("TODO") |> ignore
        analysis.DetectedKeywords.Should().Contain("FIXME") |> ignore

    [<Fact>]
    let ``CRITICAL: Must REJECT placeholder implementations`` () =
        // Arrange
        let services = createTestServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let placeholderCode = """
            let calculateResult() =
                // This is just a placeholder implementation
                // In a real implementation, this would do actual work
                42 // placeholder value
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(placeholderCode, "test.fs")
        
        // Assert - ZERO TOLERANCE
        analysis.IsSimulation.Should().BeTrue("Placeholder implementations are FORBIDDEN") |> ignore
        analysis.DetectedKeywords.Should().Contain("placeholder") |> ignore

    [<Fact>]
    let ``CRITICAL: Must REJECT mock or fake implementations`` () =
        // Arrange
        let services = createTestServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let mockCode = """
            let mockDatabaseCall() =
                // This is a mock implementation for testing
                ["fake", "data", "from", "mock", "database"]
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(mockCode, "test.fs")
        
        // Assert - ZERO TOLERANCE
        analysis.IsSimulation.Should().BeTrue("Mock implementations are FORBIDDEN") |> ignore
        analysis.DetectedKeywords.Should().Contain("mock") |> ignore
        analysis.DetectedKeywords.Should().Contain("fake") |> ignore

    [<Fact>]
    let ``CRITICAL: Must ACCEPT only real F# implementations`` () =
        // Arrange
        let services = createTestServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let realCode = """
            let quickSort arr =
                let rec qsort = function
                    | [] -> []
                    | pivot::tail ->
                        let smaller = tail |> List.filter (fun x -> x <= pivot)
                        let larger = tail |> List.filter (fun x -> x > pivot)
                        qsort smaller @ [pivot] @ qsort larger
                qsort arr
            
            let sortedNumbers = quickSort [3; 1; 4; 1; 5; 9; 2; 6]
            printfn "Sorted: %A" sortedNumbers
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(realCode, "test.fs")
        
        // Assert - ONLY REAL CODE ACCEPTED
        analysis.IsSimulation.Should().BeFalse("Real F# implementations should be accepted") |> ignore
        analysis.Verdict.Should().Be("APPROVED - REAL EXECUTION") |> ignore

    [<Fact>]
    let ``CRITICAL: Must TERMINATE execution on simulation detection`` () =
        // Arrange
        let services = createTestServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let simulatedResult = "Metascript executed successfully (simulated)"
        
        // Act
        let validation = detector.ValidateExecutionResult(simulatedResult, "Test Context")
        
        // Assert - IMMEDIATE TERMINATION
        validation.IsValid.Should().BeFalse("Simulated results must be rejected") |> ignore
        validation.IsForbidden.Should().BeTrue("Simulation must be forbidden") |> ignore
        validation.Action.Should().Be("TERMINATE_EXECUTION") |> ignore
        validation.Reason.Should().Be("SIMULATION DETECTED - FORBIDDEN OPERATION") |> ignore

    [<Fact>]
    let ``CRITICAL: Must detect semantic simulation phrases`` () =
        // Arrange
        let services = createTestServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let semanticSimulation = """
            let processRequest() =
                // For demonstration purposes, we'll return a hardcoded response
                // In a real implementation, this would connect to the actual service
                "demo response data"
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(semanticSimulation, "test.fs")
        
        // Assert - SEMANTIC DETECTION
        analysis.IsSimulation.Should().BeTrue("Semantic simulation patterns must be detected") |> ignore
        analysis.SemanticScore.Should().BeGreaterThan(0.5) |> ignore

    [<Fact>]
    let ``CRITICAL: Must have maximum confidence for obvious simulations`` () =
        // Arrange
        let services = createTestServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let obviousSimulation = """
            // TODO: This is just a placeholder for demo purposes
            let simulateComplexCalculation() =
                Thread.Sleep(1000) // Simulate processing time
                "simulated calculation result for demonstration"
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(obviousSimulation, "test.fs")
        
        // Assert - MAXIMUM CONFIDENCE
        analysis.IsSimulation.Should().BeTrue() |> ignore
        analysis.ConfidenceScore.Should().BeGreaterThan(0.9, "Obvious simulations should have maximum confidence") |> ignore
        analysis.DetectedKeywords.Should().HaveCountGreaterThan(3) |> ignore
        analysis.DetectedPatterns.Should().NotBeEmpty() |> ignore

/// <summary>
/// ENFORCEMENT INTEGRATION: Tests that verify the entire system rejects simulations
/// </summary>
module SystemWideEnforcementTests =

    [<Fact>]
    let ``SYSTEM: Must reject simulated MetascriptService results`` () =
        // This test verifies that the current MetascriptService that returns
        // "Metascript executed successfully (simulated)" is properly detected and rejected
        
        // Arrange
        let services = ServiceCollection()
        services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
            logging.SetMinimumLevel(LogLevel.Information) |> ignore
        ) |> ignore
        services.AddSingleton<SimulationDetector>() |> ignore
        services.AddSingleton<TarsEngine.FSharp.Core.Metascript.Services.IMetascriptService, TarsEngine.FSharp.Core.Metascript.Services.MetascriptService>() |> ignore
        
        let serviceProvider = services.BuildServiceProvider()
        let detector = serviceProvider.GetRequiredService<SimulationDetector>()
        let metascriptService = serviceProvider.GetRequiredService<TarsEngine.FSharp.Core.Metascript.Services.IMetascriptService>()
        
        // Act
        let result = (metascriptService.ExecuteMetascriptAsync("test content")).Result
        let validation = detector.ValidateExecutionResult(result, "System Test")
        
        // Assert - SYSTEM MUST REJECT SIMULATED SERVICE
        validation.IsForbidden.Should().BeTrue("System must reject simulated MetascriptService results") |> ignore
        validation.Action.Should().Be("TERMINATE_EXECUTION") |> ignore

    [<Fact>]
    let ``SYSTEM: Must enforce zero tolerance across all components`` () =
        // This test ensures that ALL components of TARS enforce zero simulation tolerance
        
        // Arrange
        let services = ServiceCollection()
        services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
            logging.SetMinimumLevel(LogLevel.Information) |> ignore
        ) |> ignore
        services.AddSingleton<SimulationDetector>() |> ignore
        
        let serviceProvider = services.BuildServiceProvider()
        let detector = serviceProvider.GetRequiredService<SimulationDetector>()
        
        // Test various simulation patterns that might slip through
        let forbiddenPatterns = [
            "simulated execution"
            "placeholder result"
            "mock implementation"
            "TODO: implement"
            "FIXME: broken"
            "for demonstration purposes"
            "hardcoded response"
            "fake data"
            "dummy implementation"
            "not implemented yet"
        ]
        
        // Act & Assert
        for pattern in forbiddenPatterns do
            let analysis = detector.AnalyzeForSimulation(pattern, "test.fs")
            analysis.IsSimulation.Should().BeTrue($"Pattern '{pattern}' must be detected as simulation") |> ignore
