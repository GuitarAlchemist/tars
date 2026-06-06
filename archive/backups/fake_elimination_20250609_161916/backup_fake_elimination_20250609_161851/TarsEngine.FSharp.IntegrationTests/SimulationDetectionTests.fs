namespace TarsEngine.FSharp.IntegrationTests

open System
open System.IO
open Xunit
open FluentAssertions
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core

/// <summary>
/// CRITICAL INTEGRATION TESTS: Simulation Detection System
/// These tests MUST PASS to ensure TARS never accepts simulated results
/// </summary>
module SimulationDetectionTests =

    let createSimulationDetector() =
        let services = ServiceCollection()
        services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
            logging.SetMinimumLevel(LogLevel.Information) |> ignore
        ) |> ignore
        services.AddSingleton<SimulationDetector>() |> ignore
        
        let serviceProvider = services.BuildServiceProvider()
        serviceProvider.GetRequiredService<SimulationDetector>()

    [<Fact>]
    let ``Should REJECT simulated keyword in code`` () =
        // Arrange
        let detector = createSimulationDetector()
        let simulatedCode = """
            let result = "This is simulated execution"
            printfn "%s" result
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(simulatedCode, "test.fs")
        
        // Assert
        analysis.IsSimulation.Should().BeTrue("Code contains 'simulated' keyword") |> ignore
        analysis.DetectedKeywords.Should().Contain("simulated") |> ignore
        analysis.ConfidenceScore.Should().BeGreaterThan(0.5) |> ignore
        analysis.Verdict.Should().Be("FORBIDDEN - SIMULATION DETECTED") |> ignore

    [<Fact>]
    let ``Should REJECT placeholder implementations`` () =
        // Arrange
        let detector = createSimulationDetector()
        let placeholderCode = """
            // TODO: Implement real logic
            let executeTask() = 
                // This is a placeholder implementation
                "placeholder result"
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(placeholderCode, "test.fs")
        
        // Assert
        analysis.IsSimulation.Should().BeTrue("Code contains placeholder patterns") |> ignore
        analysis.DetectedKeywords.Should().Contain("placeholder") |> ignore
        analysis.DetectedKeywords.Should().Contain("TODO") |> ignore

    [<Fact>]
    let ``Should REJECT mock implementations`` () =
        // Arrange
        let detector = createSimulationDetector()
        let mockCode = """
            let mockExecution() =
                Thread.Sleep(1000) // Fake processing time
                "mock result for testing"
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(mockCode, "test.fs")
        
        // Assert
        analysis.IsSimulation.Should().BeTrue("Code contains mock patterns") |> ignore
        analysis.DetectedKeywords.Should().Contain("mock") |> ignore

    [<Fact>]
    let ``Should REJECT Task.Delay simulation patterns`` () =
        // Arrange
        let detector = createSimulationDetector()
        let delayCode = """
            let simulateWork() = async {
                do! Task.Delay(2000) |> Async.AwaitTask
                return "simulated work complete"
            }
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(delayCode, "test.fs")
        
        // Assert
        analysis.IsSimulation.Should().BeTrue("Code uses Task.Delay for simulation") |> ignore
        analysis.DetectedPatterns.Should().NotBeEmpty() |> ignore

    [<Fact>]
    let ``Should ACCEPT real F# implementation`` () =
        // Arrange
        let detector = createSimulationDetector()
        let realCode = """
            let fibonacci n =
                let rec fib a b count =
                    if count = 0 then a
                    else fib b (a + b) (count - 1)
                fib 0 1 n
            
            let result = fibonacci 10
            printfn "Fibonacci(10) = %d" result
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(realCode, "test.fs")
        
        // Assert
        analysis.IsSimulation.Should().BeFalse("Real F# code should be accepted") |> ignore
        analysis.Verdict.Should().Be("APPROVED - REAL EXECUTION") |> ignore

    [<Fact>]
    let ``Should REJECT execution result with simulation`` () =
        // Arrange
        let detector = createSimulationDetector()
        let simulatedResult = "Metascript executed successfully (simulated)"
        
        // Act
        let validation = detector.ValidateExecutionResult(simulatedResult, "Test Execution")
        
        // Assert
        validation.IsValid.Should().BeFalse("Simulated results must be rejected") |> ignore
        validation.IsForbidden.Should().BeTrue("Simulation should be forbidden") |> ignore
        validation.Reason.Should().Be("SIMULATION DETECTED - FORBIDDEN OPERATION") |> ignore
        validation.Action.Should().Be("TERMINATE_EXECUTION") |> ignore

    [<Fact>]
    let ``Should ACCEPT real execution result`` () =
        // Arrange
        let detector = createSimulationDetector()
        let realResult = "Tower of Hanoi solved in 15 moves. Fibonacci sequence: 0,1,1,2,3,5,8,13,21,34"
        
        // Act
        let validation = detector.ValidateExecutionResult(realResult, "Real Execution")
        
        // Assert
        validation.IsValid.Should().BeTrue("Real results should be accepted") |> ignore
        validation.IsForbidden.Should().BeFalse("Real execution should not be forbidden") |> ignore
        validation.Reason.Should().Be("Real execution confirmed") |> ignore

    [<Fact>]
    let ``Should detect semantic simulation patterns`` () =
        // Arrange
        let detector = createSimulationDetector()
        let semanticSimulation = """
            // This would normally connect to a real database
            let fetchData() =
                // For demonstration purposes, return hardcoded data
                ["sample"; "data"; "for"; "testing"]
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(semanticSimulation, "test.fs")
        
        // Assert
        analysis.IsSimulation.Should().BeTrue("Semantic simulation should be detected") |> ignore
        analysis.SemanticScore.Should().BeGreaterThan(0.5) |> ignore

    [<Fact>]
    let ``Should have high confidence for obvious simulations`` () =
        // Arrange
        let detector = createSimulationDetector()
        let obviousSimulation = """
            // TODO: This is just a placeholder
            let simulateExecution() =
                Thread.Sleep(1000) // Fake processing
                "simulated result for demo purposes"
        """
        
        // Act
        let analysis = detector.AnalyzeForSimulation(obviousSimulation, "test.fs")
        
        // Assert
        analysis.IsSimulation.Should().BeTrue() |> ignore
        analysis.ConfidenceScore.Should().BeGreaterThan(0.8, "Obvious simulation should have high confidence") |> ignore
        analysis.DetectedKeywords.Should().HaveCountGreaterThan(2) |> ignore
