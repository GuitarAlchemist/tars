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
                // REMOVED: Fake simulation delay; 1; 4; 1; 5; 9; 2; 6]
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
                // REMOVED: Fake simulation delay
