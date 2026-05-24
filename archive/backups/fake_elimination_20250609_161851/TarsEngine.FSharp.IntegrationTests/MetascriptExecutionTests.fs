namespace TarsEngine.FSharp.IntegrationTests

open System
open System.IO
open Xunit
open FluentAssertions
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Metascript.Services
open TarsEngine.FSharp.Core

/// <summary>
/// CRITICAL METASCRIPT EXECUTION TESTS
/// Ensures metascripts execute REAL F# code, not simulations
/// </summary>
module MetascriptExecutionTests =

    let createMetascriptServices() =
        let services = ServiceCollection()
        services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
            logging.SetMinimumLevel(LogLevel.Information) |> ignore
        ) |> ignore
        services.AddSingleton<IMetascriptService, MetascriptService>() |> ignore
        services.AddSingleton<SimulationDetector>() |> ignore
        services.BuildServiceProvider()

    [<Fact>]
    let ``CRITICAL: Metascript with real F# code should execute and return real results`` () =
        // Arrange
        let services = createMetascriptServices()
        let metascriptService = services.GetRequiredService<IMetascriptService>()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let realMetascript = """
DESCRIBE {
    name: "Real F# Execution Test"
    version: "1.0.0"
    author: "TARS Integration Test"
}

FSHARP {
    // Real mathematical calculation
    let fibonacci n =
        let rec fib a b count =
            if count = 0 then a
            else fib b (a + b) (count - 1)
        fib 0 1 n
    
    let result = fibonacci 10
    printfn "Fibonacci(10) = %d" result
    
    // Real data processing
    let numbers = [1; 2; 3; 4; 5]
    let doubled = numbers |> List.map (fun x -> x * 2)
    printfn "Doubled: %A" doubled
    
    sprintf "Real execution completed. Fibonacci(10)=%d, Doubled=%A" result doubled
}
"""
        
        // Act
        let result = (metascriptService.ExecuteMetascriptAsync(realMetascript)).Result
        
        // Assert
        let validation = detector.ValidateExecutionResult(result, "Real F# Test")
        validation.IsValid.Should().BeTrue("Real F# code should be accepted") |> ignore
        validation.IsForbidden.Should().BeFalse("Real execution should not be forbidden") |> ignore

    [<Fact>]
    let ``CRITICAL: Metascript with simulated code should be REJECTED`` () =
        // Arrange
        let services = createMetascriptServices()
        let metascriptService = services.GetRequiredService<IMetascriptService>()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let simulatedMetascript = """
DESCRIBE {
    name: "Simulated Execution Test"
    version: "1.0.0"
}

FSHARP {
    // This is a simulated implementation
    let simulateWork() =
        // REMOVED: Fake simulation delay; 34; 25; 12; 22; 11; 90]
    
    sprintf "Primes: %A, Sorted: %A" primes sortedNumbers
}
"""
        
        // Act
        let analysis = detector.AnalyzeForSimulation(complexMetascript, "complex.trsx")
        
        // Assert
        analysis.IsSimulation.Should().BeFalse("Real complex algorithms should be accepted") |> ignore
        analysis.Verdict.Should().Be("APPROVED - REAL EXECUTION") |> ignore

    [<Fact>]
    let ``CRITICAL: Metascript should handle real file operations`` () =
        // Arrange
        let services = createMetascriptServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let fileMetascript = """
FSHARP {
    // Real file operations (safe for testing)
    let testData = "Real test data from TARS"
    let tempFile = Path.GetTempFileName()
    
    // Write real data
    File.WriteAllText(tempFile, testData)
    
    // Read real data
    let readData = File.ReadAllText(tempFile)
    
    // Clean up
    File.Delete(tempFile)
    
    sprintf "File operation completed. Data: %s" readData
}
"""
        
        // Act
        let analysis = detector.AnalyzeForSimulation(fileMetascript, "file.trsx")
        
        // Assert
        analysis.IsSimulation.Should().BeFalse("Real file operations should be accepted") |> ignore

    [<Fact>]
    let ``CRITICAL: Metascript should reject hardcoded demo data`` () =
        // Arrange
        let services = createMetascriptServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let demoMetascript = """
FSHARP {
    // This is just demo data for testing purposes
    let demoData = ["sample", "demo", "placeholder", "data"]
    
    // For demonstration purposes, return hardcoded result
    "demo result with sample data"
}
"""
        
        // Act
        let analysis = detector.AnalyzeForSimulation(demoMetascript, "demo.trsx")
        
        // Assert
        analysis.IsSimulation.Should().BeTrue("Demo/sample data should be detected as simulation") |> ignore
        analysis.DetectedKeywords.Should().Contain("demo") |> ignore
        analysis.DetectedKeywords.Should().Contain("sample") |> ignore
        analysis.DetectedKeywords.Should().Contain("placeholder") |> ignore

