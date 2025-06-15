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
        Thread.Sleep(1000) // Simulate processing time
        "simulated work result"
    
    let result = simulateWork()
    printfn "Simulated result: %s" result
    result
}
"""
        
        // Act
        let preAnalysis = detector.AnalyzeForSimulation(simulatedMetascript, "test.trsx")
        
        // Assert - Should be detected BEFORE execution
        preAnalysis.IsSimulation.Should().BeTrue("Simulated metascript should be detected") |> ignore
        preAnalysis.DetectedKeywords.Should().Contain("simulated") |> ignore
        preAnalysis.Verdict.Should().Be("FORBIDDEN - SIMULATION DETECTED") |> ignore

    [<Fact>]
    let ``CRITICAL: Metascript with placeholder code should be REJECTED`` () =
        // Arrange
        let services = createMetascriptServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let placeholderMetascript = """
DESCRIBE {
    name: "Placeholder Test"
    version: "1.0.0"
}

FSHARP {
    // TODO: Implement real algorithm
    let processData() =
        // This is just a placeholder implementation
        "placeholder result"
    
    let result = processData()
    result
}
"""
        
        // Act
        let analysis = detector.AnalyzeForSimulation(placeholderMetascript, "test.trsx")
        
        // Assert
        analysis.IsSimulation.Should().BeTrue("Placeholder metascript should be detected") |> ignore
        analysis.DetectedKeywords.Should().Contain("placeholder") |> ignore
        analysis.DetectedKeywords.Should().Contain("TODO") |> ignore

    [<Fact>]
    let ``CRITICAL: Current MetascriptService must be flagged for returning simulated results`` () =
        // This test specifically targets the current MetascriptService implementation
        // that returns "Metascript executed successfully (simulated)"
        
        // Arrange
        let services = createMetascriptServices()
        let metascriptService = services.GetRequiredService<IMetascriptService>()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        // Act
        let result = (metascriptService.ExecuteMetascriptAsync("any content")).Result
        let validation = detector.ValidateExecutionResult(result, "Current Service Test")
        
        // Assert - Current service MUST be flagged as simulation
        validation.IsForbidden.Should().BeTrue("Current MetascriptService returns simulated results") |> ignore
        validation.Reason.Should().Be("SIMULATION DETECTED - FORBIDDEN OPERATION") |> ignore
        validation.Action.Should().Be("TERMINATE_EXECUTION") |> ignore

    [<Fact>]
    let ``CRITICAL: Metascript execution should never use Thread.Sleep`` () =
        // Arrange
        let services = createMetascriptServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let sleepMetascript = """
FSHARP {
    let slowFunction() =
        Thread.Sleep(2000) // This is forbidden!
        "slow result"
    
    slowFunction()
}
"""
        
        // Act
        let analysis = detector.AnalyzeForSimulation(sleepMetascript, "test.trsx")
        
        // Assert
        analysis.IsSimulation.Should().BeTrue("Thread.Sleep should be detected as simulation") |> ignore
        analysis.DetectedPatterns.Should().NotBeEmpty() |> ignore

    [<Fact>]
    let ``CRITICAL: Metascript execution should never use Task.Delay`` () =
        // Arrange
        let services = createMetascriptServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let delayMetascript = """
FSHARP {
    let asyncWork() = async {
        do! Task.Delay(1000) |> Async.AwaitTask // Forbidden!
        return "delayed result"
    }
    
    asyncWork() |> Async.RunSynchronously
}
"""
        
        // Act
        let analysis = detector.AnalyzeForSimulation(delayMetascript, "test.trsx")
        
        // Assert
        analysis.IsSimulation.Should().BeTrue("Task.Delay should be detected as simulation") |> ignore

    [<Fact>]
    let ``CRITICAL: Real metascript should execute complex algorithms`` () =
        // Arrange
        let services = createMetascriptServices()
        let detector = services.GetRequiredService<SimulationDetector>()
        
        let complexMetascript = """
DESCRIBE {
    name: "Complex Algorithm Test"
    version: "1.0.0"
}

FSHARP {
    // Real quicksort implementation
    let rec quicksort = function
        | [] -> []
        | pivot::tail ->
            let smaller = tail |> List.filter (fun x -> x <= pivot)
            let larger = tail |> List.filter (fun x -> x > pivot)
            quicksort smaller @ [pivot] @ quicksort larger
    
    // Real prime number calculation
    let isPrime n =
        if n < 2 then false
        else
            let limit = int (sqrt (float n))
            [2..limit] |> List.forall (fun x -> n % x <> 0)
    
    let primes = [2..100] |> List.filter isPrime
    let sortedNumbers = quicksort [64; 34; 25; 12; 22; 11; 90]
    
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
