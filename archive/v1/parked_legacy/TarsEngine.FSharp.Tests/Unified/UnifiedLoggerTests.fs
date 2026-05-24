module TarsEngine.FSharp.Tests.Unified.UnifiedLoggerTests

open System
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger

/// Tests for the Unified Logger system
[<TestClass>]
type UnifiedLoggerTests() =
    
    [<Fact>]
    let ``Logger should create with valid name`` () =
        // Act
        let logger = createLogger "TestLogger"
        
        // Assert
        logger |> should not' (be null)
    
    [<Fact>]
    let ``Logger should handle correlation IDs`` () =
        // Arrange
        let logger = createLogger "TestLogger"
        let correlationId = generateCorrelationId()
        
        // Act & Assert - Should not throw
        logger.LogInformation(correlationId, "Test message")
        logger.LogWarning(correlationId, "Test warning")
        logger.LogError(correlationId, TarsError.create "TestError" "Test error" None, Exception("Test"))
    
    [<Fact>]
    let ``Logger should handle different log levels`` () =
        // Arrange
        let logger = createLogger "TestLogger"
        let correlationId = generateCorrelationId()
        
        // Act & Assert - Should not throw
        logger.LogDebug(correlationId, "Debug message")
        logger.LogInformation(correlationId, "Info message")
        logger.LogWarning(correlationId, "Warning message")
        logger.LogError(correlationId, TarsError.create "TestError" "Error message" None, Exception("Test"))
    
    [<Fact>]
    let ``Logger should handle null and empty messages`` () =
        // Arrange
        let logger = createLogger "TestLogger"
        let correlationId = generateCorrelationId()
        
        // Act & Assert - Should not throw
        logger.LogInformation(correlationId, "")
        logger.LogInformation(correlationId, null)
    
    [<Fact>]
    let ``Logger should handle concurrent logging`` () =
        // Arrange
        let logger = createLogger "TestLogger"
        
        // Act
        let tasks = [
            for i in 1..10 do
                yield async {
                    let correlationId = generateCorrelationId()
                    logger.LogInformation(correlationId, $"Concurrent message {i}")
                }
        ]
        
        // Assert - Should not throw
        tasks |> Async.Parallel |> Async.RunSynchronously |> ignore
