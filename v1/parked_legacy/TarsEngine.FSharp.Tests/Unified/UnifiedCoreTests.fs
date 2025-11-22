module TarsEngine.FSharp.Tests.Unified.UnifiedCoreTests

open System
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Core.UnifiedCore

/// Tests for the Unified Core system
[<TestClass>]
type UnifiedCoreTests() =
    
    [<Fact>]
    let ``TarsResult Success should contain value and metadata`` () =
        // Arrange
        let value = "test value"
        let metadata = Map [("key", box "value")]
        
        // Act
        let result = Success (value, metadata)
        
        // Assert
        match result with
        | Success (v, m) ->
            v |> should equal value
            m |> should equal metadata
        | Failure _ -> failwith "Expected Success"
    
    [<Fact>]
    let ``TarsResult Failure should contain error and correlation ID`` () =
        // Arrange
        let error = ValidationError ("Test error", Map.empty)
        let correlationId = "test-correlation-id"
        
        // Act
        let result = Failure (error, correlationId)
        
        // Assert
        match result with
        | Success _ -> failwith "Expected Failure"
        | Failure (e, c) ->
            e |> should equal error
            c |> should equal correlationId
    
    [<Fact>]
    let ``TarsError toString should format correctly`` () =
        // Arrange
        let error = ValidationError ("Test validation error", Map [("field", box "testField")])
        
        // Act
        let errorString = TarsError.toString error
        
        // Assert
        errorString |> should contain "ValidationError"
        errorString |> should contain "Test validation error"
    
    [<Fact>]
    let ``TarsError create should create error with message`` () =
        // Arrange
        let errorType = "TestError"
        let message = "Test error message"
        let exception = Some (Exception("Test exception"))
        
        // Act
        let error = TarsError.create errorType message exception
        
        // Assert
        match error with
        | ExecutionError (msg, ex) ->
            msg |> should contain message
            ex |> should equal exception
        | _ -> failwith "Expected ExecutionError"
    
    [<Fact>]
    let ``generateCorrelationId should generate unique IDs`` () =
        // Act
        let id1 = generateCorrelationId()
        let id2 = generateCorrelationId()
        
        // Assert
        id1 |> should not' (equal id2)
        id1.Length |> should be (greaterThan 0)
        id2.Length |> should be (greaterThan 0)
    
    [<Fact>]
    let ``createOperationContext should create valid context`` () =
        // Arrange
        let operation = "TestOperation"
        let correlationId = Some "test-correlation"
        let userId = Some "test-user"
        let sessionId = Some "test-session"
        
        // Act
        let context = createOperationContext operation correlationId userId sessionId
        
        // Assert
        context.Operation |> should equal operation
        context.CorrelationId |> should equal "test-correlation"
        context.UserId |> should equal (Some "test-user")
        context.SessionId |> should equal (Some "test-session")
        context.Timestamp |> should be (lessThanOrEqualTo DateTime.UtcNow)
    
    [<Fact>]
    let ``createOperationContext with None values should generate correlation ID`` () =
        // Arrange
        let operation = "TestOperation"
        
        // Act
        let context = createOperationContext operation None None None
        
        // Assert
        context.Operation |> should equal operation
        context.CorrelationId.Length |> should be (greaterThan 0)
        context.UserId |> should equal None
        context.SessionId |> should equal None
    
    [<Fact>]
    let ``ComponentStatus should have correct values`` () =
        // Assert
        ComponentStatus.Starting |> should not' (equal ComponentStatus.Running)
        ComponentStatus.Running |> should not' (equal ComponentStatus.Stopped)
        ComponentStatus.Stopped |> should not' (equal ComponentStatus.Error)
        ComponentStatus.Error |> should not' (equal ComponentStatus.Starting)
    
    [<Fact>]
    let ``TarsError should support all error types`` () =
        // Arrange & Act
        let validationError = ValidationError ("Validation failed", Map.empty)
        let configError = ConfigurationError ("Config error", Map.empty)
        let executionError = ExecutionError ("Execution failed", None)
        let networkError = NetworkError ("Network failed", Map.empty)
        let authError = AuthenticationError ("Auth failed", Map.empty)
        let authzError = AuthorizationError ("Authz failed", Map.empty)
        let notFoundError = NotFoundError ("Not found", Map.empty)
        let timeoutError = TimeoutError ("Timeout", Map.empty)
        
        // Assert
        TarsError.toString validationError |> should contain "ValidationError"
        TarsError.toString configError |> should contain "ConfigurationError"
        TarsError.toString executionError |> should contain "ExecutionError"
        TarsError.toString networkError |> should contain "NetworkError"
        TarsError.toString authError |> should contain "AuthenticationError"
        TarsError.toString authzError |> should contain "AuthorizationError"
        TarsError.toString notFoundError |> should contain "NotFoundError"
        TarsError.toString timeoutError |> should contain "TimeoutError"
    
    [<Fact>]
    let ``TarsResult should support map operations`` () =
        // Arrange
        let successResult = Success ("test", Map.empty)
        let failureResult = Failure (ValidationError ("error", Map.empty), "corr-id")
        
        // Act & Assert
        match successResult with
        | Success (value, metadata) ->
            value |> should equal "test"
            metadata |> should equal Map.empty
        | Failure _ -> failwith "Expected Success"
        
        match failureResult with
        | Success _ -> failwith "Expected Failure"
        | Failure (error, corrId) ->
            corrId |> should equal "corr-id"
    
    [<Fact>]
    let ``OperationContext should track timing`` () =
        // Arrange
        let startTime = DateTime.UtcNow
        
        // Act
        let context = createOperationContext "TimingTest" None None None
        
        // Assert
        context.Timestamp |> should be (greaterThanOrEqualTo startTime)
        context.Timestamp |> should be (lessThanOrEqualTo DateTime.UtcNow)
    
    [<Fact>]
    let ``TarsError metadata should be accessible`` () =
        // Arrange
        let metadata = Map [("field", box "value"); ("code", box 123)]
        let error = ValidationError ("Test error", metadata)
        
        // Act & Assert
        match error with
        | ValidationError (_, meta) ->
            meta.["field"] :?> string |> should equal "value"
            meta.["code"] :?> int |> should equal 123
        | _ -> failwith "Expected ValidationError"
    
    [<Fact>]
    let ``Multiple correlation IDs should be unique`` () =
        // Act
        let ids = [1..100] |> List.map (fun _ -> generateCorrelationId())
        
        // Assert
        let uniqueIds = ids |> List.distinct
        uniqueIds.Length |> should equal ids.Length
    
    [<Fact>]
    let ``OperationContext should support different operations`` () =
        // Arrange
        let operations = ["Create"; "Read"; "Update"; "Delete"; "Execute"; "Validate"]
        
        // Act
        let contexts = operations |> List.map (fun op -> createOperationContext op None None None)
        
        // Assert
        contexts |> List.length |> should equal operations.Length
        contexts |> List.map (fun c -> c.Operation) |> should equal operations
        
        // All should have unique correlation IDs
        let correlationIds = contexts |> List.map (fun c -> c.CorrelationId)
        let uniqueIds = correlationIds |> List.distinct
        uniqueIds.Length |> should equal correlationIds.Length
    
    [<Fact>]
    let ``TarsResult should handle complex metadata`` () =
        // Arrange
        let complexMetadata = Map [
            ("string", box "test")
            ("int", box 42)
            ("bool", box true)
            ("list", box [1; 2; 3])
            ("nested", box (Map [("inner", box "value")]))
        ]
        
        // Act
        let result = Success ("complex", complexMetadata)
        
        // Assert
        match result with
        | Success (value, metadata) ->
            value |> should equal "complex"
            metadata.["string"] :?> string |> should equal "test"
            metadata.["int"] :?> int |> should equal 42
            metadata.["bool"] :?> bool |> should equal true
        | Failure _ -> failwith "Expected Success"
    
    [<Fact>]
    let ``ComponentStatus should be comparable`` () =
        // Assert
        ComponentStatus.Starting |> should be (lessThan ComponentStatus.Running)
        ComponentStatus.Running |> should be (greaterThan ComponentStatus.Stopped)
        ComponentStatus.Error |> should not' (equal ComponentStatus.Running)
    
    [<Fact>]
    let ``TarsError should preserve exception information`` () =
        // Arrange
        let innerException = Exception("Inner exception")
        let outerException = Exception("Outer exception", innerException)
        let error = ExecutionError ("Execution failed", Some outerException)
        
        // Act & Assert
        match error with
        | ExecutionError (message, Some ex) ->
            message |> should equal "Execution failed"
            ex.Message |> should equal "Outer exception"
            ex.InnerException |> should not' (be null)
            ex.InnerException.Message |> should equal "Inner exception"
        | _ -> failwith "Expected ExecutionError with exception"
