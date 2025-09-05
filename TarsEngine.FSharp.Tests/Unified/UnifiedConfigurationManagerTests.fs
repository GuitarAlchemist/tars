module TarsEngine.FSharp.Tests.Unified.UnifiedConfigurationManagerTests

open System
open System.Threading
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager

/// Tests for the Unified Configuration Manager
[<TestClass>]
type UnifiedConfigurationManagerTests() =
    
    let createTestLogger() = createLogger "UnifiedConfigurationManagerTests"
    
    [<Fact>]
    let ``Configuration manager should initialize successfully`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            
            // Act
            let! result = configManager.InitializeAsync(CancellationToken.None)
            
            // Assert
            match result with
            | Success (_, metadata) ->
                metadata.ContainsKey("environment") |> should be True
                metadata.ContainsKey("configCount") |> should be True
            | Failure (error, _) -> failwith $"Initialization failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should set and get string configuration values`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act
            let! setResult = configManager.SetValueAsync("test.string", "test value", None)
            let retrievedValue = ConfigurationExtensions.getString configManager "test.string" "default"
            
            // Assert
            match setResult with
            | Success _ -> 
                retrievedValue |> should equal "test value"
            | Failure (error, _) -> failwith $"Set failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should set and get integer configuration values`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act
            let! setResult = configManager.SetValueAsync("test.int", 42, None)
            let retrievedValue = ConfigurationExtensions.getInt configManager "test.int" 0
            
            // Assert
            match setResult with
            | Success _ -> 
                retrievedValue |> should equal 42
            | Failure (error, _) -> failwith $"Set failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should set and get boolean configuration values`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act
            let! setResult = configManager.SetValueAsync("test.bool", true, None)
            let retrievedValue = ConfigurationExtensions.getBool configManager "test.bool" false
            
            // Assert
            match setResult with
            | Success _ -> 
                retrievedValue |> should equal true
            | Failure (error, _) -> failwith $"Set failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should set and get float configuration values`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act
            let! setResult = configManager.SetValueAsync("test.float", 3.14159, None)
            let retrievedValue = ConfigurationExtensions.getFloat configManager "test.float" 0.0
            
            // Assert
            match setResult with
            | Success _ -> 
                Math.Abs(retrievedValue - 3.14159) |> should be (lessThan 0.00001)
            | Failure (error, _) -> failwith $"Set failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should return default value for non-existent configuration`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act
            let stringValue = ConfigurationExtensions.getString configManager "non.existent.string" "default"
            let intValue = ConfigurationExtensions.getInt configManager "non.existent.int" 999
            let boolValue = ConfigurationExtensions.getBool configManager "non.existent.bool" true
            let floatValue = ConfigurationExtensions.getFloat configManager "non.existent.float" 1.23
            
            // Assert
            stringValue |> should equal "default"
            intValue |> should equal 999
            boolValue |> should equal true
            Math.Abs(floatValue - 1.23) |> should be (lessThan 0.00001)
        }
    
    [<Fact>]
    let ``Should validate configuration against schema`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act - Try to set invalid log level
            let! invalidResult = configManager.SetValueAsync("tars.core.logLevel", "InvalidLevel", None)
            let! validResult = configManager.SetValueAsync("tars.core.logLevel", "Debug", None)
            
            // Assert
            match invalidResult with
            | Success _ -> failwith "Should have failed validation"
            | Failure (ValidationError _, _) -> () // Expected
            | Failure (error, _) -> failwith $"Unexpected error: {TarsError.toString error}"
            
            match validResult with
            | Success _ -> () // Expected
            | Failure (error, _) -> failwith $"Valid value should have succeeded: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should notify subscribers of configuration changes`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            let mutable changeCount = 0
            let mutable lastChangeEvent = None
            
            let callback (changeEvent: ConfigChangeEvent) =
                changeCount <- changeCount + 1
                lastChangeEvent <- Some changeEvent
            
            configManager.SubscribeToChanges(callback)
            
            // Act
            let! _ = configManager.SetValueAsync("test.notification", "value1", None)
            let! _ = configManager.SetValueAsync("test.notification", "value2", None)
            
            // Wait a bit for notifications to process
            do! System.Threading.Tasks.Task.Delay(100)
            
            // Assert
            changeCount |> should equal 2
            lastChangeEvent |> should not' (be None)
            
            match lastChangeEvent with
            | Some event ->
                event.Key |> should equal "test.notification"
                event.NewValue |> should not' (be None)
            | None -> failwith "Expected change event"
        }
    
    [<Fact>]
    let ``Should get configuration values by category`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act
            let coreValues = configManager.GetValuesByCategory("Core")
            let agentValues = configManager.GetValuesByCategory("Agents")
            let cudaValues = configManager.GetValuesByCategory("CUDA")
            
            // Assert
            coreValues.Count |> should be (greaterThan 0)
            agentValues.Count |> should be (greaterThan 0)
            cudaValues.Count |> should be (greaterThan 0)
            
            // Check that core values contain expected keys
            coreValues |> Map.exists (fun key _ -> key.Contains("tars.core")) |> should be True
        }
    
    [<Fact>]
    let ``Should create and retrieve configuration snapshots`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Set some test values
            let! _ = configManager.SetValueAsync("snapshot.test1", "value1", None)
            let! _ = configManager.SetValueAsync("snapshot.test2", 42, None)
            
            // Act
            let! snapshotResult = configManager.CreateSnapshotAsync("test_snapshot", CancellationToken.None)
            
            // Assert
            match snapshotResult with
            | Success (snapshot, metadata) ->
                snapshot.Configuration.Count |> should be (greaterThan 0)
                snapshot.Configuration.ContainsKey("snapshot.test1") |> should be True
                snapshot.Configuration.ContainsKey("snapshot.test2") |> should be True
                metadata.ContainsKey("snapshotId") |> should be True
            | Failure (error, _) -> failwith $"Snapshot creation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should save and load configuration`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Set test values
            let! _ = configManager.SetValueAsync("save.test1", "saved_value", None)
            let! _ = configManager.SetValueAsync("save.test2", 123, None)
            
            // Act
            let! saveResult = configManager.SaveConfigurationAsync(CancellationToken.None)
            
            // Assert
            match saveResult with
            | Success (_, metadata) ->
                let configCount = metadata.["configCount"] :?> int
                configCount |> should be (greaterThan 0)
            | Failure (error, _) -> failwith $"Save failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should provide configuration statistics`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act
            let statistics = configManager.GetStatistics()
            
            // Assert
            statistics.ContainsKey("totalConfigurations") |> should be True
            statistics.ContainsKey("totalSchemas") |> should be True
            statistics.ContainsKey("totalEnvironments") |> should be True
            statistics.ContainsKey("currentEnvironment") |> should be True
            statistics.ContainsKey("isInitialized") |> should be True
            
            let isInitialized = statistics.["isInitialized"] :?> bool
            isInitialized |> should be True
            
            let currentEnv = statistics.["currentEnvironment"] :?> string
            currentEnv |> should equal "default"
        }
    
    [<Fact>]
    let ``Should handle concurrent configuration access`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act - Perform concurrent operations
            let tasks = [
                for i in 1..10 do
                    yield configManager.SetValueAsync($"concurrent.test{i}", $"value{i}", None)
            ]
            
            let! results = System.Threading.Tasks.Task.WhenAll(tasks)
            
            // Assert
            results |> Array.forall (function | Success _ -> true | Failure _ -> false) |> should be True
            
            // Verify all values were set
            for i in 1..10 do
                let value = ConfigurationExtensions.getString configManager $"concurrent.test{i}" "default"
                value |> should equal $"value{i}"
        }
    
    [<Fact>]
    let ``Should handle configuration value type conversions`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act - Set values of different types
            let! _ = configManager.SetValueAsync("type.string", "test", None)
            let! _ = configManager.SetValueAsync("type.int", 42, None)
            let! _ = configManager.SetValueAsync("type.bool", true, None)
            let! _ = configManager.SetValueAsync("type.float", 3.14, None)
            
            // Assert - Retrieve with correct types
            let stringVal = ConfigurationExtensions.getString configManager "type.string" "default"
            let intVal = ConfigurationExtensions.getInt configManager "type.int" 0
            let boolVal = ConfigurationExtensions.getBool configManager "type.bool" false
            let floatVal = ConfigurationExtensions.getFloat configManager "type.float" 0.0
            
            stringVal |> should equal "test"
            intVal |> should equal 42
            boolVal |> should equal true
            Math.Abs(floatVal - 3.14) |> should be (lessThan 0.00001)
        }
    
    [<Fact>]
    let ``Should handle invalid configuration keys gracefully`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act & Assert - These should not throw exceptions
            let emptyKey = ConfigurationExtensions.getString configManager "" "default"
            let nullKey = ConfigurationExtensions.getString configManager null "default"
            let specialChars = ConfigurationExtensions.getString configManager "key.with.special!@#$%^&*()" "default"
            
            emptyKey |> should equal "default"
            nullKey |> should equal "default"
            specialChars |> should equal "default"
        }
