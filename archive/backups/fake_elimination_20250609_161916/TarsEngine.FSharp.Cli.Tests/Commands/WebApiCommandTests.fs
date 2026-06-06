namespace TarsEngine.FSharp.Cli.Tests.Commands

open System
open System.IO
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Tests.TestHelpers

/// Comprehensive tests for WebApiCommand
module WebApiCommandTests =
    
    [<Fact>]
    let ``WebApiCommand should implement ICommand interface correctly`` () =
        // Arrange
        let logger = createMockLogger<WebApiCommand>()
        let command = WebApiCommand(logger)
        
        // Act & Assert
        (command :> ICommand).Name |> should equal "webapi"
        (command :> ICommand).Description |> should not' (be EmptyString)
        (command :> ICommand).Usage |> should not' (be EmptyString)
        (command :> ICommand).Examples |> should not' (be Empty)
    
    [<Fact>]
    let ``WebApiCommand should validate options correctly`` () =
        // Arrange
        let logger = createMockLogger<WebApiCommand>()
        let command = WebApiCommand(logger)
        let options = createCommandOptions ["rest"; "TestAPI"]
        
        // Act
        let isValid = (command :> ICommand).ValidateOptions(options)
        
        // Assert
        isValid |> should be True
    
    [<Fact>]
    let ``WebApiCommand should show help when no arguments provided`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions []
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert
            Assertions.assertSuccess result
            result.Message |> should equal "Help displayed"
        }
    
    [<Fact>]
    let ``WebApiCommand should handle REST API generation`` () =
        task {
            // Arrange
            use tempDir = createTempDirectory()
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions ["rest"; "TestAPI"]
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert
            Assertions.assertSuccess result
            result.Message |> should equal "REST API generated"
            
            // Verify output directory was created
            let outputDir = "output/webapi/testapi-rest"
            Assertions.assertDirectoryExists outputDir
        }
    
    [<Fact>]
    let ``WebApiCommand should handle GraphQL server generation`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions ["graphql"; "TestGraphQL"]
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert
            Assertions.assertSuccess result
            result.Message |> should equal "GraphQL server generated"
            
            // Verify output directory was created
            let outputDir = "output/webapi/testgraphql-graphql"
            Assertions.assertDirectoryExists outputDir
        }
    
    [<Fact>]
    let ``WebApiCommand should handle GraphQL client generation`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions ["client"; "TestClient"; "http://localhost:5000/graphql"]
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert
            Assertions.assertSuccess result
            result.Message |> should equal "GraphQL client generated"
            
            // Verify output directory was created
            let outputDir = "output/webapi/testclient-client"
            Assertions.assertDirectoryExists outputDir
        }
    
    [<Fact>]
    let ``WebApiCommand should handle hybrid API generation`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions ["hybrid"; "TestHybrid"]
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert
            Assertions.assertSuccess result
            result.Message |> should equal "Hybrid API generated"
            
            // Verify output directory was created
            let outputDir = "output/webapi/testhybrid-hybrid"
            Assertions.assertDirectoryExists outputDir
        }
    
    [<Fact>]
    let ``WebApiCommand should handle demo execution`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions ["demo"]
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert
            Assertions.assertSuccess result
            result.Message |> should equal "Demo completed"
        }
    
    [<Fact>]
    let ``WebApiCommand should list closure types`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions ["list"]
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert
            Assertions.assertSuccess result
            result.Message |> should equal "Closure types listed"
        }
    
    [<Fact>]
    let ``WebApiCommand should handle unknown subcommands gracefully`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions ["unknown"; "command"]
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert
            Assertions.assertFailure result
            Assertions.assertErrorMessage "Unknown subcommand: unknown" result
        }
    
    [<Fact>]
    let ``WebApiCommand should log appropriate messages`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions ["unknown"]
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert
            Assertions.assertLogContains "Invalid webapi command" logger
        }
    
    [<Fact>]
    let ``WebApiCommand should handle exceptions gracefully`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            
            // Create options that might cause an exception
            let options = createCommandOptions ["rest"] // Missing API name
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert - Should handle gracefully, not throw
            result |> should not' (be null)
        }
    
    [<Fact>]
    let ``WebApiCommand performance should be acceptable`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions []
            
            // Act & Assert - Should complete within 5 seconds
            let! result = Performance.assertExecutionTimeAsync (TimeSpan.FromSeconds(5)) (fun () ->
                (command :> ICommand).ExecuteAsync(options)
            )
            
            Assertions.assertSuccess result
        }
    
    [<Fact>]
    let ``WebApiCommand memory usage should be reasonable`` () =
        // Arrange
        let logger = createMockLogger<WebApiCommand>()
        
        // Act & Assert - Should use less than 50MB
        let command = Memory.assertMemoryUsage (50L * 1024L * 1024L) (fun () ->
            WebApiCommand(logger)
        )
        
        command |> should not' (be null)
    
    [<Fact>]
    let ``WebApiCommand should create valid output structure`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            let options = createCommandOptions ["rest"; "ValidationAPI"]
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert
            Assertions.assertSuccess result
            
            // Verify output structure
            let outputDir = "output/webapi/validationapi-rest"
            Assertions.assertDirectoryExists outputDir
            
            // Check for README file (placeholder implementation)
            let readmeFile = Path.Combine(outputDir, "README.md")
            Assertions.assertFileExists readmeFile
            
            // Verify README content
            let content = File.ReadAllText(readmeFile)
            Assertions.assertContains "ValidationAPI REST API" content
        }
    
    [<Fact>]
    let ``WebApiCommand should handle concurrent executions`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            
            // Create multiple concurrent tasks
            let tasks = [
                for i in 1..5 ->
                    let options = createCommandOptions ["rest"; $"ConcurrentAPI{i}"]
                    (command :> ICommand).ExecuteAsync(options)
            ]
            
            // Act
            let! results = Task.WhenAll(tasks)
            
            // Assert
            results |> Array.iter Assertions.assertSuccess
            results |> should haveLength 5
        }
    
    [<Fact>]
    let ``WebApiCommand should validate API names`` () =
        task {
            // Arrange
            let logger = createMockLogger<WebApiCommand>()
            let command = WebApiCommand(logger)
            
            // Test with empty API name
            let options = createCommandOptions ["rest"; ""]
            
            // Act
            let! result = (command :> ICommand).ExecuteAsync(options)
            
            // Assert - Should handle empty names gracefully
            result |> should not' (be null)
        }
