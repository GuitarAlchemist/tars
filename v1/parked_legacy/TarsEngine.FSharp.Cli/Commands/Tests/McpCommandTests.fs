namespace TarsEngine.FSharp.Cli.Commands.Tests

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Metascript
open TarsEngine.FSharp.Cli.Commands
open Xunit
open Moq

/// <summary>
/// Tests for the McpCommand.
/// </summary>
module McpCommandTests =
    /// <summary>
    /// Tests that the command validates options correctly.
    /// </summary>
    [<Fact>]
    let ``McpCommand validates options correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<McpCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = McpCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Test with no subcommand
        let options1 = CommandOptions.createDefault()
        Assert.False(command.ValidateOptions(options1))
        
        // Test with invalid subcommand
        let options2 = CommandOptions.createDefault()
        let options2 = CommandOptions.withArguments ["invalid"] options2
        Assert.False(command.ValidateOptions(options2))
        
        // Test start subcommand
        let options3 = CommandOptions.createDefault()
        let options3 = CommandOptions.withArguments ["start"] options3
        Assert.True(command.ValidateOptions(options3))
        
        // Test stop subcommand
        let options4 = CommandOptions.createDefault()
        let options4 = CommandOptions.withArguments ["stop"] options4
        Assert.True(command.ValidateOptions(options4))
        
        // Test status subcommand
        let options5 = CommandOptions.createDefault()
        let options5 = CommandOptions.withArguments ["status"] options5
        Assert.True(command.ValidateOptions(options5))
        
        // Test config subcommand
        let options6 = CommandOptions.createDefault()
        let options6 = CommandOptions.withArguments ["config"] options6
        Assert.True(command.ValidateOptions(options6))
        
        // Test execute subcommand with no command
        let options7 = CommandOptions.createDefault()
        let options7 = CommandOptions.withArguments ["execute"] options7
        Assert.False(command.ValidateOptions(options7))
        
        // Test execute subcommand with command
        let options8 = CommandOptions.createDefault()
        let options8 = CommandOptions.withArguments ["execute"; "echo Hello, World!"] options8
        Assert.True(command.ValidateOptions(options8))
        
        // Test code subcommand with no file path
        let options9 = CommandOptions.createDefault()
        let options9 = CommandOptions.withArguments ["code"] options9
        Assert.False(command.ValidateOptions(options9))
        
        // Test code subcommand with file path but no content
        let options10 = CommandOptions.createDefault()
        let options10 = CommandOptions.withArguments ["code"; "path/to/file.cs"] options10
        Assert.False(command.ValidateOptions(options10))
        
        // Test code subcommand with file path and content
        let options11 = CommandOptions.createDefault()
        let options11 = CommandOptions.withArguments ["code"; "path/to/file.cs"; "public class MyClass { }"] options11
        Assert.True(command.ValidateOptions(options11))
        
        // Test augment subcommand
        let options12 = CommandOptions.createDefault()
        let options12 = CommandOptions.withArguments ["augment"] options12
        Assert.True(command.ValidateOptions(options12))
        
        // Test collaborate subcommand with no subcommand
        let options13 = CommandOptions.createDefault()
        let options13 = CommandOptions.withArguments ["collaborate"] options13
        Assert.False(command.ValidateOptions(options13))
        
        // Test collaborate subcommand with invalid subcommand
        let options14 = CommandOptions.createDefault()
        let options14 = CommandOptions.withArguments ["collaborate"; "invalid"] options14
        Assert.False(command.ValidateOptions(options14))
        
        // Test collaborate subcommand with start subcommand
        let options15 = CommandOptions.createDefault()
        let options15 = CommandOptions.withArguments ["collaborate"; "start"] options15
        Assert.True(command.ValidateOptions(options15))
        
        // Test collaborate subcommand with stop subcommand
        let options16 = CommandOptions.createDefault()
        let options16 = CommandOptions.withArguments ["collaborate"; "stop"] options16
        Assert.True(command.ValidateOptions(options16))
        
        // Test collaborate subcommand with status subcommand
        let options17 = CommandOptions.createDefault()
        let options17 = CommandOptions.withArguments ["collaborate"; "status"] options17
        Assert.True(command.ValidateOptions(options17))
    
    /// <summary>
    /// Tests that the start subcommand executes successfully.
    /// </summary>
    [<Fact>]
    let ``McpCommand start executes successfully`` () =
        // Arrange
        let loggerMock = Mock<ILogger<McpCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null; Output = "MCP server started successfully"; Variables = Map.empty }))
        
        // Create the command
        let command = McpCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["start"] options
        let options = CommandOptions.withOptions (Map.ofList [("port", "8999"); ("model-provider", "DockerModelRunner")]) options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("MCP server started successfully", result.Message)
    
    /// <summary>
    /// Tests that the stop subcommand executes successfully.
    /// </summary>
    [<Fact>]
    let ``McpCommand stop executes successfully`` () =
        // Arrange
        let loggerMock = Mock<ILogger<McpCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null; Output = "MCP server stopped successfully"; Variables = Map.empty }))
        
        // Create the command
        let command = McpCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["stop"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("MCP server stopped successfully", result.Message)
    
    /// <summary>
    /// Tests that the command handles errors.
    /// </summary>
    [<Fact>]
    let ``McpCommand handles errors`` () =
        // Arrange
        let loggerMock = Mock<ILogger<McpCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a failed result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = false; ErrorMessage = "Test error"; Output = ""; Variables = Map.empty }))
        
        // Create the command
        let command = McpCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["start"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.False(result.Success)
        Assert.NotEqual(0, result.ExitCode)
        Assert.Contains("Failed to start MCP server", result.Message)
        Assert.Contains("Test error", result.Message)
    
    /// <summary>
    /// Tests that the command handles missing metascript files.
    /// </summary>
    [<Fact>]
    let ``McpCommand handles missing metascript files`` () =
        // Arrange
        let loggerMock = Mock<ILogger<McpCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = McpCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["start"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.False(result.Success)
        Assert.NotEqual(0, result.ExitCode)
        Assert.Contains("Metascript not found", result.Message)
