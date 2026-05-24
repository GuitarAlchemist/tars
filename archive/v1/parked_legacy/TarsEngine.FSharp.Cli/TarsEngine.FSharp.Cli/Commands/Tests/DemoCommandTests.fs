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
/// Tests for the DemoCommand.
/// </summary>
module DemoCommandTests =
    /// <summary>
    /// Tests that the command validates options correctly.
    /// </summary>
    [<Fact>]
    let ``DemoCommand validates options correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<DemoCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = DemoCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Test with no subcommand
        let options1 = CommandOptions.createDefault()
        Assert.False(command.ValidateOptions(options1))
        
        // Test with invalid subcommand
        let options2 = CommandOptions.createDefault()
        let options2 = CommandOptions.withArguments ["invalid"] options2
        Assert.False(command.ValidateOptions(options2))
        
        // Test with valid subcommands
        let validSubcommands = [
            "model-providers"
            "all"
            "augment-vscode"
            "a2a"
            "mcp-swarm"
            "self-coding"
            "docker-ai-agent"
            "build-fixes"
        ]
        
        for subcommand in validSubcommands do
            let options = CommandOptions.createDefault()
            let options = CommandOptions.withArguments [subcommand] options
            Assert.True(command.ValidateOptions(options))
    
    /// <summary>
    /// Tests that the all-features subcommand executes correctly.
    /// </summary>
    [<Fact>]
    let ``DemoCommand all-features executes correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<DemoCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = DemoCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["all"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("All features demo completed successfully", result.Message)
    
    /// <summary>
    /// Tests that the model-providers subcommand executes correctly.
    /// </summary>
    [<Fact>]
    let ``DemoCommand model-providers executes correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<DemoCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null; Output = "Model providers demo completed successfully"; Variables = Map.empty }))
        
        // Create the command
        let command = DemoCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["model-providers"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Model providers demo completed successfully", result.Message)
    
    /// <summary>
    /// Tests that the command handles metascript errors.
    /// </summary>
    [<Fact>]
    let ``DemoCommand handles metascript errors`` () =
        // Arrange
        let loggerMock = Mock<ILogger<DemoCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a failed result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = false; ErrorMessage = "Test error"; Output = ""; Variables = Map.empty }))
        
        // Create the command
        let command = DemoCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["model-providers"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.False(result.Success)
        Assert.NotEqual(0, result.ExitCode)
        Assert.Contains("Model providers demo failed", result.Message)
        Assert.Contains("Test error", result.Message)
    
    /// <summary>
    /// Tests that the command handles missing metascript files.
    /// </summary>
    [<Fact>]
    let ``DemoCommand handles missing metascript files`` () =
        // Arrange
        let loggerMock = Mock<ILogger<DemoCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = DemoCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["model-providers"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.False(result.Success)
        Assert.NotEqual(0, result.ExitCode)
        Assert.Contains("Metascript not found", result.Message)
