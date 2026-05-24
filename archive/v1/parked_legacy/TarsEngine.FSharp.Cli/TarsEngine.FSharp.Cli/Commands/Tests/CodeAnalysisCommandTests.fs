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
/// Tests for the CodeAnalysisCommand.
/// </summary>
module CodeAnalysisCommandTests =
    /// <summary>
    /// Tests that the command executes successfully.
    /// </summary>
    [<Fact>]
    let ``CodeAnalysisCommand executes successfully`` () =
        // Arrange
        let loggerMock = Mock<ILogger<CodeAnalysisCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null }))
        
        // Create the command
        let command = CodeAnalysisCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Code analysis completed successfully", result.Message)
    
    /// <summary>
    /// Tests that the command handles errors.
    /// </summary>
    [<Fact>]
    let ``CodeAnalysisCommand handles errors`` () =
        // Arrange
        let loggerMock = Mock<ILogger<CodeAnalysisCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a failed result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = false; ErrorMessage = "Test error" }))
        
        // Create the command
        let command = CodeAnalysisCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.False(result.Success)
        Assert.NotEqual(0, result.ExitCode)
        Assert.Contains("Code analysis failed", result.Message)
        Assert.Contains("Test error", result.Message)
    
    /// <summary>
    /// Tests that the command handles different analysis types.
    /// </summary>
    [<Fact>]
    let ``CodeAnalysisCommand handles different analysis types`` () =
        // Arrange
        let loggerMock = Mock<ILogger<CodeAnalysisCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null }))
        
        // Create the command
        let command = CodeAnalysisCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options with a specific analysis type
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withOptions (Map.ofList [("type", "complexity")]) options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Code analysis completed successfully", result.Message)
        
        // Verify that the correct metascript was executed
        metascriptExecutorMock.Verify(fun m -> m.ExecuteMetascriptAsync("TarsCli/Metascripts/Analysis/code_complexity_analysis.tars", It.IsAny<obj>()), Times.Once())
