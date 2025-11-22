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
/// Tests for the TotAutoImprovementCommand.
/// </summary>
module TotAutoImprovementCommandTests =
    /// <summary>
    /// Tests that the command executes successfully.
    /// </summary>
    [<Fact>]
    let ``TotAutoImprovementCommand executes successfully`` () =
        // Arrange
        let loggerMock = Mock<ILogger<TotAutoImprovementCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null }))
        
        // Create the command
        let command = TotAutoImprovementCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Tree-of-Thought auto-improvement pipeline completed successfully", result.Message)
    
    /// <summary>
    /// Tests that the command handles errors.
    /// </summary>
    [<Fact>]
    let ``TotAutoImprovementCommand handles errors`` () =
        // Arrange
        let loggerMock = Mock<ILogger<TotAutoImprovementCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a failed result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = false; ErrorMessage = "Test error" }))
        
        // Create the command
        let command = TotAutoImprovementCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.False(result.Success)
        Assert.NotEqual(0, result.ExitCode)
        Assert.Contains("Tree-of-Thought auto-improvement pipeline failed", result.Message)
        Assert.Contains("Test error", result.Message)
    
    /// <summary>
    /// Tests that the command handles different targets.
    /// </summary>
    [<Fact>]
    let ``TotAutoImprovementCommand handles different targets`` () =
        // Arrange
        let loggerMock = Mock<ILogger<TotAutoImprovementCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null }))
        
        // Create the command
        let command = TotAutoImprovementCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options with a specific target
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withOptions (Map.ofList [("target", "code-quality")]) options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Tree-of-Thought auto-improvement pipeline completed successfully", result.Message)
        
        // Verify that the correct metascript was executed
        metascriptExecutorMock.Verify(fun m -> m.ExecuteMetascriptAsync("TarsCli/Metascripts/Improvements/tot_code_quality_analyzer.tars", It.IsAny<obj>()), Times.Once())
