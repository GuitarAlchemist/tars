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
/// Tests for the VSCodeControlCommand.
/// </summary>
module VSCodeControlCommandTests =
    /// <summary>
    /// Tests that the command validates options correctly.
    /// </summary>
    [<Fact>]
    let ``VSCodeControlCommand validates options correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<VSCodeControlCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = VSCodeControlCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Test with no subcommand
        let options1 = CommandOptions.createDefault()
        Assert.False(command.ValidateOptions(options1))
        
        // Test with invalid subcommand
        let options2 = CommandOptions.createDefault()
        let options2 = CommandOptions.withArguments ["invalid"] options2
        Assert.False(command.ValidateOptions(options2))
        
        // Test open with no file path
        let options3 = CommandOptions.createDefault()
        let options3 = CommandOptions.withArguments ["open"] options3
        Assert.False(command.ValidateOptions(options3))
        
        // Test open with file path
        let options4 = CommandOptions.createDefault()
        let options4 = CommandOptions.withArguments ["open"; "test.cs"] options4
        Assert.True(command.ValidateOptions(options4))
        
        // Test command with no command
        let options5 = CommandOptions.createDefault()
        let options5 = CommandOptions.withArguments ["command"] options5
        Assert.False(command.ValidateOptions(options5))
        
        // Test command with command
        let options6 = CommandOptions.createDefault()
        let options6 = CommandOptions.withArguments ["command"; "workbench.action.files.newFile"] options6
        Assert.True(command.ValidateOptions(options6))
        
        // Test type with no text
        let options7 = CommandOptions.createDefault()
        let options7 = CommandOptions.withArguments ["type"] options7
        Assert.False(command.ValidateOptions(options7))
        
        // Test type with text
        let options8 = CommandOptions.createDefault()
        let options8 = CommandOptions.withArguments ["type"; "Hello, world!"] options8
        Assert.True(command.ValidateOptions(options8))
        
        // Test click with no coordinates
        let options9 = CommandOptions.createDefault()
        let options9 = CommandOptions.withArguments ["click"] options9
        Assert.False(command.ValidateOptions(options9))
        
        // Test click with invalid coordinates
        let options10 = CommandOptions.createDefault()
        let options10 = CommandOptions.withArguments ["click"; "invalid"; "invalid"] options10
        Assert.False(command.ValidateOptions(options10))
        
        // Test click with valid coordinates
        let options11 = CommandOptions.createDefault()
        let options11 = CommandOptions.withArguments ["click"; "100"; "100"] options11
        Assert.True(command.ValidateOptions(options11))
        
        // Test demo
        let options12 = CommandOptions.createDefault()
        let options12 = CommandOptions.withArguments ["demo"] options12
        Assert.True(command.ValidateOptions(options12))
        
        // Test augment with no task
        let options13 = CommandOptions.createDefault()
        let options13 = CommandOptions.withArguments ["augment"] options13
        Assert.False(command.ValidateOptions(options13))
        
        // Test augment with task
        let options14 = CommandOptions.createDefault()
        let options14 = CommandOptions.withArguments ["augment"; "Refactor this code"] options14
        Assert.True(command.ValidateOptions(options14))
    
    /// <summary>
    /// Tests that the open subcommand executes correctly.
    /// </summary>
    [<Fact(Skip = "This test requires VS Code to be installed and would launch a process")>]
    let ``VSCodeControlCommand open executes correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<VSCodeControlCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create a temporary file for testing
        let tempFile = Path.GetTempFileName()
        File.WriteAllText(tempFile, "// Test file")
        
        try
            // Create the command
            let command = VSCodeControlCommand(loggerMock.Object, metascriptExecutorMock.Object)
            
            // Create the options
            let options = CommandOptions.createDefault()
            let options = CommandOptions.withArguments ["open"; tempFile] options
            
            // Act
            let result = command.ExecuteAsync(options).Result
            
            // Assert
            Assert.True(result.Success)
            Assert.Equal(0, result.ExitCode)
            Assert.Contains("File opened successfully", result.Message)
        finally
            // Clean up
            if File.Exists(tempFile) then
                File.Delete(tempFile)
    
    /// <summary>
    /// Tests that the command subcommand executes correctly.
    /// </summary>
    [<Fact(Skip = "This test would interact with VS Code UI")>]
    let ``VSCodeControlCommand command executes correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<VSCodeControlCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = VSCodeControlCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["command"; "workbench.action.files.newFile"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Command executed successfully", result.Message)
    
    /// <summary>
    /// Tests that the type subcommand executes correctly.
    /// </summary>
    [<Fact(Skip = "This test would interact with VS Code UI")>]
    let ``VSCodeControlCommand type executes correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<VSCodeControlCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = VSCodeControlCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["type"; "Hello, world!"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Text typed successfully", result.Message)
    
    /// <summary>
    /// Tests that the click subcommand executes correctly.
    /// </summary>
    [<Fact(Skip = "This test would interact with the mouse")>]
    let ``VSCodeControlCommand click executes correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<VSCodeControlCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = VSCodeControlCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["click"; "100"; "100"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Click performed successfully", result.Message)
