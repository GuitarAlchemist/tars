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
/// Tests for the SelfCodingCommand.
/// </summary>
module SelfCodingCommandTests =
    /// <summary>
    /// Tests that the start subcommand executes successfully.
    /// </summary>
    [<Fact>]
    let ``SelfCodingCommand start executes successfully`` () =
        // Arrange
        let loggerMock = Mock<ILogger<SelfCodingCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null; Output = "Self-coding process started successfully" }))
        
        // Create the command
        let command = SelfCodingCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["start"] options
        let options = CommandOptions.withOptions (Map.ofList [("target", "src/MyProject")]) options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Self-coding process started successfully", result.Message)
    
    /// <summary>
    /// Tests that the stop subcommand executes successfully.
    /// </summary>
    [<Fact>]
    let ``SelfCodingCommand stop executes successfully`` () =
        // Arrange
        let loggerMock = Mock<ILogger<SelfCodingCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null; Output = "Self-coding process stopped successfully" }))
        
        // Create the command
        let command = SelfCodingCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["stop"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Self-coding process stopped successfully", result.Message)
    
    /// <summary>
    /// Tests that the status subcommand executes successfully.
    /// </summary>
    [<Fact>]
    let ``SelfCodingCommand status executes successfully`` () =
        // Arrange
        let loggerMock = Mock<ILogger<SelfCodingCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        let statusOutput = """
Status: Running
Current Stage: analyzing
Progress: 5/10 files processed
Start Time: 2023-06-01T12:00:00Z
Elapsed Time: 00:05:23
"""
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null; Output = statusOutput }))
        
        // Create the command
        let command = SelfCodingCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Create the options
        let options = CommandOptions.createDefault()
        let options = CommandOptions.withArguments ["status"] options
        
        // Act
        let result = command.ExecuteAsync(options).Result
        
        // Assert
        Assert.True(result.Success)
        Assert.Equal(0, result.ExitCode)
        Assert.Contains("Status: Running", result.Message)
    
    /// <summary>
    /// Tests that the improve subcommand executes successfully.
    /// </summary>
    [<Fact>]
    let ``SelfCodingCommand improve executes successfully`` () =
        // Arrange
        let loggerMock = Mock<ILogger<SelfCodingCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null; Output = "File improvement completed successfully" }))
        
        // Create a temporary file for testing
        let tempFile = Path.GetTempFileName()
        File.WriteAllText(tempFile, "// Test file")
        
        try
            // Create the command
            let command = SelfCodingCommand(loggerMock.Object, metascriptExecutorMock.Object)
            
            // Create the options
            let options = CommandOptions.createDefault()
            let options = CommandOptions.withArguments ["improve"; tempFile] options
            let options = CommandOptions.withOptions (Map.ofList [("model", "llama3"); ("auto-apply", "true")]) options
            
            // Act
            let result = command.ExecuteAsync(options).Result
            
            // Assert
            Assert.True(result.Success)
            Assert.Equal(0, result.ExitCode)
            Assert.Contains("File improvement completed successfully", result.Message)
        finally
            // Clean up
            if File.Exists(tempFile) then
                File.Delete(tempFile)
    
    /// <summary>
    /// Tests that the command validates options correctly.
    /// </summary>
    [<Fact>]
    let ``SelfCodingCommand validates options correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<SelfCodingCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = SelfCodingCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Test with no subcommand
        let options1 = CommandOptions.createDefault()
        Assert.False(command.ValidateOptions(options1))
        
        // Test with invalid subcommand
        let options2 = CommandOptions.createDefault()
        let options2 = CommandOptions.withArguments ["invalid"] options2
        Assert.False(command.ValidateOptions(options2))
        
        // Test start with no target
        let options3 = CommandOptions.createDefault()
        let options3 = CommandOptions.withArguments ["start"] options3
        Assert.False(command.ValidateOptions(options3))
        
        // Test start with target
        let options4 = CommandOptions.createDefault()
        let options4 = CommandOptions.withArguments ["start"] options4
        let options4 = CommandOptions.withOptions (Map.ofList [("target", "src/MyProject")]) options4
        Assert.True(command.ValidateOptions(options4))
        
        // Test improve with no file
        let options5 = CommandOptions.createDefault()
        let options5 = CommandOptions.withArguments ["improve"] options5
        Assert.False(command.ValidateOptions(options5))
        
        // Test improve with file
        let options6 = CommandOptions.createDefault()
        let options6 = CommandOptions.withArguments ["improve"; "src/MyProject/MyFile.cs"] options6
        Assert.True(command.ValidateOptions(options6))
        
        // Test stop
        let options7 = CommandOptions.createDefault()
        let options7 = CommandOptions.withArguments ["stop"] options7
        Assert.True(command.ValidateOptions(options7))
        
        // Test status
        let options8 = CommandOptions.createDefault()
        let options8 = CommandOptions.withArguments ["status"] options8
        Assert.True(command.ValidateOptions(options8))
        
        // Test setup
        let options9 = CommandOptions.createDefault()
        let options9 = CommandOptions.withArguments ["setup"] options9
        Assert.True(command.ValidateOptions(options9))
        
        // Test cleanup
        let options10 = CommandOptions.createDefault()
        let options10 = CommandOptions.withArguments ["cleanup"] options10
        Assert.True(command.ValidateOptions(options10))
