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
/// Tests for the TestGeneratorCommand.
/// </summary>
module TestGeneratorCommandTests =
    /// <summary>
    /// Tests that the command executes successfully.
    /// </summary>
    [<Fact>]
    let ``TestGeneratorCommand executes successfully`` () =
        // Arrange
        let loggerMock = Mock<ILogger<TestGeneratorCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a successful result
        let testFileContent = """
using Xunit;
using System;

namespace MyProject.Tests
{
    public class MyClassTests
    {
        [Fact]
        public void Test1()
        {
            // Arrange
            var myClass = new MyClass();
            
            // Act
            var result = myClass.MyMethod();
            
            // Assert
            Assert.Equal(42, result);
        }
    }
}
"""
        let variables = Map.ofList [
            ("testFilePath", { Name = "testFilePath"; Value = "src/MyProject/MyClassTests.cs"; Type = typeof<string>; IsReadOnly = false; Metadata = Map.empty })
            ("testFileContent", { Name = "testFileContent"; Value = testFileContent; Type = typeof<string>; IsReadOnly = false; Metadata = Map.empty })
            ("testCount", { Name = "testCount"; Value = 1; Type = typeof<int>; IsReadOnly = false; Metadata = Map.empty })
        ]
        
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null; Output = testFileContent; Variables = variables }))
        
        // Create a temporary file for testing
        let tempFile = Path.GetTempFileName()
        File.WriteAllText(tempFile, "// Test file")
        
        try
            // Create the command
            let command = TestGeneratorCommand(loggerMock.Object, metascriptExecutorMock.Object)
            
            // Create the options
            let options = CommandOptions.createDefault()
            let options = CommandOptions.withOptions (Map.ofList [("file", tempFile); ("generator", "improved-csharp")]) options
            
            // Act
            let result = command.ExecuteAsync(options).Result
            
            // Assert
            Assert.True(result.Success)
            Assert.Equal(0, result.ExitCode)
            Assert.Contains("Test generation completed successfully", result.Message)
        finally
            // Clean up
            if File.Exists(tempFile) then
                File.Delete(tempFile)
    
    /// <summary>
    /// Tests that the command handles errors.
    /// </summary>
    [<Fact>]
    let ``TestGeneratorCommand handles errors`` () =
        // Arrange
        let loggerMock = Mock<ILogger<TestGeneratorCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a failed result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = false; ErrorMessage = "Test error"; Output = ""; Variables = Map.empty }))
        
        // Create a temporary file for testing
        let tempFile = Path.GetTempFileName()
        File.WriteAllText(tempFile, "// Test file")
        
        try
            // Create the command
            let command = TestGeneratorCommand(loggerMock.Object, metascriptExecutorMock.Object)
            
            // Create the options
            let options = CommandOptions.createDefault()
            let options = CommandOptions.withOptions (Map.ofList [("file", tempFile)]) options
            
            // Act
            let result = command.ExecuteAsync(options).Result
            
            // Assert
            Assert.False(result.Success)
            Assert.NotEqual(0, result.ExitCode)
            Assert.Contains("Test generation failed", result.Message)
            Assert.Contains("Test error", result.Message)
        finally
            // Clean up
            if File.Exists(tempFile) then
                File.Delete(tempFile)
    
    /// <summary>
    /// Tests that the command validates options correctly.
    /// </summary>
    [<Fact>]
    let ``TestGeneratorCommand validates options correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<TestGeneratorCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = TestGeneratorCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Test with no file
        let options1 = CommandOptions.createDefault()
        Assert.False(command.ValidateOptions(options1))
        
        // Test with non-existent file
        let options2 = CommandOptions.createDefault()
        let options2 = CommandOptions.withOptions (Map.ofList [("file", "non-existent-file.cs")]) options2
        Assert.False(command.ValidateOptions(options2))
        
        // Test with existing file
        let tempFile = Path.GetTempFileName()
        File.WriteAllText(tempFile, "// Test file")
        
        try
            let options3 = CommandOptions.createDefault()
            let options3 = CommandOptions.withOptions (Map.ofList [("file", tempFile)]) options3
            Assert.True(command.ValidateOptions(options3))
        finally
            // Clean up
            if File.Exists(tempFile) then
                File.Delete(tempFile)
