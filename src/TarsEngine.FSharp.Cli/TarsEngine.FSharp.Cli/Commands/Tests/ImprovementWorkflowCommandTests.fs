namespace TarsEngine.FSharp.Cli.Commands.Tests

open System
open System.IO
open System.Threading.Tasks
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Metascript
open TarsEngine.FSharp.Cli.Commands
open Xunit
open Moq

/// <summary>
/// Tests for the ImprovementWorkflowCommand.
/// </summary>
module ImprovementWorkflowCommandTests =
    /// <summary>
    /// Tests that the command executes successfully.
    /// </summary>
    [<Fact>]
    let ``ImprovementWorkflowCommand executes successfully`` () =
        // Arrange
        let loggerMock = Mock<ILogger<ImprovementWorkflowCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create a mock improvement
        let improvement = {|
            Id = "imp-123"
            Name = "Refactor MyClass"
            Category = "Refactoring"
            PriorityScore = 0.85
            Status = "Pending"
            AffectedFiles = [| "src/MyProject/MyClass.cs" |]
        |}
        
        // Create a list of improvements
        let improvements = [improvement]
        
        // Create variables
        let variables = Map.ofList [
            ("improvements", { Name = "improvements"; Value = improvements; Type = typeof<obj list>; IsReadOnly = false; Metadata = Map.empty })
        ]
        
        // Setup the metascript executor to return a successful result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = true; ErrorMessage = null; Output = "Workflow completed successfully"; Variables = variables }))
        
        // Create a temporary directory for testing
        let tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString())
        Directory.CreateDirectory(tempDir)
        
        try
            // Create the command
            let command = ImprovementWorkflowCommand(loggerMock.Object, metascriptExecutorMock.Object)
            
            // Create the options
            let options = CommandOptions.createDefault()
            let options = CommandOptions.withOptions (Map.ofList [
                ("path", tempDir)
                ("recursive", "true")
                ("pattern", "*.cs;*.fs")
                ("max-improvements", "10")
                ("execute", "false")
                ("dry-run", "true")
                ("format", "json")
                ("verbose", "true")
            ]) options
            
            // Act
            let result = command.ExecuteAsync(options).Result
            
            // Assert
            Assert.True(result.Success)
            Assert.Equal(0, result.ExitCode)
            Assert.Contains("Improvement generation workflow completed successfully", result.Message)
        finally
            // Clean up
            if Directory.Exists(tempDir) then
                Directory.Delete(tempDir, true)
    
    /// <summary>
    /// Tests that the command handles errors.
    /// </summary>
    [<Fact>]
    let ``ImprovementWorkflowCommand handles errors`` () =
        // Arrange
        let loggerMock = Mock<ILogger<ImprovementWorkflowCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Setup the metascript executor to return a failed result
        metascriptExecutorMock.Setup(fun m -> m.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<obj>()))
            .Returns(Task.FromResult({ Success = false; ErrorMessage = "Test error"; Output = ""; Variables = Map.empty }))
        
        // Create a temporary directory for testing
        let tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString())
        Directory.CreateDirectory(tempDir)
        
        try
            // Create the command
            let command = ImprovementWorkflowCommand(loggerMock.Object, metascriptExecutorMock.Object)
            
            // Create the options
            let options = CommandOptions.createDefault()
            let options = CommandOptions.withOptions (Map.ofList [("path", tempDir)]) options
            
            // Act
            let result = command.ExecuteAsync(options).Result
            
            // Assert
            Assert.False(result.Success)
            Assert.NotEqual(0, result.ExitCode)
            Assert.Contains("Improvement generation workflow failed", result.Message)
            Assert.Contains("Test error", result.Message)
        finally
            // Clean up
            if Directory.Exists(tempDir) then
                Directory.Delete(tempDir, true)
    
    /// <summary>
    /// Tests that the command validates options correctly.
    /// </summary>
    [<Fact>]
    let ``ImprovementWorkflowCommand validates options correctly`` () =
        // Arrange
        let loggerMock = Mock<ILogger<ImprovementWorkflowCommand>>()
        let metascriptExecutorMock = Mock<IMetascriptExecutor>()
        
        // Create the command
        let command = ImprovementWorkflowCommand(loggerMock.Object, metascriptExecutorMock.Object)
        
        // Test with no path
        let options1 = CommandOptions.createDefault()
        Assert.False(command.ValidateOptions(options1))
        
        // Test with non-existent path
        let options2 = CommandOptions.createDefault()
        let options2 = CommandOptions.withOptions (Map.ofList [("path", "non-existent-path")]) options2
        Assert.False(command.ValidateOptions(options2))
        
        // Test with existing directory
        let tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString())
        Directory.CreateDirectory(tempDir)
        
        try
            let options3 = CommandOptions.createDefault()
            let options3 = CommandOptions.withOptions (Map.ofList [("path", tempDir)]) options3
            Assert.True(command.ValidateOptions(options3))
        finally
            // Clean up
            if Directory.Exists(tempDir) then
                Directory.Delete(tempDir, true)
        
        // Test with existing file
        let tempFile = Path.GetTempFileName()
        
        try
            let options4 = CommandOptions.createDefault()
            let options4 = CommandOptions.withOptions (Map.ofList [("path", tempFile)]) options4
            Assert.True(command.ValidateOptions(options4))
        finally
            // Clean up
            if File.Exists(tempFile) then
                File.Delete(tempFile)
