using System;
using System.CommandLine;
using System.CommandLine.Invocation;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Moq;
using TarsEngine.Metascripts;
using TarsCli.Commands;
using Xunit;

namespace TarsCli.Tests.Commands
{
    public class AutoImprovementCommandTests
    {
        [Fact]
        public void Constructor_ShouldCreateCommand()
        {
            // Arrange
            var loggerMock = new Mock<ILogger<AutoImprovementCommand>>();
            var executorMock = new Mock<IMetascriptExecutor>();
            
            // Act
            var command = new AutoImprovementCommand(loggerMock.Object, executorMock.Object);
            
            // Assert
            Assert.Equal("auto-improve", command.Name);
            Assert.NotNull(command.Handler);
        }
        
        [Fact]
        public void Constructor_NullLogger_ShouldThrowArgumentNullException()
        {
            // Arrange
            var executorMock = new Mock<IMetascriptExecutor>();
            
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new AutoImprovementCommand(null, executorMock.Object));
        }
        
        [Fact]
        public void Constructor_NullExecutor_ShouldThrowArgumentNullException()
        {
            // Arrange
            var loggerMock = new Mock<ILogger<AutoImprovementCommand>>();
            
            // Act & Assert
            Assert.Throws<ArgumentNullException>(() => new AutoImprovementCommand(loggerMock.Object, null));
        }
        
        [Theory]
        [InlineData("all", "TarsCli/Metascripts/Improvements/auto_improvement_pipeline.tars")]
        [InlineData("code-quality", "TarsCli/Metascripts/Improvements/code_quality_analyzer.tars")]
        [InlineData("documentation", "TarsCli/Metascripts/Improvements/documentation_generator.tars")]
        [InlineData("tests", "TarsCli/Metascripts/Improvements/test_generator.tars")]
        public async Task RunAutoImprovement_ValidTarget_ShouldExecuteCorrectMetascript(string target, string expectedMetascriptPath)
        {
            // Arrange
            var loggerMock = new Mock<ILogger<AutoImprovementCommand>>();
            var executorMock = new Mock<IMetascriptExecutor>();
            
            executorMock
                .Setup(e => e.ExecuteMetascriptAsync(It.IsAny<string>(), It.IsAny<object>()))
                .ReturnsAsync(new MetascriptExecutionResult { Success = true });
            
            var command = new AutoImprovementCommand(loggerMock.Object, executorMock.Object);
            
            // Create a mock file system to make File.Exists return true
            var fileSystemMock = new Mock<IFileSystem>();
            fileSystemMock.Setup(fs => fs.FileExists(It.IsAny<string>())).Returns(true);
            
            // Act
            await InvokeCommandHandler(command, target, false, false);
            
            // Assert
            executorMock.Verify(e => e.ExecuteMetascriptAsync(
                It.Is<string>(s => s == expectedMetascriptPath),
                It.IsAny<object>()),
                Times.Once);
        }
        
        [Fact]
        public async Task RunAutoImprovement_InvalidTarget_ShouldLogError()
        {
            // Arrange
            var loggerMock = new Mock<ILogger<AutoImprovementCommand>>();
            var executorMock = new Mock<IMetascriptExecutor>();
            
            var command = new AutoImprovementCommand(loggerMock.Object, executorMock.Object);
            
            // Act
            await InvokeCommandHandler(command, "invalid-target", false, false);
            
            // Assert
            loggerMock.Verify(
                x => x.Log(
                    LogLevel.Error,
                    It.IsAny<EventId>(),
                    It.Is<It.IsAnyType>((v, t) => v.ToString().Contains("Unknown target: invalid-target")),
                    It.IsAny<Exception>(),
                    It.IsAny<Func<It.IsAnyType, Exception, string>>()),
                Times.Once);
            
            executorMock.Verify(e => e.ExecuteMetascriptAsync(
                It.IsAny<string>(),
                It.IsAny<object>()),
                Times.Never);
        }
        
        private async Task InvokeCommandHandler(Command command, string target, bool dryRun, bool verbose)
        {
            // Get the handler
            var handler = command.Handler as CommandHandler;
            
            // Create a context
            var context = new InvocationContext(null);
            
            // Set the values in the context
            context.BindingContext.SetValueForOption(command.Options[0], target);
            context.BindingContext.SetValueForOption(command.Options[1], dryRun);
            context.BindingContext.SetValueForOption(command.Options[2], verbose);
            
            // Invoke the handler
            await handler.InvokeAsync(context);
        }
    }
    
    // Interface for mocking File.Exists
    public interface IFileSystem
    {
        bool FileExists(string path);
    }
}
