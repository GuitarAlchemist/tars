using System.CommandLine;
using System.CommandLine.Parsing;
using Microsoft.Extensions.Logging;
using Moq;
using TarsCli.Commands;
using TarsCli.Services;
using Xunit;

namespace TarsCli.Tests.Commands
{
    public class AutoImproveCommandTests
    {
        private readonly Mock<ILogger<AutoImproveCommand>> _loggerMock;
        private readonly Mock<DslService> _dslServiceMock;
        private readonly Mock<ConsoleService> _consoleServiceMock;
        private readonly AutoImproveCommand _command;

        public AutoImproveCommandTests()
        {
            _loggerMock = new Mock<ILogger<AutoImproveCommand>>();
            _dslServiceMock = new Mock<DslService>();
            _consoleServiceMock = new Mock<ConsoleService>();
            _command = new AutoImproveCommand(_loggerMock.Object, _dslServiceMock.Object, _consoleServiceMock.Object);
        }

        [Fact]
        public void Constructor_ShouldCreateCommandWithCorrectName()
        {
            // Assert
            Assert.Equal("auto-improve", _command.Name);
            Assert.Equal("Run autonomous improvement using metascripts", _command.Description);
        }

        [Fact]
        public void Constructor_ShouldAddRequiredOptions()
        {
            // Arrange
            var timeOption = _command.Options.FirstOrDefault(o => o.Name == "time" || o.Aliases.Contains("--time"));
            var modelOption = _command.Options.FirstOrDefault(o => o.Name == "model" || o.Aliases.Contains("--model"));
            var statusOption = _command.Options.FirstOrDefault(o => o.Name == "status" || o.Aliases.Contains("--status"));
            var stopOption = _command.Options.FirstOrDefault(o => o.Name == "stop" || o.Aliases.Contains("--stop"));
            var reportOption = _command.Options.FirstOrDefault(o => o.Name == "report" || o.Aliases.Contains("--report"));

            // Assert
            Assert.NotNull(timeOption);
            Assert.NotNull(modelOption);
            Assert.NotNull(statusOption);
            Assert.NotNull(stopOption);
            Assert.NotNull(reportOption);
        }

        [Fact]
        public async Task StartAsync_ShouldExecuteMetascript()
        {
            // Arrange
            _dslServiceMock.Setup(m => m.RunDslFileAsync(It.IsAny<string>(), It.IsAny<bool>()))
                .ReturnsAsync(0);

            // Act
            await InvokeCommandHandler("--time", "30", "--model", "llama3");

            // Assert
            _dslServiceMock.Verify(m => m.RunDslFileAsync(It.IsAny<string>()), Times.Once);
            _consoleServiceMock.Verify(m => m.WriteSuccess(It.IsAny<string>()), Times.Once);
        }

        [Fact]
        public async Task StartAsync_ShouldHandleErrors()
        {
            // Arrange
            _dslServiceMock.Setup(m => m.RunDslFileAsync(It.IsAny<string>(), It.IsAny<bool>()))
                .ThrowsAsync(new Exception("Test exception"));

            // Act
            await InvokeCommandHandler("--time", "30", "--model", "llama3");

            // Assert
            _consoleServiceMock.Verify(m => m.WriteError(It.IsAny<string>()), Times.Once);
        }

        private async Task InvokeCommandHandler(params string[] args)
        {
            var parser = new Parser(_command);
            var parseResult = parser.Parse(args);
            await parseResult.InvokeAsync();
        }
    }
}
