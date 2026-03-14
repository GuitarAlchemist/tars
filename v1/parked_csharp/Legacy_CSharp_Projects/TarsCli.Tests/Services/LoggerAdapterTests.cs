using Microsoft.Extensions.Logging;
using Moq;
using TarsCli.Services;
using Xunit;

namespace TarsCli.Tests.Services
{
    public class LoggerAdapterTests
    {
        [Fact]
        public void IsEnabled_DelegatesToUnderlyingLogger()
        {
            // Arrange
            var mockLogger = new Mock<ILogger>();
            mockLogger.Setup(l => l.IsEnabled(It.IsAny<LogLevel>())).Returns(true);
            var adapter = new LoggerAdapter<LoggerAdapterTests>(mockLogger.Object);

            // Act
            var result = adapter.IsEnabled(LogLevel.Information);

            // Assert
            Assert.True(result);
            mockLogger.Verify(l => l.IsEnabled(LogLevel.Information), Times.Once);
        }

        [Fact]
        public void BeginScope_DelegatesToUnderlyingLogger()
        {
            // Arrange
            var mockLogger = new Mock<ILogger>();
            var mockDisposable = new Mock<IDisposable>();
            mockLogger.Setup(l => l.BeginScope(It.IsAny<string>())).Returns(mockDisposable.Object);
            var adapter = new LoggerAdapter<LoggerAdapterTests>(mockLogger.Object);

            // Act
            var result = ((ILogger)adapter).BeginScope("test scope");

            // Assert
            Assert.Same(mockDisposable.Object, result);
            mockLogger.Verify(l => l.BeginScope("test scope"), Times.Once);
        }

        [Fact]
        public void Log_DelegatesToUnderlyingLogger()
        {
            // Arrange
            var mockLogger = new Mock<ILogger>();
            var adapter = new LoggerAdapter<LoggerAdapterTests>(mockLogger.Object);
            var eventId = new EventId(1, "Test");
            var state = "Test state";
            var exception = new Exception("Test exception");
            Func<string, Exception?, string> formatter = (s, e) => $"{s} {e?.Message}";

            // Act
            ((ILogger)adapter).Log(LogLevel.Error, eventId, state, exception, formatter);

            // Assert
            mockLogger.Verify(l => l.Log(
                LogLevel.Error,
                eventId,
                state,
                exception,
                It.IsAny<Func<string, Exception?, string>>()),
                Times.Once);
        }

        [Fact]
        public void Constructor_ThrowsArgumentNullException_WhenLoggerIsNull()
        {
            // Act & Assert
            var exception = Assert.Throws<ArgumentNullException>(() => new LoggerAdapter<LoggerAdapterTests>(null!));
            Assert.Equal("logger", exception.ParamName);
        }
    }
}
