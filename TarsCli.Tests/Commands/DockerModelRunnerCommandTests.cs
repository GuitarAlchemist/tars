using System.CommandLine;
using System.CommandLine.Invocation;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Moq;
using TarsCli.Commands;
using TarsCli.Models;
using TarsCli.Services;

namespace TarsCli.Tests.Commands;

public class DockerModelRunnerCommandTests
{
    private readonly Mock<ILogger<DockerModelRunnerCommand>> _loggerMock;
    private readonly Mock<IServiceProvider> _serviceProviderMock;
    private readonly Mock<ConsoleService> _consoleServiceMock;
    private readonly Mock<DockerModelRunnerService> _dockerModelRunnerServiceMock;
    private readonly Mock<GpuService> _gpuServiceMock;
    private readonly Mock<InvocationContext> _invocationContextMock;
    private readonly Mock<BindingContext> _bindingContextMock;

    public DockerModelRunnerCommandTests()
    {
        _loggerMock = new Mock<ILogger<DockerModelRunnerCommand>>();
        _serviceProviderMock = new Mock<IServiceProvider>();
        _consoleServiceMock = new Mock<ConsoleService>();
        _dockerModelRunnerServiceMock = new Mock<DockerModelRunnerService>(
            MockBehavior.Loose,
            Mock.Of<ILogger<DockerModelRunnerService>>(),
            Mock.Of<IConfiguration>(),
            Mock.Of<GpuService>());
        _gpuServiceMock = new Mock<GpuService>(
            MockBehavior.Loose,
            Mock.Of<ILogger<GpuService>>(),
            Mock.Of<IConfiguration>());
        _invocationContextMock = new Mock<InvocationContext>();
        _bindingContextMock = new Mock<BindingContext>();

        // Setup binding context
        _invocationContextMock.Setup(x => x.BindingContext).Returns(_bindingContextMock.Object);
        _bindingContextMock.Setup(x => x.GetService<IServiceProvider>()).Returns(_serviceProviderMock.Object);

        // Setup service provider
        _serviceProviderMock.Setup(x => x.GetService(typeof(ILogger<DockerModelRunnerCommand>))).Returns(_loggerMock.Object);
        _serviceProviderMock.Setup(x => x.GetService(typeof(ConsoleService))).Returns(_consoleServiceMock.Object);
        _serviceProviderMock.Setup(x => x.GetService(typeof(DockerModelRunnerService))).Returns(_dockerModelRunnerServiceMock.Object);
        _serviceProviderMock.Setup(x => x.GetService(typeof(GpuService))).Returns(_gpuServiceMock.Object);
    }

    [Fact]
    public async Task ListModelsCommand_ShouldDisplayModels_WhenModelsAvailable()
    {
        // Arrange
        var command = new DockerModelRunnerCommand();
        var listModelsCommand = command.Subcommands.First(c => c.Name == "list");

        // Setup mocks
        _dockerModelRunnerServiceMock.Setup(x => x.IsAvailable())
            .ReturnsAsync(true);
        _dockerModelRunnerServiceMock.Setup(x => x.GetAvailableModels())
            .ReturnsAsync(new List<ModelInfo> 
            { 
                new ModelInfo { Id = "llama3:8b", OwnedBy = "meta", Created = 1717171717 },
                new ModelInfo { Id = "mistral:7b", OwnedBy = "mistral", Created = 1717171717 }
            });

        // Act
        await listModelsCommand.InvokeAsync(_invocationContextMock.Object);

        // Assert
        _consoleServiceMock.Verify(x => x.WriteInfo("Fetching available models from Docker Model Runner..."), Times.Once);
        _consoleServiceMock.Verify(x => x.WriteSuccess("Found 2 models:"), Times.Once);
        _consoleServiceMock.Verify(x => x.WriteTable(
            It.IsAny<string[]>(),
            It.IsAny<IEnumerable<IEnumerable<string>>>()
        ), Times.Once);
        _invocationContextMock.VerifySet(x => x.ExitCode = 0, Times.Once);
    }

    [Fact]
    public async Task ListModelsCommand_ShouldHandleNoModels()
    {
        // Arrange
        var command = new DockerModelRunnerCommand();
        var listModelsCommand = command.Subcommands.First(c => c.Name == "list");

        // Setup mocks
        _dockerModelRunnerServiceMock.Setup(x => x.IsAvailable())
            .ReturnsAsync(true);
        _dockerModelRunnerServiceMock.Setup(x => x.GetAvailableModels())
            .ReturnsAsync(new List<ModelInfo>());

        // Act
        await listModelsCommand.InvokeAsync(_invocationContextMock.Object);

        // Assert
        _consoleServiceMock.Verify(x => x.WriteWarning("No models found. You may need to pull models first."), Times.Once);
        _invocationContextMock.VerifySet(x => x.ExitCode = 0, Times.Once);
    }

    [Fact]
    public async Task ListModelsCommand_ShouldHandleDockerModelRunnerUnavailable()
    {
        // Arrange
        var command = new DockerModelRunnerCommand();
        var listModelsCommand = command.Subcommands.First(c => c.Name == "list");

        // Setup mocks
        _dockerModelRunnerServiceMock.Setup(x => x.IsAvailable())
            .ReturnsAsync(false);

        // Act
        await listModelsCommand.InvokeAsync(_invocationContextMock.Object);

        // Assert
        _consoleServiceMock.Verify(x => x.WriteError("Docker Model Runner is not available. Make sure it's running and accessible."), Times.Once);
        _invocationContextMock.VerifySet(x => x.ExitCode = 1, Times.Once);
    }

    [Fact]
    public async Task StatusCommand_ShouldDisplayStatus_WhenDockerModelRunnerAvailable()
    {
        // Arrange
        var command = new DockerModelRunnerCommand();
        var statusCommand = command.Subcommands.First(c => c.Name == "status");

        // Setup mocks
        _dockerModelRunnerServiceMock.Setup(x => x.IsAvailable())
            .ReturnsAsync(true);
        _dockerModelRunnerServiceMock.SetupGet(x => x.BaseUrl).Returns("http://localhost:8080");
        _dockerModelRunnerServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3:8b");
        _gpuServiceMock.Setup(x => x.IsGpuAvailable())
            .Returns(true);
        _gpuServiceMock.Setup(x => x.GetGpuInfo())
            .Returns(new List<GpuInfo> 
            { 
                new GpuInfo { Name = "NVIDIA GeForce RTX 3080", MemoryMB = 10240 }
            });
        _gpuServiceMock.Setup(x => x.IsGpuCompatible(It.IsAny<GpuInfo>()))
            .Returns(true);
        _dockerModelRunnerServiceMock.Setup(x => x.GetAvailableModels())
            .ReturnsAsync(new List<ModelInfo> 
            { 
                new ModelInfo { Id = "llama3:8b", OwnedBy = "meta", Created = 1717171717 },
                new ModelInfo { Id = "mistral:7b", OwnedBy = "mistral", Created = 1717171717 }
            });

        // Act
        await statusCommand.InvokeAsync(_invocationContextMock.Object);

        // Assert
        _consoleServiceMock.Verify(x => x.WriteInfo("Checking Docker Model Runner status..."), Times.Once);
        _consoleServiceMock.Verify(x => x.WriteSuccess("Docker Model Runner is available"), Times.Once);
        _consoleServiceMock.Verify(x => x.WriteSuccess("GPU acceleration is available"), Times.Once);
        _consoleServiceMock.Verify(x => x.WriteInfo("Compatible GPU: NVIDIA GeForce RTX 3080 with 10240MB memory"), Times.Once);
        _consoleServiceMock.Verify(x => x.WriteInfo("Available models: 2"), Times.Once);
        _invocationContextMock.VerifySet(x => x.ExitCode = 0, Times.Once);
    }

    [Fact]
    public async Task StatusCommand_ShouldHandleNoGpu()
    {
        // Arrange
        var command = new DockerModelRunnerCommand();
        var statusCommand = command.Subcommands.First(c => c.Name == "status");

        // Setup mocks
        _dockerModelRunnerServiceMock.Setup(x => x.IsAvailable())
            .ReturnsAsync(true);
        _dockerModelRunnerServiceMock.SetupGet(x => x.BaseUrl).Returns("http://localhost:8080");
        _dockerModelRunnerServiceMock.SetupGet(x => x.DefaultModel).Returns("llama3:8b");
        _gpuServiceMock.Setup(x => x.IsGpuAvailable())
            .Returns(false);
        _dockerModelRunnerServiceMock.Setup(x => x.GetAvailableModels())
            .ReturnsAsync(new List<ModelInfo> 
            { 
                new ModelInfo { Id = "llama3:8b", OwnedBy = "meta", Created = 1717171717 }
            });

        // Act
        await statusCommand.InvokeAsync(_invocationContextMock.Object);

        // Assert
        _consoleServiceMock.Verify(x => x.WriteSuccess("Docker Model Runner is available"), Times.Once);
        _consoleServiceMock.Verify(x => x.WriteWarning("GPU acceleration is not available"), Times.Once);
        _invocationContextMock.VerifySet(x => x.ExitCode = 0, Times.Once);
    }

    [Fact]
    public async Task StatusCommand_ShouldHandleDockerModelRunnerUnavailable()
    {
        // Arrange
        var command = new DockerModelRunnerCommand();
        var statusCommand = command.Subcommands.First(c => c.Name == "status");

        // Setup mocks
        _dockerModelRunnerServiceMock.Setup(x => x.IsAvailable())
            .ReturnsAsync(false);

        // Act
        await statusCommand.InvokeAsync(_invocationContextMock.Object);

        // Assert
        _consoleServiceMock.Verify(x => x.WriteError("Docker Model Runner is not available"), Times.Once);
        _consoleServiceMock.Verify(x => x.WriteInfo("Make sure Docker Desktop is running and Docker Model Runner is enabled"), Times.Once);
        _invocationContextMock.VerifySet(x => x.ExitCode = 1, Times.Once);
    }
}
